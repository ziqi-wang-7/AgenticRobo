# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fast-Slow procedural memory evolution for LIBERO Flash plans.

System 1 (Flash) executes an anchor-relative procedural plan with zero LLM
tokens.  When Flash encounters an anomaly (missed grasp, servo stall, missing
anchor, primitive error, or an unfinished goal), it first tries any matching
learned recovery branch at zero LLM cost.  If no learned branch resolves the
episode, System 2 (a slow LLM planner such as Codex) takes over in-place
without resetting the simulator.  When System 2 succeeds, its post-handover
recovery actions are compiled into an anchor-relative recovery branch and
atomically patched back into the task's procedural memory so subsequent seeds
recover at zero LLM cost.  If a task had no prior Flash plan at all (cold
start), a solved slow-planner episode is compiled into a new Flash plan.
"""

from __future__ import annotations

import contextlib
import hashlib
import json
import math
import os
import shutil
import tempfile
import threading
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from robots.libero.flash.generate import (
    MOVE_ACTIONS,
    _attach_moves,
    _fallback_anchors,
    _merge_anchors,
    extract_relations,
    generate_flash_plan,
)
from robots.libero.flash.replay import _CELL, load, replay_with_evolution
from robots.libero.recovery_evidence import motion_alert

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows fallback
    fcntl = None  # type: ignore[assignment]

SCHEMA_VERSION = 2
PRIMITIVE_ACTIONS = frozenset(
    {
        "move_to",
        "move_pose",
        "pi0_pick",
        "pi0_doubled",
        "release",
        "set_gripper",
        "rotate_wrist",
        "rotate_pitch",
        "execute_action",
    }
)
# Maximum spatial slack (meters) when matching a live anchor/eef to a learned
# branch's observed trigger bounds.
DEFAULT_SPATIAL_MARGIN_M = 0.14
# Branches with at least this many attempts and zero successes are demoted so
# bad patches do not cause negative transfer on future seeds.
DEMOTE_MIN_ATTEMPTS = 3

_THREAD_LOCKS_GUARD = threading.Lock()
_THREAD_LOCKS: dict[Path, threading.Lock] = {}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _parse_cell_tag(cell_tag: str) -> tuple[str, str, int, int]:
    match = _CELL.match(cell_tag)
    if not match:
        raise ValueError(
            f"cannot read the cell tag {cell_tag!r}; "
            "expected <family>_<suite>_t<task>_s<seed>"
        )
    family, suite, task, seed = match.groups()
    return family, suite, int(task), int(seed)


def plan_name_for_cell(cell_tag: str) -> str:
    """Return the seed-independent Flash plan stem ``<family>_<suite>_t<task>``."""
    family, suite, task, _ = _parse_cell_tag(cell_tag)
    return f"{family}_{suite}_t{task}"


@contextlib.contextmanager
def _store_lock(store_dir: Path) -> Iterator[None]:
    """Serialize read-modify-write updates to one procedural memory store."""
    resolved = store_dir.resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    with _THREAD_LOCKS_GUARD:
        tlock = _THREAD_LOCKS.setdefault(resolved, threading.Lock())
    with tlock:
        lock_file = resolved / ".flash_evolve.lock"
        if fcntl is None:
            yield
            return
        with lock_file.open("a+") as handle:
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        tmp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _action_signature(
    actions: Sequence[Mapping[str, Any]], anchors: Sequence[Mapping[str, Any]] = ()
) -> str:
    """Stable digest of a branch's anchored action sequence for deduplication."""
    canonical = []
    for item in actions:
        args = dict(item.get("arguments") or {})
        entry: dict[str, Any] = {"action": item.get("action"), "arguments": args}
        if item.get("anchor"):
            entry["anchor"] = item["anchor"]
            entry["offset"] = item["offset"]
            # Replay replaces only the recorded XY with anchor + offset.
            entry["z"] = args.pop("xyz")[2]
        canonical.append(entry)
    surveys = sorted((a["phrase"], a.get("camera", "agentview")) for a in anchors)
    blob = json.dumps([canonical, surveys], sort_keys=True, ensure_ascii=False).encode(
        "utf-8"
    )
    return hashlib.sha256(blob).hexdigest()[:12]


def _bounds_from_xy(xy: Sequence[float] | None) -> list[list[float]] | None:
    if xy is None or len(xy) < 2:
        return None
    x, y = round(float(xy[0]), 4), round(float(xy[1]), 4)
    return [[x, x], [y, y]]


def _merge_bounds(
    existing: list[list[float]] | None, incoming: list[list[float]] | None
) -> list[list[float]] | None:
    if existing is None:
        return incoming
    if incoming is None:
        return existing
    return [
        [min(existing[0][0], incoming[0][0]), max(existing[0][1], incoming[0][1])],
        [min(existing[1][0], incoming[1][0]), max(existing[1][1], incoming[1][1])],
    ]


def _within_bounds(
    xy: Sequence[float] | None,
    bounds: list[list[float]] | None,
    margin: float = DEFAULT_SPATIAL_MARGIN_M,
) -> tuple[bool, float]:
    """Return whether ``xy`` falls within ``bounds`` expanded by ``margin``."""
    if bounds is None:
        return True, 0.0
    if xy is None:
        return False, math.inf
    x, y = float(xy[0]), float(xy[1])
    dx = max(bounds[0][0] - x, 0.0, x - bounds[0][1])
    dy = max(bounds[1][0] - y, 0.0, y - bounds[1][1])
    dist = math.hypot(dx, dy)
    return dist <= margin, dist


def bootstrap_plan_from_corpus(
    memory_root: Path,
    store_dir: Path,
    cell_tag: str,
    note: Callable[[str], None] = lambda _: None,
) -> bool:
    """Ensure ``store_dir`` holds a writable Flash plan when source assets exist.

    Search order:
    1. Already present in ``store_dir`` (previously evolved or copied).
    2. Pre-built ``flash/`` or ``task_card/`` under ``memory_root`` (copied into
       ``store_dir`` so the upstream corpus stays untouched).
    3. Seed-0 ``task-specific/<ref_tag>.json`` + ``<ref_tag>_recipe.jsonl``
       under ``memory_root`` (compiled on demand via :func:`generate_flash_plan`).

    Returns:
        ``True`` if ``store_dir`` now contains both ``_plan.json`` and
        ``_anchors.json`` for ``cell_tag``, else ``False`` (cold start).
    """
    plan_name = plan_name_for_cell(cell_tag)
    plan_file = store_dir / f"{plan_name}_plan.json"
    anchors_file = store_dir / f"{plan_name}_anchors.json"
    with _store_lock(store_dir):
        if plan_file.is_file() and anchors_file.is_file():
            return True

        # Keep incomplete copies/compilations out of the live store. Frozen
        # evaluators may be reading it without acquiring a writable lock.
        with tempfile.TemporaryDirectory(dir=store_dir) as tmp:
            staging = Path(tmp)
            for source in (memory_root / "flash", memory_root / "task_card"):
                if (source / plan_file.name).is_file() and (
                    source / anchors_file.name
                ).is_file():
                    shutil.copyfile(source / plan_file.name, staging / plan_file.name)
                    shutil.copyfile(
                        source / anchors_file.name, staging / anchors_file.name
                    )
                    note(f"bootstrapped {plan_name} from {source}")
                    break
            else:
                task_specific = memory_root / "task-specific"
                ref_tag = f"{plan_name}_s0"
                audit_path = task_specific / f"{ref_tag}.json"
                recipe_path = task_specific / f"{ref_tag}_recipe.jsonl"
                if not (audit_path.is_file() and recipe_path.is_file()):
                    return False
                try:
                    generate_flash_plan(audit_path, recipe_path, staging)
                except ValueError as exc:
                    note(f"could not compile seed-0 recipe {recipe_path.name}: {exc}")
                    return False
                trace_name = f"{plan_name}_trace.md"
                (staging / trace_name).replace(store_dir / trace_name)
                note(f"compiled initial Flash plan {plan_name} from {recipe_path.name}")
            # Plan existence is the publication marker for the completed pair.
            if not anchors_file.is_file():
                plan_file.unlink(missing_ok=True)
            (staging / anchors_file.name).replace(anchors_file)
            (staging / plan_file.name).replace(plan_file)
            return True


def load_evolving_plan(store_dir: Path, plan_name: str) -> dict[str, Any]:
    """Load a Flash plan together with its schema-v2 recovery branches."""
    raw_doc = json.loads((store_dir / f"{plan_name}_plan.json").read_text("utf-8"))
    program = load(store_dir, plan_name, plan_doc=raw_doc)
    program["schema_version"] = int(raw_doc.get("schema_version", 1))
    program["recovery_branches"] = list(raw_doc.get("recovery_branches") or [])
    program["language"] = str(raw_doc.get("language") or "")
    program["source"] = raw_doc.get("source") or {}
    program["plan_name"] = plan_name
    return program


def match_recovery_branch(
    program: Mapping[str, Any],
    failure: Mapping[str, Any],
    *,
    tried_ids: set[str] | None = None,
    spatial_margin: float = DEFAULT_SPATIAL_MARGIN_M,
) -> dict[str, Any] | None:
    """Select the highest-confidence recovery branch matching ``failure``."""
    branches = program.get("recovery_branches") or []
    if not branches:
        return None
    tried = tried_ids or set()
    failure_kind = failure.get("failure_kind")
    failed_step = failure.get("failed_step_idx")
    failed_action = failure.get("failed_action")
    anchor = failure.get("anchor")
    anchor_xy = failure.get("anchor_xy")
    eef_xy = failure.get("eef_xy")

    scored: list[tuple[tuple[int, int, float, float, str], dict[str, Any]]] = []
    for branch in branches:
        branch_id = str(branch.get("branch_id") or "")
        if not branch_id or branch_id in tried:
            continue
        if not branch.get("actions"):
            continue
        stats = branch.get("stats") or {}
        attempts = int(stats.get("attempts", 0))
        successes = int(stats.get("successes", 0))
        if attempts >= DEMOTE_MIN_ATTEMPTS and successes == 0:
            continue

        trigger = branch.get("trigger") or {}
        if trigger.get("failure_kind") != failure_kind:
            continue

        step_exact = int(trigger.get("failed_step_idx") == failed_step)
        action_match = int(
            not trigger.get("failed_action")
            or trigger.get("failed_action") == failed_action
        )
        anchor_match = int(not trigger.get("anchor") or trigger.get("anchor") == anchor)
        if not (action_match and anchor_match):
            continue
        if trigger.get("failed_step_idx") is not None and not step_exact:
            continue
        if not step_exact and not (
            trigger.get("failed_action") and trigger.get("anchor")
        ):
            continue

        bounds = trigger.get("spatial_bounds") or {}
        ok_anchor, dist_anchor = _within_bounds(
            anchor_xy, bounds.get("anchor_xy_range"), margin=spatial_margin
        )
        ok_eef, dist_eef = _within_bounds(
            eef_xy, bounds.get("eef_xy_range"), margin=spatial_margin
        )
        if not (ok_anchor and ok_eef):
            continue

        # Laplace-smoothed success rate; prioritize exact step, higher rate,
        # closer spatial trigger distance.
        rate = (successes + 1.0) / (attempts + 2.0)
        key = (
            step_exact,
            anchor_match,
            round(rate, 4),
            -round(dist_anchor + dist_eef, 4),
            branch_id,
        )
        scored.append((key, branch))

    if not scored:
        return None
    scored.sort(key=lambda pair: pair[0], reverse=True)
    return scored[0][1]


def _extract_primitive_commands(
    state: Any, *, after_step: int = -1
) -> tuple[list[dict[str, Any]], list[int]]:
    """Extract executed primitive commands from ``EnvState`` after ``after_step``."""
    commands: list[dict[str, Any]] = []
    steps: list[int] = []
    for record in state.records():
        if record.step_idx <= after_step:
            continue
        cmd = record.command
        if not isinstance(cmd, dict):
            continue
        action = cmd.get("action")
        result = record.result or {}
        if (
            action not in PRIMITIVE_ACTIONS
            or result.get("error")
            or result.get("interrupted")
        ):
            continue
        if not record.terminated and (
            (action == "pi0_pick" and result.get("success") is False)
            or motion_alert(record)
        ):
            continue
        commands.append(dict(cmd))
        steps.append(int(record.step_idx))
        if record.terminated:
            break
    return commands, steps


def _live_anchors_as_list(
    live_anchors: Mapping[str, Sequence[float]],
    existing_anchors: Sequence[Mapping[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Convert runtime ``{phrase: xy}`` map into anchor records for ``_attach_moves``."""
    locator_by_phrase = {
        str(a["phrase"]): str(a.get("locator") or "segment")
        for a in (existing_anchors or ())
        if isinstance(a, Mapping) and a.get("phrase")
    }
    records: list[dict[str, Any]] = []
    for phrase, xy in live_anchors.items():
        if xy is None or len(xy) < 2:
            continue
        pt = [float(xy[0]), float(xy[1])]
        records.append(
            {
                "phrase": str(phrase),
                "locator": locator_by_phrase.get(str(phrase), "segment"),
                "camera": "agentview",
                "score": None,
                "readings": [[pt[0], pt[1], 0.0]],
                "median_xy": pt,
                "z_top": 0.0,
                "z_span": 0.0,
            }
        )
    return records


def compile_recovery_branch(
    *,
    state: Any,
    handover_step: int,
    failure: Mapping[str, Any],
    live_anchors: Mapping[str, Sequence[float]],
    cell_tag: str,
    slow_planner: str,
    existing_anchors: Sequence[Mapping[str, Any]] | None = None,
    measured_anchors: Sequence[Mapping[str, Any]] = (),
) -> dict[str, Any] | None:
    """Compile post-handover slow-planner recovery steps into a Flash branch."""
    raw_commands, step_indices = _extract_primitive_commands(
        state, after_step=handover_step
    )
    if not raw_commands:
        return None
    anchors = _live_anchors_as_list(live_anchors, existing_anchors)
    measured_phrases = {a["phrase"] for a in measured_anchors}
    anchors = [a for a in anchors if a["phrase"] not in measured_phrases]
    anchors.extend(dict(a) for a in measured_anchors)
    anchored_actions = _attach_moves(raw_commands, anchors)
    used_phrases = {action.get("anchor") for action in anchored_actions}
    branch_anchors = [dict(a) for a in measured_anchors if a["phrase"] in used_phrases]
    sig = _action_signature(anchored_actions, branch_anchors)
    spatial_bounds: dict[str, Any] = {}
    anchor_bounds = _bounds_from_xy(failure.get("anchor_xy"))
    if anchor_bounds is not None:
        spatial_bounds["anchor_xy_range"] = anchor_bounds
    eef_bounds = _bounds_from_xy(failure.get("eef_xy"))
    if eef_bounds is not None:
        spatial_bounds["eef_xy_range"] = eef_bounds

    return {
        "branch_id": f"rb-{sig}",
        "signature": sig,
        "trigger": {
            "failure_kind": failure.get("failure_kind"),
            "failed_step_idx": failure.get("failed_step_idx"),
            "failed_action": failure.get("failed_action"),
            "anchor": failure.get("anchor"),
            "spatial_bounds": spatial_bounds,
        },
        "mode": "complete_episode",
        "anchors": branch_anchors,
        "actions": anchored_actions,
        "provenance": {
            "cell_tag": cell_tag,
            "created_at": _utc_now(),
            "slow_planner": slow_planner,
            "source_step_range": [step_indices[0], step_indices[-1]],
            "handover_step": handover_step,
        },
        "stats": {
            "attempts": 0,
            "successes": 0,
            "failures": 0,
        },
    }


def patch_plan_with_branch(
    store_dir: Path,
    plan_name: str,
    branch: Mapping[str, Any],
) -> tuple[Path, str]:
    """Atomically insert or merge a recovery branch into ``<plan_name>_plan.json``."""
    plan_path = store_dir / f"{plan_name}_plan.json"
    with _store_lock(store_dir):
        doc = json.loads(plan_path.read_text(encoding="utf-8"))
        doc["schema_version"] = SCHEMA_VERSION
        branches: list[dict[str, Any]] = list(doc.get("recovery_branches") or [])
        history: list[dict[str, Any]] = list(doc.get("evolution_history") or [])

        incoming = dict(branch)
        incoming_trigger = incoming.get("trigger") or {}
        signature = _action_signature(
            incoming["actions"], incoming.get("anchors") or ()
        )
        merged_id: str | None = None
        for existing in branches:
            ex_trigger = existing.get("trigger") or {}
            if _action_signature(
                existing["actions"], existing.get("anchors") or ()
            ) == signature and all(
                ex_trigger.get(key) == incoming_trigger.get(key)
                for key in (
                    "failure_kind",
                    "failed_step_idx",
                    "failed_action",
                    "anchor",
                )
            ):
                ex_bounds = dict(ex_trigger.get("spatial_bounds") or {})
                in_bounds = incoming_trigger.get("spatial_bounds") or {}
                for bound_key in ("anchor_xy_range", "eef_xy_range"):
                    merged_range = _merge_bounds(
                        ex_bounds.get(bound_key), in_bounds.get(bound_key)
                    )
                    if merged_range is not None:
                        ex_bounds[bound_key] = merged_range
                ex_trigger["spatial_bounds"] = ex_bounds
                existing["trigger"] = ex_trigger
                merged_id = str(existing["branch_id"])
                break

        if merged_id is None:
            branch_id = str(incoming["branch_id"])
            existing_ids = {str(b.get("branch_id")) for b in branches}
            suffix = 1
            while branch_id in existing_ids:
                branch_id = f"{incoming['branch_id']}-{suffix}"
                suffix += 1
            incoming["branch_id"] = branch_id
            branches.append(incoming)
            action_label = "added_branch"
        else:
            branch_id = merged_id
            action_label = "merged_branch_bounds"

        doc["recovery_branches"] = branches
        history.append(
            {
                "timestamp": _utc_now(),
                "event": action_label,
                "branch_id": branch_id,
                "cell_tag": (incoming.get("provenance") or {}).get("cell_tag"),
                "failure_kind": incoming_trigger.get("failure_kind"),
                "failed_step_idx": incoming_trigger.get("failed_step_idx"),
            }
        )
        doc["evolution_history"] = history
        _atomic_write_json(plan_path, doc)
    return plan_path, branch_id


def record_branch_outcome(
    store_dir: Path,
    plan_name: str,
    branch_id: str,
    *,
    succeeded: bool,
    cell_tag: str,
) -> None:
    """Update empirical success/failure counts for one executed recovery branch."""
    plan_path = store_dir / f"{plan_name}_plan.json"
    if not plan_path.is_file():
        return
    with _store_lock(store_dir):
        doc = json.loads(plan_path.read_text(encoding="utf-8"))
        changed = False
        for branch in doc.get("recovery_branches") or []:
            if branch.get("branch_id") == branch_id:
                stats = dict(branch.get("stats") or {})
                stats["attempts"] = int(stats.get("attempts", 0)) + 1
                if succeeded:
                    stats["successes"] = int(stats.get("successes", 0)) + 1
                else:
                    stats["failures"] = int(stats.get("failures", 0)) + 1
                branch["stats"] = stats
                changed = True
                break
        if changed:
            history = list(doc.get("evolution_history") or [])
            history.append(
                {
                    "timestamp": _utc_now(),
                    "event": "branch_hit_success"
                    if succeeded
                    else "branch_hit_failure",
                    "branch_id": branch_id,
                    "cell_tag": cell_tag,
                }
            )
            doc["evolution_history"] = history
            _atomic_write_json(plan_path, doc)


def compile_new_plan_from_state(
    *,
    state: Any,
    cell_tag: str,
    store_dir: Path,
    slow_planner: str,
    measured_anchors: Sequence[Mapping[str, Any]] | None = None,
) -> dict[str, Any] | None:
    """Compile a full solved slow-planner episode into a new Flash plan (cold start)."""
    family, suite, task, _ = _parse_cell_tag(cell_tag)
    plan_name = f"{family}_{suite}_t{task}"
    raw_commands, _ = _extract_primitive_commands(state, after_step=-1)
    if not raw_commands:
        return None
    records = state.records()
    language = ""
    for rec in records:
        lang = (rec.extras or {}).get("task_language")
        if isinstance(lang, str) and lang.strip():
            language = lang.strip()
            break
    if not language:
        return None

    graph = extract_relations(family, language)
    measured = [dict(a) for a in (measured_anchors or ())]
    inferred = _fallback_anchors(graph, raw_commands)
    anchors = _merge_anchors(measured, inferred)
    # Default compiled cold-start anchors to SAM3 segment so they work even
    # without an external Molmo endpoint.
    for anchor in anchors:
        if anchor.get("locator") == "molmo" and not measured:
            anchor["locator"] = "segment"
    plan = _attach_moves(raw_commands, anchors)
    task_name = f"{family}/{suite}_t{task}"
    provenance = {
        "cell": cell_tag,
        "compiled_by": "flash_evolve_cold_start",
        "slow_planner": slow_planner,
        "created_at": _utc_now(),
    }
    plan_doc = {
        "schema_version": SCHEMA_VERSION,
        "task": task_name,
        "language": language,
        "source": provenance,
        "goal_graph": graph.as_dict(),
        "plan": plan,
        "recovery_branches": [],
        "evolution_history": [
            {
                "timestamp": _utc_now(),
                "event": "compiled_new_plan",
                "cell_tag": cell_tag,
                "slow_planner": slow_planner,
                "actions": len(plan),
                "anchors": len(anchors),
            }
        ],
    }
    anchors_doc = {
        "task": task_name,
        "language": language,
        "source": provenance,
        "anchors": anchors,
        "prompts_without_readings": [],
    }
    with _store_lock(store_dir):
        # Another episode may have learned and patched this task while the
        # slow planner was running. Keep the first complete plan and its history.
        plan_path = store_dir / f"{plan_name}_plan.json"
        anchors_path = store_dir / f"{plan_name}_anchors.json"
        if plan_path.is_file() and anchors_path.is_file():
            return None
        if not anchors_path.is_file():
            plan_path.unlink(missing_ok=True)
        _atomic_write_json(anchors_path, anchors_doc)
        _atomic_write_json(plan_path, plan_doc)
    return {
        "plan_name": plan_name,
        "anchors": len(anchors),
        "actions": len(plan),
        "attached_moves": sum("anchor" in item for item in plan),
        "total_moves": sum(item["action"] in MOVE_ACTIONS for item in plan),
    }


def build_handover_message(
    base_user_message: str,
    *,
    plan_name: str,
    replay_outcome: Mapping[str, Any],
) -> str:
    """Compose a concise in-place continuation message for the slow planner."""
    failure = replay_outcome.get("failure") or {}
    handover_step = replay_outcome.get("handover_step", 0)
    live_anchors = replay_outcome.get("live_anchors") or {}
    remaining = replay_outcome.get("remaining_plan") or []
    executed = replay_outcome.get("executed_trace") or []

    anchor_lines = (
        "\n".join(
            f"  - {phrase}: [{xy[0]:+.4f}, {xy[1]:+.4f}]"
            for phrase, xy in sorted(live_anchors.items())
        )
        or "  (none)"
    )
    executed_tail = executed[-5:]
    executed_lines = (
        "\n".join(
            f"  - step {item.get('plan_idx')}: {item.get('action')} "
            f"({item.get('status', 'ok')})"
            for item in executed_tail
        )
        or "  (none)"
    )
    remaining_head = remaining[:6]
    remaining_lines = (
        "\n".join(
            f"  - plan[{item.get('plan_idx')}]: {item.get('action')} "
            f"anchor={item.get('anchor')!r}"
            for item in remaining_head
        )
        or "  (plan exhausted)"
    )

    handover_block = (
        "\n\n=== SYSTEM-1 FLASH HANDOVER (IN-PLACE SINGLE-ATTEMPT RECOVERY) ===\n"
        f"The fast procedural memory ({plan_name}) already executed up to env step "
        f"{handover_step} in this LIVE episode and paused for your recovery.\n"
        "DO NOT call `reset`. DO NOT re-execute already completed subtasks.\n"
        f"- Failure kind: {failure.get('failure_kind', 'unknown')}\n"
        f"- Failed plan step: {failure.get('failed_step_idx')} "
        f"({failure.get('failed_action')})\n"
        f"- Detail: {failure.get('detail', 'n/a')}\n"
        f"- Live localized anchors from Flash:\n{anchor_lines}\n"
        f"- Recent Flash actions executed:\n{executed_lines}\n"
        f"- Remaining unexecuted Flash plan steps:\n{remaining_lines}\n\n"
        "IMMEDIATE NEXT STEP:\n"
        '1. Call `view_env_state({"step": -1})` to inspect the exact live scene '
        f"(step {handover_step}) and gripper state.\n"
        "2. Recover in-place from the current physical state to reach "
        "`terminated == true` (e.g., re-localize shifted objects, adjust pre-grasp "
        "pose/offset, switch to `move_pose` near IK boundaries, or complete the "
        "remaining placement/articulation).\n"
        "3. Keep your recovery sequence clean and concise — if you solve the task, "
        "your recovery actions will be compiled into procedural memory for future seeds.\n"
        "=====================================================================\n"
    )
    return base_user_message + handover_block


def _extract_segment_anchors_from_state(
    state: Any, *, from_step: int = 0, exclude: set[str] | None = None
) -> list[dict[str, Any]]:
    """Read the per-step JSON artifacts written by the read-only SAM3 tool."""
    anchors: list[dict[str, Any]] = []
    seen_phrases: set[str] = set()
    for record in state.records():
        if record.step_idx < from_step:
            continue
        for name in sorted(record.artifacts):
            if not (name.startswith("segment_") and name.endswith(".json")) or (
                exclude and name in exclude
            ):
                continue
            result = state.load(name, step=record.step_idx)
            xyz = result.get("world_xyz")
            prompt = str(result.get("prompt") or "").strip()
            if (
                result.get("found") is not True
                or result.get("mode") != "text"
                or not prompt
                or prompt in seen_phrases
                or not isinstance(xyz, list)
                or len(xyz) < 3
                or not all(
                    isinstance(v, (int, float)) and math.isfinite(v) for v in xyz[:3]
                )
            ):
                continue
            seen_phrases.add(prompt)
            anchors.append(
                {
                    "phrase": prompt,
                    "locator": "segment",
                    "camera": str(result.get("camera") or "agentview"),
                    "score": result.get("score"),
                    "readings": [[float(xyz[0]), float(xyz[1]), float(xyz[2])]],
                    "median_xy": [float(xyz[0]), float(xyz[1])],
                    "z_top": float(xyz[2]),
                    "z_span": 0.0,
                }
            )
    return anchors


@contextlib.contextmanager
def _record_evolution_errors(
    trace: dict[str, Any], note: Callable[[str], None]
) -> Iterator[None]:
    """Keep the environment outcome when an optional memory update fails."""
    try:
        yield
    except (OSError, ValueError) as exc:
        trace["evolution_error"] = str(exc)
        note(f"Flash memory update failed: {exc}")


def run_flash_evolve(
    toolkit: Any,
    cell_tag: str,
    *,
    slow_solve: Callable[[str], Any],
    user_message: str,
    flash_memory_dir: Path | str,
    evolve_enabled: bool = True,
    slow_planner_name: str = "codex",
    output_dir: Path | str | None = None,
    note: Callable[[str], None] = lambda _: None,
) -> dict[str, Any]:
    """Run System-1 Flash + System-2 Handover + Closed-Loop Procedural Evolution."""
    started = time.time()
    plan_name = plan_name_for_cell(cell_tag)
    store_dir = Path(flash_memory_dir).expanduser().resolve()
    memory_root = Path(toolkit.memory.root).resolve()
    program = None
    if evolve_enabled:
        if bootstrap_plan_from_corpus(memory_root, store_dir, cell_tag, note=note):
            program = load_evolving_plan(store_dir, plan_name)
    elif all(
        (store_dir / f"{plan_name}_{suffix}.json").is_file()
        for suffix in ("plan", "anchors")
    ):
        program = load_evolving_plan(store_dir, plan_name)
    else:
        # A frozen evaluation can still use seed-0 assets without modifying
        # its (possibly read-only) input store.
        with tempfile.TemporaryDirectory() as staging:
            if bootstrap_plan_from_corpus(
                memory_root, Path(staging), cell_tag, note=note
            ):
                program = load_evolving_plan(Path(staging), plan_name)
    molmo = getattr(getattr(toolkit, "primitives", None), "molmo_client", None)

    trace_summary: dict[str, Any] = {
        "cell_tag": cell_tag,
        "plan_name": plan_name,
        "flash_memory_dir": str(store_dir),
        "evolve_enabled": bool(evolve_enabled),
        "slow_planner": slow_planner_name,
        "had_initial_plan": program is not None,
        "execution_mode": "failed",
        "evolved_action": "none",
        "branch_attempts": [],
        "fast_actions": 0,
        "slow_result": None,
    }

    if program is None:
        note(f"no Flash plan for {plan_name}; running cold-start {slow_planner_name}")
        slow_result = slow_solve(user_message)
        solved = bool(toolkit.solved())
        trace_summary["slow_result"] = {
            "error": getattr(slow_result, "error", None),
            "stats": dict(getattr(slow_result, "stats", None) or {}),
        }
        trace_summary["execution_mode"] = (
            "slow_cold_start_solved" if solved else "slow_cold_start_failed"
        )
        if solved and evolve_enabled:
            with _record_evolution_errors(trace_summary, note):
                compiled = compile_new_plan_from_state(
                    state=toolkit.state,
                    cell_tag=cell_tag,
                    store_dir=store_dir,
                    slow_planner=slow_planner_name,
                    measured_anchors=_extract_segment_anchors_from_state(toolkit.state),
                )
                if compiled is not None:
                    trace_summary["evolved_action"] = "compiled_new_plan"
                    trace_summary["compiled_plan"] = compiled
                    note(
                        f"compiled new Flash plan {plan_name} "
                        f"({compiled['actions']} actions, {compiled['anchors']} anchors)"
                    )
        trace_summary["done"] = solved
        trace_summary["elapsed_s"] = round(time.time() - started, 2)
        _write_trace(output_dir, trace_summary)
        return {
            "done": solved,
            "program": plan_name,
            "trace": trace_summary,
            "slow_planner_result": slow_result,
        }

    # Stage 1: Fast-path execution (with 0-token learned recovery branches).
    note(
        f"running System-1 Flash for {plan_name} "
        f"({len(program['plan'])} steps, "
        f"{len(program.get('recovery_branches', []))} learned branches)"
    )
    replay_outcome = replay_with_evolution(
        toolkit,
        molmo,
        program,
        note=note,
        branch_matcher=match_recovery_branch,
        allow_sam3_fallback=True,
    )
    trace_summary["fast_actions"] = int(replay_outcome.get("executed_steps", 0))
    trace_summary["anchors"] = int(replay_outcome.get("anchors", 0))
    trace_summary["branch_attempts"] = list(replay_outcome.get("branch_attempts") or [])
    trace_summary["failure"] = replay_outcome.get("failure")
    trace_summary["handover_step"] = replay_outcome.get("handover_step")

    # Persist empirical hit/miss stats for any recovery branches tried in Stage 1.
    if evolve_enabled:
        for attempt in trace_summary["branch_attempts"]:
            with _record_evolution_errors(trace_summary, note):
                record_branch_outcome(
                    store_dir,
                    plan_name,
                    str(attempt["branch_id"]),
                    succeeded=bool(attempt["succeeded"]),
                    cell_tag=cell_tag,
                )

    if replay_outcome.get("done") or replay_outcome.get("mode") == "flash_truncated":
        trace_summary["done"] = bool(replay_outcome.get("done"))
        trace_summary["execution_mode"] = str(replay_outcome.get("mode", "flash_fast"))
        if (
            trace_summary["branch_attempts"]
            and evolve_enabled
            and not trace_summary.get("evolution_error")
        ):
            trace_summary["evolved_action"] = "updated_branch_stats"
        trace_summary["elapsed_s"] = round(time.time() - started, 2)
        _write_trace(output_dir, trace_summary)
        return {
            "done": trace_summary["done"],
            "program": plan_name,
            "trace": trace_summary,
            "slow_planner_result": None,
        }

    # Stage 2: Just-in-Time Slow-System Handover (same live toolkit, NO RESET).
    handover_step = int(replay_outcome.get("handover_step", 0))
    handover_msg = build_handover_message(
        user_message,
        plan_name=plan_name,
        replay_outcome=replay_outcome,
    )
    note(
        f"Flash handover at env step {handover_step} "
        f"({(replay_outcome.get('failure') or {}).get('failure_kind', 'unknown')}) "
        f"-> invoking {slow_planner_name}"
    )
    preexisting_artifacts = (
        toolkit.state.get(handover_step).artifacts if evolve_enabled else set()
    )
    slow_result = slow_solve(handover_msg)
    solved = bool(toolkit.solved())
    trace_summary["slow_result"] = {
        "error": getattr(slow_result, "error", None),
        "stats": dict(getattr(slow_result, "stats", None) or {}),
    }
    trace_summary["done"] = solved
    trace_summary["execution_mode"] = (
        "slow_handover_rescue" if solved else "slow_handover_failed"
    )

    # Stage 3: Compile slow-planner rescue into a procedural recovery branch.
    if solved and evolve_enabled and replay_outcome.get("failure"):
        # A tried branch may already have moved or grasped the object. Its
        # rescue suffix cannot run directly from the main plan's trigger.
        if replay_outcome.get("failure_step", handover_step) != handover_step:
            trace_summary["evolution_skipped"] = "branch_changed_state_before_handover"
        else:
            with _record_evolution_errors(trace_summary, note):
                branch = compile_recovery_branch(
                    state=toolkit.state,
                    handover_step=handover_step,
                    failure=replay_outcome["failure"],
                    live_anchors=replay_outcome.get("live_anchors") or {},
                    cell_tag=cell_tag,
                    slow_planner=slow_planner_name,
                    measured_anchors=_extract_segment_anchors_from_state(
                        toolkit.state,
                        from_step=handover_step,
                        exclude=preexisting_artifacts,
                    ),
                )
                if branch is not None:
                    _, branch_id = patch_plan_with_branch(store_dir, plan_name, branch)
                    trace_summary["evolved_action"] = "patched_branch"
                    trace_summary["patched_branch_id"] = branch_id
                    trace_summary["patched_branch_actions"] = len(branch["actions"])
                    note(
                        f"patched {plan_name} with recovery branch {branch_id} "
                        f"({len(branch['actions'])} actions)"
                    )

    trace_summary["elapsed_s"] = round(time.time() - started, 2)
    _write_trace(output_dir, trace_summary)
    return {
        "done": solved,
        "program": plan_name,
        "trace": trace_summary,
        "slow_planner_result": slow_result,
    }


def _write_trace(output_dir: Path | str | None, payload: Mapping[str, Any]) -> None:
    if output_dir is None:
        return
    try:
        _atomic_write_json(Path(output_dir) / "flash_evolve_trace.json", payload)
    except OSError:
        pass


__all__ = [
    "SCHEMA_VERSION",
    "bootstrap_plan_from_corpus",
    "build_handover_message",
    "compile_new_plan_from_state",
    "compile_recovery_branch",
    "load_evolving_plan",
    "match_recovery_branch",
    "patch_plan_with_branch",
    "plan_name_for_cell",
    "record_branch_outcome",
    "run_flash_evolve",
]
