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
"""One LIBERO Flash plan: what it holds, and how it is replayed.

The program supplies the actions; perception supplies the coordinates. Each anchor
is re-read through the interface it was first read through -- a `segment` anchor
is re-segmented, because its offsets are relative to a mask centroid, and a wide
container seen at an angle has that centroid some way from where a pointing
model points. Point-grounded anchors are localized through the grounder.

Localization is two-stage: a coarse survey of the opening frame, then the arm
parks over each point-grounded anchor and asks again from the wrist, where the
object fills the view. The close reading is kept only when it agrees with the
coarse one, so a wrist view that found something else cannot overwrite a
correct answer.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from robots.libero import tools as libero_tools
from robots.libero.flash.prompts import build as prompt_for
from robots.libero.memory import replay_directory
from robots.libero.recovery_evidence import motion_alert
from rpent.robots.components.molmo_client import MolmoClient
from rpent.session import EnvState

#: ``<family>_<suite>_t<task>_s<seed>``, the tag the CLI builds per cell.
_CELL = re.compile(r"^(10|goal|object|spatial)_(task|swap)_t(\d+)_s(\d+)$")


#: A close reading further than this from the coarse one has found something
#: else, so the coarse one stands.
REFINE_ACCEPT = 0.05
#: Beyond this a waypoint was not written relative to any located object.
MAX_ATTACH = 0.20
#: The reachable workspace. A reading outside it is not a position.
REACH = 0.45
#: How far a held object can plausibly sit from the gripper holding it.
MAX_HELD = 0.06
#: A wrist reading of a held object lands short of its centre, further the
#: taller it is. These coefficients correct that height-dependent parallax.
PARALLAX = {"x": (0.0231, 0.0610), "y": (-0.0029, 0.2056)}
#: How many times a pick that did not take hold is retried by replaying its
#: approach. No reset is involved; the episode continues.
PICK_ATTEMPTS = 3
#: Preserve the Flash Mode replay's established grasp acceptance thresholds,
#: while leaving ``pi0_pick`` as the single owner of the success decision.
FLASH_PICK_THRESHOLDS = {
    "lift_thresh": 0.04,
    "gripper_closed_thresh": 0.07,
    "gripper_open_thresh": 0.003,
    "descent_thresh": 0.0,
}


def profile(
    state: EnvState,
    step: int,
    camera: str,
    col: float,
    row: float,
    spread: int = 45,
    samples: int = 9,
):
    """A phrase's pixel, and what the depth map says along a line through it.

    One reading names a point on a surface; several down the object's own
    height say where its top is and whether the line stayed on one surface at
    all. Readings that fall far below the highest have left the object for the
    table behind it and are dropped.
    """
    points = []
    for offset in np.linspace(-spread, spread, samples):
        found = libero_tools.back_project(
            row=int(round(row + offset)),
            col=int(round(col)),
            step=step,
            camera=camera,
            resolution="high",
            state=state,
        )
        world = found.data.get("world_xyz") if not found.is_error else None
        if isinstance(world, list) and len(world) >= 3 and all(np.isfinite(world[:3])):
            points.append([float(v) for v in world[:3]])
    if not points:
        return None
    array = np.array(points)
    keep = array[array[:, 2] > array[:, 2].max() - 0.03]
    span = float(np.ptp(keep[:, 2]))
    centre = np.median(keep[:, :2], 0)
    return {
        "xy": centre,
        "z_top": float(keep[:, 2].max()),
        "z_span": span,
        "corrected": centre
        - np.array(
            [
                PARALLAX["x"][0] + PARALLAX["x"][1] * span,
                PARALLAX["y"][0] + PARALLAX["y"][1] * span,
            ]
        ),
    }


def locate(molmo: MolmoClient, state: EnvState, step: int, camera: str, query: str):
    image = _image_bytes(state, step, camera)
    if image is None:
        return None
    found = molmo.ground(image, query)
    if not found.found:
        return None
    col, row = found.point_xy
    return profile(state, step, camera, col, row)


def held_body(molmo: MolmoClient, state: EnvState, step: int, query: str):
    """What is in the gripper, sampled on a grid because a line may miss it."""
    image = _image_bytes(state, step, "wrist")
    if image is None:
        return None
    found = molmo.ground(image, query)
    if not found.found:
        return None
    col, row = found.point_xy
    points = []
    for dc in (-40, 0, 40):
        for dr in (-40, 0, 40):
            got = libero_tools.back_project(
                row=int(round(row + dr)),
                col=int(round(col + dc)),
                step=step,
                camera="wrist",
                resolution="high",
                state=state,
            )
            world = got.data.get("world_xyz") if not got.is_error else None
            if (
                isinstance(world, list)
                and len(world) >= 3
                and all(np.isfinite(world[:3]))
            ):
                points.append([float(v) for v in world[:3]])
    if len(points) < 4:
        return None
    array = np.array(points)
    centre = np.median(array[:, :2], 0)
    near = array[np.linalg.norm(array[:, :2] - centre, axis=1) < 0.05]
    if len(near) < 3:
        return None
    span = float(np.ptp(array[:, 2]))
    return {
        "xy": np.median(near[:, :2], 0)
        - np.array(
            [
                PARALLAX["x"][0] + PARALLAX["x"][1] * span,
                PARALLAX["y"][0] + PARALLAX["y"][1] * span,
            ]
        )
    }


def _image_bytes(state: EnvState, step: int, camera: str):
    name = "agentview_high.png" if camera == "agentview" else "wrist_high.png"
    try:
        return state.load_bytes(name, step=step)
    except Exception:
        return None


def action_result(raw):
    if not isinstance(raw, dict):
        return {}
    log = raw.get("log")
    if isinstance(log, dict) and isinstance(log.get("result"), dict):
        return log["result"]
    return raw


def pick_succeeded(raw) -> bool:
    """Return the success decision made by ``pi0_pick`` itself."""
    return action_result(raw).get("success") is True


def execute(toolkit: Any, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
    """Execute one toolkit action and turn its error result into an exception."""
    result = toolkit.execute_tool(name, arguments)
    if result.is_error:
        raise RuntimeError(f"{name} failed: {result.data['error']}")
    return result.data


def plans(root: Path) -> Path:
    """Require Flash plans in the selected memory; synchronization belongs to the CLI."""
    if not any(root.glob("*_plan.json")):
        raise FileNotFoundError(
            f"no Flash plans found under {root}; use "
            "python -m robots.libero.memory sync --memory-version GPT_5.5_xhigh "
            "and pass its output as --memory-profile local --memory-dir <root>"
        )
    return root


def load(root: Path, plan_name: str) -> dict:
    """Read one program: its plan, and the anchors its coordinates were written against."""
    plan = json.loads((root / f"{plan_name}_plan.json").read_text())["plan"]
    anchors = json.loads((root / f"{plan_name}_anchors.json").read_text())["anchors"]
    return {
        "plan": plan,
        "reference": {a["phrase"]: np.array(a["median_xy"]) for a in anchors},
        "locator_of": {a["phrase"]: a["locator"] for a in anchors},
    }


def replay(
    toolkit: Any,
    molmo: MolmoClient,
    program: dict,
    note: Callable[[str], None] = lambda _: None,
) -> dict:
    """Run one program against the scene the toolkit is holding open.

    The episode is single-attempt: a pick that does not take hold is retried in
    place by replaying its approach, and its carry is skipped if it still does
    not, but the environment is never reset back to a clean state.
    """
    plan = program["plan"]
    reference = program["reference"]
    locator_of = program["locator_of"]

    state = toolkit.state

    def look() -> int:
        """The step the toolkit recorded for the action just executed.

        Every tool that moves the arm records one on its way out, so this only
        has to name it -- dumping again would file the same observation twice.
        """
        step = state.latest_step
        if step is None:
            raise RuntimeError("Flash Mode replay requires an initialized environment")
        return step

    def finished() -> bool:
        return bool(toolkit.solved())

    # Toolkit initialization resets the environment and records its opening
    # observation before the planner starts.
    opening = look()

    live: dict[str, np.ndarray] = {}
    for phrase in reference:
        if locator_of.get(phrase) == "segment":
            # One phrase the segmenter cannot place must not end the episode:
            # the rest of the anchors, and the actions that hang off them,
            # are still worth running.
            try:
                found = execute(
                    toolkit,
                    "segment",
                    {
                        "prompt": phrase,
                        "camera": "agentview",
                        "step": opening,
                        "min_score": 0.2,
                    },
                )
            except RuntimeError as exc:
                note(f"      {phrase[:26]!r} segmentation failed: {exc}")
                found = {}
            world = found.get("world_xyz")
            xy = (
                np.array(world[:2], dtype=float)
                if isinstance(world, list) and len(world) >= 2
                else None
            )
        else:
            got = locate(
                molmo, state, opening, "agentview", prompt_for("survey", phrase)
            )
            xy = got["xy"] if got else None
        if xy is None or max(abs(xy[0]), abs(xy[1])) > REACH:
            note(f"      {phrase[:26]!r} not located, or out of reach")
            continue
        if any(np.linalg.norm(a - xy) < 0.03 for a in live.values()):
            continue
        live[phrase] = xy
    note(
        "    survey "
        + "  ".join(f"{p[:18]}=({a[0]:+.3f},{a[1]:+.3f})" for p, a in live.items())
    )

    # Park over each point-grounded anchor and read it again from close range,
    # keeping the second reading only when it agrees with the first.
    hover = max(
        (
            s["arguments"]["xyz"][2]
            for s in plan
            if s["action"] in {"move_to", "move_pose"} and s["arguments"].get("xyz")
        ),
        default=0.72,
    )
    for phrase in list(live):
        if locator_of.get(phrase) == "segment":
            continue
        coarse = live[phrase]
        try:
            execute(
                toolkit,
                "move_to",
                {
                    "xyz": [
                        round(float(coarse[0]), 4),
                        round(float(coarse[1]), 4),
                        float(hover),
                    ],
                    "gripper": -1,
                    "step_clip": 0.02,
                    "max_steps": 150,
                    "tol": 0.012,
                },
            )
        except Exception:
            continue
        close = locate(molmo, state, look(), "wrist", prompt_for("refine", phrase))
        if close is None:
            continue
        gap = float(np.linalg.norm(close["xy"] - coarse))
        if gap > REFINE_ACCEPT:
            note(f"      {phrase[:20]!r} close reading {gap:.3f} away, rejected")
            continue
        note(f"      {phrase[:20]!r} refined by {gap:.3f}")
        live[phrase] = close["xy"]

    offset = np.zeros(2)
    held_phrase = "object"
    recent: list[tuple[str, dict]] = []
    skip_suffix = False
    for entry in plan:
        if finished():
            break
        name = entry["action"]
        arguments = dict(entry["arguments"])
        if skip_suffix:
            # A pick that never took hold must not fall through into its
            # carry: an empty gripper crossing the workspace knocks over
            # whatever an earlier transaction put down.
            if name == "release":
                skip_suffix = False
                recent.clear()
                continue
            if name != "pi0_pick":
                continue
            skip_suffix = False
        if name in {"move_to", "move_pose"}:
            xyz = arguments.get("xyz") or []
            if len(xyz) != 3:
                continue
            target = np.array(xyz[:2], dtype=float)
            phrase = entry.get("anchor")
            attached = (
                phrase is not None and entry.get("anchor_distance", 9) <= MAX_ATTACH
            )
            if attached and phrase not in live:
                note(f"      {phrase[:26]!r} unavailable; stopping replay")
                break
            if attached:
                target = live[phrase] + np.array(entry["offset"])
            held = arguments.get("gripper", -1) == 1
            if held:
                target = target - offset
            if max(abs(target[0]), abs(target[1])) > REACH:
                continue
            arguments["xyz"] = [
                round(float(target[0]), 4),
                round(float(target[1]), 4),
                float(xyz[2]),
            ]
            execute(toolkit, name, arguments)
            look()
        elif name in {"segment", "segment_point"}:
            continue
        elif name in {"pi0_pick", "pi0_doubled"}:
            prompt = str(arguments.get("prompt", ""))
            stripped = re.sub(r"^(pick up|grasp)\s+the\s+", "", prompt, flags=re.I)
            held_phrase = re.split(r"\b(on|in|into|inside|by|and)\b", stripped)[
                0
            ].strip()
            if name == "pi0_pick":
                arguments.update(FLASH_PICK_THRESHOLDS)
            raw = execute(toolkit, name, arguments)
            look()
            if name == "pi0_pick" and not pick_succeeded(raw):
                for _ in range(PICK_ATTEMPTS - 1):
                    if finished():
                        break
                    for again, again_args in recent:
                        execute(toolkit, again, dict(again_args))
                    raw = execute(toolkit, name, dict(arguments))
                    look()
                    if pick_succeeded(raw):
                        break
                if not pick_succeeded(raw):
                    note("      pick unconfirmed, skipping its carry")
                    skip_suffix = True
        elif name == "set_gripper":
            execute(toolkit, name, arguments)
            held_step = look()
            body = held_body(molmo, state, held_step, prompt_for("held", held_phrase))
            eef = np.asarray(state.get(held_step).state["robot0_eef_pos"][:2])
            candidate = body["xy"] - eef if body is not None else None
            if candidate is not None and np.linalg.norm(candidate) <= MAX_HELD:
                offset = candidate
                note(f"      held offset ({offset[0]:+.4f},{offset[1]:+.4f})")
            else:
                offset = np.zeros(2)
        elif name == "release":
            execute(toolkit, name, arguments)
            offset = np.zeros(2)
            look()
        else:
            execute(toolkit, name, arguments)
            look()
        if name in {"release", "pi0_pick", "pi0_doubled"}:
            recent.clear()
        elif name in {"move_to", "move_pose", "set_gripper", "rotate_wrist"}:
            recent.append((name, dict(arguments)))
            recent = recent[-6:]

    return {"done": finished(), "anchors": len(live), "plan": len(plan)}


def _eef_xy(state: EnvState) -> list[float] | None:
    record = state.latest_record()
    if record is None:
        return None
    pos = (record.state or {}).get("robot0_eef_pos")
    if isinstance(pos, (list, tuple, np.ndarray)) and len(pos) >= 2:
        return [round(float(pos[0]), 4), round(float(pos[1]), 4)]
    return None


def _survey_live_anchors(
    toolkit: Any,
    molmo: MolmoClient | None,
    *,
    reference: dict[str, np.ndarray],
    locator_of: dict[str, str],
    plan: list[dict[str, Any]],
    opening: int,
    note: Callable[[str], None],
    allow_sam3_fallback: bool,
) -> dict[str, np.ndarray]:
    """Localize plan anchors via Molmo+SAM3, or SAM3-only when Molmo is absent."""
    state = toolkit.state
    live: dict[str, np.ndarray] = {}
    for phrase in reference:
        use_segment = locator_of.get(phrase) == "segment" or (
            molmo is None and allow_sam3_fallback
        )
        xy: np.ndarray | None = None
        if use_segment:
            try:
                found = execute(
                    toolkit,
                    "segment",
                    {
                        "prompt": phrase,
                        "camera": "agentview",
                        "step": opening,
                        "min_score": 0.15 if molmo is None else 0.2,
                    },
                )
            except RuntimeError as exc:
                note(f"      {phrase[:26]!r} segmentation failed: {exc}")
                found = {}
            world = found.get("world_xyz")
            xy = (
                np.array(world[:2], dtype=float)
                if isinstance(world, list) and len(world) >= 2
                else None
            )
        elif molmo is not None:
            got = locate(
                molmo, state, opening, "agentview", prompt_for("survey", phrase)
            )
            xy = got["xy"] if got else None
        if xy is None or max(abs(xy[0]), abs(xy[1])) > REACH:
            note(f"      {phrase[:26]!r} not located, or out of reach")
            continue
        if any(np.linalg.norm(a - xy) < 0.03 for a in live.values()):
            continue
        live[phrase] = xy

    note(
        "    survey "
        + "  ".join(f"{p[:18]}=({a[0]:+.3f},{a[1]:+.3f})" for p, a in live.items())
    )

    if molmo is None:
        return live

    hover = max(
        (
            s["arguments"]["xyz"][2]
            for s in plan
            if s["action"] in {"move_to", "move_pose"} and s["arguments"].get("xyz")
        ),
        default=0.72,
    )
    for phrase in list(live):
        if locator_of.get(phrase) == "segment":
            continue
        coarse = live[phrase]
        try:
            execute(
                toolkit,
                "move_to",
                {
                    "xyz": [
                        round(float(coarse[0]), 4),
                        round(float(coarse[1]), 4),
                        float(hover),
                    ],
                    "gripper": -1,
                    "step_clip": 0.02,
                    "max_steps": 150,
                    "tol": 0.012,
                },
            )
        except Exception:
            continue
        close = locate(
            molmo,
            state,
            state.latest_step or opening,
            "wrist",
            prompt_for("refine", phrase),
        )
        if close is None:
            continue
        gap = float(np.linalg.norm(close["xy"] - coarse))
        if gap > REFINE_ACCEPT:
            note(f"      {phrase[:20]!r} close reading {gap:.3f} away, rejected")
            continue
        note(f"      {phrase[:20]!r} refined by {gap:.3f}")
        live[phrase] = close["xy"]
    return live


def _run_anchored_sequence(
    toolkit: Any,
    molmo: MolmoClient | None,
    actions: list[dict[str, Any]],
    live: dict[str, np.ndarray],
    *,
    note: Callable[[str], None],
    offset: np.ndarray,
    held_phrase: str,
    recent: list[tuple[str, dict[str, Any]]],
    start_idx: int = 0,
    allow_pick_retry: bool = True,
) -> dict[str, Any]:
    """Execute anchored actions until completion or first physical anomaly."""
    state = toolkit.state

    def look() -> int:
        step = state.latest_step
        if step is None:
            raise RuntimeError("Flash Mode replay requires an initialized environment")
        return step

    def finished() -> bool:
        return bool(toolkit.solved())

    executed_trace: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    last_anchor: str | None = None

    for rel_idx, entry in enumerate(actions):
        plan_idx = start_idx + rel_idx
        if finished():
            break
        name = entry["action"]
        arguments = dict(entry.get("arguments") or {})
        phrase = entry.get("anchor")
        if phrase:
            last_anchor = str(phrase)
        anchor_xy = (
            [
                round(float(live[last_anchor][0]), 4),
                round(float(live[last_anchor][1]), 4),
            ]
            if last_anchor and last_anchor in live
            else None
        )

        if name in {"move_to", "move_pose"}:
            xyz = arguments.get("xyz") or []
            if len(xyz) != 3:
                continue
            target = np.array(xyz[:2], dtype=float)
            attached = (
                phrase is not None and entry.get("anchor_distance", 9) <= MAX_ATTACH
            )
            if attached and phrase not in live:
                failure = {
                    "failure_kind": "anchor_missing",
                    "failed_step_idx": plan_idx,
                    "failed_action": name,
                    "anchor": phrase,
                    "anchor_xy": None,
                    "eef_xy": _eef_xy(state),
                    "detail": f"attached anchor {phrase!r} could not be localized",
                }
                executed_trace.append(
                    {"plan_idx": plan_idx, "action": name, "status": "anchor_missing"}
                )
                break
            if attached:
                target = live[phrase] + np.array(entry["offset"])
            held = arguments.get("gripper", -1) == 1
            if held:
                target = target - offset
            if max(abs(target[0]), abs(target[1])) > REACH:
                failure = {
                    "failure_kind": "motion_stalled",
                    "failed_step_idx": plan_idx,
                    "failed_action": name,
                    "anchor": phrase,
                    "anchor_xy": anchor_xy,
                    "eef_xy": _eef_xy(state),
                    "detail": f"target xy {target.tolist()} exceeds reachable workspace {REACH}",
                }
                executed_trace.append(
                    {"plan_idx": plan_idx, "action": name, "status": "out_of_reach"}
                )
                break
            arguments["xyz"] = [
                round(float(target[0]), 4),
                round(float(target[1]), 4),
                float(xyz[2]),
            ]
            try:
                execute(toolkit, name, arguments)
            except RuntimeError as exc:
                failure = {
                    "failure_kind": "action_error",
                    "failed_step_idx": plan_idx,
                    "failed_action": name,
                    "anchor": phrase,
                    "anchor_xy": anchor_xy,
                    "eef_xy": _eef_xy(state),
                    "detail": str(exc),
                }
                executed_trace.append(
                    {"plan_idx": plan_idx, "action": name, "status": "error"}
                )
                break
            look()
            latest = state.latest_record()
            if latest is not None and motion_alert(latest):
                failure = {
                    "failure_kind": "motion_stalled",
                    "failed_step_idx": plan_idx,
                    "failed_action": name,
                    "anchor": phrase or last_anchor,
                    "anchor_xy": anchor_xy,
                    "eef_xy": _eef_xy(state),
                    "detail": f"{name} stalled with servo residual exceeding tolerance",
                }
                executed_trace.append(
                    {"plan_idx": plan_idx, "action": name, "status": "motion_stalled"}
                )
                break
            executed_trace.append(
                {"plan_idx": plan_idx, "action": name, "status": "ok"}
            )
        elif name in {"segment", "segment_point"}:
            continue
        elif name in {"pi0_pick", "pi0_doubled"}:
            prompt = str(arguments.get("prompt", ""))
            stripped = re.sub(r"^(pick up|grasp)\s+the\s+", "", prompt, flags=re.I)
            held_phrase = re.split(r"\b(on|in|into|inside|by|and)\b", stripped)[
                0
            ].strip()
            if name == "pi0_pick":
                arguments.update(FLASH_PICK_THRESHOLDS)
            try:
                raw = execute(toolkit, name, arguments)
            except RuntimeError as exc:
                failure = {
                    "failure_kind": "action_error",
                    "failed_step_idx": plan_idx,
                    "failed_action": name,
                    "anchor": last_anchor,
                    "anchor_xy": anchor_xy,
                    "eef_xy": _eef_xy(state),
                    "detail": str(exc),
                }
                executed_trace.append(
                    {"plan_idx": plan_idx, "action": name, "status": "error"}
                )
                break
            look()
            if name == "pi0_pick" and not pick_succeeded(raw):
                if allow_pick_retry:
                    for _ in range(PICK_ATTEMPTS - 1):
                        if finished():
                            break
                        for again, again_args in recent:
                            execute(toolkit, again, dict(again_args))
                        raw = execute(toolkit, name, dict(arguments))
                        look()
                        if pick_succeeded(raw):
                            break
                if not pick_succeeded(raw):
                    failure = {
                        "failure_kind": "pick_failed",
                        "failed_step_idx": plan_idx,
                        "failed_action": name,
                        "anchor": last_anchor,
                        "anchor_xy": anchor_xy,
                        "eef_xy": _eef_xy(state),
                        "detail": (
                            f"pi0_pick({prompt!r}) failed after "
                            f"{PICK_ATTEMPTS if allow_pick_retry else 1} attempt(s)"
                        ),
                    }
                    executed_trace.append(
                        {"plan_idx": plan_idx, "action": name, "status": "pick_failed"}
                    )
                    break
            executed_trace.append(
                {"plan_idx": plan_idx, "action": name, "status": "ok"}
            )
        elif name == "set_gripper":
            try:
                execute(toolkit, name, arguments)
            except RuntimeError as exc:
                failure = {
                    "failure_kind": "action_error",
                    "failed_step_idx": plan_idx,
                    "failed_action": name,
                    "anchor": last_anchor,
                    "anchor_xy": anchor_xy,
                    "eef_xy": _eef_xy(state),
                    "detail": str(exc),
                }
                executed_trace.append(
                    {"plan_idx": plan_idx, "action": name, "status": "error"}
                )
                break
            held_step = look()
            if molmo is not None:
                body = held_body(
                    molmo, state, held_step, prompt_for("held", held_phrase)
                )
                eef = np.asarray(state.get(held_step).state["robot0_eef_pos"][:2])
                candidate = body["xy"] - eef if body is not None else None
                if candidate is not None and np.linalg.norm(candidate) <= MAX_HELD:
                    offset = candidate
                    note(f"      held offset ({offset[0]:+.4f},{offset[1]:+.4f})")
                else:
                    offset = np.zeros(2)
            executed_trace.append(
                {"plan_idx": plan_idx, "action": name, "status": "ok"}
            )
        elif name == "release":
            try:
                execute(toolkit, name, arguments)
            except RuntimeError as exc:
                failure = {
                    "failure_kind": "action_error",
                    "failed_step_idx": plan_idx,
                    "failed_action": name,
                    "anchor": last_anchor,
                    "anchor_xy": anchor_xy,
                    "eef_xy": _eef_xy(state),
                    "detail": str(exc),
                }
                executed_trace.append(
                    {"plan_idx": plan_idx, "action": name, "status": "error"}
                )
                break
            offset = np.zeros(2)
            look()
            executed_trace.append(
                {"plan_idx": plan_idx, "action": name, "status": "ok"}
            )
        else:
            try:
                execute(toolkit, name, arguments)
            except RuntimeError as exc:
                failure = {
                    "failure_kind": "action_error",
                    "failed_step_idx": plan_idx,
                    "failed_action": name,
                    "anchor": last_anchor,
                    "anchor_xy": anchor_xy,
                    "eef_xy": _eef_xy(state),
                    "detail": str(exc),
                }
                executed_trace.append(
                    {"plan_idx": plan_idx, "action": name, "status": "error"}
                )
                break
            look()
            executed_trace.append(
                {"plan_idx": plan_idx, "action": name, "status": "ok"}
            )

        if name in {"release", "pi0_pick", "pi0_doubled"}:
            recent.clear()
        elif name in {"move_to", "move_pose", "set_gripper", "rotate_wrist"}:
            recent.append((name, dict(arguments)))
            recent = recent[-6:]

    return {
        "done": finished(),
        "failure": failure,
        "executed_trace": executed_trace,
        "offset": offset,
        "held_phrase": held_phrase,
        "recent": recent,
        "last_anchor": last_anchor,
    }


def replay_with_evolution(
    toolkit: Any,
    molmo: MolmoClient | None,
    program: dict[str, Any],
    *,
    note: Callable[[str], None] = lambda _: None,
    branch_matcher: Callable[..., dict[str, Any] | None] | None = None,
    allow_sam3_fallback: bool = True,
) -> dict[str, Any]:
    """Replay a Flash plan with anomaly detection and learned recovery branches.

    Unlike :func:`replay`, this function never silently discards failed
    subtasks via ``skip_suffix``: on the first anomaly (or when the plan ends
    without ``terminated == true``), it tries a matching learned recovery
    branch at zero LLM cost, and if the task is still unsolved it returns a
    structured handover snapshot for the slow planner to take over in-place.
    """
    plan = list(program["plan"])
    reference = program["reference"]
    locator_of = program["locator_of"]
    state = toolkit.state

    opening = state.latest_step
    if opening is None:
        raise RuntimeError("Flash Mode replay requires an initialized environment")

    live = _survey_live_anchors(
        toolkit,
        molmo,
        reference=reference,
        locator_of=locator_of,
        plan=plan,
        opening=opening,
        note=note,
        allow_sam3_fallback=allow_sam3_fallback,
    )
    live_serializable = {
        phrase: [round(float(xy[0]), 4), round(float(xy[1]), 4)]
        for phrase, xy in live.items()
    }

    offset = np.zeros(2)
    held_phrase = "object"
    recent: list[tuple[str, dict[str, Any]]] = []
    main_run = _run_anchored_sequence(
        toolkit,
        molmo,
        plan,
        live,
        note=note,
        offset=offset,
        held_phrase=held_phrase,
        recent=recent,
        start_idx=0,
        allow_pick_retry=True,
    )
    executed_trace = list(main_run["executed_trace"])
    branch_attempts: list[dict[str, Any]] = []

    if main_run["done"]:
        return {
            "done": True,
            "mode": "flash_fast",
            "anchors": len(live),
            "live_anchors": live_serializable,
            "plan": len(plan),
            "executed_steps": len(executed_trace),
            "handover_step": int(state.latest_step or 0),
            "failure": None,
            "executed_trace": executed_trace,
            "remaining_plan": [],
            "branch_attempts": branch_attempts,
        }

    failure = main_run["failure"]
    if failure is None:
        last_anchor = main_run.get("last_anchor")
        failure = {
            "failure_kind": "plan_incomplete",
            "failed_step_idx": len(plan),
            "failed_action": plan[-1]["action"] if plan else None,
            "anchor": last_anchor,
            "anchor_xy": live_serializable.get(last_anchor) if last_anchor else None,
            "eef_xy": _eef_xy(state),
            "detail": "Flash plan finished all steps but environment did not terminate",
        }

    # Try matching learned recovery branch at zero LLM tokens before handover.
    tried_ids: set[str] = set()
    if branch_matcher is not None:
        branch = branch_matcher(program, failure, tried_ids=tried_ids)
        if branch is not None:
            branch_id = str(branch["branch_id"])
            tried_ids.add(branch_id)
            note(
                f"    matched learned recovery branch {branch_id} "
                f"for {failure['failure_kind']} at step {failure['failed_step_idx']}"
            )
            branch_run = _run_anchored_sequence(
                toolkit,
                molmo,
                list(branch.get("actions") or []),
                live,
                note=note,
                offset=main_run["offset"],
                held_phrase=main_run["held_phrase"],
                recent=[],
                start_idx=0,
                allow_pick_retry=False,
            )
            branch_succeeded = bool(branch_run["done"])
            branch_attempts.append(
                {
                    "branch_id": branch_id,
                    "failure_kind": failure["failure_kind"],
                    "succeeded": branch_succeeded,
                    "actions_executed": len(branch_run["executed_trace"]),
                }
            )
            executed_trace.extend(
                {**item, "branch_id": branch_id}
                for item in branch_run["executed_trace"]
            )
            if branch_succeeded:
                return {
                    "done": True,
                    "mode": "flash_branch",
                    "anchors": len(live),
                    "live_anchors": live_serializable,
                    "plan": len(plan),
                    "executed_steps": len(executed_trace),
                    "handover_step": int(state.latest_step or 0),
                    "failure": failure,
                    "executed_trace": executed_trace,
                    "remaining_plan": [],
                    "branch_attempts": branch_attempts,
                }
            if branch_run["failure"] is not None:
                failure = {
                    **failure,
                    "branch_failure_detail": branch_run["failure"].get("detail"),
                }

    failed_idx = int(failure.get("failed_step_idx", len(plan)))
    remaining = [
        {
            "plan_idx": idx,
            "action": entry["action"],
            "anchor": entry.get("anchor"),
        }
        for idx, entry in enumerate(plan[failed_idx:], start=failed_idx)
    ]
    return {
        "done": False,
        "mode": "needs_handover",
        "anchors": len(live),
        "live_anchors": live_serializable,
        "plan": len(plan),
        "executed_steps": len(executed_trace),
        "handover_step": int(state.latest_step or 0),
        "failure": failure,
        "executed_trace": executed_trace,
        "remaining_plan": remaining,
        "branch_attempts": branch_attempts,
    }


def run_flash(
    toolkit: Any, cell_tag: str, note: Callable[[str], None] = lambda _: None
) -> dict:
    """Replay the program for one cell, named the way the CLI names its cells.

    The ``RobotSpec`` hook behind ``--planner flash``: everything LIBERO
    knows about its own plans -- how a tag maps to a program, and where its
    grounder is -- stays here, so the planner needs to know none of it.
    """
    match = _CELL.match(cell_tag)
    if not match:
        raise ValueError(
            f"cannot read the cell tag {cell_tag!r}; "
            "expected <family>_<suite>_t<task>_s<seed>"
        )
    # The seed selects the layout to solve, not the plan used to solve it.
    family, suite, task, _ = match.groups()
    key = f"{suite}_t{task}"
    root = plans(replay_directory(toolkit.memory.root))
    plan_name = f"{family}_{key}"
    if not all(
        (root / f"{plan_name}_{suffix}.json").is_file()
        for suffix in ("plan", "anchors")
    ):
        raise FileNotFoundError(
            f"no complete Flash plan for {family}/{key} under {root}; "
            "both plan and anchors files are required"
        )

    molmo = toolkit.primitives.molmo_client
    if molmo is None:
        raise RuntimeError(
            "no grounder: Flash plans need a Molmo server named by --molmo-endpoint"
        )
    note(f"replaying the {family}/{key} program")
    return {
        **replay(toolkit, molmo, load(root, plan_name), note),
        "program": f"{family}/{key}",
    }
