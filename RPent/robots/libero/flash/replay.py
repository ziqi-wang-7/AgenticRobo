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
from robots.libero.flash.generate import _is_contact_pick
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


def load(root: Path, plan_name: str, *, plan_doc: dict[str, Any] | None = None) -> dict:
    """Read one program: its plan, and the anchors its coordinates were written against."""
    if plan_doc is None:
        plan_doc = json.loads((root / f"{plan_name}_plan.json").read_text("utf-8"))
    anchors = json.loads((root / f"{plan_name}_anchors.json").read_text("utf-8"))[
        "anchors"
    ]
    return {
        "plan": plan_doc["plan"],
        "reference": {a["phrase"]: np.array(a["median_xy"]) for a in anchors},
        "locator_of": {a["phrase"]: a["locator"] for a in anchors},
        "camera_of": {a["phrase"]: a.get("camera", "agentview") for a in anchors},
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


def _episode_ended(toolkit: Any) -> bool:
    record = toolkit.state.latest_record()
    return bool(toolkit.solved() or (record is not None and record.truncated))


def _segment_xy(
    call_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
    phrase: str,
    camera: str,
    step: int,
    min_score: float,
    note: Callable[[str], None],
) -> np.ndarray | None:
    try:
        found = call_tool(
            "segment",
            {"prompt": phrase, "camera": camera, "step": step, "min_score": min_score},
        )
    except RuntimeError as exc:
        note(f"      {phrase[:26]!r} segmentation failed: {exc}")
        return None
    world = found.get("world_xyz")
    if isinstance(world, list) and len(world) >= 2:
        xy = np.asarray(world[:2], dtype=float)
        if np.isfinite(xy).all() and np.max(np.abs(xy)) <= REACH:
            return xy
    return None


def _survey_live_anchors(
    toolkit: Any,
    molmo: MolmoClient | None,
    *,
    reference: dict[str, np.ndarray],
    locator_of: dict[str, str],
    camera_of: dict[str, str],
    opening: int,
    note: Callable[[str], None],
    allow_sam3_fallback: bool,
    call_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> dict[str, np.ndarray]:
    """Localize anchors from the recorded opening frame without moving the robot."""
    state = toolkit.state
    live: dict[str, np.ndarray] = {}
    for phrase in reference:
        if _episode_ended(toolkit):
            break
        use_segment = locator_of.get(phrase) == "segment" or (
            molmo is None and allow_sam3_fallback
        )
        xy = None
        if use_segment:
            camera = camera_of.get(phrase, "agentview")
            min_score = 0.15 if molmo is None else 0.2
            xy = _segment_xy(call_tool, phrase, camera, opening, min_score, note)
            # A wrist observation can come from a later pose in the teaching
            # episode. The fixed camera can still see the object at reset.
            if xy is None and camera == "wrist":
                xy = _segment_xy(
                    call_tool, phrase, "agentview", opening, min_score, note
                )
        elif molmo is not None:
            try:
                got = locate(
                    molmo, state, opening, "agentview", prompt_for("survey", phrase)
                )
                xy = got["xy"] if got else None
            except (RuntimeError, OSError, ValueError) as exc:
                note(f"      {phrase[:26]!r} point grounding failed: {exc}")
        if xy is None or not np.isfinite(xy).all() or np.max(np.abs(xy)) > REACH:
            note(f"      {phrase[:26]!r} not located, or out of reach")
            continue
        live[phrase] = xy

    note(
        "    survey "
        + "  ".join(f"{p[:18]}=({a[0]:+.3f},{a[1]:+.3f})" for p, a in live.items())
    )

    return live


def _refine_live_anchors(
    toolkit: Any,
    molmo: MolmoClient,
    live: dict[str, np.ndarray],
    locator_of: dict[str, str],
    plan: list[dict[str, Any]],
    note: Callable[[str], None],
    call_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
) -> None:
    """Refine point-grounded anchors from the wrist after the plan passes preflight."""
    state = toolkit.state
    hover = max(
        (
            s["arguments"]["xyz"][2]
            for s in plan
            if s["action"] in {"move_to", "move_pose"} and s["arguments"].get("xyz")
        ),
        default=0.72,
    )
    for phrase, coarse in live.items():
        if _episode_ended(toolkit):
            break
        if locator_of.get(phrase) == "segment":
            continue
        try:
            call_tool(
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
            if _episode_ended(toolkit):
                break
            close = locate(
                molmo, state, state.latest_step, "wrist", prompt_for("refine", phrase)
            )
        except (RuntimeError, OSError, ValueError) as exc:
            note(f"      {phrase[:26]!r} wrist refinement failed: {exc}")
            continue
        if close is None:
            continue
        gap = float(np.linalg.norm(close["xy"] - coarse))
        if not np.isfinite(gap) or gap > REFINE_ACCEPT:
            note(f"      {phrase[:20]!r} close reading {gap:.3f} away, rejected")
            continue
        note(f"      {phrase[:20]!r} refined by {gap:.3f}")
        live[phrase] = close["xy"]


def _attached_anchor(entry: dict[str, Any]) -> str | None:
    phrase = entry.get("anchor")
    if (
        entry["action"] in {"move_to", "move_pose"}
        and phrase is not None
        and entry.get("anchor_distance", 9) <= MAX_ATTACH
    ):
        return phrase
    return None


def _missing_plan_anchors(
    plan: list[dict[str, Any]], live: dict[str, np.ndarray], state: EnvState
) -> list[dict[str, Any]]:
    """Build step-zero recovery triggers for unresolved attached moves in plan order."""
    missing = []
    for idx, entry in enumerate(plan):
        phrase = _attached_anchor(entry)
        if phrase is None or phrase in live:
            continue
        missing.append(
            {
                "failure_kind": "anchor_missing",
                "failed_step_idx": 0,
                "missing_plan_idx": idx,
                "failed_action": entry["action"],
                "anchor": phrase,
                "anchor_xy": None,
                "eef_xy": _eef_xy(state),
                "detail": (
                    f"Opening survey: anchor {phrase!r} required by plan[{idx}] was not localized. "
                    "Flash executed zero main plan actions."
                ),
            }
        )
    return missing


def _run_anchored_sequence(
    toolkit: Any,
    molmo: MolmoClient | None,
    actions: list[dict[str, Any]],
    live: dict[str, np.ndarray],
    *,
    note: Callable[[str], None],
    call_tool: Callable[[str, dict[str, Any]], dict[str, Any]],
    recorded_commands: bool,
) -> dict[str, Any]:
    """Execute anchored actions until completion or first physical anomaly."""
    state = toolkit.state
    offset = np.zeros(2)
    held_phrase = "object"
    recent: list[tuple[int, str, dict[str, Any], str | None]] = []
    trace: list[dict[str, Any]] = []
    failure: dict[str, Any] | None = None
    last_anchor: str | None = None

    def fail(idx: int, name: str, anchor: str | None, kind: str, detail: str) -> None:
        nonlocal failure
        xy = live.get(anchor) if anchor is not None else None
        failure = {
            "failure_kind": kind,
            "failed_step_idx": idx,
            "failed_action": name,
            "anchor": anchor,
            "anchor_xy": xy.round(4).tolist() if xy is not None else None,
            "eef_xy": _eef_xy(state),
            "detail": detail,
        }

    def perform(
        idx: int,
        name: str,
        args: dict[str, Any],
        anchor: str | None,
        *,
        check_grasp: bool = False,
    ) -> dict[str, Any]:
        entry = {"plan_idx": idx, "action": name, "status": "ok"}
        trace.append(entry)
        try:
            raw = call_tool(name, args)
        except RuntimeError as exc:
            entry["status"] = "error"
            fail(idx, name, anchor, "action_error", str(exc))
            return {}
        record = state.latest_record()
        if (
            name in {"move_to", "move_pose"}
            and record is not None
            and motion_alert(record)
        ):
            entry["status"] = "motion_stalled"
            fail(
                idx,
                name,
                anchor,
                "motion_stalled",
                f"{name} stalled with servo residual exceeding tolerance",
            )
        elif check_grasp and not pick_succeeded(raw):
            entry["status"] = "pick_failed"
        return raw

    for idx, entry in enumerate(actions):
        if _episode_ended(toolkit):
            break
        name = entry["action"]
        if name in {"segment", "segment_point"}:
            continue
        args = dict(entry.get("arguments") or {})
        check_grasp = name == "pi0_pick" and not _is_contact_pick(args)
        phrase = entry.get("anchor")
        last_anchor = phrase or last_anchor

        if name in {"move_to", "move_pose"}:
            xyz = args.get("xyz") or []
            if len(xyz) != 3:
                fail(
                    idx,
                    name,
                    last_anchor,
                    "action_error",
                    f"{name} requires a three-coordinate xyz",
                )
                break
            target = np.array(xyz[:2], dtype=float)
            if _attached_anchor(entry) is not None:
                if phrase not in live:
                    fail(
                        idx,
                        name,
                        phrase,
                        "anchor_missing",
                        f"attached anchor {phrase!r} could not be localized",
                    )
                    break
                target = live[phrase] + np.array(entry["offset"])
            if not recorded_commands and args.get("gripper", -1) == 1:
                target = target - offset
            if not np.isfinite(target).all() or np.max(np.abs(target)) > REACH:
                fail(
                    idx,
                    name,
                    last_anchor,
                    "motion_stalled",
                    f"target xy {target.tolist()} exceeds reachable workspace {REACH}",
                )
                break
            args["xyz"] = [
                round(float(target[0]), 4),
                round(float(target[1]), 4),
                float(xyz[2]),
            ]
        elif name in {"pi0_pick", "pi0_doubled"}:
            prompt = str(args.get("prompt", ""))
            stripped = re.sub(r"^(pick up|grasp)\s+the\s+", "", prompt, flags=re.I)
            held_phrase = re.split(r"\b(on|in|into|inside|by|and)\b", stripped)[
                0
            ].strip()
            if check_grasp and not recorded_commands:
                args.update(FLASH_PICK_THRESHOLDS)

        raw = perform(idx, name, args, last_anchor, check_grasp=check_grasp)
        if failure or _episode_ended(toolkit):
            break
        if check_grasp and not pick_succeeded(raw):
            attempts = 1
            while not recorded_commands and attempts < PICK_ATTEMPTS:
                for again in recent:
                    perform(*again)
                    if failure or _episode_ended(toolkit):
                        break
                if failure or _episode_ended(toolkit):
                    break
                raw = perform(idx, name, args, last_anchor, check_grasp=True)
                attempts += 1
                if failure or _episode_ended(toolkit) or pick_succeeded(raw):
                    break
            if failure or _episode_ended(toolkit):
                break
            if not pick_succeeded(raw):
                fail(
                    idx,
                    name,
                    last_anchor,
                    "pick_failed",
                    f"pi0_pick({prompt!r}) failed after {attempts} attempt(s)",
                )
                break
        elif name == "set_gripper" and not recorded_commands:
            offset = np.zeros(2)
            if molmo is not None and args.get("gripper", -1) == 1:
                try:
                    body = held_body(
                        molmo, state, state.latest_step, prompt_for("held", held_phrase)
                    )
                except (RuntimeError, OSError, ValueError) as exc:
                    note(f"      held object grounding failed: {exc}")
                    body = None
                eef = _eef_xy(state)
                if body is not None and eef is not None:
                    candidate = body["xy"] - eef
                    if np.linalg.norm(candidate) <= MAX_HELD:
                        offset = candidate
                        note(f"      held offset ({offset[0]:+.4f},{offset[1]:+.4f})")
        elif name == "release":
            offset = np.zeros(2)

        if name in {"release", "pi0_pick", "pi0_doubled"}:
            recent.clear()
        elif name in {
            "move_to",
            "move_pose",
            "set_gripper",
            "rotate_wrist",
            "rotate_pitch",
        }:
            recent.append((idx, name, args, last_anchor))
            recent = recent[-6:]

    return {"failure": failure, "executed_trace": trace, "last_anchor": last_anchor}


def replay_with_evolution(
    toolkit: Any,
    molmo: MolmoClient | None,
    program: dict[str, Any],
    *,
    note: Callable[[str], None] = lambda _: None,
    branch_matcher: Callable[..., dict[str, Any] | None] | None = None,
    allow_sam3_fallback: bool = True,
) -> dict[str, Any]:
    """Replay one plan, try a learned recovery and report the live handover state."""
    plan = list(program["plan"])
    state = toolkit.state
    opening = state.latest_step
    if opening is None:
        raise RuntimeError("Flash Mode replay requires an initialized environment")

    tool_calls = 0

    def call_tool(name: str, args: dict[str, Any]) -> dict[str, Any]:
        nonlocal tool_calls
        tool_calls += 1
        return execute(toolkit, name, args)

    live = _survey_live_anchors(
        toolkit,
        molmo,
        reference=program["reference"],
        locator_of=program["locator_of"],
        camera_of=program.get("camera_of", {}),
        opening=opening,
        note=note,
        allow_sam3_fallback=allow_sam3_fallback,
        call_tool=call_tool,
    )
    missing = (
        [] if _episode_ended(toolkit) else _missing_plan_anchors(plan, live, state)
    )
    if missing:
        failure, trace = missing[0], []
        note(f"    anchor preflight: {failure['detail']}")
    else:
        if molmo is not None:
            _refine_live_anchors(
                toolkit, molmo, live, program["locator_of"], plan, note, call_tool
            )
        main_run = _run_anchored_sequence(
            toolkit,
            molmo,
            plan,
            live,
            note=note,
            call_tool=call_tool,
            recorded_commands=(program.get("source") or {}).get("compiled_by")
            == "flash_evolve_cold_start",
        )
        failure, trace = main_run["failure"], main_run["executed_trace"]
        if failure is None and not _episode_ended(toolkit):
            last_anchor = main_run["last_anchor"]
            xy = live.get(last_anchor)
            failure = {
                "failure_kind": "plan_incomplete",
                "failed_step_idx": len(plan),
                "failed_action": plan[-1]["action"] if plan else None,
                "anchor": last_anchor,
                "anchor_xy": xy.round(4).tolist() if xy is not None else None,
                "eef_xy": _eef_xy(state),
                "detail": "Flash plan finished all steps but environment did not terminate",
            }

    failure_step = state.latest_step
    branch_attempts: list[dict[str, Any]] = []
    if failure and not _episode_ended(toolkit) and branch_matcher is not None:
        for candidate in missing or [failure]:
            branch = branch_matcher(program, candidate, tried_ids=set())
            if branch is not None:
                failure = candidate
                break
        if branch is not None:
            branch_id = str(branch["branch_id"])
            note(
                f"    matched learned recovery branch {branch_id} "
                f"for {failure['failure_kind']} at step {failure['failed_step_idx']}"
            )
            # These anchors were first observed by the slow planner at handover.
            # Survey them in this frame without moving the gripper or changing its grasp.
            calls_before_branch = tool_calls
            for anchor in branch.get("anchors", ()):
                phrase = anchor["phrase"]
                xy = _segment_xy(
                    call_tool,
                    phrase,
                    anchor.get("camera", "agentview"),
                    failure_step,
                    0.2,
                    note,
                )
                if xy is None:
                    live.pop(phrase, None)
                else:
                    live[phrase] = xy
            branch_run = _run_anchored_sequence(
                toolkit,
                molmo,
                list(branch["actions"]),
                live,
                note=note,
                call_tool=call_tool,
                recorded_commands=True,
            )
            branch_attempts.append(
                {
                    "branch_id": branch_id,
                    "failure_kind": failure["failure_kind"],
                    "succeeded": bool(toolkit.solved()),
                    "actions_executed": tool_calls - calls_before_branch,
                }
            )
            trace.extend(
                {**item, "branch_id": branch_id}
                for item in branch_run["executed_trace"]
            )
            if branch_run["failure"] is not None:
                failure["branch_failure_detail"] = branch_run["failure"]["detail"]

    done = bool(toolkit.solved())
    if done:
        mode = "flash_branch" if branch_attempts else "flash_fast"
    elif _episode_ended(toolkit):
        mode = "flash_truncated"
    else:
        mode = "needs_handover"

    failed_idx = (
        int(failure["failed_step_idx"])
        if failure and mode == "needs_handover"
        else len(plan)
    )
    return {
        "done": done,
        "mode": mode,
        "anchors": len(live),
        "live_anchors": {phrase: xy.round(4).tolist() for phrase, xy in live.items()},
        "plan": len(plan),
        # Step-0 segmentation reads are recorded by the toolkit but never moved the arm.
        "executed_steps": 0 if missing and not branch_attempts else tool_calls,
        "handover_step": state.latest_step,
        "failure_step": failure_step,
        "failure": failure,
        "executed_trace": trace,
        "remaining_plan": [
            {"plan_idx": idx, "action": entry["action"], "anchor": entry.get("anchor")}
            for idx, entry in enumerate(plan[failed_idx:], start=failed_idx)
        ],
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
