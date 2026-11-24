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

"""Flash grounding and handover from recorded observations, without simulator or models."""

from __future__ import annotations

import io
import json
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import pytest

from robots.libero.flash.evolve import build_handover_message, run_flash_evolve
from robots.libero.flash.replay import replay_with_evolution
from robots.libero.prompts import evaluate
from robots.libero.tools import LiberoPrimitives
from rpent.robots.components.sam3_client import Sam3Result
from tests.unit_tests.robots.libero.test_flash_evolve import FakeLiberoToolkit


class ImageSegmenter:
    """Return the highest-scoring colored instance still visible in the submitted image."""

    def __init__(self):
        self.prompts = {"moka pot"}
        self.scores = {240: 0.95, 160: 0.8}
        self.requests = []

    def segment(self, image, *, text_prompt, min_score, point=None):
        if isinstance(image, bytes):
            image = imageio.imread(io.BytesIO(image))
        self.requests.append((text_prompt, image.copy()))
        if text_prompt in self.prompts:
            for color, score in self.scores.items():
                mask = image[..., 0] == color
                if mask.any():
                    # Return the raw score even when low, to test the client's evidence check.
                    return Sam3Result(found=True, mask=mask, score=score)
        return Sam3Result(found=False, reason="no matching instance")


@pytest.fixture
def scene(tmp_path, monkeypatch):
    tk = FakeLiberoToolkit(tmp_path, memory_root=tmp_path, anchors_xyz={})
    sam = ImageSegmenter()
    tk.primitives = LiberoPrimitives(
        env=None, model=None, sam3_client=sam, check_cancelled=lambda: None
    )
    move = tk.execute_tool

    def execute(name, args):
        if name == "segment":
            return tk.primitives.segment(state=tk.state, **args)
        return move(name, args)

    monkeypatch.setattr(tk, "execute_tool", execute)
    image = np.full((26, 38, 3), 20, dtype=np.uint8)
    image[3:11, 3:12] = [240, 20, 10]
    image[5:7, 5:7] = 20  # A hole inside the first mask's bounding box.
    image[14:23, 23:32] = [160, 70, 100]
    rows, cols = np.indices(image.shape[:2])
    world = np.stack(
        [(rows - 13) * 0.01, (cols - 19) * 0.008, 0.6 + rows * 0.001], axis=-1
    )
    tk.state.save("agentview_high.png", image)
    tk.state.save("agentview_world_high.npz", world)
    return tk, sam, image, world


def anchored_plan(phrase):
    return {
        "reference": {phrase: np.array([-0.31, 0.31])},
        "locator_of": {phrase: "segment"},
        "plan": [
            {
                "action": "move_to",
                "arguments": {"xyz": [-0.30, 0.30, 0.72]},
                "anchor": phrase,
                "anchor_distance": 0.02,
                "offset": [0.01, -0.01],
            }
        ],
    }


@pytest.mark.parametrize(
    "direction,selected",
    [("left", 160), ("right", 240), ("front", 240), ("closer", 240), ("back", 160)],
)
def test_disambiguation_masks_a_copy_and_uses_world_axes(scene, direction, selected):
    tk, sam, image, world = scene
    phrase = f"the {direction} moka pot"
    notes = []
    out = replay_with_evolution(tk, None, anchored_plan(phrase), note=notes.append)
    assert [p for p, _ in sam.requests] == [phrase, "moka pot", "moka pot"]
    expected_image = image.copy()
    expected_image[3:11, 3:12] = image.mean(axis=(0, 1))
    np.testing.assert_array_equal(sam.requests[2][1], expected_image)
    np.testing.assert_array_equal(tk.state.load("agentview_high.png", step=0), image)
    points = world[image[..., 0] == selected]
    selected_xy = np.median(points[:, :2], axis=0)
    assert out["live_anchors"][phrase] == pytest.approx(selected_xy)
    assert out["live_anchor_details"][phrase]["z_top"] == pytest.approx(
        np.percentile(points[:, 2], 95)
    )
    assert out["live_anchor_details"][phrase]["source"] == "sam3_directional_pair"
    assert tk.executed[0][1]["xyz"][:2] == pytest.approx(selected_xy + [0.01, -0.01])
    assert any("sam3_directional_pair" in line for line in notes)
    assert not tk.reset_called


@pytest.mark.parametrize(
    "case",
    [
        "single",
        "low_score",
        "ambiguous_axis",
        "four_cm",
        "unreachable_peer",
        "invalid_depth",
    ],
)
def test_uncertain_direction_fails_before_motion(scene, case):
    tk, sam, image, world = scene
    if case == "single":
        image[image[..., 0] == 160] = 20
        tk.state.save("agentview_high.png", image)
    elif case == "low_score":
        sam.scores[160] = 0.3
    elif case == "ambiguous_axis":
        world[..., 1] = 0  # Separated in x, tied on the requested left/right axis.
    elif case == "four_cm":
        world[..., 1] = 0.10
        world[image[..., 0] == 160, 1] = 0.14
    elif case == "unreachable_peer":
        world[image[..., 0] == 240, 0] = 1.0
    else:
        world[image[..., 0] == 160] = np.nan
    tk.state.save("agentview_world_high.npz", world)
    out = replay_with_evolution(tk, None, anchored_plan("the left moka pot"))
    assert out["failure"]["failure_kind"] == "anchor_missing"
    assert out["handover_step"] == out["physical_action_calls"] == 0
    assert not tk.executed and not out["live_anchors"]


def test_plural_anchor_cannot_bind_even_when_sam_would_return_a_mask(scene):
    tk, sam, _, _ = scene
    phrase = "the both moka pots"
    sam.prompts.add(phrase)
    out = replay_with_evolution(tk, None, anchored_plan(phrase))
    assert not out["live_anchors"] and not tk.executed
    assert out["failure"]["anchor"] == phrase


@pytest.mark.parametrize("molmo,enabled", [(None, False), (object(), True)])
def test_geometric_fallback_only_in_sam3_only_mode(scene, molmo, enabled):
    tk, sam, _, _ = scene
    phrase = "the left moka pot"
    out = replay_with_evolution(
        tk, molmo, anchored_plan(phrase), allow_sam3_fallback=enabled
    )
    assert [p for p, _ in sam.requests] == [phrase]
    assert not out["live_anchors"]


def test_successful_original_phrase_keeps_its_projection_and_skips_second_pass(scene):
    tk, sam, image, world = scene
    phrase = "the right moka pot"
    sam.prompts.add(phrase)
    out = replay_with_evolution(tk, None, anchored_plan(phrase))
    assert [p for p, _ in sam.requests] == [phrase]
    assert out["live_anchor_details"][phrase]["z_top"] == pytest.approx(
        np.percentile(world[image[..., 0] == 240][:, 2], 95), abs=0.0001
    )


def test_unreachable_direct_detection_is_not_replaced_by_a_different_object(scene):
    tk, sam, image, world = scene
    phrase = "the left moka pot"
    sam.prompts.add(phrase)
    world[image[..., 0] == 240, 0] = 1.0
    tk.state.save("agentview_world_high.npz", world)
    out = replay_with_evolution(tk, None, anchored_plan(phrase))
    assert [p for p, _ in sam.requests] == [phrase]
    assert not out["live_anchors"] and not tk.executed


@pytest.mark.parametrize("quality", ["visible", "facade", "sparse", "clipped"])
def test_container_body_requires_supported_horizontal_center(scene, quality):
    tk, sam, image, world = scene
    phrase = "the inside of the microwave"
    sam.prompts = {"the microwave"}
    if quality == "facade":
        world[..., 0] = world[
            ..., 1
        ]  # An oblique face has no footprint around its center.
    elif quality == "sparse":
        world[4:, :, :] = np.nan
    elif quality == "clipped":
        image[:4, 3:12] = [240, 20, 10]
        tk.state.save("agentview_high.png", image)
    tk.state.save("agentview_world_high.npz", world)
    notes = []
    out = replay_with_evolution(tk, None, anchored_plan(phrase), note=notes.append)
    assert [p for p, _ in sam.requests] == [phrase, "the microwave"]
    if quality == "visible":
        assert out["live_anchors"][phrase] == pytest.approx([-0.065, -0.096])
        assert any("sam3_container_body" in line for line in notes)
        assert "Body center only" in build_handover_message(
            "", plan_name="microwave", replay_outcome=out
        )
    else:
        assert out["failure"]["failure_kind"] == "anchor_missing"
        assert not tk.executed


def test_survey_only_handoff_contains_only_measured_anchors(tmp_path: Path):
    tk = FakeLiberoToolkit(
        tmp_path, memory_root=tmp_path, anchors_xyz={"plate": [0.12, 0.11, 0.65]}
    )
    plan = anchored_plan("the mug")
    plan["reference"]["plate"] = np.array([0.39, 0.39])
    store = tmp_path / "store"
    store.mkdir()
    (store / "10_swap_t8_plan.json").write_text(json.dumps({"plan": plan["plan"]}))
    (store / "10_swap_t8_anchors.json").write_text(
        json.dumps(
            {
                "anchors": [
                    {"phrase": p, "median_xy": xy.tolist(), "locator": "segment"}
                    for p, xy in plan["reference"].items()
                ]
            }
        )
    )
    received = []
    run_flash_evolve(
        tk,
        "10_swap_t8_s1",
        slow_solve=received.append,
        user_message="Task",
        flash_memory_dir=store,
        evolve_enabled=False,
        output_dir=tmp_path,
    )
    out = json.loads((tmp_path / "flash_evolve_trace.json").read_text())
    assert out["fast_actions"] == 2  # Both survey calls were read-only.
    assert out["handoff_type"] == "pre_execution"
    assert out["physical_action_calls"] == out["handover_step"] == 0
    assert out["failure"]["failed_target_xyz"] is None
    (msg,) = received
    assert "PRE-EXECUTION HANDOFF (NO MOTION EXECUTED YET)" in msg
    assert "initial step 0" in msg and "back_project" in msg and "finish" in msg
    assert "plate: xy=[+0.1200, +0.1100]" in msg
    assert all(
        word not in msg
        for word in ("SYSTEM-1 FLASH HANDOVER", "clean and concise", "the mug", "0.39")
    )
    assert not tk.reset_called


def test_anchor_miss_after_gripper_action_uses_in_place_recovery(tmp_path):
    tk = FakeLiberoToolkit(tmp_path, memory_root=tmp_path, anchors_xyz={})
    plan = anchored_plan("the top layer of the drawer handle")
    plan["plan"].insert(0, {"action": "set_gripper", "arguments": {"gripper": 1}})
    out = replay_with_evolution(tk, None, plan)
    assert out["physical_action_calls"] == 1
    assert out["failure"]["failure_kind"] == "anchor_missing"
    msg = build_handover_message("Task", plan_name="drawer", replay_outcome=out)
    assert "SYSTEM-1 FLASH HANDOVER" in msg and "PRE-EXECUTION" not in msg
    assert "TARGET DRAWER LEVEL: top" in msg
    assert not tk.reset_called


def test_stalled_drawer_reports_retargeted_xyz_and_instruction_level(tmp_path: Path):
    phrase = "the top drawer handle"  # The historical anchor conflicts with the live instruction.
    tk = FakeLiberoToolkit(
        tmp_path,
        memory_root=tmp_path,
        anchors_xyz={phrase: [0.08, -0.32, 0.51]},
        task_language="open the bottom drawer of the cabinet",
        stall_move_to_y_above=0.25,
    )
    out = replay_with_evolution(tk, None, anchored_plan(phrase))
    failure = out["failure"]
    assert failure["failure_kind"] == "motion_stalled"
    assert failure["eef_xyz"] == [0, 0, 0.72]
    assert failure["failed_target_xyz"] == pytest.approx([0.09, -0.33, 0.72])
    assert failure["anchor_xy"] == [0.08, -0.32]
    assert out["physical_action_calls"] == out["handover_step"] == 1
    msg = build_handover_message("Task", plan_name="drawer", replay_outcome=out)
    assert "SYSTEM-1 FLASH HANDOVER" in msg and "PRE-EXECUTION" not in msg
    assert "eef_xyz (m): [0.0, 0.0, 0.72]" in msg
    assert "failed_target_xyz (m): [0.09, -0.33, 0.72]" in msg
    assert (
        "TARGET DRAWER LEVEL: bottom" in msg and "TARGET DRAWER LEVEL: top" not in msg
    )
    assert all(
        lever in msg
        for lever in (
            "move_pose",
            "target_pitch",
            "target_yaw",
            "back_project",
            "pi0_pick",
            "pi0_doubled",
        )
    )
    assert "clean and concise" not in msg and not tk.reset_called


def test_microwave_prompt_guides_a_physical_attempt():
    levers = evaluate.PROVEN_LEVERS
    assert "only UNSOLVED seed-0 cell" not in levers
    advice = levers.split("mug→microwave + close (t9)", 1)[1]
    assert all(
        term in advice
        for term in (
            "agentview_high.png",
            "back_project",
            "Grasp",
            "move_pose",
            "capped",
            "pi0_doubled",
        )
    )
