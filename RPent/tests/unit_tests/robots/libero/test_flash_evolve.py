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

"""Offline unit tests for Flash-Evolve fast-slow procedural memory evolution."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from robots.libero.flash.evolve import (
    bootstrap_plan_from_corpus,
    load_evolving_plan,
    match_recovery_branch,
    patch_plan_with_branch,
    record_branch_outcome,
    run_flash_evolve,
)
from rpent.planner.base import Planner, PlannerResult
from rpent.planner.flash_evolve import FlashEvolvePlanner
from rpent.session import EnvState
from rpent.tools import ToolResult


class FakeLiberoToolkit:
    """In-memory toolkit recording EnvState transitions without MuJoCo or VLA."""

    def __init__(
        self,
        tmp_dir: Path,
        *,
        memory_root: Path,
        anchors_xyz: dict[str, list[float]],
        task_language: str = "pick up the black bowl and place it on the plate",
        fail_first_pick_prompt: str | None = None,
        stall_move_to_y_above: float | None = None,
        solve_on_release: bool = True,
    ) -> None:
        self.state = EnvState(tmp_dir / "state")
        self.memory = SimpleNamespace(root=memory_root)
        self.primitives = SimpleNamespace(molmo_client=None)
        self._anchors_xyz = {k: list(v) for k, v in anchors_xyz.items()}
        self._fail_first_pick_prompt = fail_first_pick_prompt
        self._stall_move_to_y_above = stall_move_to_y_above
        self._solve_on_release = solve_on_release
        self._solved = False
        self._holding = False
        self._eef = [0.0, 0.0, 0.72]
        self.executed: list[tuple[str, dict[str, Any]]] = []
        self.reset_called = False

        with self.state.record_step(
            state={"robot0_eef_pos": list(self._eef)},
            terminated=False,
            extras={"task_language": task_language},
        ):
            pass

    def solved(self) -> bool:
        return self._solved

    def execute_tool(self, name: str, arguments: dict[str, Any]) -> ToolResult:
        args = dict(arguments)
        self.executed.append((name, args))
        if name == "reset":
            self.reset_called = True
            raise AssertionError("Single-attempt mode must never call reset")
        if name == "segment":
            prompt = str(args.get("prompt", "")).strip()
            xyz = self._anchors_xyz.get(prompt)
            if xyz is None:
                # Fallback substring match for simplified phrases
                for key, value in self._anchors_xyz.items():
                    if key in prompt or prompt in key:
                        xyz = value
                        break
            if xyz is None:
                return ToolResult(data={"found": False})
            return ToolResult(
                data={"found": True, "world_xyz": list(xyz), "score": 0.85}
            )

        result_payload: dict[str, Any] = {"ok": True}
        if name in {"move_to", "move_pose"}:
            xyz = list(args["xyz"])
            if (
                name == "move_to"
                and self._stall_move_to_y_above is not None
                and abs(float(xyz[1])) > self._stall_move_to_y_above
            ):
                result_payload = {
                    "final_dist_m": 0.08,
                    "steps_used": int(args.get("max_steps", 80)),
                }
            else:
                self._eef = xyz
                result_payload = {"final_dist_m": 0.004, "steps_used": 12}
        elif name == "pi0_pick":
            prompt = str(args.get("prompt", ""))
            if (
                self._fail_first_pick_prompt is not None
                and prompt == self._fail_first_pick_prompt
            ):
                result_payload = {"success": False}
            else:
                self._holding = True
                result_payload = {"success": True}
        elif name == "set_gripper":
            result_payload = {"ok": True}
        elif name == "release":
            if self._holding and self._solve_on_release:
                self._solved = True
            self._holding = False
            result_payload = {"ok": True}

        cmd = {"action": name, **args}
        with self.state.record_step(
            state={"robot0_eef_pos": list(self._eef)},
            terminated=self._solved,
            command=cmd,
            result=result_payload,
            extras={"task_language": self.state.get(0).extras["task_language"]},
        ):
            pass
        return ToolResult(
            data={
                **result_payload,
                "log": {"command": cmd, "result": result_payload},
                "terminated": self._solved,
            }
        )


@pytest.mark.parametrize("outcome", ["tool_error", "truncation", "solved"])
def test_pick_retry_stops_at_the_first_error_or_terminal_step(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from robots.libero.flash.replay import replay_with_evolution

    toolkit = FakeLiberoToolkit(
        tmp_path, memory_root=tmp_path, anchors_xyz={}, fail_first_pick_prompt="pick"
    )
    execute_tool = toolkit.execute_tool
    calls = []

    def execute_with_retry_event(name: str, args: dict[str, Any]) -> ToolResult:
        calls.append(name)
        if calls == ["move_to", "pi0_pick", "move_to"] and outcome == "tool_error":
            return ToolResult(data={"error": "unreachable on retry"})
        raw = execute_tool(name, args)
        if len(calls) == 3:
            if outcome == "truncation":
                toolkit.state.latest_record().truncated = True
            elif outcome == "solved":
                toolkit._solved = True
        return raw

    monkeypatch.setattr(toolkit, "execute_tool", execute_with_retry_event)
    result = replay_with_evolution(
        toolkit,
        None,
        {
            "plan": [
                {"action": "move_to", "arguments": {"xyz": [0.0, 0.0, 0.7]}},
                {"action": "pi0_pick", "arguments": {"prompt": "pick"}},
                {"action": "release", "arguments": {}},
            ],
            "reference": {},
            "locator_of": {},
        },
    )
    assert calls == ["move_to", "pi0_pick", "move_to"]
    assert result["executed_steps"] == len(calls)
    if outcome == "tool_error":
        assert result["mode"] == "needs_handover"
        assert result["failure"]["failure_kind"] == "action_error"
        assert result["failure"]["failed_action"] == "move_to"
        assert "unreachable on retry" in result["failure"]["detail"]
    else:
        assert result["mode"] == (
            "flash_truncated" if outcome == "truncation" else "flash_fast"
        )


def test_recovery_uses_its_observed_anchor_and_recorded_grasp(tmp_path: Path) -> None:
    from robots.libero.flash.replay import replay_with_evolution

    toolkit = FakeLiberoToolkit(
        tmp_path, memory_root=tmp_path, anchors_xyz={"rim": [0.16, 0.02, 0.5]}
    )
    # The slow planner used a wrist-only reading after the main path moved the arm.
    # Its taught grasp has a different acceptance threshold from the base Flash pick.
    branch = {
        "branch_id": "rim-grasp",
        "anchors": [{"phrase": "rim", "camera": "wrist"}],
        "actions": [
            {
                "action": "move_to",
                "anchor": "rim",
                "anchor_distance": 0.01,
                "offset": [0, 0.01],
                "arguments": {"xyz": [0.25, 0.20, 0.7]},
            },
            {
                "action": "pi0_pick",
                "arguments": {"prompt": "pick", "lift_thresh": 0.08},
            },
            {"action": "release", "arguments": {}},
        ],
    }
    result = replay_with_evolution(
        toolkit,
        None,
        {
            "plan": [{"action": "move_to", "arguments": {"xyz": [0.0, 0.0, 0.7]}}],
            "reference": {},
            "locator_of": {},
        },
        branch_matcher=lambda *_args, **_kwargs: branch,
    )
    assert result["mode"] == "flash_branch"
    assert result["failure_step"] == 1
    assert [name for name, _ in toolkit.executed] == [
        "move_to",
        "segment",
        "move_to",
        "pi0_pick",
        "release",
    ]
    assert toolkit.executed[1][1]["step"] == result["failure_step"]
    assert toolkit.executed[1][1]["camera"] == "wrist"
    assert toolkit.executed[2][1]["xyz"] == pytest.approx([0.16, 0.03, 0.7])
    assert toolkit.executed[3][1] == {"prompt": "pick", "lift_thresh": 0.08}
    assert result["executed_steps"] == len(toolkit.executed)


def _write_seed0_task_specific(memory_root: Path) -> None:
    """Create a seed-0 audit + recipe pair mimicking GPT_6_astra_low."""
    ts = memory_root / "task-specific"
    ts.mkdir(parents=True, exist_ok=True)
    tag = "spatial_task_t0_s0"
    audit = {
        "suite": "libero_spatial_task",
        "task_id": 0,
        "seed": 0,
        "task_language": "pick up the black bowl and place it on the plate",
        "libero_terminated": True,
    }
    (ts / f"{tag}.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
    recipe = [
        {"action": "move_to", "xyz": [0.10, 0.05, 0.70], "gripper": -1},
        {"action": "pi0_pick", "prompt": "pick up the black bowl"},
        {"action": "set_gripper", "gripper": 1},
        {"action": "move_to", "xyz": [-0.10, 0.15, 0.70], "gripper": 1},
        {"action": "release"},
    ]
    (ts / f"{tag}_recipe.jsonl").write_text(
        "\n".join(json.dumps(row) for row in recipe) + "\n", encoding="utf-8"
    )
    # Use segment-measured anchors so SAM3-only grounding has exact phrases
    seg_dir = ts / "segments"
    seg_dir.mkdir(exist_ok=True)
    (seg_dir / "segment_00.json").write_text(
        json.dumps(
            {
                "found": True,
                "mode": "text",
                "prompt": "the black bowl",
                "world_xyz": [0.10, 0.05, 0.50],
            }
        ),
        encoding="utf-8",
    )
    (seg_dir / "segment_01.json").write_text(
        json.dumps(
            {
                "found": True,
                "mode": "text",
                "prompt": "the plate",
                "world_xyz": [-0.10, 0.15, 0.50],
            }
        ),
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    "plate_xy", [[-0.05, 0.22], [0.20, -0.02]], ids=["apart", "adjacent"]
)
def test_bootstrap_and_fast_zero_token_solve(
    tmp_path: Path, plate_xy: list[float]
) -> None:
    memory_root = tmp_path / "corpus"
    store_dir = tmp_path / "flash_evolved"
    _write_seed0_task_specific(memory_root)

    assert bootstrap_plan_from_corpus(memory_root, store_dir, "spatial_task_t0_s1")
    assert (store_dir / "spatial_task_t0_plan.json").is_file()
    assert (store_dir / "spatial_task_t0_anchors.json").is_file()

    # Shifted object layout on seed 1
    tk = FakeLiberoToolkit(
        tmp_path / "ep1",
        memory_root=memory_root,
        anchors_xyz={
            "the black bowl": [0.18, -0.02, 0.50],
            "the plate": [*plate_xy, 0.50],
        },
    )
    slow_calls = 0

    def slow_solve(_msg: str) -> PlannerResult:
        nonlocal slow_calls
        slow_calls += 1
        return PlannerResult(finish_result={"status": "success"})

    outcome = run_flash_evolve(
        tk,
        "spatial_task_t0_s1",
        slow_solve=slow_solve,
        user_message="Solve task",
        flash_memory_dir=store_dir,
        output_dir=tmp_path / "ep1",
    )
    assert outcome["done"] is True
    assert outcome["trace"]["execution_mode"] == "flash_fast"
    assert slow_calls == 0
    assert not tk.reset_called
    # Verify anchor-relative re-grounding moved to shifted bowl/plate coords
    moves = [args["xyz"] for name, args in tk.executed if name == "move_to"]
    assert moves[0][:2] == pytest.approx([0.18, -0.02], abs=1e-3)
    assert moves[1][:2] == pytest.approx(plate_xy, abs=1e-3)


def test_closed_loop_handover_patch_and_zero_token_branch_hit(tmp_path: Path) -> None:
    """Episode 1 rescues pick_failed via slow planner -> Episode 2 hits branch at 0 tokens."""
    memory_root = tmp_path / "corpus"
    store_dir = tmp_path / "flash_evolved"
    _write_seed0_task_specific(memory_root)

    # --- Episode 1 (seed 1): original prompt fails; slow planner rescues ---
    tk1 = FakeLiberoToolkit(
        tmp_path / "ep1",
        memory_root=memory_root,
        anchors_xyz={
            "the black bowl": [0.16, 0.02, 0.50],
            "the plate": [-0.08, 0.18, 0.50],
        },
        fail_first_pick_prompt="pick up the black bowl",
    )
    received_handover: list[str] = []

    def slow_rescue_ep1(handover_msg: str) -> PlannerResult:
        received_handover.append(handover_msg)
        # Slow planner uses rim offset (+0.045 y) + short prompt + carry + release
        tk1.execute_tool("move_pose", {"xyz": [0.16, 0.065, 0.66], "gripper": -1})
        tk1.execute_tool("pi0_pick", {"prompt": "grasp the bowl rim"})
        tk1.execute_tool("set_gripper", {"gripper": 1})
        tk1.execute_tool("move_to", {"xyz": [-0.08, 0.18, 0.70], "gripper": 1})
        tk1.execute_tool("release", {})
        return PlannerResult(
            finish_result={"status": "success"},
            stats={
                "total_input_tokens": 12000,
                "total_output_tokens": 600,
                "turns_used": 4,
            },
        )

    out1 = run_flash_evolve(
        tk1,
        "spatial_task_t0_s1",
        slow_solve=slow_rescue_ep1,
        user_message="Solve task",
        flash_memory_dir=store_dir,
        output_dir=tmp_path / "ep1",
    )
    assert out1["done"] is True
    assert out1["trace"]["execution_mode"] == "slow_handover_rescue"
    assert out1["trace"]["evolved_action"] == "patched_branch"
    assert not tk1.reset_called
    assert len(received_handover) == 1
    assert "SYSTEM-1 FLASH HANDOVER" in received_handover[0]
    assert "pick_failed" in received_handover[0]

    # Verify branch persisted with anchor-relative offsets in *_plan.json
    evolved = load_evolving_plan(store_dir, "spatial_task_t0")
    assert evolved["schema_version"] == 2
    assert len(evolved["recovery_branches"]) == 1
    branch = evolved["recovery_branches"][0]
    assert branch["trigger"]["failure_kind"] == "pick_failed"
    assert branch["actions"][0]["action"] == "move_pose"
    assert branch["actions"][0]["anchor"] == "the black bowl"
    assert branch["actions"][0]["offset"] == pytest.approx([0.0, 0.045], abs=1e-3)

    # --- Episode 2 (seed 2, shifted layout!): hits learned branch at 0 LLM tokens ---
    tk2 = FakeLiberoToolkit(
        tmp_path / "ep2",
        memory_root=memory_root,
        anchors_xyz={
            "the black bowl": [0.20, 0.04, 0.50],
            "the plate": [-0.04, 0.20, 0.50],
        },
        fail_first_pick_prompt="pick up the black bowl",
    )
    slow_ep2_called = False

    def slow_should_not_run(_msg: str) -> PlannerResult:
        nonlocal slow_ep2_called
        slow_ep2_called = True
        return PlannerResult(finish_result={"status": "failure"})

    out2 = run_flash_evolve(
        tk2,
        "spatial_task_t0_s2",
        slow_solve=slow_should_not_run,
        user_message="Solve task",
        flash_memory_dir=store_dir,
        output_dir=tmp_path / "ep2",
    )
    assert out2["done"] is True
    assert out2["trace"]["execution_mode"] == "flash_branch"
    assert out2["trace"]["evolved_action"] == "updated_branch_stats"
    assert slow_ep2_called is False
    # Verify the learned rim offset (+0.045 y) was transferred to the NEW bowl position!
    move_poses = [args["xyz"] for name, args in tk2.executed if name == "move_pose"]
    assert len(move_poses) == 1
    assert move_poses[0][:2] == pytest.approx([0.20, 0.085], abs=1e-3)

    # Empirical stats updated on disk
    evolved2 = load_evolving_plan(store_dir, "spatial_task_t0")
    stats = evolved2["recovery_branches"][0]["stats"]
    assert stats["attempts"] == 1
    assert stats["successes"] == 1
    assert stats["failures"] == 0


def test_motion_stall_triggers_handover_and_bad_branch_demotion(tmp_path: Path) -> None:
    memory_root = tmp_path / "corpus"
    store_dir = tmp_path / "flash_evolved"
    _write_seed0_task_specific(memory_root)
    bootstrap_plan_from_corpus(memory_root, store_dir, "spatial_task_t0_s1")

    # Seed with plate near workspace edge (y=0.30 > 0.25) stalls move_to
    tk = FakeLiberoToolkit(
        tmp_path / "ep_stall",
        memory_root=memory_root,
        anchors_xyz={
            "the black bowl": [0.10, 0.05, 0.50],
            "the plate": [-0.08, 0.30, 0.50],
        },
        stall_move_to_y_above=0.25,
    )

    def slow_use_move_pose(_msg: str) -> PlannerResult:
        tk.execute_tool("move_pose", {"xyz": [-0.08, 0.30, 0.70], "gripper": 1})
        tk.execute_tool("release", {})
        return PlannerResult(finish_result={"status": "success"})

    out = run_flash_evolve(
        tk,
        "spatial_task_t0_s1",
        slow_solve=slow_use_move_pose,
        user_message="Solve task",
        flash_memory_dir=store_dir,
    )
    assert out["done"] is True
    assert out["trace"]["failure"]["failure_kind"] == "motion_stalled"
    branch_id = out["trace"]["patched_branch_id"]

    # Record 3 consecutive failures -> branch must be automatically demoted
    for idx in range(3):
        record_branch_outcome(
            store_dir,
            "spatial_task_t0",
            branch_id,
            succeeded=False,
            cell_tag=f"spatial_task_t0_s{idx + 2}",
        )
    program = load_evolving_plan(store_dir, "spatial_task_t0")
    assert match_recovery_branch(program, out["trace"]["failure"]) is None


@pytest.mark.parametrize(
    "mug_visible_in",
    [None, "wrist", "agentview"],
    ids=["inferred", "wrist-reading", "wrist-reading-out-of-view"],
)
def test_cold_start_compiles_new_plan_for_subsequent_seeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, mug_visible_in: str | None
) -> None:
    """When a task has no seed-0 plan (like libero_10_swap/t6), slow solve compiles one."""
    from robots.libero.tools import LiberoPrimitives

    empty_corpus = tmp_path / "empty_corpus"
    empty_corpus.mkdir()
    store_dir = tmp_path / "flash_evolved"

    tk1 = FakeLiberoToolkit(
        tmp_path / "cold1",
        memory_root=empty_corpus,
        anchors_xyz={
            "the yellow and white mug": [0.12, -0.04, 0.50],
            "the plate": [-0.06, 0.14, 0.50],
        },
        task_language="put the yellow and white mug on the plate",
    )

    def slow_cold(_msg: str) -> PlannerResult:
        if mug_visible_in is not None:
            tk1.state.save(
                "segment_00.json",
                {
                    "found": True,
                    "mode": "text",
                    "prompt": "the yellow and white mug",
                    "camera": "wrist",
                    "world_xyz": [0.12, -0.04, 0.5],
                },
            )
        tk1.execute_tool("move_to", {"xyz": [0.12, -0.04, 0.70], "gripper": -1})
        tk1.execute_tool("pi0_pick", {"prompt": "pick up the yellow and white mug"})
        tk1.execute_tool("set_gripper", {"gripper": 1})
        tk1.execute_tool("move_to", {"xyz": [-0.06, 0.14, 0.70], "gripper": 1})
        tk1.execute_tool("release", {})
        return PlannerResult(
            finish_result={"status": "success"},
            stats={
                "total_input_tokens": 9000,
                "total_output_tokens": 400,
                "turns_used": 5,
            },
        )

    out1 = run_flash_evolve(
        tk1,
        "goal_task_t3_s1",
        slow_solve=slow_cold,
        user_message="Solve task",
        flash_memory_dir=store_dir,
        output_dir=tmp_path / "cold1",
    )
    assert out1["done"] is True
    assert out1["trace"]["execution_mode"] == "slow_cold_start_solved"
    assert out1["trace"]["evolved_action"] == "compiled_new_plan"
    assert (store_dir / "goal_task_t3_plan.json").is_file()
    assert (store_dir / "goal_task_t3_anchors.json").is_file()

    # Subsequent seed runs on the newly compiled Flash plan with 0 LLM calls!
    tk2 = FakeLiberoToolkit(
        tmp_path / "cold2",
        memory_root=empty_corpus,
        anchors_xyz={
            "the yellow and white mug": [0.15, 0.01, 0.50],
            "the plate": [-0.02, 0.17, 0.50],
        },
        task_language="put the yellow and white mug on the plate",
    )
    execute_tool = tk2.execute_tool
    mug_cameras = []

    def replay_tool(name: str, args: dict[str, Any]) -> ToolResult:
        getattr(LiberoPrimitives, name).args_schema.model_validate(args, strict=True)
        if name == "segment" and args["prompt"] == "the yellow and white mug":
            mug_cameras.append(args["camera"])
            if mug_visible_in and args["camera"] != mug_visible_in:
                return ToolResult(data={"found": False})
        return execute_tool(name, args)

    monkeypatch.setattr(tk2, "execute_tool", replay_tool)
    out2 = run_flash_evolve(
        tk2,
        "goal_task_t3_s2",
        slow_solve=lambda _: (_ for _ in ()).throw(
            AssertionError("should not call slow")
        ),
        user_message="Solve task",
        flash_memory_dir=store_dir,
        output_dir=tmp_path / "cold2",
    )
    assert out2["done"] is True
    assert out2["trace"]["execution_mode"] == "flash_fast"
    expected_cameras = ["wrist"] if mug_visible_in else ["agentview"]
    if mug_visible_in == "agentview":
        expected_cameras.append("agentview")
    assert mug_cameras == expected_cameras
    assert next(args["xyz"] for name, args in tk2.executed if name == "move_to")[
        :2
    ] == pytest.approx([0.15, 0.01])


def test_patch_merges_spatial_bounds_on_duplicate_signature(tmp_path: Path) -> None:
    memory_root = tmp_path / "corpus"
    store_dir = tmp_path / "flash_evolved"
    _write_seed0_task_specific(memory_root)
    bootstrap_plan_from_corpus(memory_root, store_dir, "spatial_task_t0_s1")

    base_branch = {
        "branch_id": "rb-abc",
        "signature": "abc",
        "trigger": {
            "failure_kind": "pick_failed",
            "failed_step_idx": 1,
            "failed_action": "pi0_pick",
            "anchor": "the black bowl",
            "spatial_bounds": {"anchor_xy_range": [[0.10, 0.10], [0.05, 0.05]]},
        },
        "mode": "complete_episode",
        "actions": [{"action": "release", "arguments": {}}],
        "provenance": {"cell_tag": "spatial_task_t0_s1"},
        "stats": {"attempts": 0, "successes": 0, "failures": 0},
    }
    _, id1 = patch_plan_with_branch(store_dir, "spatial_task_t0", base_branch)
    second = json.loads(json.dumps(base_branch))
    second["trigger"]["spatial_bounds"]["anchor_xy_range"] = [
        [0.20, 0.20],
        [0.00, 0.00],
    ]
    second["provenance"]["cell_tag"] = "spatial_task_t0_s2"
    _, id2 = patch_plan_with_branch(store_dir, "spatial_task_t0", second)
    assert id1 == id2
    program = load_evolving_plan(store_dir, "spatial_task_t0")
    assert len(program["recovery_branches"]) == 1
    bounds = program["recovery_branches"][0]["trigger"]["spatial_bounds"][
        "anchor_xy_range"
    ]
    assert bounds == [[0.10, 0.20], [0.00, 0.05]]


def test_flash_evolve_planner_wrapper_and_zero_token_stats(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    memory_root = tmp_path / "corpus"
    store_dir = tmp_path / "flash_evolved"
    _write_seed0_task_specific(memory_root)
    tk = FakeLiberoToolkit(
        tmp_path / "ep_wrap",
        memory_root=memory_root,
        anchors_xyz={
            "the black bowl": [0.10, 0.05, 0.50],
            "the plate": [-0.10, 0.15, 0.50],
        },
    )

    class DummySlowPlanner(Planner):
        def solve(self, **kwargs: Any) -> PlannerResult:
            raise AssertionError("Fast path should not invoke slow planner")

    import rpent.planner.flash_evolve as fe_mod

    monkeypatch.setattr(
        fe_mod,
        "get_robot_spec",
        lambda _name: SimpleNamespace(run_flash_evolve=run_flash_evolve),
    )

    planner = FlashEvolvePlanner(
        recipe_tag="spatial_task_t0_s1",
        robot_name="libero",
        slow_planner=DummySlowPlanner(),
        slow_planner_name="codex",
        flash_memory_dir=store_dir,
        output_dir=tmp_path / "ep_wrap",
        evolve_enabled=True,
    )
    res = planner.solve(
        system_prompt="sys",
        user_message="usr",
        toolkit=tk,  # type: ignore[arg-type]
        max_turns=20,
    )
    assert res.error is None
    assert res.finish_result is not None and res.finish_result["status"] == "success"
    assert res.stats["total_input_tokens"] == 0
    assert res.stats["execution_mode"] == "flash_fast"


def test_claude_handover_accounts_for_cached_input_in_local_store(
    tmp_path, monkeypatch
):
    from rpent.dashboard.events import NullDashboardEventSink
    from rpent.planner.base import build_planner
    from rpent.planner.claude_code import ClaudeCodePlanner

    corpus = tmp_path / "corpus"
    toolkit = FakeLiberoToolkit(
        tmp_path / "episode", memory_root=corpus, anchors_xyz={}
    )
    monkeypatch.setattr(
        ClaudeCodePlanner,
        "solve",
        lambda self, **kwargs: PlannerResult(
            stats={
                "total_input_tokens": 5,
                "total_cache_creation_input_tokens": 7,
                "total_cache_read_input_tokens": 11,
                "total_output_tokens": 3,
                "total_cost_usd": 0.01,
            }
        ),
    )
    planner = build_planner(
        "flash_evolve",
        flash_slow_planner="claude_code",
        memory_dir=corpus,
        output_dir=tmp_path,
        recipe_tag="spatial_task_t0_s1",
        robot_name="libero",
        dashboard_events=NullDashboardEventSink(),
    )
    result = planner.solve(
        toolkit=toolkit, system_prompt="", user_message="task", max_turns=1
    )
    assert result.error is None
    assert result.stats["execution_mode"] == "slow_cold_start_failed"
    assert Path(result.stats["flash_memory_dir"]) == corpus / "flash_evolved"
    assert result.stats["total_input_tokens"] == 23
    assert result.stats["total_cached_input_tokens"] == 11
    assert result.stats["total_cost_usd"] == 0.01


def test_libero_parse_config_allows_flash_evolve_without_molmo(tmp_path: Path) -> None:
    from robots.libero.robot_spec import _parse_config

    args = argparse.Namespace(
        suite="libero_spatial_task",
        task=0,
        seed=1,
        planner="flash_evolve",
        explore=False,
        molmo_endpoint=None,
        env_endpoint=None,
        initial_scene=None,
        memory_profile="hf",
        memory_dir=None,
        collect_flywheel_data=False,
        enable_direct_action=False,
        explore_sessions=1,
        output_dir=str(tmp_path / "out"),
    )
    cfg = _parse_config(args)
    assert cfg.recipe_tag == "spatial_task_t0_s1"

    # Strict legacy flash mode still requires --molmo-endpoint
    args.planner = "flash"
    with pytest.raises(ValueError, match="molmo-endpoint"):
        _parse_config(args)


def test_repro_command_and_summarize_batch(tmp_path: Path) -> None:
    from repro.flash_evolve import summarize_batch
    from repro.liberopro_baseline import command

    batch = tmp_path / "batch"
    batch.mkdir()
    plan = {
        "condition": "flash_evolve",
        "flash_slow_planner": "codex",
        "flash_memory_dir": str(batch / "flash_memory"),
        "flash_evolve_enabled": True,
        "planned_episodes": 4,
        "episodes": [
            {
                "suite": "libero_spatial_task",
                "task": 0,
                "seed": 1,
                "episode_id": "s1-t0",
            },
            {
                "suite": "libero_spatial_task",
                "task": 0,
                "seed": 2,
                "episode_id": "s2-t0",
            },
            {
                "suite": "libero_spatial_task",
                "task": 1,
                "seed": 1,
                "episode_id": "s1-t1",
            },
            {
                "suite": "libero_spatial_task",
                "task": 1,
                "seed": 2,
                "episode_id": "s2-t1",
            },
        ],
    }
    (batch / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    argv = command(
        batch,
        plan["episodes"][0],
        {"environment_gpu": 1, "vla_port": 8420, "sam3_port": 8414},
    )
    assert "--planner" in argv and argv[argv.index("--planner") + 1] == "flash_evolve"
    assert "--flash-memory-dir" in argv

    ep1 = batch / "episodes" / "s1-t0"
    ep2 = batch / "episodes" / "s2-t0"
    ep1.mkdir(parents=True)
    ep2.mkdir(parents=True)
    for directory, input_tokens, output_tokens in [(ep1, 10000, 500), (ep2, 0, 0)]:
        (directory / "metrics.json").write_text(
            json.dumps(
                {
                    "valid": True,
                    "native_success": True,
                    "total_input_tokens": input_tokens,
                    "total_output_tokens": output_tokens,
                }
            ),
            encoding="utf-8",
        )
    for episode_id in ("s1-t1", "s2-t1"):
        directory = batch / "episodes" / episode_id
        directory.mkdir()
        (directory / "flash_evolve_trace.json").write_text(
            json.dumps({"done": True, "execution_mode": "flash_fast"}),
            encoding="utf-8",
        )
    (batch / "episodes" / "s1-t1" / "metrics.json").write_text(
        json.dumps({"valid": False, "native_success": True}), encoding="utf-8"
    )
    (ep1 / "flash_evolve_trace.json").write_text(
        json.dumps(
            {
                "done": True,
                "execution_mode": "slow_handover_rescue",
                "evolved_action": "patched_branch",
                "slow_result": {
                    "stats": {"total_input_tokens": 10000, "total_output_tokens": 500}
                },
            }
        ),
        encoding="utf-8",
    )
    (ep2 / "flash_evolve_trace.json").write_text(
        json.dumps(
            {
                "done": True,
                "execution_mode": "flash_branch",
                "evolved_action": "updated_branch_stats",
                "slow_result": None,
            }
        ),
        encoding="utf-8",
    )
    summary = summarize_batch(batch)
    assert summary["completed"] == 2
    assert summary["infrastructure_errors"] == 1
    assert summary["pending"] == 1
    assert summary["solved"] == 2
    assert summary["success_rate"] == 1.0
    assert summary["per_seed"]["1"]["zero_token_solved"] == 0
    assert summary["per_seed"]["2"]["zero_token_solved"] == 1
    assert summary["per_seed"]["2"]["branch_hit_solved"] == 1
    assert summary["total_input_tokens"] == 10000


@pytest.mark.parametrize(
    ("condition", "slow_planner", "slow_result", "requires_spans"),
    [
        ("original_harnessvla_full", "codex", None, True),
        ("flash_evolve", "codex", None, False),
        ("flash_evolve", "codex", {"stats": {}}, True),
        ("flash_evolve", "api", {"stats": {}}, False),
        ("flash_evolve", "claude_code", {"stats": {}}, False),
    ],
)
def test_batch_telemetry_matches_executed_planner(
    tmp_path, condition, slow_planner, slow_result, requires_spans
) -> None:
    from repro.liberopro_baseline import _telemetry_error

    plan = {"condition": condition, "flash_slow_planner": slow_planner}
    coverage = {
        "sampling_spans": 0,
        "llm_request_spans": 0,
        "vla_client_calls": 2,
        "vla_server_calls": 2,
    }
    (tmp_path / "flash_evolve_trace.json").write_text(
        json.dumps({"slow_result": slow_result}), encoding="utf-8"
    )
    error = _telemetry_error(plan, tmp_path, coverage)
    assert error == ("Codex request telemetry missing" if requires_spans else None)
    coverage.update(sampling_spans=1, llm_request_spans=1, vla_server_calls=1)
    assert (
        _telemetry_error(plan, tmp_path, coverage) == "VLA telemetry coverage mismatch"
    )


def test_prepare_stream_uses_backend_model_and_unique_episode_ids(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from repro.flash_evolve import prepare
    from repro.liberopro_baseline import write

    monkeypatch.setattr("repro.flash_evolve.now", lambda: "test-time")
    monkeypatch.setattr("repro.liberopro_baseline.now", lambda: "test-time")
    baseline = tmp_path / "baseline"
    snapshot = tmp_path / "frozen-hf"
    write(
        baseline / "plan.json",
        {
            "model": "gpt-6-astra",
            "condition": "original_harnessvla_full",
            "planned_episodes": 80,
            "runtime_memory_root": str(snapshot),
            "episodes": [
                {
                    "suite": f"suite_{s}",
                    "task": t,
                    "seed": 1,
                    "initial_scene": "frozen.npz",
                }
                for s in range(8)
                for t in range(10)
            ],
            "deployment": {"environment_gpu": 1, "vla_port": 8420, "sam3_port": 8414},
        },
    )
    output = tmp_path / "batch"
    args = argparse.Namespace(
        baseline=baseline,
        output=output,
        suite="suite_0",
        task=0,
        seeds=[1, 1],
        flash_memory_dir=snapshot / "flash",
        slow_planner="claude_code",
        model=None,
        evolve=False,
        molmo_endpoint=None,
    )
    with pytest.raises(ValueError, match="unique"):
        prepare(args)
    args.seeds = [1, 2]
    args.evolve = True
    with pytest.raises(ValueError, match="outside the frozen HF snapshot"):
        prepare(args)
    args.evolve = False
    prepare(args)
    assert not args.flash_memory_dir.exists()
    plan = json.loads((output / "plan.json").read_text(encoding="utf-8"))
    assert plan["model"] == "sonnet"
    argv = plan["example_command"]
    assert argv[argv.index("--model") + 1] == plan["model"]
    first, second = plan["episodes"]
    assert first["episode_id"] != second["episode_id"]
    assert "initial_scene" in first and "initial_scene" not in second
