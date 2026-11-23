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

"""Regression tests for learning from the live LIBERO trace."""

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from robots.libero.flash.evolve import (
    bootstrap_plan_from_corpus,
    compile_new_plan_from_state,
    compile_recovery_branch,
    load_evolving_plan,
    match_recovery_branch,
    patch_plan_with_branch,
    run_flash_evolve,
)
from robots.libero.flash.replay import replay_with_evolution
from robots.libero.tools import LiberoPrimitives
from rpent.tools import ToolResult
from tests.unit_tests.robots.libero.test_flash_evolve import (
    FakeLiberoToolkit,
    _write_seed0_task_specific,
)


def test_rescue_measured_anchor_is_relocalized_before_branch(tmp_path: Path) -> None:
    corpus, store = tmp_path / "corpus", tmp_path / "learned"
    _write_seed0_task_specific(corpus)
    xy = {"the black bowl": [0.1, 0.05, 0.5], "the plate": [-0.1, 0.15, 0.5]}
    tag = "spatial_task_t0"

    def episode(seed: int, mug_x: float, solve_on_release: bool = True):
        return FakeLiberoToolkit(
            tmp_path / f"ep{seed}",
            memory_root=corpus,
            anchors_xyz={**xy, "the mug handle": [mug_x, -0.2, 0.5]},
            fail_first_pick_prompt="pick up the black bowl",
            solve_on_release=solve_on_release,
        )

    first = episode(1, 0.22)

    def rescue(_msg: str) -> None:
        # The real segment tool saves JSON on the photographed step without
        # advancing the environment or adding a command/result to the record.
        first.state.save(
            "segment_03.json",
            {
                "found": True,
                "mode": "text",
                "prompt": "the mug handle",
                "camera": "wrist",
                "world_xyz": [0.22, -0.2, 0.5],
            },
        )
        first.execute_tool("pi0_pick", {"prompt": "pick up the black bowl"})
        first.execute_tool("move_pose", {"xyz": [0.23, -0.2, 0.66]})
        first.execute_tool("execute_action", {"values": [0.0] * 7})
        first.execute_tool("pi0_pick", {"prompt": "grasp rim"})
        first.execute_tool("release", {})

    result = run_flash_evolve(
        first, tag + "_s1", slow_solve=rescue, user_message="", flash_memory_dir=store
    )
    assert result["trace"]["evolved_action"] == "patched_branch"
    (branch,) = load_evolving_plan(store, tag)["recovery_branches"]
    assert [s["action"] for s in branch["actions"]] == [
        "move_pose",
        "execute_action",
        "pi0_pick",
        "release",
    ]
    assert branch["actions"][0]["offset"] == pytest.approx([0.01, 0.0])
    assert branch["anchors"][0]["camera"] == "wrist"

    second = episode(2, 0.30)
    result = run_flash_evolve(
        second,
        tag + "_s2",
        slow_solve=lambda _: pytest.fail("unexpected handover"),
        user_message="",
        flash_memory_dir=store,
    )
    assert result["trace"]["execution_mode"] == "flash_branch"
    (args,) = [args for tool, args in second.executed if tool == "move_pose"]
    assert args["xyz"] == pytest.approx([0.31, -0.2, 0.66])

    third = episode(3, 0.28, solve_on_release=False)

    def rescue_after_attempt(_msg: str) -> None:
        third._solve_on_release = True
        third.execute_tool("pi0_pick", {"prompt": "pick up from the current pose"})
        third.execute_tool("release", {})

    result = run_flash_evolve(
        third,
        tag + "_s3",
        slow_solve=rescue_after_attempt,
        user_message="",
        flash_memory_dir=store,
    )
    assert result["done"] is True
    assert (
        result["trace"]["evolution_skipped"] == "branch_changed_state_before_handover"
    )
    assert len(load_evolving_plan(store, tag)["recovery_branches"]) == 1


@pytest.mark.parametrize(
    ("route", "contact_args"),
    [
        ("cold_start", {"prompt": "press the knob", "lift_thresh": 999.0}),
        ("branch", {"prompt": "turn on the stove"}),
    ],
)
def test_learned_contact_pick_continues_without_a_grasp(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, route: str, contact_args: dict
) -> None:
    contact_args = {**contact_args, "max_chunks": 2}

    def episode(path: Path) -> FakeLiberoToolkit:
        tk = FakeLiberoToolkit(
            path,
            memory_root=tmp_path,
            anchors_xyz={
                "the black bowl": [-0.25, -0.25, 0.5],
                "the plate": [0.1, -0.1, 0.5],
            },
        )
        ordinary_tool = tk.execute_tool
        # The real contact action consumes policy chunks without lifting an object.
        static_arm = SimpleNamespace(
            _last_obs_eef_z=0.70,
            _last_obs_gripper=0.08,
            _vlm_chunk=lambda _: None,
            env=SimpleNamespace(terminated=False, truncated=False),
        )

        def execute(name: str, args: dict) -> ToolResult:
            if name != "pi0_pick" or args["prompt"] != contact_args["prompt"]:
                return ordinary_tool(name, args)
            assert args == contact_args
            result = LiberoPrimitives.pi0_pick(static_arm, **args).data
            assert result["success"] is False and result["chunks_used"] == 2
            tk.executed.append((name, dict(args)))
            command = {"action": name, **args}
            with tk.state.record_step(
                state=tk.state.get().state, command=command, result=result
            ):
                pass
            return ToolResult(data={"log": {"command": command, "result": result}})

        monkeypatch.setattr(tk, "execute_tool", execute)
        return tk

    taught = episode(tmp_path / "teach")
    taught.execute_tool("move_to", {"xyz": [0.25, 0.25, 0.7]})
    taught.execute_tool("pi0_pick", contact_args)
    taught.execute_tool("move_to", {"xyz": [-0.25, -0.25, 0.7]})
    taught.execute_tool("pi0_pick", {"prompt": "grasp the bowl"})
    taught.execute_tool("move_to", {"xyz": [0.1, -0.1, 0.7], "gripper": 1})
    taught.execute_tool("release", {})
    assert taught.solved()
    tag = "spatial_task_t0_s1"
    if route == "cold_start":
        compile_new_plan_from_state(
            state=taught.state, cell_tag=tag, store_dir=tmp_path, slow_planner="codex"
        )
        program = load_evolving_plan(tmp_path, "spatial_task_t0")
        assert program["reference"]["the black bowl"] == pytest.approx([-0.25, -0.25])
    else:
        failure = {"failure_kind": "plan_incomplete", "failed_step_idx": 0}
        branch = compile_recovery_branch(
            state=taught.state,
            handover_step=0,
            failure=failure,
            live_anchors={},
            cell_tag=tag,
            slow_planner="codex",
        )
        assert branch is not None
        program = {
            "plan": [],
            "reference": {},
            "locator_of": {},
            "recovery_branches": [branch],
        }

    tk = episode(tmp_path / "replay")
    result = replay_with_evolution(
        tk, None, program, branch_matcher=match_recovery_branch
    )
    assert result["mode"] == ("flash_fast" if route == "cold_start" else "flash_branch")
    assert result["done"] is True
    assert [item["status"] for item in result["executed_trace"]] == ["ok"] * 6
    assert [args["prompt"] for name, args in tk.executed if name == "pi0_pick"] == [
        contact_args["prompt"],
        "grasp the bowl",
    ]


def test_frozen_evaluation_does_not_create_store(tmp_path: Path) -> None:
    corpus, store = tmp_path / "corpus", tmp_path / "frozen"
    _write_seed0_task_specific(corpus)
    tk = FakeLiberoToolkit(
        tmp_path / "run",
        memory_root=corpus,
        anchors_xyz={
            "the black bowl": [0.1, 0.05, 0.5],
            "the plate": [-0.1, 0.15, 0.5],
        },
    )
    result = run_flash_evolve(
        tk,
        "spatial_task_t0_s1",
        slow_solve=lambda _: pytest.fail("unexpected handover"),
        user_message="",
        flash_memory_dir=store,
        evolve_enabled=False,
    )
    assert result["trace"]["execution_mode"] == "flash_fast"
    assert not store.exists()


def test_memory_compilation_failure_preserves_env_success(tmp_path: Path) -> None:
    tk = FakeLiberoToolkit(
        tmp_path / "run",
        memory_root=tmp_path / "corpus",
        task_language="unsupported instruction",
        anchors_xyz={},
    )

    def solve(_msg: str) -> None:
        tk.execute_tool("pi0_pick", {"prompt": "grasp"})
        tk.execute_tool("release", {})

    result = run_flash_evolve(
        tk,
        "goal_task_t0_s1",
        slow_solve=solve,
        user_message="",
        flash_memory_dir=tmp_path / "store",
    )
    assert result["trace"]["execution_mode"] == "slow_cold_start_solved"
    assert result["done"] is True
    assert "unsupported LIBERO" in result["trace"]["evolution_error"]
    assert not (tmp_path / "store" / "goal_task_t0_plan.json").exists()


def test_distinct_rescues_survive_merge_and_late_cold_start(tmp_path: Path) -> None:
    store, corpus = tmp_path / "store", tmp_path / "corpus"
    _write_seed0_task_specific(corpus)
    tag = "spatial_task_t0"
    bootstrap_plan_from_corpus(corpus, store, tag + "_s1")
    original = {
        "branch_id": "rb-legacy",
        "signature": "legacy",
        "trigger": {
            "failure_kind": "pick_failed",
            "failed_step_idx": 1,
            "failed_action": "pi0_pick",
            "anchor": "the bowl",
            "spatial_bounds": {"eef_xy_range": [[0, 0], [0, 0]]},
        },
        "actions": [
            {
                "action": "move_pose",
                "arguments": {"xyz": [0.1, 0.2, 0.62]},
                "anchor": "the bowl",
                "offset": [0.02, 0.03],
                "anchor_distance": 0.036,
            },
            {"action": "rotate_wrist", "arguments": {"delta_yaw": 0.3}},
            {"action": "move_to", "arguments": {"xyz": [-0.2, 0.0, 0.8]}},
        ],
    }
    _, original_id = patch_plan_with_branch(store, tag, original)
    shifted = deepcopy(original)
    shifted["actions"][0]["arguments"]["xyz"][0] = 0.3
    _, merged_id = patch_plan_with_branch(store, tag, shifted)
    assert merged_id == original_id
    for index, key, value in [
        (0, "xyz", [0.1, 0.2, 0.70]),
        (1, "delta_yaw", -0.3),
        (2, "xyz", [0.2, 0.0, 0.8]),
    ]:
        distinct = deepcopy(original)
        distinct["actions"][index]["arguments"][key] = value
        _, branch_id = patch_plan_with_branch(store, tag, distinct)
        assert branch_id != original_id
    # A parallel seed that started cold finishes after the preceding patches.
    tk = FakeLiberoToolkit(tmp_path / "late", memory_root=corpus, anchors_xyz={})
    tk.execute_tool("pi0_pick", {"prompt": "pick up bowl"})
    tk.execute_tool("release", {})
    assert (
        compile_new_plan_from_state(
            state=tk.state, cell_tag=tag + "_s2", store_dir=store, slow_planner="codex"
        )
        is None
    )
    program = load_evolving_plan(store, tag)
    assert len(program["recovery_branches"]) == 4
    trigger = original["trigger"]
    assert match_recovery_branch(program, {**trigger, "eef_xy": None}) is None
    assert (
        match_recovery_branch(program, {**trigger, "eef_xy": [0, 0], "anchor": "other"})
        is None
    )
    assert (
        match_recovery_branch(
            program, {**trigger, "eef_xy": [0, 0], "failed_step_idx": 4}
        )
        is None
    )
    assert match_recovery_branch(program, {**trigger, "eef_xy": [0, 0]}) is not None
