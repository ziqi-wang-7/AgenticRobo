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

"""Drawer identity, progress and recovery evidence stay current and attributed."""

from types import SimpleNamespace

import pytest

from robots.libero.action_evidence import (
    ActionCandidate,
    ActionVisualCheck,
    build_action_query,
)
from robots.libero.recovery_evidence import (
    DrawerProgress,
    drawer_progress_failure,
    motion_alert,
)
from robots.libero.tools import LiberoPrimitives
from rpent.session import StepRecord
from tests.unit_tests.rpent.memory import test_memcodex_m2_contracts as m2

indexed = m2.indexed
offline_tokenizer = m2.offline_tokenizer
online = m2.online


def candidate(tool="pi0_doubled", **drawer):
    args = (
        {"prompt": "open the bottom drawer", "max_chunks": 8}
        if tool == "pi0_doubled"
        else {"xyz": [0.1, 0.2, 1.0], "gripper": -1.0}
    )
    return ActionCandidate(
        tool=tool,
        args=args,
        roles={"handle": "bottom_drawer_handle"},
        drawer={
            "target_level": "bottom",
            "operation": "open",
            "stage": "actuate",
            "direction": "outward",
            **drawer,
        },
    )


def query(c, checks=()):
    tools = {
        t.name: t
        for t in (
            LiberoPrimitives.pi0_doubled,
            LiberoPrimitives.move_pose,
            LiberoPrimitives.move_to,
        )
    }
    return build_action_query(
        c,
        tools=tools,
        step=2,
        task_objects={"bottom_drawer_handle", "top_drawer_handle", "drawer_handle"},
        drawer_goal={"level": "bottom", "operation": "open"},
        localizations={"bottom_drawer_handle": {"id": "loc-1"}},
        artifacts={"agentview_high.png"},
        visual_checks=list(checks),
    )


@pytest.mark.parametrize("tool", ["pi0_doubled", "move_pose", "move_to"])
def test_drawer_requires_level_access_and_contact_even_with_geometry(tool):
    _, q = query(candidate(tool))
    assert "handle_level:bottom_drawer_handle" in q.required_facts
    assert "access_clear:bottom_drawer_handle" in q.required_facts
    assert "contact_ready:bottom_drawer_handle" in q.required_facts
    assert {f.key for f in q.facts if f.status == "verified"} == {
        "localized:bottom_drawer_handle"
    }


def test_wrong_handle_level_or_direction_is_rejected():
    with pytest.raises(ValueError, match="identity.*level"):
        query(candidate(target_level="top"))
    with pytest.raises(ValueError, match="direction"):
        query(candidate(direction="inward"))


def test_visual_level_claim_is_current_and_never_becomes_physical_verification():
    check = ActionVisualCheck(
        fact="handle_level",
        object_id="bottom_drawer_handle",
        step=2,
        artifact="agentview_high.png",
        value="bottom",
    )
    _, q = query(candidate(), [check])
    assert (
        next(f for f in q.facts if f.key.startswith("handle_level:")).status
        == "claimed"
    )
    with pytest.raises(ValueError, match="conflicts"):
        query(candidate(), [check.model_copy(update={"value": "middle"})])
    with pytest.raises(ValueError, match="current camera"):
        query(candidate(), [check.model_copy(update={"step": 1})])


@pytest.mark.parametrize(
    "level,direction,kind",
    [
        ("middle", "outward", "wrong_drawer"),
        ("bottom", "none", "contact_stalled"),
        ("bottom", "inward", "contact_stalled"),
        ("bottom", "outward", None),
        ("unknown", "unknown", None),
    ],
)
def test_drawer_specific_progress_predicates(level, direction, kind):
    previous = StepRecord(1, {}, artifacts={"agentview.png"})
    current = StepRecord(2, {}, artifacts={"agentview.png"})
    progress = DrawerProgress(
        previous_step=1,
        step=2,
        artifact="agentview.png",
        target_level="bottom",
        observed_level=level,
        expected_direction="outward",
        observed_direction=direction,
    )
    assert drawer_progress_failure(progress, current, previous) == kind
    with pytest.raises(ValueError, match="consecutive"):
        drawer_progress_failure(
            progress.model_copy(update={"previous_step": 0}), current, previous
        )


def test_motion_residual_uses_command_tolerance_and_preserves_native_success():
    record = StepRecord(
        2,
        {},
        command={"action": "move_to", "max_steps": 50, "tol": 0.005},
        result={"final_dist_m": 0.02, "steps_used": 50},
    )
    assert motion_alert(record) == "contact_stalled"
    record.terminated = True
    assert motion_alert(record) is None
    record.terminated = False
    record.command = {"action": "pi0_doubled"}
    record.result = {"success": False}
    assert motion_alert(record) is None


def test_online_failure_keeps_object_and_phase_and_rejects_stale_image(online):
    kit, record, config = online
    record.artifacts.add("agentview.png")
    report = {
        "kind": "object_slipped",
        "object_id": "milk",
        "phase": "transport",
        "step": 0,
        "artifact": "agentview.png",
    }
    response = kit.retrieve_memory(step=0, failure=report)
    assert "object_slipped" in response.data["memory_context"]
    with pytest.raises(ValueError, match="current camera"):
        kit.retrieve_memory(step=0, failure={**report, "step": 1})
    with pytest.raises(ValueError, match="object"):
        kit.retrieve_memory(step=0, failure={**report, "object_id": "imaginary"})


def test_online_wrong_drawer_enters_recovery_even_without_recover_purpose(online):
    kit, record, config = online
    profile = kit.memory.controller.indexed.profile(m2.TASK)
    profile.update(
        drawer_goal={"level": "bottom"}, context_objects=["bottom_drawer_handle"]
    )
    record.step_idx = 1
    record.artifacts.add("agentview.png")
    record.command = {"action": "pi0_doubled"}
    previous = StepRecord(0, {}, artifacts={"agentview.png"})
    kit._state = SimpleNamespace(latest_record=lambda: record, get=lambda s: previous)
    response = kit.retrieve_memory(
        step=1,
        drawer_progress={
            "previous_step": 0,
            "step": 1,
            "artifact": "agentview.png",
            "target_level": "bottom",
            "observed_level": "middle",
            "expected_direction": "outward",
            "observed_direction": "outward",
        },
    )
    assert "recovery_case:wrong_drawer" in response.data["memory_context"]
