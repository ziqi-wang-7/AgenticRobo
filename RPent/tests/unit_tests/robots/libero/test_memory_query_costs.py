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

"""Real tool declarations and warnings must not force speculative retrieval."""

import json

import pytest
from pydantic import ValidationError

from robots.libero.robot_spec import get_robot_spec
from robots.libero.toolkit import LiberoToolkit
from rpent.tools.base import ToolResult
from tests.unit_tests.robots.libero import test_action_evidence_contracts as actions
from tests.unit_tests.rpent.memory import test_memcodex_m2_contracts as m2

indexed = m2.indexed
offline_tokenizer = m2.offline_tokenizer
online = m2.online
action_run = actions.action_run


def test_normal_declaration_dispatches_with_geometry_and_rejects_drawer_inputs(
    action_run,
):
    kit, record, config = action_run
    declaration = kit._retrieval_tool()
    schema = json.dumps(declaration.input_schema)
    assert "DrawerAction" not in schema and "DrawerProgress" not in schema
    assert "drawer_progress" not in declaration.input_schema["properties"]
    with pytest.raises(ValidationError):
        declaration.args_schema.model_validate(
            {"step": 0, "drawer_progress": {}}, strict=True
        )
    args = declaration.args_schema.model_validate(
        {
            "step": 0,
            "candidate": actions._pick(),
            "localizations": {"milk": m2.localize(kit)},
        },
        strict=True,
    )
    result = declaration(
        **{field: getattr(args, field) for field in type(args).model_fields}
    )
    assert not result.is_error
    assert actions._trace(config)["memory_read_complete"]


def test_drawer_task_still_exposes_validated_drawer_contract(online):
    kit, _, _ = online
    profile = kit.memory.controller.indexed.profile(m2.TASK)
    profile["drawer_goal"] = {"level": "bottom", "operation": "open"}
    declaration = kit._retrieval_tool()
    assert "drawer_progress" in declaration.input_schema["properties"]
    assert "DrawerAction" in json.dumps(declaration.input_schema)
    with pytest.raises(ValidationError):
        declaration.args_schema.model_validate(
            {"step": 0, "drawer_progress": {"target_level": "invented"}}, strict=True
        )


def test_servo_warning_does_not_force_execute_into_recovery(online):
    kit, record, config = online
    record.command = {"action": "move_to", "max_steps": 80, "tol": 0.01}
    record.result = {"final_dist_m": 0.1, "steps_used": 80}
    kit.retrieve_memory(
        step=0, localizations={"milk": m2.localize(kit), "basket": m2.localize(kit)}
    )
    trace = actions._trace(config)
    assert not trace["state"]["needs_recovery"] and not trace["last_action_failed"]
    assert "failure_kind" not in trace["state"]
    assert trace["memory_read_complete"]
    kit.retrieve_memory(
        step=0, purpose="recover", issue="object slipped during transport"
    )
    assert actions._trace(config)["state"]["needs_recovery"]


@pytest.mark.parametrize("name", ["pi0_doubled", "move_to"])
def test_primitive_result_does_not_order_recovery_retrieval(online, monkeypatch, name):
    kit, record, _ = online
    record.command = {"action": name, "max_steps": 80, "tol": 0.01}
    record.result = {"success": False, "final_dist_m": 0.1, "steps_used": 80}
    monkeypatch.setattr(
        LiberoToolkit,
        "_execute_primitive",
        lambda *a, **k: ToolResult(data={"state": "observed"}),
    )
    result = kit._execute_primitive(name, lambda: None)
    assert (
        "retrieve_memory" not in result.to_text()
        and "purpose=recover" not in result.to_text()
    )
    assert "progress_check" not in result.data
    if name == "move_to":
        assert result.data["motion_check"]["failure_confirmed"] is False
    else:
        assert result.data == {"state": "observed"}


def test_task_scoped_prompt_and_single_perception_protocol():
    variables = {
        "memory_mode": "m2",
        "memcodex_action_evidence": True,
        "recipe_tag": "test",
        "output_dir": "/tmp/test",
    }
    normal = get_robot_spec().prompts.render("system", variables=variables)
    drawer = get_robot_spec().prompts.render(
        "system", variables={**variables, "memcodex_drawer": True}
    )
    assert "drawer_progress=" not in normal and "drawer_progress=" in drawer
    assert normal.count("Identify the task's targets, destinations and relation") == 1
    assert drawer.count("Drawer candidates also support") == 1
    assert "unclassified" in normal.lower() and "EXHAUSTED" in normal
