# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO proposed-action evidence is checked without executing primitives."""

import json
from argparse import ArgumentParser
from dataclasses import replace

import pytest
from pydantic import ValidationError

from repro.liberopro_baseline import command, report
from robots.libero.memcodex_online import prepare_indexed_memory
from robots.libero.robot_spec import get_robot_spec
from robots.libero.tools import LiberoPrimitives, view_env_state
from rpent.robots.robot_spec import RunConfig
from robots.libero.recovery_evidence import classify_grasp
from tests.unit_tests.rpent.memory import test_memcodex_m2_contracts as m2

indexed = m2.indexed
offline_tokenizer = m2.offline_tokenizer
online = m2.online


@pytest.fixture
def action_run(online):
    kit, record, config = online

    def unexpected_execution(**kwargs):
        pytest.fail(f"A memory query executed a tool with {kwargs!r}")

    kit._tools.update(
        {
            primitive.name: primitive.with_handler(unexpected_execution)
            for primitive in (
                LiberoPrimitives.pi0_pick,
                LiberoPrimitives.move_to,
                LiberoPrimitives.release,
                view_env_state,
            )
        }
    )
    return kit, record, config


def _trace(config):
    path = config.output_dir / "m2-retrievals.jsonl"
    return json.loads(path.read_text(encoding="utf-8").splitlines()[-1])


def _pick(**args):
    return {
        "tool": "pi0_pick",
        "args": {"prompt": "pick up the milk", **args},
        "roles": {"target_object": "milk"},
    }


def _carry(**args):
    return {
        "tool": "move_to",
        "args": {"xyz": [1.0, 2.0, 3.0], **args},
        "roles": {"held_object": "milk", "destination": "basket"},
        "expected_effect": "hold the milk while moving to the basket",
    }


def test_pick_uses_registered_defaults_without_executing(action_run):
    kit, record, config = action_run
    evidence = m2.localize(kit)
    # Use the same strict outer input model and parameter extraction as Toolkit.
    params = kit.retrieve_memory.args_schema.model_validate(
        {"step": 0, "candidate": _pick(), "localizations": {"milk": evidence}},
        strict=True,
    )
    kit.retrieve_memory(
        **{field: getattr(params, field) for field in type(params).model_fields}
    )
    result = _trace(config)

    defaults = (
        kit._tools["pi0_pick"]
        .args_schema.model_validate({"prompt": "pick up the milk"}, strict=True)
        .model_dump(mode="json")
    )
    assert result["candidate"]["args"] == defaults
    assert result["memory_read_complete"]
    assert not result["action_preconditions_met"]
    assert result["action_unverified_facts"] == ["identity:milk"]
    assert result["action_missing_facts"] == []
    assert record.step_idx == 0


@pytest.mark.parametrize("tool_name", ["unknown_tool", "view_env_state", "pi0_pick"])
def test_unregistered_or_readonly_candidate_rejected(action_run, tool_name):
    kit, _, _ = action_run
    if tool_name == "pi0_pick":
        kit._tools[tool_name] = replace(kit._tools[tool_name], readonly=True)

    with pytest.raises(ValueError, match="Unsupported action evidence primitive"):
        kit.retrieve_memory(step=0, candidate={**_pick(), "tool": tool_name})


def test_strict_tool_validation_and_carry_default_open_rejected(action_run):
    kit, _, _ = action_run
    with pytest.raises(ValidationError):
        kit.retrieve_memory(step=0, candidate=_pick(max_chunks="4"))
    with pytest.raises(ValidationError):
        kit.retrieve_memory(
            step=0, candidate=_carry(xyz=[1.0, float("inf"), 3.0], gripper=1.0)
        )
    with pytest.raises(ValueError, match="gripper=1.0"):
        kit.retrieve_memory(step=0, candidate=_carry())
    with pytest.raises(ValueError, match="gripper=1.0"):
        kit.retrieve_memory(step=0, candidate=_carry(gripper=-1.0))


def test_geometry_does_not_verify_held_object_or_identity(action_run):
    kit, record, config = action_run
    record.artifacts.add("wrist.png")
    localizations = {"milk": m2.localize(kit), "basket": m2.localize(kit)}
    candidate = _carry(gripper=1.0)
    response = kit.retrieve_memory(
        step=0, candidate=candidate, localizations=localizations
    )
    geometry_only = _trace(config)
    assert not response.images
    assert geometry_only["routing_decision"] == "OBSERVE"
    planner_header = json.loads(response.data["memory_context"].splitlines()[1])
    assert planner_header["routing_decision"] == "OBSERVE"
    assert planner_header["routing_reason"] == geometry_only["routing_reason"]
    assert geometry_only["memory_read_complete"]
    assert geometry_only["action_missing_facts"] == ["holding:milk"]
    assert geometry_only["action_unverified_facts"] == ["identity:basket"]

    check = {"fact": "holding", "object_id": "milk", "step": 0, "artifact": "wrist.png"}
    response = kit.retrieve_memory(
        step=0,
        candidate=candidate,
        localizations=localizations,
        visual_checks=[check],
    )
    with_claim = _trace(config)
    assert not response.images
    assert with_claim["routing_decision"] == "OBSERVE"
    assert not with_claim["action_preconditions_met"]
    assert with_claim["action_missing_facts"] == []
    assert set(with_claim["action_unverified_facts"]) == {
        "identity:basket",
        "holding:milk",
    }
    assert next(
        item
        for item in with_claim["action_query"]["facts"]
        if item["key"] == "holding:milk"
    ) == {
        "key": "holding:milk",
        "status": "claimed",
        "evidence_ref": "0:wrist.png",
        "source": "planner_wrist_grasp_assessment",
    }


@pytest.mark.parametrize(
    ("overrides", "message"),
    [
        ({"step": 0}, "current camera artifact"),
        ({"artifact": "unrecorded.png"}, "current camera artifact"),
        ({"object_id": "basket"}, "Unexpected or duplicate visual check"),
    ],
)
def test_visual_check_requires_current_recorded_camera_and_correct_role(
    action_run, overrides, message
):
    kit, record, _ = action_run
    record.step_idx = 1
    record.artifacts.add("wrist.png")
    check = {
        "fact": "holding",
        "object_id": "milk",
        "step": 1,
        "artifact": "wrist.png",
        **overrides,
    }

    with pytest.raises(ValueError, match=message):
        kit.retrieve_memory(
            step=1, candidate=_carry(gripper=1.0), visual_checks=[check]
        )


def test_release_requires_correct_holding_and_placement_claims(action_run):
    kit, record, config = action_run
    record.artifacts.update({"wrist.png", "agentview.png"})
    candidate = {"tool": "release", "args": {}, "roles": _carry()["roles"]}
    localizations = {"basket": m2.localize(kit)}
    holding = {
        "fact": "holding",
        "object_id": "milk",
        "step": 0,
        "artifact": "wrist.png",
    }
    placement = {
        "fact": "placement_ready",
        "object_id": "basket",
        "step": 0,
        "artifact": "agentview.png",
    }

    kit.retrieve_memory(
        step=0,
        candidate=candidate,
        localizations=localizations,
        visual_checks=[holding],
    )
    incomplete = _trace(config)
    assert incomplete["memory_read_complete"]
    assert incomplete["action_missing_facts"] == ["placement_ready:basket"]
    assert incomplete["routing_decision"] == "OBSERVE"

    with pytest.raises(ValueError, match="Unexpected or duplicate visual check"):
        kit.retrieve_memory(
            step=0,
            candidate=candidate,
            localizations=localizations,
            visual_checks=[holding, {**placement, "object_id": "milk"}],
        )

    kit.retrieve_memory(
        step=0,
        candidate=candidate,
        localizations=localizations,
        visual_checks=[holding, placement],
    )
    review = _trace(config)
    assert review["routing_decision"] == "OBSERVE"
    assert not review["action_preconditions_met"]
    assert not review["action_missing_facts"]
    assert {
        fact["key"]: fact["evidence_ref"]
        for fact in review["action_query"]["facts"]
        if fact["status"] == "claimed"
    } == {
        "identity:basket": localizations["basket"],
        "holding:milk": "0:wrist.png",
        "placement_ready:basket": "0:agentview.png",
    }


def test_identity_requires_current_agentview_attestation(action_run):
    kit, record, config = action_run
    record.artifacts.add("agentview_high.png")
    evidence = m2.localize(kit)
    candidate = _pick()
    uncertain = {
        "fact": "identity",
        "object_id": "milk",
        "value": "milk",
        "status": "uncertain",
        "step": 0,
        "artifact": "agentview_high.png",
    }
    kit.retrieve_memory(
        step=0,
        candidate=candidate,
        localizations={"milk": evidence},
        visual_checks=[uncertain],
    )
    assert _trace(config)["action_unverified_facts"] == ["identity:milk"]

    verified = {**uncertain, "status": "verified"}
    kit.retrieve_memory(
        step=0,
        candidate=candidate,
        localizations={"milk": evidence},
        visual_checks=[verified],
    )
    trace = _trace(config)
    assert trace["action_preconditions_met"]
    identity = next(
        fact for fact in trace["action_query"]["facts"]
        if fact["key"] == "identity:milk"
    )
    assert identity["status"] == "verified"
    assert identity["source"] == "planner_agentview_identity_attestation"


def test_identity_rejection_is_not_promoted_to_verified(action_run):
    kit, record, config = action_run
    record.artifacts.add("agentview.png")
    evidence = m2.localize(kit)
    rejected = {
        "fact": "identity",
        "object_id": "milk",
        "value": "butter",
        "status": "rejected",
        "step": 0,
        "artifact": "agentview.png",
    }
    kit.retrieve_memory(
        step=0,
        candidate=_pick(),
        localizations={"milk": evidence},
        visual_checks=[rejected],
    )
    trace = _trace(config)
    assert not trace["action_preconditions_met"]
    identity = next(
        fact for fact in trace["action_query"]["facts"]
        if fact["key"] == "identity:milk"
    )
    assert identity["status"] == "claimed"
    assert identity["source"] == "planner_agentview_identity_rejected"


def test_tri_state_grasp_does_not_trust_pi0_boolean_alone():
    assert classify_grasp(pi0_success=False, visual_status=None) == "uncertain"
    assert classify_grasp(pi0_success=True, visual_status=None) == "uncertain"
    assert classify_grasp(pi0_success=False, visual_status="verified") == "verified"
    assert classify_grasp(pi0_success=True, visual_status="failed") == "failed"
    assert classify_grasp(pi0_success=False, visual_status="uncertain") == "uncertain"


def test_confirmed_failed_grasp_requests_recovery(action_run):
    kit, record, config = action_run
    record.command = {"action": "pi0_pick"}
    record.result = {"success": False}
    record.artifacts.add("wrist.png")
    localizations = {"milk": m2.localize(kit), "basket": m2.localize(kit)}
    failed_check = {
        "fact": "holding",
        "object_id": "milk",
        "step": 0,
        "artifact": "wrist.png",
        "status": "failed",
    }
    kit.retrieve_memory(
        step=0,
        candidate=_carry(gripper=1.0),
        localizations=localizations,
        visual_checks=[failed_check],
    )
    trace = _trace(config)
    assert trace["grasp_status"] == "failed"
    assert trace["last_action_failed"]
    assert trace["state"]["needs_recovery"]


def test_memory_review_is_scoped_to_normalized_candidate(action_run):
    kit, _, config = action_run
    indexed_corpus = kit.memory.controller.indexed
    for item in indexed_corpus.profile(m2.TASK)["layers"]["skills"]:
        indexed_corpus.index["entries"][item["path"]]["review_required"] = True
    localizations = {"milk": m2.localize(kit)}
    kit.retrieve_memory(step=0, candidate=_pick(), localizations=localizations)
    initial = _trace(config)
    selected = [
        entry for entry in initial["selected_ids"] if entry.startswith("skills:")
    ]
    assert selected
    default = initial["candidate"]["args"]["max_chunks"]

    kit.retrieve_memory(
        step=0,
        candidate=_pick(max_chunks=default),
        localizations=localizations,
        applicable_entries=selected,
    )
    assert (
        _trace(config)["action_query"]["fingerprint"]
        == initial["action_query"]["fingerprint"]
    )
    with pytest.raises(ValueError, match="this step and candidate"):
        kit.retrieve_memory(
            step=0,
            candidate=_pick(max_chunks=default + 1),
            localizations=localizations,
            applicable_entries=selected,
        )


def test_action_evidence_flag_requires_m2_and_changes_only_opted_in_prompt(
    indexed, tmp_path
):
    root, index = indexed
    spec = get_robot_spec()
    parser = ArgumentParser()
    parser.add_argument("--planner", default="codex")
    spec.add_cli_args(parser, use_dashboard=False)
    base = ["--suite", "libero_object_task", "--task", "0"]
    action = "--memcodex-action-evidence"
    with pytest.raises(ValueError, match="requires --memcodex-mode m2"):
        spec.validate_args(parser.parse_args([*base, action]))

    base += ["--memcodex-root", str(root), "--memcodex-index", str(index)]
    base += ["--memcodex-mode", "m2"]
    prompts = {}
    for enabled in (False, True):
        args = parser.parse_args([*base, action] if enabled else base)
        spec.validate_args(args)
        output = tmp_path / f"run-{enabled}"
        config = RunConfig(
            "test", output, {"output_dir": str(output), "recipe_tag": "test"}, {}
        )
        prepare_indexed_memory(args, config)
        recorded = json.loads((output / "m2-config.json").read_text(encoding="utf-8"))
        assert config.prompt_vars["memcodex_action_evidence"] is enabled
        assert recorded["action_evidence"] is enabled
        prompts[enabled] = spec.prompts.render("system", variables=config.prompt_vars)

    assert "Bind each required object" in prompts[False]
    assert "target_object" not in prompts[False]
    assert "target_object" in prompts[True]
    assert "Bind each required object" not in prompts[True]
    assert "empty-gripper approach to the already" in prompts[True]
    assert "localize EVERYTHING first" not in prompts[True]


@pytest.mark.parametrize("enabled", [False, True])
def test_repro_command_passes_action_evidence_opt_in(tmp_path, enabled, monkeypatch):
    monkeypatch.setattr("repro.liberopro_baseline.now", lambda: "test-time")
    cell = {"suite": "libero_object_task", "task": 0, "seed": 1, "episode_id": "test"}
    plan = {
        "condition": "memcodex_m2_indexed",
        "memcodex_enabled": True,
        "memcodex_root": "corpus",
        "memory_token_budget": 32768,
        "memcodex_index": "INDEX.json",
        "action_evidence": enabled,
        "episodes": [cell],
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan), encoding="utf-8")
    argv = command(
        tmp_path,
        cell,
        {"environment_gpu": 1, "vla_port": 8420, "sam3_port": 8414},
    )

    assert ("--memcodex-action-evidence" in argv) is enabled
    assert argv[argv.index("--memcodex-mode") + 1] == "m2"
    assert report(tmp_path)["action_evidence"] is enabled
