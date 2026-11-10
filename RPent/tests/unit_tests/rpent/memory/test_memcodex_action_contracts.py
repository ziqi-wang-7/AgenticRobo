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

"""Action-specific memory coverage and independent evidence validation."""

import pytest

from rpent.memory.memcodex import MemCodexController
from rpent.memory.memcodex.policy import ActionFact, ActionQuery, RoutingDecision
from rpent.tools.base import ToolResult
from tests.unit_tests.rpent.memory import test_memcodex_m2_contracts as m2

indexed = m2.indexed
offline_tokenizer = m2.offline_tokenizer


def test_pick_reads_for_target_only_and_preserves_full_task_query(indexed):
    control = m2.controller(indexed)
    observation = {"task_language": m2.LANGUAGE, "localized_objects": ["milk"]}
    state = {"available_skills": ["pi0_pick", "move_to", "release"]}
    pick = ActionQuery(
        tool="pi0_pick",
        step=0,
        required_facts=("localized:milk", "identity:milk"),
        facts=(
            ActionFact("localized:milk", "verified", "loc-0001", "perception"),
            ActionFact("identity:milk", "claimed", "loc-0001", "planner"),
        ),
    )

    legacy = control.retrieve_with_trace(m2.TASK, observation, state)
    assert not legacy.sufficient and "object_localization" in legacy.missing

    candidate = control.retrieve_with_trace(
        m2.TASK, observation, state, action_query=pick
    )
    assert candidate.memory_read_complete
    assert not candidate.action_preconditions_met and not candidate.sufficient
    assert not candidate.action_missing_facts
    assert candidate.action_unverified_facts == ("identity:milk",)
    assert "raw" not in candidate.layers_visited


def test_custom_stop_cannot_certify_missing_or_claimed_facts(indexed):
    root, index = indexed
    attempted = []

    def always_stop(context):
        attempted.append(context)
        return RoutingDecision("STOP")

    control = MemCodexController(
        root,
        20000,
        mode="m2",
        index_path=index,
        route=always_stop,
        route_name="always_stop",
    )
    result = control.retrieve_with_trace(
        m2.TASK,
        {"task_language": m2.LANGUAGE},
        {"available_skills": ["move_to"]},
        action_query=ActionQuery(
            tool="move_to",
            step=0,
            required_facts=("localized:basket", "holding:milk"),
            facts=(ActionFact("holding:milk", "claimed", "0:wrist.png", "planner"),),
        ),
    )

    assert not any(context.memory_read_complete for context in attempted)
    assert result.layers_visited[-1] == "relational"
    assert result.routing_decision == "OBSERVE"
    assert not result.sufficient and not result.action_preconditions_met
    assert result.action_missing_facts == ("localized:basket",)
    assert result.action_unverified_facts == ("holding:milk",)


def test_legacy_query_never_invokes_action_router(indexed):
    root, index = indexed

    def unexpected_route(context):
        pytest.fail(f"Legacy query invoked the action router: {context!r}")

    control = MemCodexController(
        root, 20000, mode="m2", index_path=index, route=unexpected_route
    )
    legacy = m2.query(control)

    assert legacy.sufficient
    assert legacy.routing_decision is None
    assert '"routing_decision"' not in legacy.context


def test_rejecting_all_reviewed_candidates_can_reach_raw(indexed):
    control = m2.controller(indexed)
    for item in control.indexed.profile(m2.TASK)["layers"]["skills"]:
        control.indexed.index["entries"][item["path"]]["review_required"] = True

    initial = m2.query(control)
    assert not initial.sufficient and initial.layers_visited[-1] == "skills"
    rejected = [entry for entry in initial.selected_ids if entry.startswith("skills:")]
    assert rejected

    fallback = m2.query(control, rejected_entries=rejected)
    assert fallback.sufficient and fallback.layers_visited[-1] == "raw"
    assert set(rejected).isdisjoint(fallback.selected_ids)


def test_action_routing_header_counts_against_wire_budget(indexed):
    action = ActionQuery(
        tool="pi0_pick", step=0, required_facts=("localized:milk",), facts=()
    )
    observation = {"task_language": m2.LANGUAGE}
    state = {"available_skills": ["pi0_pick"]}
    baseline = m2.controller(indexed).retrieve_with_trace(
        m2.TASK, observation, state, action_query=action
    )
    assert not baseline.memory_read_complete and baseline.routing_decision == "OBSERVE"

    budget = baseline.token_count - 1
    limited = m2.controller(indexed, budget).retrieve_with_trace(
        m2.TASK, observation, state, action_query=action
    )
    wire = ToolResult(data={"memory_context": limited.context}).to_text()
    assert len(wire.encode()) == limited.token_count <= budget
    assert "[truncated]" not in wire
    assert limited.omitted_ids
