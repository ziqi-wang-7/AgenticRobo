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

"""Unsupported recovery searches terminate honestly without dumping the corpus."""

import json

import pytest

from rpent.memory.memcodex.policy import ActionQuery, RoutingDecision
from tests.unit_tests.rpent.memory import test_memcodex_m2_contracts as m2
from tests.unit_tests.rpent.memory.test_memcodex_descent_contracts import add_entry

indexed = m2.indexed
offline_tokenizer = m2.offline_tokenizer


def header(result):
    return json.loads(result.context.splitlines()[1])


@pytest.mark.parametrize(
    "state,gap",
    [
        ({"issue": "unexpected behavior near the object"}, "unclassified_issue"),
        ({"failure_kind": "wrong_drawer"}, "recovery_case:wrong_drawer"),
    ],
)
def test_unchanged_unresolvable_query_is_bounded_and_keeps_full_guidance(
    indexed, state, gap
):
    control = m2.controller(indexed)
    for _ in range(3):
        result = m2.query(control, purpose="recover", localized=False, **state)
        assert not result.sufficient and not result.memory_read_complete
        assert gap in result.memory_missing
        assert header(result)["search_exhausted"]
        assert gap in header(result)["unavailable_recovery"]
        assert "unclassified" not in header(result)["recovery_requirements"]
        assert result.routing_steps[-1]["action"] == "EXHAUSTED"
        assert "raw" not in result.layers_visited
        assert result.files_read <= 4
        assert (
            "Use pi0_pick" in result.context and "If the milk slips" in result.context
        )


def test_case_found_only_in_raw_skips_irrelevant_skill_bodies(indexed):
    for number in range(10):
        add_entry(
            indexed,
            f"skills/irrelevant{number}",
            source="task-family/object.md",
            text="Generic pi0_pick move_to release recovery.",
            skills=("pi0_pick", "move_to", "release"),
            provides=("skill", "recovery"),
        )
    wanted = add_entry(
        indexed,
        "raw/case",
        source="task-family/object.md",
        text="Wrong drawer opened. Retreat and re-localize the bottom handle before retrying.",
        skills=("pi0_pick", "move_to", "release"),
        provides=("skill", "recovery"),
    )
    result = m2.query(
        m2.controller(indexed),
        purpose="recover",
        failure_kind="wrong_drawer",
        localized=False,
    )
    assert wanted in result.selected_ids and result.memory_read_complete
    assert result.layers_visited[-1] == "raw"
    assert result.files_read <= 5
    assert "before retrying" in result.context


def test_unavailable_case_does_not_override_conditions_or_rejections(indexed):
    wanted = add_entry(
        indexed,
        "skills/conditional",
        source="task-family/object.md",
        text="Wrong drawer opened. Retreat and re-localize the handle.",
        skills=("pi0_pick", "move_to", "release"),
        provides=("skill", "recovery"),
        conditions={"drawer_visible": True},
    )
    missing = m2.query(
        m2.controller(indexed), purpose="recover", failure_kind="wrong_drawer"
    )
    assert header(missing)["search_exhausted"] and wanted not in missing.selected_ids
    found = m2.query(
        m2.controller(indexed),
        purpose="recover",
        failure_kind="wrong_drawer",
        conditions={"drawer_visible": True},
    )
    assert found.memory_read_complete and wanted in found.selected_ids
    rejected = m2.query(
        m2.controller(indexed),
        purpose="recover",
        failure_kind="wrong_drawer",
        conditions={"drawer_visible": True},
        rejected_entries=[wanted],
    )
    assert not rejected.memory_read_complete and wanted not in rejected.selected_ids


def test_legacy_index_unknown_case_still_has_bounded_raw_fallback(indexed):
    for number in range(4):
        add_entry(
            indexed,
            f"raw/unrelated{number}",
            source="task-family/object.md",
            text="Generic pi0_pick move_to release recovery.",
            skills=("pi0_pick", "move_to", "release"),
            provides=("skill", "recovery"),
        )
    _, index = indexed
    data = json.loads(index.read_text())
    data.update(schema_version=2, version="m2-index-v2")
    for item in data["entries"].values():
        item.pop("recovery_cases")
    index.write_text(json.dumps(data))
    result = m2.query(
        m2.controller(indexed), purpose="recover", failure_kind="wrong_drawer"
    )
    assert result.layers_visited[-1] == "raw"
    assert len([p for p in result.files_opened if p.startswith("raw/")]) == 2
    assert not result.memory_read_complete
    assert not header(result)["search_exhausted"]
    assert any(
        s["action"] == "LIMIT" and s["layer"] == "raw" and s["candidate_ids"]
        for s in result.routing_steps
    )


def test_new_index_can_resolve_previously_unavailable_case(indexed):
    control = m2.controller(indexed)
    missing = m2.query(control, purpose="recover", failure_kind="wrong_drawer")
    assert header(missing)["search_exhausted"]
    wanted = add_entry(
        indexed,
        "raw/newcase",
        source="task-family/object.md",
        text="Wrong drawer opened. Retreat and re-localize the bottom handle before retrying.",
        skills=("pi0_pick", "move_to", "release"),
        provides=("skill", "recovery"),
    )
    with pytest.raises(RuntimeError, match="changed"):
        m2.query(control, purpose="recover", failure_kind="wrong_drawer")
    found = m2.query(
        m2.controller(indexed), purpose="recover", failure_kind="wrong_drawer"
    )
    assert found.memory_read_complete and wanted in found.selected_ids
    assert not header(found)["unavailable_recovery"]


def test_spatial_failure_uses_bounded_raw_fallback_when_index_case_missing(indexed):
    from repro.memcodex_corpus import encoded

    wanted = add_entry(
        indexed,
        "raw/spatial",
        source="task-family/libero_10_swap_t6.md",
        text=(
            "Right-side relation unmet for the chocolate pudding box; there is not "
            "enough grasp space near the mug. Push the pudding box aside, rotate "
            "the wrist, create more space, re-localize and retry the grasp."
        ),
        skills=("pi0_pick", "move_to", "release"),
        provides=("skill", "recovery"),
    )
    _, index = indexed
    data = json.loads(index.read_text())
    entry = next(
        item for item in data["entries"].values() if item["key"] == wanted
    )
    # Simulate an index whose structured recovery annotation missed the
    # narrative remedy; the hash-checked raw body remains authoritative.
    entry["recovery_cases"] = []
    index.write_bytes(encoded(data))

    control = m2.controller(indexed)
    control.indexed.index["tasks"][m2.TASK]["layers"]["raw"] = [
        item
        for item in control.indexed.index["tasks"][m2.TASK]["layers"]["raw"]
        if item["key"] == wanted
    ]
    result = m2.query(
        control,
        purpose="recover",
        issue="right-side relation unmet",
        failure_object="chocolate_pudding",
        phase="recovery",
    )
    assert wanted in result.selected_ids
    assert result.layers_visited[-1] == "raw"
    assert result.memory_read_complete
    assert "Push the pudding box aside" in result.context
    assert result.files_read <= 5


def test_spatial_raw_fallback_does_not_claim_unrelated_body_as_coverage(indexed):
    add_entry(
        indexed,
        "raw/unrelated_spatial",
        source="task-family/libero_10_swap_t6.md",
        text="The chocolate pudding box is visible. Use pi0_pick and move_to.",
        skills=("pi0_pick", "move_to", "release"),
        provides=("skill", "recovery"),
    )
    control = m2.controller(indexed)
    control.indexed.index["tasks"][m2.TASK]["layers"]["raw"] = [
        item
        for item in control.indexed.index["tasks"][m2.TASK]["layers"]["raw"]
        if item["key"] == "raw:unrelated_spatial"
    ]
    result = m2.query(
        control,
        purpose="recover",
        issue="right-side relation unmet",
        failure_object="chocolate_pudding",
        phase="recovery",
    )
    assert not result.memory_read_complete
    assert "recovery_case:wrong_spatial_relation" in result.memory_missing
    assert result.routing_steps[-1]["action"] == "EXHAUSTED"
    assert "raw" in result.layers_visited


def test_custom_exhausted_cannot_hide_known_guidance(indexed):
    control = m2.controller(indexed)
    control.route = lambda ctx: RoutingDecision("EXHAUSTED")
    result = control.retrieve_with_trace(
        m2.TASK,
        {"task_language": m2.LANGUAGE},
        {"purpose": "recover", "available_skills": ["pi0_pick"]},
        action_query=ActionQuery("pi0_pick", 0, ("localized:milk",), ()),
    )
    assert result.memory_read_complete and "Use pi0_pick" in result.context
    assert result.routing_decision == "OBSERVE"


def test_ordinary_execute_defers_only_until_localization(indexed):
    control = m2.controller(indexed)
    absent = m2.query(control, localized=False)
    assert absent.layers_visited == ("summary", "relational")
    assert (
        not absent.memory_read_complete
        and absent.routing_steps[-1]["action"] == "OBSERVE"
    )
    ready = m2.query(control)
    assert ready.memory_read_complete and "Use pi0_pick" in ready.context
    recovery = m2.query(
        control, localized=False, purpose="recover", issue="milk slipped"
    )
    assert recovery.memory_read_complete and "If the milk slips" in recovery.context


def test_custom_descent_respects_ordinary_localization_gate(indexed):
    control = m2.controller(indexed)
    control.route = lambda ctx: RoutingDecision("DESCEND")
    result = control.retrieve_with_trace(
        m2.TASK,
        {"task_language": m2.LANGUAGE},
        {"purpose": "execute", "available_skills": ["pi0_pick"]},
        action_query=ActionQuery("pi0_pick", 0, ("localized:milk",), ()),
    )
    assert result.layers_visited == ("summary", "relational")
    assert result.routing_decision == "OBSERVE"
    assert not result.memory_read_complete
