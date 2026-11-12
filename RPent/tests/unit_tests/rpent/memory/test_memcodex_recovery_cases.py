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

"""Particular failures cannot be covered by unrelated recovery text."""

import json

import pytest

from rpent.memory.memcodex.recovery import (
    covers,
    failure_kinds,
    object_types,
    recovery_cases,
    requirements,
)
from tests.unit_tests.rpent.memory import test_memcodex_m2_contracts as m2
from tests.unit_tests.rpent.memory.test_memcodex_descent_contracts import add_entry

indexed = m2.indexed
offline_tokenizer = m2.offline_tokenizer


def test_slip_is_covered_without_unnecessary_raw(indexed):
    result = m2.query(
        m2.controller(indexed),
        purpose="recover",
        issue="milk slipped during carry",
        failure_object="milk",
        phase="transport",
    )
    assert result.memory_read_complete
    assert result.layers_visited[-1] == "skills"


def test_absent_drawer_case_reports_exhaustion_without_reading_raw(indexed):
    result = m2.query(
        m2.controller(indexed),
        purpose="recover",
        failure_kind="wrong_drawer",
        failure_object="bottom_drawer_handle",
        phase="articulate",
        localized=False,
    )
    assert result.layers_visited[-1] == "skills"
    assert result.routing_steps[-1]["action"] == "EXHAUSTED"
    assert not result.memory_read_complete
    assert "recovery_case:wrong_drawer" in result.memory_missing
    assert "object_localization" in result.state_missing


def test_specific_raw_remedy_is_delivered(indexed):
    wanted = add_entry(
        indexed,
        "raw/drawer",
        source="task-family/drawer.md",
        text="Wrong drawer opened. Retreat, re-localize the bottom handle and retry with a tilted approach.",
        skills=("pi0_pick", "move_to", "release"),
        provides=("skill", "recovery"),
    )
    # Raw bindings derive from sibling non-raw source records.
    add_entry(
        indexed,
        "skills/drawer",
        source="task-family/drawer.md",
        text="Use pi0_pick, move_to and release.",
        skills=("pi0_pick", "move_to", "release"),
    )
    result = m2.query(
        m2.controller(indexed), purpose="recover", failure_kind="wrong_drawer"
    )
    assert wanted in result.selected_ids
    assert result.memory_read_complete
    assert result.layers_visited[-1] == "raw"


@pytest.mark.parametrize(
    ("issue", "expected"),
    [
        ("right-side relation unmet", ("wrong_spatial_relation",)),
        ("the pudding box collided with the mug", ("object_collision",)),
        ("not enough grasp space around the pudding box", ("grasp_space_insufficient",)),
        ("the cup remained tilted", ("object_tipped",)),
        ("placement goal not satisfied after release", ("placement_not_satisfied",)),
    ],
)
def test_t6_failure_language_maps_to_structured_recovery(issue, expected):
    assert failure_kinds(issue) == list(expected)
    assert requirements({"issue": issue}) == expected


def test_explicit_failure_keeps_additional_causes_from_issue_text():
    assert requirements(
        {
            "failure_kind": "placement_not_satisfied",
            "issue": (
                "Pudding released on negative-y side. Need spatial placement "
                "correction toward positive-y."
            ),
        }
    ) == ("placement_not_satisfied", "wrong_spatial_relation")
    assert requirements(
        {
            "failure_kind": "object_collision",
            "issue": "Gripper housing brushed the mug; the mug now leans over the plate.",
        }
    ) == ("object_collision", "object_tipped")


def test_narrative_recovery_cases_keep_sentence_level_object_scope():
    text = (
        "The gripper housing contacts the white mug. "
        "The white mug leans slightly. "
        "Move the pudding box aside, rotate the wrist, re-localize and retry."
    )
    cases = recovery_cases(text)
    collision = next(case for case in cases if case["kind"] == "object_collision")
    tipped = next(case for case in cases if case["kind"] == "object_tipped")
    assert collision["object_types"] == ["mug"]
    assert tipped["object_types"] == ["mug"]


def test_slight_lean_is_classified_as_tipped_and_multi_object_scope_is_separate():
    assert "object_tipped" in failure_kinds(
        "The white mug leans slightly after the gripper housing contacts it."
    )
    mug_case = {
        "kind": "object_tipped",
        "object_types": ["mug"],
        "phases": ["transport", "recovery"],
    }
    pudding_case = {
        "kind": "object_tipped",
        "object_types": ["pudding_box"],
        "phases": ["transport", "recovery"],
    }
    state = {
        "failure_cases": [
            {
                "failure_kind": "object_collision",
                "failure_object": "white_mug",
                "phase": "transport",
            },
            {
                "failure_kind": "object_tipped",
                "failure_object": "white_mug",
                "phase": "transport",
            },
            {
                "failure_kind": "grasp_space_insufficient",
                "failure_object": "chocolate_pudding",
                "phase": "grasp",
            },
        ]
    }
    assert covers(mug_case, "object_tipped", state)
    assert not covers(pudding_case, "object_tipped", state)
    assert requirements(state) == (
        "grasp_space_insufficient",
        "object_collision",
        "object_tipped",
    )


def test_spatial_recovery_case_extracts_push_rotate_and_relocalize_remedy():
    text = (
        "Right-side relation unmet for the chocolate pudding box. "
        "Push the pudding box aside, rotate the wrist, create more space, "
        "re-localize and retry the grasp."
    )
    cases = recovery_cases(text)
    assert any(case["kind"] == "wrong_spatial_relation" for case in cases)
    assert any(case["kind"] == "grasp_space_insufficient" for case in cases)
    assert "pudding_box" in object_types(text)
    assert any(
        covers(
            case,
            "wrong_spatial_relation",
            {"failure_object": "chocolate_pudding", "phase": "recovery"},
        )
        for case in cases
    )


def test_collision_and_tipped_recovery_cases_are_not_generic_grasp_cases():
    text = (
        "The pudding box collided with the mug and the cup remained tilted. "
        "Move the mug aside, re-localize the box, rotate the wrist and retry."
    )
    kinds = {case["kind"] for case in recovery_cases(text)}
    assert "object_collision" in kinds
    assert "object_tipped" in kinds
    assert "grasp_not_closed" not in kinds


def test_recovery_cannot_cross_explicit_object_or_phase():
    case = recovery_cases("If the milk slips, re-localize and retry the grasp.")[0]
    assert covers(
        case, "object_slipped", {"failure_object": "milk", "phase": "transport"}
    )
    assert not covers(case, "object_slipped", {"failure_object": "moka_pot"})
    assert not covers(case, "object_slipped", {"phase": "release"})


@pytest.mark.parametrize(
    "text",
    [
        "No observed slip; retry unrelated skill.",
        "The object slipped. Cause unknown.",
        "| Pot slipped | Unknown | No validated remedy |",
    ],
)
def test_negated_or_remedy_free_passages_are_not_coverage(text):
    assert not recovery_cases(text)


def test_tiny_budget_does_not_claim_missing_case_covered(indexed):
    result = m2.query(
        m2.controller(indexed, budget=1500),
        purpose="recover",
        failure_kind="wrong_drawer",
    )
    assert not result.memory_read_complete
    assert result.token_count <= 1500


def test_schema_two_index_remains_readable(indexed):
    from repro.index_memcodex_memory import validate_index

    root, index = indexed
    data = json.loads(index.read_text())
    data.update(schema_version=2, version="m2-index-v2")
    for entry in data["entries"].values():
        entry.pop("recovery_cases")
    index.write_text(json.dumps(data))
    assert validate_index(root, index)["status"] == "passed"
    result = m2.query(m2.controller(indexed), purpose="recover", issue="milk slipped")
    assert result.memory_read_complete
