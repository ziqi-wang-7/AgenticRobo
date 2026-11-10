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

from rpent.memory.memcodex.recovery import covers, recovery_cases
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
