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

"""Recovery descent, provenance scopes and bounded narrowing without a robot."""

import json

import pytest

from repro.index_memcodex_memory import build_index, validate_index
from repro.memcodex_corpus import digest, encoded
from rpent.memory.memcodex.policy import ActionQuery, RoutingDecision
from tests.unit_tests.rpent.memory import test_memcodex_m2_contracts as m2

indexed = m2.indexed
offline_tokenizer = m2.offline_tokenizer
online = m2.online


def add_entry(indexed, name, *, source, text, skills=(), provides=(), conditions=None):
    """Add a hashed synthetic entry and rebuild the real metadata index."""
    root, index = indexed
    layer, stem = name.split("/")
    data = {
        "id": stem,
        "text": text,
        "task_ids": [m2.TASK],
        "applies_when": conditions or {},
        "skill_ids": list(skills),
        "provides": list(provides),
        "source_refs": [source + "#L1-L20"],
    }
    path = name + ".json"
    payload = encoded(data)
    (root / path).write_bytes(payload)
    manifest = json.loads((root / "MANIFEST.json").read_text())
    manifest["files"][path] = digest(payload)
    (root / "MANIFEST.json").write_bytes(encoded(manifest))
    index.write_bytes(encoded(build_index(root)))
    return layer + ":" + stem


def test_recovery_without_localization_reads_skill_and_keeps_state_gap(indexed):
    result = m2.query(m2.controller(indexed), localized=False, purpose="recover")
    assert result.memory_read_complete and not result.sufficient
    assert result.state_missing == ("object_localization",)
    assert result.memory_missing == ()
    assert "If the milk slips" in result.context
    assert "Use pi0_pick" in result.context
    assert result.layers_visited[-1] == "skills"
    assert not any(p.startswith("raw/") for p in result.files_opened)


def test_recovery_without_localization_reaches_raw_when_skills_absent(indexed):
    control = m2.controller(indexed)
    control.indexed.profile(m2.TASK)["layers"]["skills"] = []
    result = m2.query(control, localized=False, purpose="recover")
    assert result.layers_visited[-1] == "raw"
    assert result.memory_read_complete and not result.state_evidence_complete
    assert any(p.startswith("raw/") for p in result.files_opened)
    assert "If the milk slips" in result.context


def test_online_empty_localizations_do_not_discard_recovery_issue(online):
    kit, record, config = online
    record.command = {"action": "move_to"}
    record.result = {"final_dist_m": 0.12}
    kit.retrieve_memory(step=0, purpose="recover", issue="insertion stalled at roof")
    trace = json.loads((config.output_dir / "m2-retrievals.jsonl").read_text())
    assert trace["state"]["issue"] == "insertion stalled at roof"
    assert trace["state"]["last_action"] == "move_to"
    # A milk-slip remedy does not cover blocked microwave insertion.
    assert not trace["memory_read_complete"]
    assert "recovery_case:insertion_blocked" in trace["memory_missing"]
    assert trace["layers_visited"][-1] == "raw"
    assert trace["state_missing"] == ["object_localization"]
    assert trace["state"]["needs_recovery"]
    assert "If the milk slips" in trace["context"]


def test_narrow_follows_source_and_does_not_read_unrelated_task_memory(indexed):
    unrelated = add_entry(
        indexed,
        "skills/unrelated",
        source="task-family/unrelated.md",
        text="Unrelated recovery technique using pi0_pick move_to release.",
        skills=("pi0_pick", "move_to", "release"),
        provides=("skill", "recovery"),
    )
    result = m2.query(m2.controller(indexed), purpose="recover", localized=False)
    assert result.memory_read_complete
    assert unrelated not in result.selected_ids
    assert "skills/unrelated.json" not in result.files_opened
    assert any(
        s["action"] == "NARROW" and "source_links" in s["reason"]
        for s in result.routing_steps
    )


def test_failure_topic_narrows_then_widens_to_complete_skill_guidance(indexed):
    contact = add_entry(
        indexed,
        "skills/contact",
        source="task-family/object.md",
        text="Roof contact stalled insertion: withdraw and check cavity clearance.",
        provides=("recovery",),
    )
    result = m2.query(
        m2.controller(indexed),
        purpose="recover",
        localized=False,
        issue="roof contact stalled insertion",
        last_action="move_to",
    )
    narrow = [s for s in result.routing_steps if s["action"] == "NARROW"]
    assert any(
        "failure_topic" in s["reason"] and s["candidate_ids"] == [contact]
        for s in narrow
    )
    assert contact in result.selected_ids
    assert any(
        s["action"] == "WIDEN" and s["layer"] == "skills" for s in result.routing_steps
    )
    assert result.memory_read_complete and "Use pi0_pick" in result.context
    assert "raw" not in result.layers_visited


def test_widen_can_find_recovery_outside_selected_source(indexed):
    needed = add_entry(
        indexed,
        "skills/extra",
        source="task-family/extra.md",
        text="Use move_pose for the supported recovery maneuver.",
        skills=("move_pose",),
        provides=("skill", "recovery"),
    )
    control = m2.controller(indexed)
    control.indexed.profile(m2.TASK)["required_skills"].append("move_pose")
    result = control.retrieve_with_trace(
        m2.TASK,
        {"task_language": m2.LANGUAGE},
        {
            "purpose": "recover",
            "available_skills": ["pi0_pick", "move_to", "release", "move_pose"],
        },
    )
    assert result.memory_read_complete and needed in result.selected_ids
    assert any(s["action"] == "WIDEN" for s in result.routing_steps)


def test_narrow_never_overrides_rejection_or_applicability(indexed):
    blocked = add_entry(
        indexed,
        "skills/blocked",
        source="task-family/object.md",
        text="Roof contact recovery.",
        provides=("recovery",),
        conditions={"door_open": True},
    )
    rejected = add_entry(
        indexed,
        "skills/rejected",
        source="task-family/object.md",
        text="Roof contact recovery.",
        provides=("recovery",),
    )
    result = m2.query(
        m2.controller(indexed),
        purpose="recover",
        issue="roof contact",
        rejected_entries=[rejected],
    )
    assert blocked not in result.selected_ids and rejected not in result.selected_ids
    assert "skills/blocked.json" not in result.files_opened
    assert "skills/rejected.json" not in result.files_opened
    assert not result.memory_read_complete
    assert "recovery_case:unclassified" in result.memory_missing


def test_source_links_include_skipped_layers_and_raw_but_not_other_sources(indexed):
    root, index = indexed
    data = json.loads(index.read_text())
    summary = data["tasks"][m2.TASK]["layers"]["summary"][0]["path"]
    links = data["source_links"][summary]
    assert all(links[layer] for layer in ("relational", "skills", "raw"))
    for layer, paths in links.items():
        for path in paths:
            assert path.startswith(layer + "/")
            assert all(
                ref.startswith("task-family/object.md#")
                for ref in data["entries"][path]["source_refs"]
            )
    assert validate_index(root, index)["status"] == "passed"


def test_tampered_source_links_rejected(indexed):
    _, index = indexed
    data = json.loads(index.read_text())
    summary = data["tasks"][m2.TASK]["layers"]["summary"][0]["path"]
    data["source_links"][summary]["raw"] = []
    index.write_bytes(encoded(data))
    with pytest.raises(ValueError, match="source links"):
        m2.controller(indexed)


def test_frozen_v1_index_still_valid_and_does_not_need_corpus_rewrite(indexed):
    root, index = indexed
    before = (root / "MANIFEST.json").read_bytes()
    data = json.loads(index.read_text())
    data.update(schema_version=1, version="m2-index-v1")
    data.pop("source_links")
    for entry in data["entries"].values():
        entry.pop("issue_tags")
        entry.pop("recovery_cases")
    index.write_bytes(encoded(data))
    assert validate_index(root, index)["status"] == "passed"
    result = m2.query(m2.controller(indexed), localized=False, purpose="recover")
    assert result.memory_read_complete and not result.state_evidence_complete
    assert (root / "MANIFEST.json").read_bytes() == before


@pytest.mark.parametrize("proposal", ["OBSERVE", "REVIEW", "NARROW"])
def test_custom_router_cannot_skip_missing_guidance(indexed, proposal):
    control = m2.controller(indexed)
    control.route = lambda context: RoutingDecision(proposal)
    result = control.retrieve_with_trace(
        m2.TASK,
        {"task_language": m2.LANGUAGE},
        {"available_skills": ["pi0_pick"]},
        action_query=ActionQuery("pi0_pick", 0, ("localized:milk",), ()),
    )
    assert result.memory_read_complete
    assert result.action_missing_facts == ("localized:milk",)
    assert not result.action_preconditions_met
    assert any(p.startswith("skills/") for p in result.files_opened)


def test_narrow_budget_omission_never_counts_as_evidence(indexed):
    generous = m2.query(m2.controller(indexed), purpose="overview")
    result = m2.query(
        m2.controller(indexed, generous.token_count + 8),
        purpose="recover",
        localized=False,
    )
    assert result.token_count <= result.token_budget
    assert not result.memory_read_complete
    assert "recovery_guidance" in result.memory_missing
    assert result.omitted_ids and "[truncated]" not in result.context


def test_summary_review_does_not_stop_missing_skill_descent(indexed):
    control = m2.controller(indexed)
    profile = control.indexed.profile(m2.TASK)
    profile["layers"]["skills"] = []
    for item in profile["layers"]["summary"]:
        control.indexed.index["entries"][item["path"]]["review_required"] = True
    result = m2.query(control, purpose="recover", localized=False)
    assert result.layers_visited[-1] == "raw"
    assert "If the milk slips" in result.context
    assert "task_summary" in result.memory_missing
    assert "recovery_guidance" not in result.memory_missing
    assert not result.sufficient
