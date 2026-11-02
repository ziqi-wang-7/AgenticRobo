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

"""Offline behavioral contracts for budgeted, state-dependent memory retrieval."""

import json
from dataclasses import replace
from pathlib import Path

import pytest

from profiling.analyze import analyze
from profiling.profiler import MemoryProfiler
from rpent.memory import MemoryManager
from rpent.memory.memcodex import MemCodexController
from rpent.memory.memcodex.cache import RetrievalCache

TASK = "demo/mug-to-plate"
STATE = {"objects_localized": True, "available_skills": ["pick", "place"]}


def entry(root: Path, layer: str = "summary", name: str = "base", **kwargs) -> Path:
    data = {
        "id": name,
        "text": "Identify the mug, grip its rim, carry closed, then release on the plate.",
        "task_ids": [TASK],
        "applies_when": {},
        "provides": ["task", "skill"],
        "skill_ids": ["pick"],
        "objects": ["mug"],
        **kwargs,
    }
    folder = root / layer
    folder.mkdir(exist_ok=True)
    path = folder / f"{name}.json"
    path.write_text(json.dumps(data, ensure_ascii=False))
    return path


def test_summary_short_circuit_does_not_read_raw_or_call_a_model(tmp_path):
    entry(tmp_path)
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "broken.json").write_text("malformed raw must not be read")
    controller = MemCodexController(tmp_path, 1024)
    result = controller.retrieve_with_trace(TASK, {"localized_objects": ["mug"]}, STATE)
    assert result.sufficient
    assert result.layers_visited == ("summary",)
    assert result.files_read == 1
    assert result.token_count == len(result.context.encode()) <= 1024
    assert controller.retrieve(TASK, {}, STATE) == result.context


def test_escalates_to_relations_and_skills_then_stops(tmp_path):
    entry(tmp_path, provides=["task"], skill_ids=[])
    entry(
        tmp_path,
        "relational",
        "rim",
        text="Mug rim is the grasp target.",
        provides=[],
        skill_ids=["pick"],
    )
    entry(
        tmp_path,
        "skills",
        "pick",
        text="Close at the rim, lift, and keep the gripper closed.",
        provides=["skill"],
    )
    result = MemCodexController(tmp_path, 2048).retrieve_with_trace(TASK, {}, STATE)
    assert result.sufficient
    assert result.layers_visited == ("summary", "relational", "skills")
    assert "skills:pick" in result.selected_ids


def test_anomaly_requires_recovery_and_reads_raw(tmp_path):
    entry(tmp_path)
    entry(
        tmp_path,
        "raw",
        "recovery",
        text="After a missed grasp, re-localize the target before approaching.",
        provides=["anomaly", "recovery"],
        skill_ids=[],
        applies_when={"failure": "missed_grasp"},
    )
    result = MemCodexController(tmp_path, 2048).retrieve_with_trace(
        TASK,
        {},
        {
            **STATE,
            "anomaly": True,
            "needs_recovery": True,
            "conditions": {"failure": "missed_grasp"},
        },
    )
    assert result.sufficient
    assert result.layers_visited[-1] == "raw"
    assert result.context.startswith("[raw:recovery]")


@pytest.mark.parametrize("state", [{}, {"objects_localized": False}])
def test_memory_does_not_invent_localized_objects(tmp_path, state):
    entry(tmp_path)
    result = MemCodexController(tmp_path, 1024).retrieve_with_trace(TASK, {}, state)
    assert not result.sufficient
    assert "object_localization" in result.missing
    assert result.layers_visited[-1] == "raw"


def test_unknown_and_false_conditions_reject_memory(tmp_path):
    entry(tmp_path, applies_when={"holding": True})
    controller = MemCodexController(tmp_path, 1024)
    for facts in ({}, {"holding": False}, {"holding": 1}):
        result = controller.retrieve_with_trace(
            TASK, {}, {**STATE, "conditions": facts}
        )
        assert result.context == ""
        assert not result.sufficient
    result = controller.retrieve_with_trace(
        TASK, {}, {**STATE, "conditions": {"holding": True}}
    )
    assert result.sufficient


def test_state_overrides_old_observation_conditions(tmp_path):
    entry(tmp_path, applies_when={"holding": True})
    result = MemCodexController(tmp_path, 1024).retrieve_with_trace(
        TASK,
        {"conditions": {"holding": True}},
        {**STATE, "conditions": {"holding": False}},
    )
    assert not result.selected_ids


def test_wrong_task_and_global_only_evidence_cannot_claim_match(tmp_path):
    entry(tmp_path, task_ids=["libero_10_swap/t6"])
    result = MemCodexController(tmp_path, 1024).retrieve_with_trace(TASK, {}, STATE)
    assert result.context == ""
    entry(tmp_path, task_ids=["*"])
    result = MemCodexController(tmp_path, 1024).retrieve_with_trace(TASK, {}, STATE)
    assert result.context
    assert "task_match" in result.missing


def test_missing_required_skill_prevents_sufficiency(tmp_path):
    entry(tmp_path)
    result = MemCodexController(tmp_path, 1024).retrieve_with_trace(
        TASK, {}, {**STATE, "required_skills": ["pick", "place"]}
    )
    assert "applicable_skill" in result.missing


def test_unavailable_skill_is_not_returned(tmp_path):
    entry(tmp_path)
    result = MemCodexController(tmp_path, 1024).retrieve_with_trace(
        TASK, {}, {**STATE, "available_skills": []}
    )
    assert not result.context
    assert "applicable_skill" in result.missing


def test_compression_reassesses_sufficiency_after_dropping_skill(tmp_path):
    entry(tmp_path, text="Task: mug to plate.", provides=["task"], skill_ids=[])
    entry(
        tmp_path,
        "skills",
        "large",
        text="Keep this indivisible sequence. " * 100,
        provides=["skill"],
    )
    result = MemCodexController(tmp_path, 100).retrieve_with_trace(TASK, {}, STATE)
    assert result.token_count <= 100
    assert "skills:large" in result.omitted_ids
    assert "applicable_skill" in result.missing
    assert result.layers_visited[-1] == "raw"
    assert "Keep this" not in result.context


def test_recovery_survives_tight_budget_ahead_of_summary(tmp_path):
    entry(tmp_path, text="task " * 30)
    entry(
        tmp_path,
        "raw",
        "recover",
        text="Release the failed grasp. Re-localize before retrying.",
        provides=["recovery", "anomaly"],
        skill_ids=[],
    )
    result = MemCodexController(tmp_path, 150).retrieve_with_trace(
        TASK, {}, {**STATE, "anomaly": True}
    )
    assert result.selected_ids == ("raw:recover",)
    assert not result.sufficient
    assert result.token_count <= 150


@pytest.mark.parametrize(
    "text", ["Close, lift, release.", "夹紧杯沿，提起后保持闭合。", "🥣→🍽️"]
)
def test_headers_and_unicode_are_charged_to_budget(tmp_path, text):
    entry(tmp_path, text=text)
    base = MemCodexController(tmp_path, 2048).retrieve(TASK, {}, STATE)
    cost = len(base.encode())
    assert cost > len(text.encode())
    exact = MemCodexController(tmp_path, cost).retrieve_with_trace(TASK, {}, STATE)
    assert exact.context == base
    too_small = MemCodexController(tmp_path, cost - 1).retrieve_with_trace(
        TASK, {}, STATE
    )
    assert too_small.context == ""
    assert not too_small.sufficient


def test_injected_counter_measures_whole_rendered_string(tmp_path):
    entry(tmp_path)
    controller = MemCodexController(
        tmp_path,
        100,
        token_counter=lambda text: len(text.split()),
        counter_name="test_words",
    )
    result = controller.retrieve_with_trace(TASK, {}, STATE)
    assert result.token_count == len(result.context.split())
    assert result.counter_name == "test_words"


@pytest.mark.parametrize("count", [True, -1, 1.5, 1])
def test_invalid_counter_cannot_charge_empty_context(tmp_path, count):
    controller = MemCodexController(tmp_path, 0, token_counter=lambda text: count)
    with pytest.raises(ValueError, match="token_counter"):
        controller.retrieve(TASK, {}, STATE)


def test_cache_invalidates_on_state_observation_and_corpus_changes(tmp_path):
    path = entry(tmp_path)
    controller = MemCodexController(tmp_path, 1024)
    first = controller.retrieve_with_trace(TASK, {"localized_objects": ["mug"]}, {})
    hit = controller.retrieve_with_trace(
        TASK, {"localized_objects": ["mug"], "rgb": object()}, {}
    )
    assert (
        hit.cache_hit and hit.files_read == 0 and hit.token_count == first.token_count
    )
    assert not controller.retrieve_with_trace(TASK, {}, {}).cache_hit
    assert not controller.retrieve_with_trace(
        TASK, {}, {**STATE, "needs_recovery": True}
    ).cache_hit
    data = json.loads(path.read_text())
    data["text"] = "New validated technique."
    path.write_text(json.dumps(data))
    changed = controller.retrieve_with_trace(TASK, {"localized_objects": ["mug"]}, {})
    assert not changed.cache_hit and "New validated" in changed.context
    path.unlink()
    assert controller.retrieve(TASK, {}, STATE) == ""


def test_zero_budget_never_reads_entry_contents(tmp_path):
    (tmp_path / "summary").mkdir()
    (tmp_path / "summary/bad.json").write_text("invalid")
    result = MemCodexController(tmp_path, 0).retrieve_with_trace(TASK, {}, STATE)
    assert (
        result.token_count == 0 and result.files_read == 0 and not result.layers_visited
    )
    assert not result.sufficient


def test_cache_eviction_and_disabling():
    cache = RetrievalCache(1)
    cache.put((1,), "first")
    cache.put((2,), "second")
    assert cache.get((1,)) is None and cache.get((2,)) == "second"
    cache = RetrievalCache(0)
    cache.put((1,), "first")
    assert cache.get((1,)) is None


@pytest.mark.parametrize("budget", [-1, True, 1.5, "1024"])
def test_invalid_budget_rejected(tmp_path, budget):
    with pytest.raises(ValueError, match="token_budget"):
        MemCodexController(tmp_path, budget)


@pytest.mark.parametrize(
    "state",
    [
        {"anomaly": "false"},
        {"conditions": {"x": float("nan")}},
        {"required_skills": "pick"},
    ],
)
def test_malformed_state_rejected(tmp_path, state):
    with pytest.raises(ValueError):
        MemCodexController(tmp_path, 1024).retrieve(TASK, {}, state)


def test_symlink_escape_and_duplicate_ids_rejected(tmp_path):
    outside = tmp_path / "outside"
    outside.mkdir()
    root = tmp_path / "corpus"
    root.mkdir()
    (root / "raw").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        MemCodexController(root, 100).retrieve(TASK, {}, STATE)
    (root / "raw").unlink()
    path = entry(root)
    (path.parent / "copy.json").write_bytes(path.read_bytes())
    with pytest.raises(ValueError, match="Duplicate"):
        MemCodexController(root, 1000).retrieve(TASK, {}, STATE)


def test_changes_during_retrieval_are_not_cached(tmp_path, monkeypatch):
    path = entry(tmp_path)
    controller = MemCodexController(tmp_path, 1024)
    original = controller.hierarchy.read_layer

    def read_and_change(layer, snapshot):
        result = original(layer, snapshot)
        path.write_text(path.read_text() + " ")
        return result

    monkeypatch.setattr(controller.hierarchy, "read_layer", read_and_change)
    with pytest.raises(RuntimeError, match="changed during retrieval"):
        controller.retrieve(TASK, {}, STATE)


def test_original_memory_manager_remains_available(tmp_path):
    assert MemoryManager(tmp_path).root == tmp_path
    assert not list(tmp_path.iterdir())


def test_profiler_counts_cache_hits_and_native_outcomes_once(tmp_path):
    entry(tmp_path)
    controller = MemCodexController(tmp_path, 1024)
    path = tmp_path / "profile.jsonl"
    profiler = MemoryProfiler(path, "v1")
    one = controller.retrieve_with_trace(TASK, {}, STATE)
    two = controller.retrieve_with_trace(TASK, {}, STATE)
    profiler.record("episode1", one)
    profiler.record("episode1", two)
    profiler.outcome("episode1", True)
    report = analyze(path, 0.95)
    variant = report["variants"][0]
    assert variant["memory_tokens_total"] == one.token_count * 2
    assert variant["completed"] == 1 and variant["cache_hits"] == 1
    assert report["lowest_memory_variant_meeting_observed_tau"] == "v1"
    assert '"context"' not in path.read_text()


def test_partial_or_unpaired_profiles_cannot_select_a_winner(tmp_path):
    entry(tmp_path)
    result = MemCodexController(tmp_path, 1024).retrieve_with_trace(TASK, {}, STATE)
    path = tmp_path / "profile.jsonl"
    one = MemoryProfiler(path, "one")
    two = MemoryProfiler(path, "two")
    one.record("ep1", result)
    one.outcome("ep1", True)
    two.record("ep2", result)
    report = analyze(path, 0.95)
    assert not report["paired_complete_coverage"]
    assert report["lowest_memory_variant_meeting_observed_tau"] is None
    two.outcome("ep2", True)
    assert analyze(path, 0.95)["lowest_memory_variant_meeting_observed_tau"] is None


def test_duplicate_outcome_rejected(tmp_path):
    path = tmp_path / "profile.jsonl"
    profiler = MemoryProfiler(path, "v1")
    profiler.outcome("ep1", True)
    profiler.outcome("ep1", True)
    with pytest.raises(ValueError, match="one boolean"):
        analyze(path, 0.95)


def test_paired_selection_obeys_success_threshold_before_cost(tmp_path):
    entry(tmp_path)
    result = MemCodexController(tmp_path, 1024).retrieve_with_trace(TASK, {}, STATE)
    path = tmp_path / "profile.jsonl"
    for variant, outcomes, cost in (
        ("cheap", [True, False], 50),
        ("expensive", [True, True], 100),
    ):
        profiler = MemoryProfiler(path, variant)
        for index, success in enumerate(outcomes):
            # Synthetic measurements test the analyzer, not native success.
            profiler.record(str(index), replace(result, token_count=cost))
            profiler.outcome(str(index), success)
    assert (
        analyze(path, 0.95)["lowest_memory_variant_meeting_observed_tau"] == "expensive"
    )
    assert analyze(path, 0.5)["lowest_memory_variant_meeting_observed_tau"] == "cheap"


def test_profiles_with_different_tokenizers_are_not_comparable(tmp_path):
    entry(tmp_path)
    result = MemCodexController(tmp_path, 1024).retrieve_with_trace(TASK, {}, STATE)
    path = tmp_path / "profile.jsonl"
    for name in ("tokenizer-one", "tokenizer-two"):
        profiler = MemoryProfiler(path, name)
        profiler.record("paired-episode", replace(result, counter_name=name))
        profiler.outcome("paired-episode", True)
    report = analyze(path, 0.95)
    assert not report["paired_complete_coverage"]
    assert report["lowest_memory_variant_meeting_observed_tau"] is None
