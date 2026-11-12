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

"""M2 evidence boundaries, task selection, budgets and online delivery."""

import json
import sys
from threading import RLock
from types import SimpleNamespace

import pytest

from repro.index_memcodex_memory import build_index, task_requirements, validate_index
from repro.memcodex_corpus import build, encoded
from robots.libero.memcodex_online import (
    IndexedMemoryManager,
    M2LiberoToolkit,
    _reviewable_memory_ids,
    make_controller,
    prepare_indexed_memory,
)
from rpent.memory.memcodex.policy import RoutingContext, manual_route
from rpent.robots.robot_spec import RunConfig
from rpent.session.base import StepRecord
from rpent.tools.base import ToolResult, tool

TASK = "libero_object_task/t0"
LANGUAGE = "pick the milk and place it in the basket"


@pytest.fixture(autouse=True)
def offline_tokenizer(monkeypatch):
    """Exercise response accounting without downloading optional tokenizer data."""
    encoder = SimpleNamespace(encode=lambda text, **kwargs: list(text.encode()))
    monkeypatch.setitem(
        sys.modules, "tiktoken", SimpleNamespace(get_encoding=lambda name: encoder)
    )


@pytest.fixture
def indexed(tmp_path):
    files = {
        "MEMORY.md": b"# Index\n",
        "task-family/object.md": (
            "---\nid: object\nsuite: libero_object\nregime: task\ntask_id: 0\ntask_language: "
            + LANGUAGE
            + "\nrelated: []\n---\n"
            "## Applicable pattern\nPick milk and place in basket.\n"
            "## Re-localization per scene\nLocate milk and basket from current images.\n"
            "## Winning technique\nUse pi0_pick, move_to and release, carrying closed.\n"
            "## Failure modes\nIf the milk slips, re-localize and retry the grasp.\n"
        ).encode(),
        "task-family/other.md": b"---\nid: other\nsuite: libero_object\nregime: task\ntask_id: 1\ntask_language: pick the butter\nrelated: []\n---\n## Applicable pattern\nPick butter.\n## Winning technique\nUse pi0_pick.\n",
    }
    catalog = [
        {"suite": "libero_object_task", "task": 0, "language": LANGUAGE},
        {"suite": "libero_object_task", "task": 1, "language": "pick the butter"},
    ]
    entries, manifest = build(files, catalog)
    root = tmp_path / "corpus"
    for name, payload in entries.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    (root / "MANIFEST.json").write_bytes(encoded(manifest))
    index = tmp_path / "INDEX.json"
    index.write_bytes(encoded(build_index(root)))
    return root, index


def controller(indexed, budget=20000):
    root, index = indexed
    return make_controller(
        {
            "corpus": str(root),
            "index": str(index),
            "token_budget": budget,
        }
    )


def query(control, *, localized=True, **state):
    return control.retrieve_with_trace(
        TASK,
        {
            "task_language": LANGUAGE,
            "localized_objects": ["milk", "basket"] if localized else [],
        },
        {"available_skills": ["pi0_pick", "move_to", "release"], **state},
    )


def test_early_stop_and_only_selected_bodies_open(indexed, monkeypatch):
    control = controller(indexed)
    opened = []
    original = control.indexed.read

    def read(item):
        opened.append(item["path"])
        return original(item)

    monkeypatch.setattr(control.indexed, "read", read)
    overview = query(control, purpose="overview", localized=False)
    assert overview.sufficient and overview.layers_visited == ("summary",)
    assert len(opened) == overview.files_read == 1
    missing_perception = query(control, localized=False)
    assert missing_perception.layers_visited == ("summary", "relational")
    assert not missing_perception.memory_read_complete
    assert not missing_perception.state_evidence_complete
    assert "skill_guidance" in missing_perception.memory_missing
    assert not missing_perception.sufficient
    assert "object_localization" in missing_perception.missing
    ready = query(control)
    assert ready.sufficient and ready.layers_visited == (
        "summary",
        "relational",
        "skills",
    )
    assert all(not p.startswith("raw/") for p in opened)
    task_paths = {
        x["path"]
        for layer in control.indexed.profile(TASK)["layers"].values()
        for x in layer
    }
    assert set(opened) <= task_paths
    assert not query(control).cache_hit


def test_raw_fallback_and_recovery_use_retained_evidence(indexed):
    control = controller(indexed)
    control.indexed.index["tasks"][TASK]["layers"]["skills"] = []
    result = query(control, needs_recovery=True)
    assert result.sufficient and result.layers_visited[-1] == "raw"
    assert any(key.startswith("raw:") for key in result.selected_ids)


def test_budget_whole_entries_and_serialized_counter(indexed):
    import tiktoken

    generous = query(controller(indexed), purpose="overview")
    control = controller(indexed, generous.token_count + 8)
    result = query(control)
    assert result.token_count <= control.token_budget
    assert result.omitted_ids and not result.sufficient
    wire = ToolResult(data={"memory_context": result.context}).to_text()
    assert len(tiktoken.get_encoding("o200k_base").encode(wire)) == result.token_count
    assert "[truncated]" not in wire
    with pytest.raises(ValueError, match="envelope"):
        query(controller(indexed, 5))


def test_unknown_conditions_need_review_before_raw(indexed):
    control = controller(indexed)
    items = control.indexed.index["tasks"][TASK]["layers"]["skills"]
    for item in items:
        control.indexed.index["entries"][item["path"]]["review_required"] = True
    initial = query(control)
    assert not initial.sufficient and initial.layers_visited[-1] == "skills"
    assert "review_entries" in initial.context
    assert query(control, applicable_entries=list(initial.selected_ids)).sufficient


def test_index_validation_and_task_mismatch(indexed):
    root, index = indexed
    assert validate_index(root, index)["tasks"] == 2
    control = controller(indexed)
    with pytest.raises(ValueError, match="language"):
        control.retrieve_with_trace(TASK, {"task_language": "pick butter"})
    with pytest.raises(ValueError, match="Unknown indexed task"):
        control.retrieve_with_trace("libero_object_swap/t0")
    index.write_text(index.read_text() + "\n")
    with pytest.raises(RuntimeError, match="changed"):
        query(control)


def test_corrupt_selected_body_rejected_unrelated_body_not_read(indexed):
    root, _ = indexed
    control = controller(indexed)
    other = control.indexed.profile("libero_object_task/t1")["layers"]["summary"][0][
        "path"
    ]
    (root / other).write_text("invalid unrelated JSON")
    assert query(control, purpose="overview").sufficient
    own = control.indexed.profile(TASK)["layers"]["summary"][0]["path"]
    (root / own).write_text("invalid selected JSON")
    with pytest.raises(ValueError, match="changed"):
        query(control, purpose="overview")


@pytest.fixture
def online(indexed, tmp_path):
    root, index = indexed
    output = tmp_path / "run"
    config = RunConfig(
        "test", output, {"output_dir": str(output), "recipe_tag": "test"}, {}
    )
    prepare_indexed_memory(
        SimpleNamespace(
            memcodex_root=root,
            memcodex_index=index,
            suite="libero_object_task",
            task=0,
            memory_token_budget=20000,
        ),
        config,
    )
    memory = IndexedMemoryManager(
        output / "memory-view", output_dir=output, guides=tmp_path / "guides"
    )
    kit = object.__new__(M2LiberoToolkit)
    kit._memory = memory
    record = StepRecord(0, {}, extras={"task_language": LANGUAGE})
    kit._state = SimpleNamespace(latest_record=lambda: record)
    kit._memory_lock = RLock()
    kit._localization_evidence = {}
    kit._memory_seen = {}
    kit._tools = dict.fromkeys(["pi0_pick", "move_to", "release"])
    return kit, record, config


def localize(kit, xyz=(1.0, 2.0, 3.0)):
    @tool(readonly=True)
    def back_project() -> ToolResult:
        """Return a simulated depth-tool result for contract tests."""
        return ToolResult(
            data={
                "step": kit.state.latest_record().step_idx,
                "world_xyz": list(xyz),
                "camera": "agentview",
            }
        )

    return kit._capture_localization(back_project).data.get("memory_evidence_id")


def test_online_requires_real_current_evidence_and_tracks_provenance(online):
    kit, record, config = online
    kit.retrieve_memory(step=0, purpose="overview")
    with pytest.raises(ValueError, match="perception evidence"):
        kit.retrieve_memory(step=0, localizations={"milk": "made-up"})
    with pytest.raises(ValueError, match="already returned"):
        kit.retrieve_memory(step=0, applicable_entries=["skills:made-up"])
    milk, basket = localize(kit), localize(kit)
    result = kit.retrieve_memory(step=0, localizations={"milk": milk, "basket": basket})
    assert '"sufficient": true' in result.data["memory_context"]
    trace = json.loads(
        (config.output_dir / "m2-retrievals.jsonl").read_text().splitlines()[-1]
    )
    assert trace["localization_evidence"]["milk"]["tool"] == "back_project"
    assert trace["semantic_binding_source"] == "planner_image_interpretation"
    record.step_idx = 1
    with pytest.raises(ValueError, match="Stale memory query"):
        kit.retrieve_memory(step=0)
    with pytest.raises(ValueError, match="stale perception"):
        kit.retrieve_memory(step=1, localizations={"milk": milk})
    assert localize(kit, xyz=(float("nan"), 2, 3)) is None


def test_online_failed_grasp_requests_recovery_without_extra_llm(online):
    kit, record, config = online
    record.command = {"action": "pi0_pick"}
    record.result = {"success": False}
    kit.retrieve_memory(
        step=0, localizations={"milk": localize(kit), "basket": localize(kit)}
    )
    trace = json.loads((config.output_dir / "m2-retrievals.jsonl").read_text())
    assert trace["state"]["needs_recovery"] and trace["last_action_failed"]
    assert trace["sufficient"]


def test_m2_prompt_and_access_boundaries(online):
    from robots.libero.robot_spec import get_robot_spec

    kit, _, config = online
    prompt = get_robot_spec().prompts.render("system", variables=config.prompt_vars)
    assert "retrieve_memory" in prompt and "memory_evidence_id" in prompt
    assert "{{m1_context}}" not in prompt and "NEVER read the BDDL" in prompt
    assert not (config.output_dir / "m1-context.json").exists()
    for name in ("m2-config.json", "m2-retrievals.jsonl"):
        for method in (kit.memory.authorize_read, kit.memory.authorize_write):
            with pytest.raises(PermissionError):
                method(config.output_dir / name)


def test_overview_memory_ids_are_reviewable_only_within_same_step():
    seen = {
        (0, ""): {"summary:task", "skills:grasp"},
        (0, "candidate-A"): {"skills:grasp", "skills:place"},
        (0, "candidate-B"): {"skills:drawer"},
        (1, ""): {"summary:next-step"},
    }

    assert _reviewable_memory_ids(seen, 0, "candidate-A") == {
        "summary:task",
        "skills:grasp",
        "skills:place",
    }
    assert _reviewable_memory_ids(seen, 0, "candidate-B") == {
        "summary:task",
        "skills:drawer",
    }
    assert _reviewable_memory_ids(seen, 1, "candidate-A") == {
        "summary:next-step",
    }


def test_unverified_action_facts_request_observation_not_memory_review():
    context = RoutingContext(
        layer="skills",
        query=None,
        memory_missing=(),
        action_missing_facts=(),
        action_unverified_facts=("identity:white_mug",),
        review_entries=(),
        selected_ids=("skills:grasp",),
        recovery=False,
    )

    assert manual_route(context).action == "OBSERVE"


def test_missing_action_facts_request_observation():
    context = RoutingContext(
        layer="skills",
        query=None,
        memory_missing=(),
        action_missing_facts=("holding:white_mug",),
        action_unverified_facts=(),
        review_entries=(),
        selected_ids=("skills:place",),
        recovery=False,
    )

    assert manual_route(context).action == "OBSERVE"


def test_memory_applicability_review_remains_review():
    context = RoutingContext(
        layer="skills",
        query=None,
        memory_missing=(),
        action_missing_facts=(),
        action_unverified_facts=(),
        review_entries=("skills:analogy",),
        selected_ids=("skills:analogy",),
        recovery=False,
    )

    assert manual_route(context).action == "REVIEW"


def test_task_requirements_keep_distinct_targets():
    result = task_requirements(
        "put the white mug on the left plate and the chocolate pudding to the right of the right plate"
    )
    assert set(result["required_objects"]) == {
        "white_mug",
        "left_plate",
        "chocolate_pudding",
        "right_plate",
    }
