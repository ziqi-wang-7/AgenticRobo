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

"""M1 source fidelity, fixed selection, budget and online access contracts."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from repro.memcodex_corpus import REVISION, VERSION, build, digest, encoded, validate
from rpent.memory.memcodex import MemCodexController
from rpent.memory.memcodex.adapter import FixedMemoryManager


@pytest.fixture
def corpus(tmp_path):
    files = {
        "MEMORY.md": b"\n# Original index\n\n",
        "global/grasp.md": b"---\nid: grasp\nscope: global\nkind: primitive\napplies_when: Only after a verified grasp.\nevidence:\n  cells: [spatial_task_t0_s0]\nrelated: []\n---\nCarry closed; never carry with an open gripper.\n",
        "task-family/spatial.md": b"---\nid: spatial\nsuite: spatial\nregime: task\ntask_id: 0\ntask_language: pick bowl\nrelated: [grasp]\n---\n## Applicable pattern\nPick the bowl, not the plate.\n\n## Winning technique\nUse pi0_pick, then carry closed.\n\n## Re-localization per scene\nLocalize the bowl in the current frame.\n\n## Failure modes\nIf the bowl slips, re-localize before retrying.\n",
        "task-specific/spatial_task_t0_s0.json": encoded(
            {
                "suite": "libero_spatial_task",
                "task_id": 0,
                "seed": 0,
                "strategy_notes": "Never reuse the seed-0 location.",
                "final_state": {"xyz": [1, 2, 3]},
            }
        ),
        "task-specific/spatial_task_t0_s0_recipe.jsonl": b'{"action":"move_to","xyz":[1,2,3],"gripper":1}\n{"action":"release"}\n',
    }
    catalog = [
        {"suite": "libero_spatial_task", "task": 0, "language": "pick bowl"},
        {
            "suite": "libero_10_swap",
            "task": 6,
            "language": "put chocolate beside plate",
        },
    ]
    entries, manifest = build(files, catalog)
    root, source = tmp_path / "converted", tmp_path / "official"
    for base, contents in ((root, entries), (source, files)):
        for name, value in contents.items():
            path = base / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(value)
    (root / "MANIFEST.json").write_bytes(encoded(manifest))
    receipt = tmp_path / "receipt.json"
    receipt.write_bytes(
        encoded(
            {
                "version": VERSION,
                "revision": REVISION,
                "files": {k: digest(v) for k, v in files.items()},
            }
        )
    )
    return root, source, receipt, files, catalog


def test_conversion_preserves_sources_conditions_and_recipe_order(corpus):
    root, source, receipt, files, _ = corpus
    report = validate(root, source, receipt)
    assert report["missing_task_family"] == ["libero_10_swap/t6"]
    raw = [json.loads(p.read_text()) for p in (root / "raw").glob("*.json")]
    assert {r["text"] for r in raw} == {v.decode() for v in files.values()}
    result = MemCodexController(root, 20000, mode="fixed").retrieve_with_trace(
        "libero_spatial_task/t0"
    )
    assert "Only after a verified grasp" in result.context
    assert "not the plate" in result.context
    assert "If the bowl slips" in result.context
    assert '"xyz"' not in result.context
    assert result.context.index('"action": "move_to"') < result.context.index(
        '"action": "release"'
    )
    assert all(json.loads(p.read_text())["source_refs"] for p in root.glob("*/*.json"))


def test_fixed_mode_ignores_state_and_never_uses_raw_or_cache(corpus):
    root, *_ = corpus
    (root / "raw/broken.json").write_text(
        "invalid JSON that fixed retrieval must not decode"
    )
    controller = MemCodexController(root, 20000, mode="fixed", max_per_layer=1)
    first = controller.retrieve_with_trace("libero_spatial_task/t0", {}, {})
    second = controller.retrieve_with_trace(
        "libero_spatial_task/t0", {}, {"anomaly": True, "needs_recovery": True}
    )
    assert first.context == second.context
    assert not first.cache_hit and not second.cache_hit
    assert first.layers_visited == ("summary", "relational", "skills")
    assert len(first.selected_ids) > 3  # Fixed mode does not apply progressive top-k.
    assert not first.sufficient and first.missing == ("not_assessed_in_fixed_mode",)


def test_fixed_budget_overflow_fails_without_silently_dropping_instructions(corpus):
    root, *_ = corpus
    with pytest.raises(ValueError, match="No entries were truncated"):
        MemCodexController(root, 1, mode="fixed").retrieve("libero_spatial_task/t0")


def test_natural_language_conditions_are_not_unconditionally_routed(corpus):
    root, *_ = corpus
    result = MemCodexController(root, 20000).retrieve_with_trace(
        "libero_spatial_task/t0", {}, {"objects_localized": True}
    )
    assert not result.selected_ids


def test_validator_detects_tampering_even_after_manifest_hash_is_replaced(corpus):
    root, source, receipt, *_ = corpus
    path = next((root / "summary").glob("*.json"))
    entry = json.loads(path.read_text())
    entry["text"] = "Pick the wrong target."
    path.write_bytes(encoded(entry))
    manifest = json.loads((root / "MANIFEST.json").read_text())
    manifest["files"][path.relative_to(root).as_posix()] = digest(path.read_bytes())
    (root / "MANIFEST.json").write_bytes(encoded(manifest))
    with pytest.raises(ValueError, match="deterministic conversion"):
        validate(root, source, receipt)


def test_source_hash_and_task_language_must_match(corpus):
    root, source, receipt, files, catalog = corpus
    catalog[0]["language"] = "pick the wrong object"
    with pytest.raises(ValueError, match="task language"):
        build(files, catalog)
    (source / "MEMORY.md").write_text("changed source")
    with pytest.raises(ValueError, match="source inventory or hash"):
        validate(root, source, receipt)


def test_file_access_blocks_original_corpus_previous_runs_and_symlink_escape(
    tmp_path, monkeypatch
):
    import rpent.memory.memcodex.adapter as adapter

    monkeypatch.setattr(adapter, "get_repo_root", lambda: tmp_path)
    output = tmp_path / "logs/current"
    view = output / "memory-view"
    view.mkdir(parents=True)
    (view / "MEMORY.md").write_text("fixed memory")
    guides = tmp_path / "robots/libero/guides"
    guides.mkdir(parents=True)
    guide = guides / "guide.md"
    guide.write_text("guide")
    original = tmp_path / "memory/original.md"
    original.parent.mkdir()
    original.write_text("original full memory")
    (output / "escape.md").symlink_to(original)
    memory = FixedMemoryManager(view, output_dir=output, guides=guides)
    assert memory.authorize_read(view / "MEMORY.md") == view / "MEMORY.md"
    assert memory.authorize_read(guide) == guide
    assert memory.authorize_write(output / "audit.json") == output / "audit.json"
    for path in (original, output / "escape.md", tmp_path / "logs/previous/audit.json"):
        with pytest.raises(PermissionError):
            memory.authorize_read(path)
    for path in (view / "MEMORY.md", guide, output / "m1-context.json"):
        with pytest.raises(PermissionError):
            memory.authorize_write(path)


def test_codex_shell_setting_is_scoped_to_restricted_memory(tmp_path):
    from rpent.planner.codex import build_codex_config

    kwargs = {
        "mcp_url": "http://127.0.0.1:1234/mcp/",
        "base_url": None,
        "api_key": None,
        "cwd": str(tmp_path),
    }
    original = build_codex_config(**kwargs)
    restricted = build_codex_config(**kwargs, native_file_access=False)
    assert "features.shell_tool=false" not in original.config_overrides
    assert "features.shell_tool=false" in restricted.config_overrides


def test_m1_rejects_exploration_and_other_planners():
    from robots.libero.memory import validate_options

    for planner, explore in (("api", False), ("codex", True)):
        with pytest.raises(ValueError, match="Codex evaluation only"):
            validate_options(
                SimpleNamespace(
                    memcodex_root=Path("example"), planner=planner, explore=explore
                )
            )


def test_fixed_context_is_in_the_rendered_planner_prompt(corpus, tmp_path, monkeypatch):
    from robots.libero.memcodex import prepare_fixed_memory
    from robots.libero.robot_spec import get_robot_spec
    from rpent.robots.robot_spec import RunConfig

    root, *_ = corpus
    monkeypatch.setitem(
        sys.modules,
        "tiktoken",
        SimpleNamespace(
            get_encoding=lambda _: SimpleNamespace(
                encode=lambda text, **_: list(text.encode())
            )
        ),
    )
    config = RunConfig(
        "test",
        tmp_path / "run",
        {"output_dir": str(tmp_path / "run"), "recipe_tag": "test"},
        {},
    )
    args = SimpleNamespace(
        memcodex_root=root,
        suite="libero_spatial_task",
        task=0,
        memory_token_budget=20000,
    )
    prepare_fixed_memory(args, config)
    prompt = get_robot_spec().prompts.render("system", variables=config.prompt_vars)
    record = json.loads((config.output_dir / "m1-context.json").read_text())
    assert record["context"] in prompt
    assert "READ EACH AVAILABLE LOCAL MEMORY LAYER" not in prompt
    assert record["token_count"] == len(record["context"].encode())
    assert record["token_count"] <= args.memory_token_budget
    assert record["delivery"] == "initial_planner_prompt"
    assert "NEVER read the BDDL" in prompt
    assert "SINGLE ATTEMPT, NO RESET" in prompt


def test_fixed_prompt_budget_includes_instructions_and_missing_source_notice(
    corpus, tmp_path, monkeypatch
):
    from robots.libero.memcodex import prepare_fixed_memory
    from rpent.robots.robot_spec import RunConfig

    root, *_ = corpus
    monkeypatch.setitem(
        sys.modules,
        "tiktoken",
        SimpleNamespace(
            get_encoding=lambda _: SimpleNamespace(
                encode=lambda text, **_: list(text.encode())
            )
        ),
    )
    count = (
        MemCodexController(root, 20000, mode="fixed")
        .retrieve_with_trace("libero_10_swap/t6")
        .token_count
    )
    args = SimpleNamespace(
        memcodex_root=root, suite="libero_10_swap", task=6, memory_token_budget=count
    )
    config = RunConfig("test", tmp_path / "run", {}, {})
    with pytest.raises(ValueError, match="plus missing-source notice"):
        prepare_fixed_memory(args, config)
    assert not config.output_dir.exists()
