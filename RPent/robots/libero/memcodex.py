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

"""Opt-in fixed-memory preparation for LIBERO evaluation."""

from __future__ import annotations

import json
import time
from argparse import Namespace
from dataclasses import asdict
from pathlib import Path

from rpent.memory.memcodex import MemCodexController
from rpent.robots.robot_spec import RunConfig

SOURCE_REVISION = "47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1"

MEMORY_INSTRUCTIONS = (
    "The complete fixed memory context for this task is supplied below. "
    "Check natural-language applicability conditions against the current scene. "
    "Re-localize all coordinates; historical numbers are evidence, not current targets. "
    "Record the entry IDs used in strategy_notes. Raw sources are reserved for offline audit.\n\n"
)


def prepare_fixed_memory(args: Namespace, config: RunConfig) -> None:
    """Validate a frozen corpus and supply one complete fixed context."""
    import hashlib

    import tiktoken

    root = Path(args.memcodex_root).expanduser().resolve()
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if manifest.get("status") != "converted" or manifest.get("schema_version") != 2:
        raise ValueError("M1 requires a converted, validated schema-2 corpus")
    if (
        manifest.get("source_version") != "GPT_6_astra_low"
        or manifest.get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("M1 requires the official Astra corpus")
    task_id = f"{args.suite}/t{args.task}"
    if not any(f"{c['suite']}/t{c['task']}" == task_id for c in manifest["catalog"]):
        raise ValueError(f"Task is absent from M1 catalog: {task_id}")
    for name in manifest["files"]:
        relative = Path(name)
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or len(relative.parts) != 2
            or relative.parts[0] not in {"summary", "relational", "skills", "raw"}
            or relative.suffix != ".json"
        ):
            raise ValueError(f"Invalid M1 entry path: {name}")
    started = time.perf_counter()
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("M1 corpus must not contain symlinks")
        if path.is_file() and path.name != "MANIFEST.json":
            relative = path.relative_to(root).as_posix()
            actual.add(relative)
            if hashlib.sha256(path.read_bytes()).hexdigest() != manifest["files"].get(
                relative
            ):
                raise ValueError(f"M1 corpus hash mismatch: {relative}")
    if actual != set(manifest["files"]):
        raise ValueError("M1 corpus file inventory mismatch")
    encoder = tiktoken.get_encoding("o200k_base")
    controller = MemCodexController(
        root,
        args.memory_token_budget,
        mode="fixed",
        cache_size=0,
        token_counter=lambda text: len(encoder.encode(text, disallowed_special=())),
        counter_name="o200k_base_estimate",
    )
    result = controller.retrieve_with_trace(task_id)
    context = MEMORY_INSTRUCTIONS + result.context
    if task_id in manifest["missing_task_family"]:
        context = (
            "No task-specific source experience exists for this task. The following is shared guidance, not a recorded solution.\n\n"
            + context
        )
    count = len(encoder.encode(context, disallowed_special=()))
    if count > args.memory_token_budget:
        raise ValueError("Fixed memory plus missing-source notice exceeds its budget")
    output = Path(config.output_dir).resolve()
    view = output / "memory-view"
    view.mkdir(parents=True, exist_ok=False)
    (view / "MEMORY.md").write_text(context)
    record = {
        **asdict(result),
        "mode": "fixed",
        "context": context,
        "token_count": count,
        "memory_root": str(view),
        "source_revision": manifest["source_revision"],
        "corpus_manifest_sha256": hashlib.sha256(
            (root / "MANIFEST.json").read_bytes()
        ).hexdigest(),
        "cache_enabled": False,
        "adaptive_routing": False,
        "early_stop": False,
        "native_command_execution": False,
        "delivery": "initial_planner_prompt",
        "prepare_elapsed_ms": (time.perf_counter() - started) * 1000,
    }
    (output / "m1-context.json").write_text(
        json.dumps(record, ensure_ascii=False, indent=2)
    )
    config.prompt_vars.update(
        memory_mode="m1", memory_dir=str(view), m1_context=context
    )
