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

"""Observe an unmodified production LIBERO CLI episode in a dedicated process."""

from __future__ import annotations

import hashlib
import sys
import time
from dataclasses import replace
from pathlib import Path

from profiling.codex_telemetry import CodexTelemetry, EventWriter


def primitive_result(data: dict) -> dict:
    """Unwrap the original primitive payload retained inside the state view."""
    return data.get("log", {}).get("result") or data


def install(writer: EventWriter, receiver: CodexTelemetry) -> None:
    """Wrap measurements only; retain original tools, outputs, prompts and policy."""
    import numpy as np
    import tiktoken

    import rpent.planner.codex as codex
    from robots.libero.env_client import LiberoEnvClient
    from robots.libero.tools import _is_primitive_action
    from rpent.robots.components.pi05_vla_client import Pi05VLAClient
    from rpent.tools.toolkit import Toolkit
    from rpent.utils.config import get_repo_root

    tokenizer = tiktoken.get_encoding("o200k_base")
    original_config = codex.build_codex_config

    def build_config(**kwargs):
        config = original_config(**kwargs)
        return replace(
            config, config_overrides=(*config.config_overrides, *receiver.overrides())
        )

    codex.build_codex_config = build_config
    original_solve = codex.CodexPlanner.solve

    def solve(self, **kwargs):
        memory = kwargs["toolkit"].memory
        if (
            not getattr(memory, "native_file_access", True)
            and getattr(memory, "memory_mode", "m1") == "m1"
        ):
            import json

            record = json.loads((memory.output_dir / "m1-context.json").read_text())
            if record["context"] not in kwargs["system_prompt"]:
                raise RuntimeError(
                    "Fixed memory is missing from the actual planner prompt"
                )
            writer.emit(
                "memory_injection",
                memory_tokens=record["token_count"],
                duration_ms=record["prepare_elapsed_ms"],
                context_sha256=hashlib.sha256(record["context"].encode()).hexdigest(),
                counter=record["counter_name"],
                mode="fixed",
                cache_hit=False,
                source_revision=record["source_revision"],
            )
        return original_solve(self, **kwargs)

    codex.CodexPlanner.solve = solve
    original_message = codex._message_to_json

    def message_to_json(message):
        value = original_message(message)
        if value["method"] in {
            "turn/started",
            "turn/completed",
            "thread/tokenUsage/updated",
            "error",
        }:
            payload = value.get("payload", {})
            writer.emit("codex_event", method=value["method"], payload=payload)
        return value

    codex._message_to_json = message_to_json
    original_tool = Toolkit.execute_tool
    failed = set()

    def execute_tool(self, name, input_dict):
        primitive = _is_primitive_action(name)
        identity = (name, str(input_dict.get("prompt", "")))
        recovery = primitive and identity in failed
        memory_operation = name == "retrieve_memory"
        path = None
        if name in {"read_text_file", "list_dir"}:
            path = (get_repo_root() / input_dict.get("path", "")).resolve()
            memory_operation = path.is_relative_to(self.memory.root)
        started = time.perf_counter_ns()
        result = None
        try:
            result = original_tool(self, name, input_dict)
            return result
        finally:
            elapsed = (time.perf_counter_ns() - started) / 1e6
            data = result.data if result is not None else {}
            primitive_data = primitive_result(data)
            explicit_failure = (
                result is None
                or result.is_error
                or (name == "pi0_pick" and primitive_data.get("success") is False)
            )
            if primitive:
                if explicit_failure:
                    failed.add(identity)
                elif name != "pi0_doubled" or primitive_data.get("success") is True:
                    failed.discard(identity)
            # Count exactly the returned text, including JSON envelope and truncation.
            content = (
                result.to_text()
                if memory_operation and result is not None and not result.is_error
                else ""
            )
            writer.emit(
                "tool",
                name=name,
                duration_ms=elapsed,
                primitive=primitive,
                recovery_retry=recovery,
                explicit_failure=explicit_failure,
                memory_operation=memory_operation,
                path=str(path) if path else None,
                memory_tokens=len(tokenizer.encode(content, disallowed_special=()))
                if content
                else 0,
                memory_counter="o200k_base_estimate_not_provider_attribution",
                response_sha256=hashlib.sha256(content.encode()).hexdigest()
                if content
                else None,
                result_success=primitive_data.get("success"),
            )

    Toolkit.execute_tool = execute_tool
    for name in ("step", "chunk_step", "reset"):
        original = getattr(LiberoEnvClient, name)

        def wrap(self, *args, _original=original, _name=name, **kwargs):
            started = time.perf_counter_ns()
            result = _original(self, *args, **kwargs)
            fields = {}
            if _name != "reset":
                fields = {
                    "steps": int(np.asarray(result[2]).size),
                    "native_success": bool(np.any(result[2])),
                    "truncated": bool(np.any(result[3])),
                }
            writer.emit(
                "env_" + _name,
                duration_ms=(time.perf_counter_ns() - started) / 1e6,
                **fields,
            )
            return result

        setattr(LiberoEnvClient, name, wrap)
    original_predict = Pi05VLAClient.predict

    def predict(self, *args, **kwargs):
        started = time.perf_counter_ns()
        try:
            return original_predict(self, *args, **kwargs)
        finally:
            writer.emit(
                "vla_client", duration_ms=(time.perf_counter_ns() - started) / 1e6
            )

    Pi05VLAClient.predict = predict


def main() -> int:
    """Run the official CLI with the supplied original arguments."""
    # The batch driver sets --output-dir to an episode-specific location.
    output = Path(sys.argv[sys.argv.index("--output-dir") + 1]).resolve()
    writer = EventWriter(output / "telemetry.jsonl")
    receiver = CodexTelemetry(writer)
    receiver.start()
    started = time.perf_counter()
    code = 1
    try:
        install(writer, receiver)
        from rpent.cli.main import main as official_main

        code = official_main()
        return code
    finally:
        writer.emit(
            "episode_process",
            duration_ms=(time.perf_counter() - started) * 1000,
            exit_code=code,
        )
        receiver.close()
        writer.close()


if __name__ == "__main__":
    raise SystemExit(main())
