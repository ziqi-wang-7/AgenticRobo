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

"""Metrics from native outcomes and measured events, with explicit coverage."""

from __future__ import annotations

import json
import math
import os
import shlex
from collections import Counter
from collections.abc import Callable
from pathlib import Path


def read_events(path: Path) -> list[dict]:
    """Read complete JSONL records; ignore only an interrupted final line."""
    if not path.exists():
        return []
    lines = path.read_text().splitlines(keepends=True)
    records = []
    for index, line in enumerate(lines):
        try:
            records.append(json.loads(line))
        except ValueError:
            if index != len(lines) - 1 or line.endswith("\n"):
                raise
    return records


def distribution(values: list[float]) -> dict:
    """Report linear-interpolated quantiles, retaining null for no measurements."""
    values = sorted(values)
    if not values:
        return {"n": 0, "total": 0, "mean": None, "p50": None, "p95": None}

    def percentile(q):
        position = (len(values) - 1) * q
        lo, hi = math.floor(position), math.ceil(position)
        return values[lo] + (values[hi] - values[lo]) * (position - lo)

    return {
        "n": len(values),
        "total": sum(values),
        "mean": sum(values) / len(values),
        "p50": percentile(0.5),
        "p95": percentile(0.95),
    }


def native_memory_reads(
    events: list[dict], root: Path, counter: Callable[[str], int]
) -> list[dict]:
    """Attribute native shell output only when all file actions read this corpus."""
    commands = {}
    for event in events:
        item = event.get("payload", {}).get("item", {})
        if (
            event.get("method") == "item/completed"
            and item.get("type") == "commandExecution"
        ):
            commands[item["id"]] = item
    reads = []
    root = root.resolve()
    for identifier, item in commands.items():
        actions = item.get("command_actions", [])
        cwd = Path(item.get("cwd") or Path.cwd())
        mentions = str(root) in item.get("command", "") or cwd.is_relative_to(root)
        if root.is_relative_to(cwd):
            mentions |= str(root.relative_to(cwd)) in item.get("command", "")
        if not mentions:
            continue
        memory_only = bool(actions)
        content_read = False
        for action in actions:
            try:
                tokens = shlex.split(action.get("command", ""))
            except ValueError:
                tokens = []
            paths = []
            for token in tokens[1:]:
                candidate = (cwd / token).resolve()
                try:
                    if not token.startswith("-") and (
                        candidate.exists() or str(root) in token
                    ):
                        paths.append(candidate)
                except OSError:
                    memory_only = False
            action_ok = bool(tokens) and tokens[0] in {
                "cat",
                "ls",
                "find",
                "head",
                "tail",
                "sed",
                "rg",
                "grep",
                "wc",
            }
            action_ok &= bool(paths) and all(
                path.is_relative_to(root) for path in paths
            )
            action_ok &= "-i" not in tokens and not any(
                token in {">", ">>", "|", "&&", ";"} for token in tokens
            )
            memory_only &= action_ok
            content_read |= action.get("type") == "read" or (
                bool(tokens)
                and tokens[0] in {"cat", "head", "tail", "sed", "rg", "grep"}
            )
        output = item.get("aggregated_output", item.get("aggregatedOutput", ""))
        reads.append(
            {
                "id": identifier,
                "channel": "codex_native_shell",
                "memory_only": memory_only,
                "content_read": content_read,
                "duration_ms": item.get("duration_ms"),
                "memory_tokens": (counter(output) if item.get("exit_code") == 0 else 0)
                if memory_only
                else None,
                "output_chars": len(output),
                "exit_code": item.get("exit_code"),
            }
        )
    return reads


def collect(
    run: Path, cell: dict, services: Path, exit_code: int, elapsed_s: float
) -> dict:
    """Classify task failures separately from interrupted infrastructure."""
    events = read_events(run / "telemetry.jsonl")
    paths = list(run.glob("transcript_*.json"))
    transcript = json.loads(paths[0].read_text()) if len(paths) == 1 else {}
    stats = transcript.get("stats", {})
    error = transcript.get("error")
    env = [e for e in events if e["kind"] in {"env_step", "env_chunk_step"}]
    env_ready = any(e["kind"] == "env_reset" for e in events)
    native = transcript.get("environment_success")
    if native is None and env:
        native = any(e["native_success"] for e in env)
    budget = error and any(
        s in error.lower()
        for s in ("timed out after 1200", "max_turns", "max turns", "maximum turns")
    )
    infra = (
        not transcript
        or not env_ready
        or (bool(error) and not budget)
        or (exit_code != 0 and not error)
    )
    if native is None and not infra:
        native = False
    tools = [e for e in events if e["kind"] == "tool"]
    memory = [e for e in tools if e["memory_operation"]]
    injections = [e for e in events if e["kind"] == "memory_injection"]
    stream_events = [
        e for stream in run.glob("*.stream.jsonl") for e in read_events(stream)
    ]
    native_reads = []
    plan_path = services.parent / "plan.json"
    if plan_path.exists():
        import tiktoken

        os.environ.setdefault(
            "TIKTOKEN_CACHE_DIR",
            str(Path(__file__).resolve().parents[1] / "repro/tokenizer-cache"),
        )
        encoding = tiktoken.get_encoding("o200k_base")
        root = Path(json.loads(plan_path.read_text())["runtime_memory_root"])
        native_reads = native_memory_reads(
            stream_events,
            root,
            lambda text: len(encoding.encode(text, disallowed_special=())),
        )
    ambiguous_native = sum(not item["memory_only"] for item in native_reads)
    known_memory_tokens = sum(e["memory_tokens"] for e in memory + injections) + sum(
        e["memory_tokens"] or 0 for e in native_reads
    )
    primitives = [e for e in tools if e["primitive"]]
    spans = [e for e in events if e["kind"] == "otel_span"]
    sampling = [e for e in spans if e["name"] == "codex.sampling"]
    requests = [
        e
        for e in spans
        if e["name"]
        in {"responses_websocket.stream_request", "responses_sse.stream_request"}
    ]
    vla = [
        e
        for e in read_events(services / "vla.jsonl")
        if e.get("episode_id") == cell["episode_id"]
    ]
    latency = {
        "llm_request": [e["duration_ms"] / 1000 for e in requests],
        "llm_sampling": [e["duration_ms"] / 1000 for e in sampling],
        "memory_retrieval": [e["duration_ms"] / 1000 for e in memory + injections]
        + [
            e["duration_ms"] / 1000
            for e in native_reads
            if e["duration_ms"] is not None
        ],
        "vla_inference": [e["duration_ms"] / 1000 for e in vla],
        "vla_client_roundtrip": [
            e["duration_ms"] / 1000 for e in events if e["kind"] == "vla_client"
        ],
    }
    calls = len(sampling) if sampling else stats.get("turns_used")
    native_commands = 0
    for stream in run.glob("*.stream.jsonl"):
        for event in read_events(stream):
            if event.get("method") == "item/completed":
                item = event.get("payload", {}).get("item", {})
                native_commands += item.get("type") == "commandExecution"
    return {
        **cell,
        "valid": not infra,
        "native_success": native,
        "status": "infrastructure_error"
        if infra
        else "success"
        if native
        else "task_failure",
        "error": error
        or ("missing transcript or environment initialization" if infra else None),
        "exit_code": exit_code,
        "budget_exhausted": bool(budget),
        "episode_length_env_steps": sum(e["steps"] for e in env),
        "total_input_tokens": stats.get("total_input_tokens"),
        "total_output_tokens": stats.get("total_output_tokens"),
        "cached_input_tokens": stats.get("total_cached_input_tokens"),
        "reasoning_tokens": stats.get("total_reasoning_output_tokens"),
        "memory_tokens": known_memory_tokens if not ambiguous_native else None,
        "memory_tokens_known_lower_bound": known_memory_tokens,
        "memory_token_counter": "o200k_base estimate of injected memory, MCP text and SDK-recorded native memory output, no history multiplier",
        "initial_memory_injection_tokens": sum(e["memory_tokens"] for e in injections),
        "native_memory_reads": native_reads,
        "llm_calls": calls,
        "llm_completed_responses": stats.get("turns_used"),
        "llm_calls_source": "codex.sampling spans"
        if sampling
        else "token-usage updates (completed only)",
        "agent_latency_s": stats.get("elapsed_s"),
        "end_to_end_latency_s": elapsed_s,
        "memory_retrieval_frequency": len(memory) + len(native_reads) + len(injections),
        "memory_content_reads": sum(
            e["name"] in {"read_text_file", "retrieve_memory"} for e in memory
        )
        + sum(e["content_read"] for e in native_reads),
        "primitive_calls": len(primitives),
        "primitive_counts": dict(Counter(e["name"] for e in primitives)),
        "recovery_calls": sum(e["recovery_retry"] for e in primitives),
        "recovery_definition": "same primitive/prompt retried after an explicit error or failed pi0_pick",
        "latency_samples_s": latency,
        "latency_s": {name: distribution(values) for name, values in latency.items()},
        "telemetry_coverage": {
            "native_command_executions": native_commands,
            "memory_tools_only_coverage": native_commands == 0,
            "native_memory_commands": len(native_reads),
            "ambiguous_native_memory_commands": ambiguous_native,
            "llm_request_spans": len(requests),
            "sampling_spans": len(sampling),
            "memory_ops": len(memory) + len(native_reads) + len(injections),
            "memory_injections": len(injections),
            "indexed_memory_reads": sum(
                e["name"] == "retrieve_memory" and not e.get("explicit_failure")
                for e in memory
            ),
            "memory_ops_mcp": len(memory),
            "vla_server_calls": len(vla),
            "vla_client_calls": len(latency["vla_client_roundtrip"]),
            "otel_decode_errors": sum(e["kind"] == "otel_decode_error" for e in events),
        },
        "video_paths": [str(p.relative_to(run)) for p in run.glob("*.mp4")],
    }


def summarize(results: list[dict], planned: int) -> dict:
    """Use only valid finished episodes for success; expose unfinished coverage."""
    valid = [r for r in results if r["valid"]]
    n = len(valid)
    successes = sum(r["native_success"] is True for r in valid)
    latency = {}
    for name in (
        "llm_request",
        "llm_sampling",
        "memory_retrieval",
        "vla_inference",
        "vla_client_roundtrip",
    ):
        latency[name] = distribution(
            [v for r in valid for v in r["latency_samples_s"][name]]
        )
    for name in ("agent", "end_to_end"):
        latency[name] = distribution(
            [
                r[name + "_latency_s"]
                for r in valid
                if r[name + "_latency_s"] is not None
            ]
        )
    totals = {}
    for field in (
        "total_input_tokens",
        "total_output_tokens",
        "reasoning_tokens",
        "memory_tokens",
        "llm_calls",
        "primitive_calls",
        "recovery_calls",
        "memory_retrieval_frequency",
    ):
        measured = [r[field] for r in valid if r[field] is not None]
        totals[field] = {
            "total": sum(measured),
            "episodes_measured": len(measured),
            "mean_per_episode": sum(measured) / len(measured) if measured else None,
        }
    return {
        "planned": planned,
        "finished_valid": n,
        "infrastructure_errors": len(results) - n,
        "pending": planned - len(results),
        "successes": successes,
        "failures": n - successes,
        "task_success_rate": successes / n if n else None,
        "failure_rate": (n - successes) / n if n else None,
        "episode_length_env_steps": distribution(
            [r["episode_length_env_steps"] for r in valid]
        ),
        "totals": totals,
        "latency_s": latency,
    }
