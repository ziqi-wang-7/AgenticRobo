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

"""Offline baseline contracts: denominator, scope and measurement provenance."""

import json

import pytest

from profiling.codex_telemetry import attributes
from profiling.libero_capture import primitive_result
from profiling.libero_metrics import (
    collect,
    distribution,
    native_memory_reads,
    read_events,
    summarize,
)
from repro.liberopro_baseline import command


def fixture_run(tmp_path, *, error=None, native=False, code=0):
    run = tmp_path / "episode"
    run.mkdir()
    services = tmp_path / "services"
    services.mkdir()
    cell = {
        "episode_id": "libero_spatial_swap-t0-s1",
        "suite": "libero_spatial_swap",
        "task": 0,
        "seed": 1,
    }
    stats = {
        "turns_used": 2,
        "elapsed_s": 3.5,
        "total_input_tokens": 100,
        "total_output_tokens": 20,
        "total_reasoning_output_tokens": 5,
    }
    (run / "transcript_test.json").write_text(
        json.dumps({"error": error, "environment_success": native, "stats": stats})
    )
    events = [
        {"kind": "env_reset"},
        {"kind": "env_step", "steps": 1, "native_success": native},
        {"kind": "env_chunk_step", "steps": 10, "native_success": native},
        {
            "kind": "tool",
            "name": "read_text_file",
            "memory_operation": True,
            "memory_tokens": 7,
            "duration_ms": 20,
            "primitive": False,
            "recovery_retry": False,
        },
        {
            "kind": "tool",
            "name": "pi0_pick",
            "memory_operation": False,
            "memory_tokens": 0,
            "duration_ms": 2000,
            "primitive": True,
            "recovery_retry": True,
        },
        {"kind": "otel_span", "name": "codex.sampling", "duration_ms": 1000},
        {
            "kind": "otel_span",
            "name": "responses_websocket.stream_request",
            "duration_ms": 900,
        },
    ]
    (run / "telemetry.jsonl").write_text("".join(json.dumps(e) + "\n" for e in events))
    return collect(run, cell, services, code, 5.0)


def test_metrics_use_native_outcome_and_primitive_step_units(tmp_path):
    r = fixture_run(tmp_path, native=True)
    assert r["native_success"] and r["valid"]
    assert r["episode_length_env_steps"] == 11
    assert r["memory_tokens"] == 7 and r["reasoning_tokens"] == 5
    assert r["primitive_calls"] == 1 and r["recovery_calls"] == 1
    assert r["latency_s"]["llm_request"]["mean"] == 0.9
    assert r["agent_latency_s"] == 3.5 and r["end_to_end_latency_s"] == 5


def test_budget_failure_counts_but_quota_error_is_infrastructure(tmp_path):
    first = tmp_path / "one"
    first.mkdir()
    second = tmp_path / "two"
    second.mkdir()
    timeout = fixture_run(first, error="Codex SDK timed out after 1200s", code=1)
    quota = fixture_run(second, error="usageLimitExceeded", code=1)
    assert timeout["valid"] and not timeout["native_success"]
    assert not quota["valid"]
    s = summarize([timeout, quota], 80)
    assert (
        s["finished_valid"] == 1
        and s["infrastructure_errors"] == 1
        and s["pending"] == 78
    )
    assert s["task_success_rate"] == 0 and s["failure_rate"] == 1


def test_p95_is_pooled_over_calls_not_mean_of_episode_percentiles(tmp_path):
    r = fixture_run(tmp_path)
    r["latency_samples_s"]["llm_request"] = [1, 2, 100]
    s = summarize([r], 80)
    assert s["latency_s"]["llm_request"]["p50"] == 2
    assert s["latency_s"]["llm_request"]["p95"] == pytest.approx(90.2)
    assert distribution([])["p95"] is None


def test_only_an_interrupted_final_jsonl_record_is_ignored(tmp_path):
    p = tmp_path / "events.jsonl"
    p.write_text('{"kind":"ok"}\n{"incomplete"')
    assert read_events(p) == [{"kind": "ok"}]
    p.write_text('{"broken"\n')
    with pytest.raises(ValueError):
        read_events(p)


def test_telemetry_drops_prompts_credentials_and_output_bodies():
    values = [
        {"key": k, "value": {"stringValue": v}}
        for k, v in [
            ("event.name", "codex.api_request"),
            ("duration_ms", "2"),
            ("prompt", "private"),
            ("authorization", "secret"),
            ("output", "private"),
        ]
    ]
    assert attributes(values) == {"event.name": "codex.api_request", "duration_ms": "2"}


def test_recovery_reads_nested_original_primitive_result():
    assert primitive_result({"log": {"result": {"success": False}}})["success"] is False


def test_native_shell_memory_is_counted_and_mixed_output_is_not_fabricated(tmp_path):
    memory = tmp_path / "memory"
    memory.mkdir()
    (memory / "MEMORY.md").write_text("experience")
    guide = tmp_path / "guide.md"
    guide.write_text("guide")
    action = {
        "type": "read",
        "command": f"cat {memory}/MEMORY.md",
        "path": str(memory / "MEMORY.md"),
    }
    item = {
        "id": "one",
        "type": "commandExecution",
        "cwd": str(tmp_path),
        "command": action["command"],
        "command_actions": [action],
        "aggregated_output": "experience",
        "duration_ms": 251,
        "exit_code": 0,
    }
    event = {"method": "item/completed", "payload": {"item": item}}
    reads = native_memory_reads([event, event], memory, len)
    assert len(reads) == 1 and reads[0]["memory_tokens"] == 10
    assert reads[0]["duration_ms"] == 251 and reads[0]["content_read"]
    item["command_actions"].append(
        {
            "type": "read",
            "command": f"cat {guide}",
            "path": str(guide),
        }
    )
    mixed = native_memory_reads([event], memory, len)
    assert not mixed[0]["memory_only"] and mixed[0]["memory_tokens"] is None


def test_command_uses_hf_pro_original_defaults_without_memcodex(tmp_path):
    argv = command(
        tmp_path,
        {"suite": "libero_goal_swap", "task": 7, "seed": 1, "episode_id": "test"},
        {"environment_gpu": 1, "vla_port": 8420, "sam3_port": 8414},
    )
    for flag, value in [
        ("--libero-type", "pro"),
        ("--model", "gpt-6-astra"),
        ("--reasoning-effort", "low"),
        ("--memory-profile", "hf"),
        ("--memory-version", "GPT_6_astra_low"),
        ("--planner-timeout-s", "1200"),
    ]:
        assert argv[argv.index(flag) + 1] == value
    assert "--explore" not in argv and "--memory-dir" not in argv
    assert "--no-auto-merge-memory" in argv


def test_m1_initial_prompt_memory_is_counted_once_alongside_later_reads(tmp_path):
    fixture_run(tmp_path)
    run = tmp_path / "episode"
    with (run / "telemetry.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "memory_injection",
                    "memory_tokens": 1234,
                    "duration_ms": 50,
                }
            )
            + "\n"
        )
    result = collect(run, {"episode_id": "test"}, tmp_path / "services", 0, 5)
    assert result["memory_tokens"] == 1241
    assert result["initial_memory_injection_tokens"] == 1234
    assert result["memory_retrieval_frequency"] == 2
    assert result["telemetry_coverage"]["memory_injections"] == 1
    assert result["latency_samples_s"]["memory_retrieval"] == [0.02, 0.05]


def test_m1_command_selects_fixed_corpus_and_preserves_policy_limits(tmp_path):
    plan = {
        "memcodex_enabled": True,
        "condition": "memcodex_m1_fixed",
        "memcodex_root": "/frozen/m1",
        "memory_token_budget": 32768,
    }
    (tmp_path / "plan.json").write_text(json.dumps(plan))
    argv = command(
        tmp_path,
        {"suite": "libero_goal_swap", "task": 7, "seed": 2, "episode_id": "test"},
        {"environment_gpu": 1, "vla_port": 8420, "sam3_port": 8414},
    )
    assert argv[argv.index("--memcodex-root") + 1] == "/frozen/m1"
    assert argv[argv.index("--memory-token-budget") + 1] == "32768"
    assert argv[argv.index("--max-episode-steps") + 1] == "10000"
    assert "--memory-repo" not in argv


def test_m2_command_and_tool_memory_accounting(tmp_path):
    fixture_run(tmp_path)
    run = tmp_path / "episode"
    with (run / "telemetry.jsonl").open("a") as stream:
        stream.write(
            json.dumps(
                {
                    "kind": "tool",
                    "name": "retrieve_memory",
                    "memory_operation": True,
                    "memory_tokens": 200,
                    "duration_ms": 25,
                    "primitive": False,
                    "recovery_retry": False,
                    "explicit_failure": False,
                }
            )
            + "\n"
        )
    result = collect(run, {"episode_id": "test"}, tmp_path / "services", 0, 5)
    assert result["memory_tokens"] == 207
    assert result["memory_content_reads"] == 2
    assert result["telemetry_coverage"]["indexed_memory_reads"] == 1
    assert result["telemetry_coverage"]["memory_injections"] == 0
    (tmp_path / "plan.json").write_text(
        json.dumps(
            {
                "memcodex_enabled": True,
                "condition": "memcodex_m2_indexed",
                "memcodex_root": "/frozen/m1",
                "memcodex_index": "/frozen/m2/INDEX.json",
                "memory_token_budget": 32768,
            }
        )
    )
    argv = command(
        tmp_path,
        {"suite": "libero_goal_swap", "task": 7, "seed": 2, "episode_id": "test"},
        {"environment_gpu": 1, "vla_port": 8420, "sam3_port": 8414},
    )
    assert argv[argv.index("--memcodex-mode") + 1] == "m2"
    assert argv[argv.index("--memcodex-index") + 1] == "/frozen/m2/INDEX.json"
