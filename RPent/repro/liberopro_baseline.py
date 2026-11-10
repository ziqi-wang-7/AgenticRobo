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

"""Frozen, resumable 80-episode official-memory LIBERO-PRO baseline."""

from __future__ import annotations

import argparse
import csv
import fcntl
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from profiling.libero_metrics import collect, summarize

ROOT = Path(__file__).resolve().parents[1]
REVISION = "47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1"
VERSION = "GPT_6_astra_low"


def now() -> str:
    """Return experiment time in the user's timezone."""
    return datetime.now(ZoneInfo("Asia/Shanghai")).isoformat()


def write(path: Path, data: object) -> None:
    """Atomically replace a structured artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def sha(path: Path) -> str:
    """Hash source and checkpoint files with bounded memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def validate_scenes(plan: dict) -> None:
    """Check every private reset input, including the rendered reference."""
    for cell in plan["episodes"]:
        if not cell.get("initial_scene"):
            continue
        scene = Path(cell["initial_scene"])
        expected = cell["initial_scene_sha256"]
        for suffix in (".npz", ".observation.json", ".cameras.npz"):
            path = scene.with_suffix(suffix)
            if sha(path) != expected[suffix]:
                raise RuntimeError(f"Frozen initial scene changed: {path}")


def command(batch: Path, cell: dict, deployment: dict) -> list[str]:
    """Use the original CLI flags, with only an observation wrapper."""
    argv = [
        sys.executable,
        "-u",
        "-m",
        "profiling.libero_capture",
        "--robot",
        "libero",
        "--libero-type",
        "pro",
        "--suite",
        cell["suite"],
        "--task",
        str(cell["task"]),
        "--seed",
        str(cell["seed"]),
        "--planner",
        "codex",
        "--model",
        "gpt-6-astra",
        "--reasoning-effort",
        "low",
        "--memory-profile",
        "hf",
        "--memory-version",
        VERSION,
        "--memory-repo",
        "RLinf/RPent-memory@" + REVISION,
        "--no-auto-merge-memory",
        "--planner-timeout-s",
        "1200",
        "--max-turns",
        "100",
        "--max-tokens",
        "8192",
        "--max-episode-steps",
        "10000",
        "--cuda-device",
        str(deployment["environment_gpu"]),
        "--vla-endpoint",
        f"http://127.0.0.1:{deployment['vla_port']}",
        "--sam3-endpoint",
        f"http://127.0.0.1:{deployment['sam3_port']}",
        "--output-dir",
        str(batch / "episodes" / cell["episode_id"]),
    ]
    plan_path = batch / "plan.json"
    if cell.get("initial_scene"):
        argv.extend(["--initial-scene", cell["initial_scene"]])
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan.get("memcodex_enabled"):
            if plan.get("condition") not in {
                "memcodex_m1_fixed",
                "memcodex_m2_indexed",
            }:
                raise ValueError("Unsupported MemCodex evaluation condition")
            index = argv.index("--memory-repo")
            del argv[index : index + 2]
            argv.extend(
                [
                    "--memcodex-root",
                    plan["memcodex_root"],
                    "--memory-token-budget",
                    str(plan["memory_token_budget"]),
                ]
            )
            if plan["condition"] == "memcodex_m2_indexed":
                argv.extend(
                    [
                        "--memcodex-mode",
                        "m2",
                        "--memcodex-index",
                        plan["memcodex_index"],
                    ]
                )
                if plan.get("action_evidence", False):
                    argv.append("--memcodex-action-evidence")
    return argv


def prepare(batch: Path) -> None:
    """Freeze the authorized scope, official memory, and current production sources."""
    if (batch / "plan.json").exists():
        raise FileExistsError("The existing plan is immutable; use run to resume it")
    catalog = json.loads((batch / "task-catalog.json").read_text())
    assert len(catalog) == 80 and len({c["episode_id"] for c in catalog}) == 80
    assert {c["seed"] for c in catalog} == {1}
    names = (
        subprocess.check_output(["git", "ls-files", "-z", "rpent", "robots"], cwd=ROOT)
        .decode()
        .split("\0")
    )
    subprocess.run(
        ["git", "diff", "--exit-code", "HEAD", "--", "rpent", "robots"],
        cwd=ROOT,
        check=True,
        stdout=subprocess.DEVNULL,
    )
    names = [n for n in names if n and (ROOT / n).is_file()]
    names += [
        "profiling/codex_telemetry.py",
        "profiling/libero_capture.py",
        "profiling/libero_metrics.py",
        "profiling/vla_server.py",
        "repro/liberopro_baseline.py",
    ]
    receipt = json.loads(
        (ROOT / f"memory/libero/{VERSION}.download-receipt.json").read_text()
    )
    assert receipt["revision"] == REVISION and receipt["version"] == VERSION
    weights = [
        w
        for w in json.loads(
            (ROOT / "logs/install-smoke/weights-manifest.json").read_text()
        )
        if w.get("component") in {"pi05", "sam3"}
    ]
    deployment = {
        "pi05_gpu": 0,
        "sam3_gpu": 1,
        "environment_gpu": 1,
        "vla_port": 8420,
        "sam3_port": 8414,
    }
    suite_order = [
        f"libero_{family}_{perturbation}"
        for perturbation in ("task", "swap")
        for family in ("spatial", "object", "goal", "10")
    ]
    plan = {
        "created_at": now(),
        "benchmark": "LIBERO-PRO",
        "model": "gpt-6-astra",
        "planner": "codex",
        "reasoning_effort": "low",
        "condition": "original_harnessvla_full",
        "memcodex_enabled": False,
        "spatial_repair_enabled": False,
        "memory_profile": "hf",
        "memory_version": VERSION,
        "memory_revision": REVISION,
        "memory_update": False,
        "exploration": False,
        "memory_receipt": receipt,
        "runtime_memory_root": str(
            ROOT
            / "memory/libero/.versions"
            / hashlib.sha256(b"RLinf/RPent-memory").hexdigest()[:20]
            / "snapshots"
            / REVISION
            / VERSION
        ),
        "episodes": sorted(
            catalog, key=lambda c: (c["task"], suite_order.index(c["suite"]))
        ),
        "planned_episodes": 80,
        "seeds": [1],
        "weights": weights,
        "source_revision": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
        ).strip(),
        "source_sha256": {n: sha(ROOT / n) for n in names},
        "bddl_sha256": {c["bddl_path"]: sha(Path(c["bddl_path"])) for c in catalog},
        "deployment": deployment,
        "max_turns": 100,
        "max_tokens": 8192,
        "max_episode_steps": 10000,
        "planner_timeout_s": 1200,
        "infra_policy": "stop and preserve interrupted episode; never count as policy failure or automatically retry",
        "metrics_definitions": str(batch / "METRICS.md"),
    }
    plan["example_command"] = command(batch, plan["episodes"][0], deployment)
    write(batch / "plan.json", plan)
    report(batch)


def report(batch: Path) -> dict:
    """Refresh per-episode, per-suite and overall reports without hiding omissions."""
    plan = json.loads((batch / "plan.json").read_text())
    results = []
    for cell in plan["episodes"]:
        p = batch / "episodes" / cell["episode_id"] / "metrics.json"
        if p.exists():
            results.append(json.loads(p.read_text()))
    summary = summarize(results, len(plan["episodes"]))
    summary.update(
        updated_at=now(),
        complete=summary["finished_valid"] == len(plan["episodes"]),
        benchmark="LIBERO-PRO",
        condition=plan.get("condition", "original_harnessvla_full"),
        memcodex_enabled=plan.get("memcodex_enabled", False),
        action_evidence=plan.get("action_evidence", False),
    )
    suites = {
        suite: summarize(
            [r for r in results if r["suite"] == suite],
            sum(c["suite"] == suite for c in plan["episodes"]),
        )
        for suite in sorted({c["suite"] for c in plan["episodes"]})
    }
    write(batch / "summary.json", summary)
    write(batch / "by-suite.json", suites)
    columns = [
        "episode_id",
        "suite",
        "task",
        "seed",
        "status",
        "valid",
        "native_success",
        "episode_length_env_steps",
        "total_input_tokens",
        "total_output_tokens",
        "cached_input_tokens",
        "memory_tokens",
        "reasoning_tokens",
        "llm_calls",
        "agent_latency_s",
        "end_to_end_latency_s",
        "memory_retrieval_frequency",
        "primitive_calls",
        "recovery_calls",
    ]
    for name in ("llm_request", "memory_retrieval", "vla_inference"):
        columns += [f"{name}_{stat}_s" for stat in ("mean", "p50", "p95")]
    with (batch / "episodes.csv").open("w") as stream:
        csvwriter = csv.DictWriter(stream, fieldnames=columns)
        csvwriter.writeheader()
        for r in results:
            row = {k: r.get(k) for k in columns}
            for name in ("llm_request", "memory_retrieval", "vla_inference"):
                for stat in ("mean", "p50", "p95"):
                    row[f"{name}_{stat}_s"] = r["latency_s"][name][stat]
            csvwriter.writerow(row)
    return summary


def run(batch: Path) -> None:
    """Own services and sequential workers, stopping explicitly on infra failures."""
    from robots.libero.memory import _matches_files
    from rpent.utils.daemon import ProcessDaemon
    from rpent.utils.rpc.client_utils import make_rpc_client, wait_for_ready

    plan = json.loads((batch / "plan.json").read_text())
    validate_scenes(plan)
    for name, digest in plan["source_sha256"].items():
        if sha(ROOT / name) != digest:
            raise RuntimeError(f"Frozen source changed: {name}")
    for name, digest in plan["bddl_sha256"].items():
        if sha(Path(name)) != digest:
            raise RuntimeError(f"Frozen task changed: {name}")
    memory = Path(plan["runtime_memory_root"])
    if not _matches_files(memory, plan["memory_receipt"]["files"]):
        raise RuntimeError("Official memory changed")
    if plan.get("memcodex_enabled"):
        from repro.memcodex_corpus import validate

        corpus = Path(plan["memcodex_root"])
        if sha(corpus / "MANIFEST.json") != plan["corpus_manifest_sha256"]:
            raise RuntimeError("Frozen M1 corpus changed")
        validate(
            corpus,
            ROOT / f"memory/libero/{VERSION}",
            ROOT / f"memory/libero/{VERSION}.download-receipt.json",
        )
        if plan["condition"] == "memcodex_m2_indexed":
            from repro.index_memcodex_memory import validate_index

            index = Path(plan["memcodex_index"])
            if sha(index) != plan["index_sha256"]:
                raise RuntimeError("Frozen M2 index changed")
            validate_index(corpus, index)
    services = batch / "services"
    services.mkdir(exist_ok=True)
    output = batch / "episodes"
    output.mkdir(exist_ok=True)
    deployment = plan["deployment"]
    # RLinf is installed in the selected interpreter. Treating its whole
    # site-packages directory as a checkout changes LIBERO asset paths when
    # ProcessDaemon resolves a symlinked virtualenv, breaking frozen resets.
    env = {
        **os.environ,
        "RPENT_REPO_ROOT": str(ROOT),
        "PYTHONPATH": str(ROOT),
        "LIBERO_TYPE": "pro",
        "LIBERO_CONFIG_PATH": str(ROOT / "repro/assets/libero-pro/config"),
        "LIBERO_PRO_ASSET_PATH": str(ROOT / "repro/assets/libero-pro/tree"),
        "PI05_CHECKPOINT_PATH": str(ROOT / "checkpoints/pi05"),
        "SAM3_CHECKPOINT_PATH": str(ROOT / "checkpoints/sam3/sam3.pt"),
        "MUJOCO_GL": "egl",
        "PYOPENGL_PLATFORM": "egl",
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "TIKTOKEN_CACHE_DIR": str(ROOT / "repro/tokenizer-cache"),
        "RPENT_BASELINE_SERVICES": str(services),
        "LD_LIBRARY_PATH": str(ROOT / "repro/system-lib/lib")
        + os.pathsep
        + os.environ.get("LD_LIBRARY_PATH", ""),
        "NO_PROXY": "127.0.0.1,localhost",
        "no_proxy": "127.0.0.1,localhost",
        "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
    }
    for key in (
        "CODEX_TIMEOUT_S",
        "CELL_TIMEOUT_S",
        "LIBERO_SUFFIX",
        "LIBERO_PERTURBATION",
        "CODEX_MODEL_CONTEXT_WINDOW",
        "CODEX_AUTO_COMPACT_TOKEN_LIMIT",
    ):
        env.pop(key, None)
    stopping = []
    daemons = []
    child = None

    def stop(signum, _frame):
        stopping.append(signum)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    lock = (batch / "run.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        summary = report(batch)
        if summary["complete"]:
            return
        if summary["infrastructure_errors"]:
            raise RuntimeError(
                "Preserved infrastructure failure requires diagnosis before resume"
            )
        write(
            batch / "status.json",
            {"phase": "checking_weights", "pid": os.getpid(), "at": now()},
        )
        for weight in plan["weights"]:
            if sha(ROOT / weight["weight_file"]) != weight["sha256"]:
                raise RuntimeError("Checkpoint mismatch")
        write(
            batch / "status.json",
            {"phase": "starting_services", "pid": os.getpid(), "at": now()},
        )
        import socket

        for port in (deployment["vla_port"], deployment["sam3_port"]):
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        for name, module, gpu, port, extra in (
            (
                "pi05",
                "profiling.vla_server",
                deployment["pi05_gpu"],
                deployment["vla_port"],
                ["--embodiment", "libero"],
            ),
            (
                "sam3",
                "rpent.robots.components.sam3_server",
                deployment["sam3_gpu"],
                deployment["sam3_port"],
                [],
            ),
        ):
            daemon = ProcessDaemon(
                name,
                [
                    sys.executable,
                    "-u",
                    "-m",
                    module,
                    *extra,
                    "--cuda-device",
                    str(gpu),
                    "--transport",
                    "http",
                    "--host",
                    "127.0.0.1",
                    "--port",
                    str(port),
                    "--parent-watch",
                ],
                cwd=str(ROOT),
                env_overrides={**env, "CUDA_VISIBLE_DEVICES": str(gpu)},
                log_path=str(services / f"{name}.log"),
            )
            daemon.start()
            daemons.append((daemon, port))
        write(services / "owned-pids.json", {d.name: d._proc.pid for d, _ in daemons})
        for daemon, port in daemons:
            wait_for_ready(
                make_rpc_client(f"http://127.0.0.1:{port}"),
                daemon=daemon,
                timeout_s=900,
            )
        write(services / "ready.json", {"at": now(), **deployment})
        for cell in plan["episodes"]:
            run_dir = output / cell["episode_id"]
            if (run_dir / "metrics.json").exists():
                continue
            if run_dir.exists():
                raise RuntimeError(f"Unfinished episode preserved: {run_dir}")
            if stopping or (batch / "STOP").exists():
                raise InterruptedError("Stop requested")
            for name, digest in plan["source_sha256"].items():
                if sha(ROOT / name) != digest:
                    raise RuntimeError(f"Frozen source changed: {name}")
            validate_scenes({"episodes": [cell]})
            write(services / "current-episode.json", cell)
            write(
                batch / "status.json",
                {
                    "phase": "running",
                    "at": now(),
                    "pid": os.getpid(),
                    "active": cell,
                    "completed": report(batch)["finished_valid"],
                },
            )
            argv = command(batch, cell, deployment)
            logdir = batch / "worker-logs"
            logdir.mkdir(exist_ok=True)
            started = time.perf_counter()
            with (logdir / (cell["episode_id"] + ".log")).open("w") as log:
                child = subprocess.Popen(
                    argv,
                    cwd=ROOT,
                    env={
                        **env,
                        "CUDA_VISIBLE_DEVICES": str(deployment["environment_gpu"]),
                        "MUJOCO_EGL_DEVICE_ID": str(deployment["environment_gpu"]),
                    },
                    stdin=subprocess.DEVNULL,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                )
                write(
                    batch / "active-process.json",
                    {"pid": child.pid, "episode_id": cell["episode_id"], "at": now()},
                )
                while child.poll() is None:
                    if stopping or (batch / "STOP").exists():
                        raise InterruptedError("Stop requested")
                    if any(d.poll() is not None for d, _ in daemons):
                        raise RuntimeError("Model service exited")
                    if time.perf_counter() - started > 2100:
                        raise TimeoutError(
                            "Episode worker watchdog; infrastructure timeout"
                        )
                    time.sleep(2)
            result = collect(
                run_dir, cell, services, child.returncode, time.perf_counter() - started
            )
            child = None
            if plan.get("memcodex_enabled"):
                coverage = result["telemetry_coverage"]
                m2 = plan["condition"] == "memcodex_m2_indexed"
                invalid = bool(coverage["native_command_executions"])
                invalid |= coverage["memory_injections"] != (0 if m2 else 1)
                if m2:
                    invalid |= coverage["indexed_memory_reads"] < 1
                if invalid:
                    result.update(
                        valid=False,
                        status="infrastructure_error",
                        error="MemCodex memory delivery or native-tool isolation check failed",
                    )
            write(run_dir / "metrics.json", result)
            if not _matches_files(memory, plan["memory_receipt"]["files"]):
                raise RuntimeError("Official HF memory changed during evaluation")
            summary = report(batch)
            print(
                json.dumps(
                    {
                        "at": now(),
                        "episode_id": cell["episode_id"],
                        "status": result["status"],
                        "completed": summary["finished_valid"],
                        "native_success": result["native_success"],
                    }
                ),
                flush=True,
            )
            if not result["valid"]:
                raise RuntimeError(
                    f"Infrastructure failure: {cell['episode_id']}: {result['error']}"
                )
            if (
                not result["telemetry_coverage"]["sampling_spans"]
                or not result["telemetry_coverage"]["llm_request_spans"]
            ):
                raise RuntimeError(
                    "Request telemetry missing; preserve episode and diagnose before continuing"
                )
            if (
                result["telemetry_coverage"]["vla_client_calls"]
                != result["telemetry_coverage"]["vla_server_calls"]
            ):
                raise RuntimeError("VLA telemetry coverage mismatch")
        write(
            batch / "status.json",
            {"phase": "complete", "at": now(), "completed": len(plan["episodes"])},
        )
    except BaseException as exc:
        write(
            batch / "status.json",
            {
                "phase": "stopped",
                "at": now(),
                "error": str(exc),
                "completed": report(batch)["finished_valid"],
            },
        )
        raise
    finally:
        if child is not None and child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        cleanup = []
        for daemon, _ in reversed(daemons):
            daemon.stop()
            cleanup.append(
                {
                    "name": daemon.name,
                    "pid": daemon._proc.pid,
                    "exit_code": daemon.poll(),
                }
            )
        write(services / "cleanup.json", cleanup)
        write(
            batch / "memory-integrity-after.json",
            {
                "at": now(),
                "unchanged": _matches_files(memory, plan["memory_receipt"]["files"]),
            },
        )
        lock.close()


def main() -> None:
    """Prepare, run, or summarize this baseline."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=["prepare", "run", "report"])
    parser.add_argument("batch", type=Path)
    args = parser.parse_args()
    batch = args.batch.resolve()
    {"prepare": prepare, "run": run, "report": report}[args.action](batch)


if __name__ == "__main__":
    main()
