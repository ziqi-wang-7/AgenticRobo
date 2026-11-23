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

"""Prepare and summarize paired/streaming Flash-Evolve batches on LIBERO-PRO."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from repro.liberopro_baseline import ROOT, command, now, report, sha, write


def prepare(args: argparse.Namespace) -> None:
    """Create a Flash-Evolve evaluation batch paired with an M0 baseline manifest."""
    baseline = args.baseline.resolve()
    output = args.output.resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    if (args.suite is None) != (args.task is None):
        raise ValueError("Select both --suite and --task, or neither for all 80 tasks")
    seeds = list(args.seeds or [1])
    if any(seed < 1 for seed in seeds):
        raise ValueError("Evaluation seeds must be >= 1 (seed 0 is reference)")

    plan = json.loads((baseline / "plan.json").read_text(encoding="utf-8"))
    if (
        plan.get("condition") != "original_harnessvla_full"
        or plan.get("planned_episodes") != 80
    ):
        raise ValueError("Expected the original 80-task M0 baseline plan.json")

    base_cells = [
        dict(cell)
        for cell in plan["episodes"]
        if args.suite is None
        or (cell["suite"], cell["task"]) == (args.suite, args.task)
    ]
    if not base_cells:
        raise ValueError("Requested task is absent from the M0 catalog")

    # Seed-major ordering enables continual procedural evolution across seeds:
    # rescues learned on seed s become zero-token branches on seed s+1.
    episodes: list[dict[str, Any]] = []
    for seed in seeds:
        for cell in base_cells:
            item = dict(cell)
            item.update(
                seed=seed,
                episode_id=f"{cell['suite']}-t{cell['task']}-s{seed}",
            )
            if seed != 1:
                item.pop("initial_scene", None)
                item.pop("initial_scene_sha256", None)
            episodes.append(item)

    flash_memory_dir = (
        args.flash_memory_dir.resolve()
        if args.flash_memory_dir is not None
        else output / "flash_memory"
    )

    names = set(plan.get("source_sha256") or ())
    names.update(
        {
            "rpent/planner/base.py",
            "rpent/planner/flash_evolve.py",
            "rpent/cli/main.py",
            "rpent/robots/robot_spec.py",
            "robots/libero/robot_spec.py",
            "robots/libero/memory.py",
            "robots/libero/flash/__init__.py",
            "robots/libero/flash/replay.py",
            "robots/libero/flash/evolve.py",
            "repro/liberopro_baseline.py",
            "repro/flash_evolve.py",
        }
    )

    plan.update(
        created_at=now(),
        condition="flash_evolve",
        planner="flash_evolve",
        flash_slow_planner=args.slow_planner,
        flash_memory_dir=str(flash_memory_dir),
        flash_evolve_enabled=bool(args.evolve),
        molmo_endpoint=args.molmo_endpoint,
        memcodex_enabled=False,
        memory_update=bool(args.evolve),
        baseline_batch=str(baseline),
        baseline_plan_sha256=sha(baseline / "plan.json"),
        episodes=episodes,
        planned_episodes=len(episodes),
        seeds=seeds,
        source_sha256={
            name: sha(ROOT / name) for name in sorted(names) if (ROOT / name).is_file()
        },
        metrics_definitions=str(output / "METRICS.md"),
    )

    output.mkdir(parents=True)
    flash_memory_dir.mkdir(parents=True, exist_ok=True)
    write(output / "plan.json", plan)
    plan["example_command"] = command(output, episodes[0], plan["deployment"])
    write(output / "plan.json", plan)
    if (baseline / "METRICS.md").is_file():
        (output / "METRICS.md").write_text(
            (baseline / "METRICS.md").read_text(encoding="utf-8")
            + "\nFlash-Evolve: each episode writes flash_evolve_trace.json recording "
            "execution_mode (flash_fast / flash_branch / slow_handover_rescue / "
            "slow_cold_start_solved), branch hits, slow-planner token usage, and "
            "procedural memory patches.\n",
            encoding="utf-8",
        )
    report(output)
    print(
        json.dumps(
            {
                "batch": str(output),
                "flash_memory_dir": str(flash_memory_dir),
                "episodes": len(episodes),
                "seeds": seeds,
                "next": f".venv/bin/python -m repro.liberopro_baseline run {output}",
            },
            indent=2,
        )
    )


def summarize_batch(batch: Path) -> dict[str, Any]:
    """Aggregate per-seed success rates, execution modes, and evolution counts."""
    batch = batch.resolve()
    plan = json.loads((batch / "plan.json").read_text(encoding="utf-8"))
    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    modes: Counter[str] = Counter()
    evolved: Counter[str] = Counter()
    total_input_tokens = 0
    total_output_tokens = 0
    completed = 0
    solved = 0

    for cell in plan["episodes"]:
        ep_dir = batch / "episodes" / cell["episode_id"]
        trace_path = ep_dir / "flash_evolve_trace.json"
        if not trace_path.is_file():
            continue
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        completed += 1
        is_solved = bool(trace.get("done"))
        solved += int(is_solved)
        mode = str(trace.get("execution_mode", "unknown"))
        modes[mode] += 1
        evolved[str(trace.get("evolved_action", "none"))] += 1
        slow_stats = (trace.get("slow_result") or {}).get("stats") or {}
        in_tok = int(slow_stats.get("total_input_tokens", 0))
        out_tok = int(slow_stats.get("total_output_tokens", 0))
        total_input_tokens += in_tok
        total_output_tokens += out_tok
        by_seed[int(cell["seed"])].append(
            {
                "episode_id": cell["episode_id"],
                "done": is_solved,
                "mode": mode,
                "evolved_action": trace.get("evolved_action", "none"),
                "input_tokens": in_tok,
                "output_tokens": out_tok,
            }
        )

    per_seed: dict[str, Any] = {}
    for seed in sorted(by_seed):
        rows = by_seed[seed]
        n = len(rows)
        s = sum(int(r["done"]) for r in rows)
        zero_tok = sum(
            int(r["done"] and r["mode"] in {"flash_fast", "flash_branch"}) for r in rows
        )
        branch_wins = sum(int(r["done"] and r["mode"] == "flash_branch") for r in rows)
        per_seed[str(seed)] = {
            "completed": n,
            "solved": s,
            "success_rate": round(s / n, 4) if n else 0.0,
            "zero_token_solved": zero_tok,
            "branch_hit_solved": branch_wins,
            "avg_input_tokens": round(sum(r["input_tokens"] for r in rows) / n, 1)
            if n
            else 0.0,
        }

    return {
        "batch": str(batch),
        "planned": int(plan.get("planned_episodes", len(plan["episodes"]))),
        "completed": completed,
        "solved": solved,
        "success_rate": round(solved / completed, 4) if completed else 0.0,
        "execution_modes": dict(modes),
        "evolved_actions": dict(evolved),
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "per_seed": per_seed,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    prep = sub.add_parser("prepare", help="Prepare a Flash-Evolve batch.")
    prep.add_argument("--baseline", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--flash-memory-dir", type=Path, default=None)
    prep.add_argument(
        "--slow-planner", choices=("codex", "api", "claude_code"), default="codex"
    )
    prep.add_argument("--seeds", type=int, nargs="+", default=[1])
    prep.add_argument("--suite", default=None)
    prep.add_argument("--task", type=int, default=None)
    prep.add_argument("--molmo-endpoint", default=None)
    prep.add_argument("--evolve", action=argparse.BooleanOptionalAction, default=True)

    summ = sub.add_parser("summarize", help="Summarize a Flash-Evolve batch.")
    summ.add_argument("batch", type=Path)

    args = parser.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    else:
        print(json.dumps(summarize_batch(args.batch), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
