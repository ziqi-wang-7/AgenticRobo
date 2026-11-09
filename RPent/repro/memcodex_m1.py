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

"""Prepare a fixed-memory evaluation using an existing M0 task/weight manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from repro.liberopro_baseline import ROOT, VERSION, command, now, report, sha, write
from repro.memcodex_corpus import validate


def prepare(args: argparse.Namespace) -> None:
    """Freeze M1 source hashes and task selection without altering the M0 batch."""
    baseline, output, corpus = (
        args.baseline.resolve(),
        args.output.resolve(),
        args.corpus.resolve(),
    )
    if output.exists():
        raise FileExistsError("Choose a fresh M1 output directory")
    if args.token_budget <= 0 or args.seed < 1:
        raise ValueError("Use a positive memory budget and evaluation seed >= 1")
    if (args.suite is None) != (args.task is None):
        raise ValueError("Select both --suite and --task, or neither for all 80 tasks")
    source = ROOT / f"memory/libero/{VERSION}"
    validation = validate(corpus, source, source.with_suffix(".download-receipt.json"))
    m2 = getattr(args, "mode", "fixed") == "m2"
    index_validation = None
    if m2:
        from repro.index_memcodex_memory import validate_index

        index_validation = validate_index(corpus, args.index.resolve())
    plan = json.loads((baseline / "plan.json").read_text())
    if (
        plan.get("condition") != "original_harnessvla_full"
        or plan["planned_episodes"] != 80
    ):
        raise ValueError("Expected the original 80-task M0 manifest")
    episodes = [
        dict(cell)
        for cell in plan["episodes"]
        if args.suite is None
        or (cell["suite"], cell["task"]) == (args.suite, args.task)
    ]
    if not episodes:
        raise ValueError("Requested task is absent from the M0 catalog")
    for cell in episodes:
        cell.update(
            seed=args.seed, episode_id=f"{cell['suite']}-t{cell['task']}-s{args.seed}"
        )
    names = set(plan["source_sha256"])
    names.update(
        str(p.relative_to(ROOT)) for p in (ROOT / "rpent/memory/memcodex").glob("*.py")
    )
    if m2:
        names.update(
            {
                "robots/libero/memcodex_online.py",
                "robots/libero/action_evidence.py",
                "repro/index_memcodex_memory.py",
                "repro/memcodex_m2.py",
            }
        )
    names.update(
        {
            "robots/libero/memcodex.py",
            "repro/memcodex_m1.py",
            "repro/memcodex_corpus.py",
        }
    )
    plan.update(
        created_at=now(),
        condition="memcodex_m1_fixed",
        memcodex_enabled=True,
        baseline_batch=str(baseline),
        baseline_plan_sha256=sha(baseline / "plan.json"),
        memcodex_root=str(corpus),
        memory_token_budget=args.token_budget,
        corpus_manifest_sha256=sha(corpus / "MANIFEST.json"),
        episodes=episodes,
        planned_episodes=len(episodes),
        seeds=[args.seed],
        source_sha256={name: sha(ROOT / name) for name in sorted(names)},
        metrics_definitions=str(output / "METRICS.md"),
        adaptive_routing=False,
        cache_enabled=False,
        early_stop=False,
        native_command_execution=False,
        comparison_limitations=[
            "M1 changes memory organization, fixed delivery and native file-tool access together; it is not a pure hierarchy-only ablation.",
            "Memory tokens use o200k_base estimates, without a repeated-history multiplier; M0 ambiguous native reads remain missing.",
        ],
    )
    if m2:
        plan.update(
            condition="memcodex_m2_indexed",
            memcodex_index=str(args.index.resolve()),
            index_sha256=index_validation["index_sha256"],
            early_stop=True,
            action_evidence=bool(getattr(args, "action_evidence", False)),
            comparison_limitations=[
                "M2 uses indexed progressive tool delivery and step-bound perception evidence; M1 injects a fixed context once.",
                "Sufficiency is a deterministic coverage rule, not a task-success guarantee. No result cache or extra judging LLM.",
                "Historical raw sources and marked analogies are available only on escalation. Memory tokens count each serialized delivery without a history multiplier.",
            ],
        )
        if plan["action_evidence"]:
            plan["comparison_limitations"].append(
                "Action evidence changes the planner's perception workflow and candidate-scoped retrieval together; compare routing policies with this workflow fixed."
            )
    output.mkdir(parents=True)
    write(output / "plan.json", plan)
    plan["example_command"] = command(output, episodes[0], plan["deployment"])
    write(output / "plan.json", plan)
    write(output / "corpus-validation.json", validation)
    if m2:
        write(output / "index-validation.json", index_validation)
    (output / "METRICS.md").write_text(
        (baseline / "METRICS.md").read_text()
        + (
            "\nM2: memory tokens count serialized retrieve_memory responses, including repeated deliveries. Retrieval latency measures tool execution; index validation at startup is included only in end-to-end latency. No initial memory injection.\n"
            if m2
            else "\nM1: include the fixed memory context on actual planner submission. "
            "Memory retrieval latency includes corpus integrity checks and tokenization. "
            "Raw archives are hashed but never supplied to the planner.\n"
        )
    )
    report(output)
    print(
        json.dumps(
            {
                "batch": str(output),
                "episodes": len(episodes),
                "next": f".venv/bin/python -m repro.liberopro_baseline run {output}",
            }
        )
    )


def main() -> None:
    """Prepare a paired batch, or one development episode with an explicit seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--corpus", type=Path, default=Path("memory/memcodex_v1"))
    parser.add_argument("--token-budget", type=int, default=32768)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--suite")
    parser.add_argument("--task", type=int)
    prepare(parser.parse_args())


if __name__ == "__main__":
    main()
