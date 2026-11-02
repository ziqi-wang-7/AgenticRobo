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

"""Report observed success constraints and memory cost for paired variants."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path


def analyze(path: str | Path, tau: float) -> dict:
    """Select minimum observed memory cost only on identical complete cohorts.

    This is descriptive sample analysis, not a statistical success guarantee.
    Missing outcomes never count as failures or disappear from the denominator.
    """
    if (
        isinstance(tau, bool)
        or not isinstance(tau, (int, float))
        or not math.isfinite(tau)
        or not 0 <= tau <= 1
    ):
        raise ValueError("tau must be between 0 and 1")
    groups = defaultdict(dict)
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        event = json.loads(line)
        variant, episode_id = event["variant"], event["episode_id"]
        episode = groups[variant].setdefault(
            episode_id, {"retrievals": [], "outcome": None}
        )
        if event["event"] == "retrieval":
            if event["token_count"] > event["token_budget"]:
                raise ValueError("Recorded memory context exceeds its budget")
            episode["retrievals"].append(event)
        elif event["event"] == "outcome":
            if (
                episode["outcome"] is not None
                or type(event["native_success"]) is not bool
            ):
                raise ValueError("Each episode must have one boolean native outcome")
            episode["outcome"] = event["native_success"]
        else:
            raise ValueError(f"Unknown profiling event: {event['event']}")
    rows = []
    for variant, episodes in sorted(groups.items()):
        calls = [r for e in episodes.values() for r in e["retrievals"]]
        configs = {(r["token_budget"], r["counter_name"]) for r in calls}
        if len(configs) > 1:
            raise ValueError(
                "Use separate variants for different budgets or token counters"
            )
        completed = [e for e in episodes.values() if e["outcome"] is not None]
        rate = (
            sum(e["outcome"] for e in completed) / len(completed) if completed else None
        )
        complete = len(completed) == len(episodes) and all(
            e["retrievals"] for e in episodes.values()
        )
        total_tokens = sum(r["token_count"] for r in calls)
        rows.append(
            {
                "variant": variant,
                "episodes": len(episodes),
                "completed": len(completed),
                "retrieval_calls": len(calls),
                "success_rate_completed": rate,
                "complete": complete,
                "observed_constraint_met": complete
                and rate is not None
                and rate >= tau,
                "memory_tokens_total": total_tokens,
                "mean_memory_tokens_per_episode": total_tokens / len(episodes),
                "mean_retrieval_ms": sum(r["elapsed_ms"] for r in calls) / len(calls)
                if calls
                else None,
                "cache_hits": sum(r["cache_hit"] for r in calls),
                "counter_name": next(iter(configs))[1] if configs else None,
            }
        )
    cohorts = {tuple(sorted(episodes)) for episodes in groups.values()}
    comparable = (
        bool(rows) and len(cohorts) == 1 and all(row["complete"] for row in rows)
    )
    comparable &= len({row["counter_name"] for row in rows}) == 1
    eligible = (
        [row for row in rows if row["observed_constraint_met"]] if comparable else []
    )
    best = (
        min(eligible, key=lambda r: (r["mean_memory_tokens_per_episode"], r["variant"]))
        if eligible
        else None
    )
    return {
        "tau": tau,
        "variants": rows,
        "paired_complete_coverage": comparable,
        "lowest_memory_variant_meeting_observed_tau": best["variant"] if best else None,
        "interpretation": "Observed episode outcomes only. Retrieval sufficiency is not native success. Same-tokenizer, paired complete cohorts are required for selection.",
    }


def main() -> None:
    """Analyze a local profiling file without invoking any model."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path)
    parser.add_argument("--tau", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(analyze(args.path, args.tau), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
