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

"""Fast-Slow hybrid planner combining Flash replay with slow-planner evolution.

System 1 executes anchor-relative procedural memory at zero LLM token cost,
including learned recovery branches.  On an unresolved anomaly (or cold start
with no prior Flash plan), it hands the live toolkit in-place to the wrapped
slow planner (e.g. Codex) without resetting the environment, and compiles any
successful rescue back into procedural memory for future episodes.
"""

from __future__ import annotations

import queue
import time
from pathlib import Path
from typing import Any

from rpent.dashboard.interaction import DashboardInteractionPort
from rpent.planner.base import Planner, PlannerResult
from rpent.robots.base import get_robot_spec
from rpent.tools.toolkit import Toolkit
from rpent.utils.logging import get_logger

logger = get_logger("flash_evolve")


class FlashEvolvePlanner(Planner):
    """Run System-1 Flash replay, escalating to a slow planner on anomaly."""

    def __init__(
        self,
        *,
        recipe_tag: str,
        robot_name: str,
        slow_planner: Planner,
        slow_planner_name: str,
        flash_memory_dir: str | Path,
        output_dir: str | Path,
        evolve_enabled: bool = True,
    ) -> None:
        self._recipe_tag = recipe_tag
        self._robot_name = robot_name
        self._slow_planner = slow_planner
        self._slow_planner_name = slow_planner_name
        self._flash_memory_dir = Path(flash_memory_dir)
        self._output_dir = Path(output_dir)
        self._evolve_enabled = bool(evolve_enabled)

    def solve(
        self,
        *,
        system_prompt: str,
        user_message: str,
        toolkit: Toolkit,
        max_turns: int,
        input_queue: queue.Queue[str | None] | None = None,
        dashboard_interaction: DashboardInteractionPort | None = None,
    ) -> PlannerResult:
        """Execute Flash fast-path, optional slow handover, and memory patching."""
        run_flash_evolve = get_robot_spec(self._robot_name).run_flash_evolve
        if run_flash_evolve is None:
            return PlannerResult(
                error=(
                    f"the {self._robot_name} robot does not implement "
                    "Flash-Evolve procedural memory"
                )
            )

        def slow_solve(handover_user_message: str) -> PlannerResult:
            return self._slow_planner.solve(
                system_prompt=system_prompt,
                user_message=handover_user_message,
                toolkit=toolkit,
                max_turns=max_turns,
                input_queue=input_queue,
                dashboard_interaction=dashboard_interaction,
            )

        started = time.time()
        try:
            outcome = run_flash_evolve(
                toolkit,
                self._recipe_tag,
                slow_solve=slow_solve,
                user_message=user_message,
                flash_memory_dir=self._flash_memory_dir,
                evolve_enabled=self._evolve_enabled,
                slow_planner_name=self._slow_planner_name,
                output_dir=self._output_dir,
                note=logger.info,
            )
        except Exception as exc:
            return PlannerResult(error=f"{type(exc).__name__}: {exc}")
        elapsed = time.time() - started

        solved = bool(outcome.get("done"))
        trace: dict[str, Any] = dict(outcome.get("trace") or {})
        slow_res: PlannerResult | None = outcome.get("slow_planner_result")
        slow_stats = dict(slow_res.stats) if slow_res is not None else {}

        stats: dict[str, Any] = {
            "backend": "flash_evolve",
            "slow_planner": self._slow_planner_name,
            "execution_mode": trace.get("execution_mode", "unknown"),
            "evolved_action": trace.get("evolved_action", "none"),
            "evolve_enabled": self._evolve_enabled,
            "flash_memory_dir": str(self._flash_memory_dir),
            "fast_actions": int(trace.get("fast_actions", 0)),
            "branch_hits": len(trace.get("branch_attempts") or []),
            "total_input_tokens": int(slow_stats.get("total_input_tokens", 0)),
            "total_cached_input_tokens": int(
                slow_stats.get("total_cached_input_tokens", 0)
            ),
            "total_output_tokens": int(slow_stats.get("total_output_tokens", 0)),
            "total_reasoning_output_tokens": int(
                slow_stats.get("total_reasoning_output_tokens", 0)
            ),
            "turns_used": int(slow_stats.get("turns_used", 0)),
            "tool_calls": int(trace.get("fast_actions", 0))
            + int(slow_stats.get("tool_calls", 0)),
            "elapsed_s": round(elapsed, 1),
        }
        if trace.get("patched_branch_id"):
            stats["patched_branch_id"] = trace["patched_branch_id"]
        if trace.get("handover_step") is not None:
            stats["handover_step"] = trace["handover_step"]

        finish_result = (
            slow_res.finish_result
            if slow_res is not None and slow_res.finish_result is not None
            else {
                "status": "success" if solved else "failure",
                "summary": (
                    f"flash_evolve ({trace.get('execution_mode', 'unknown')}): "
                    f"program={outcome.get('program', self._recipe_tag)}, "
                    f"fast_actions={stats['fast_actions']}, "
                    f"evolved={stats['evolved_action']}"
                ),
            }
        )
        return PlannerResult(
            finish_result=finish_result,
            messages=list(slow_res.messages) if slow_res is not None else [],
            stats=stats,
            error=slow_res.error if slow_res is not None else None,
        )


__all__ = ["FlashEvolvePlanner"]
