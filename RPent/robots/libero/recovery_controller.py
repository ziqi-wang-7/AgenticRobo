# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Task-skill-specific recovery routing with bounded, auditable transitions.

This module selects a recovery *workflow*, not robot coordinates or permission
to execute a primitive. Every transition still requires current perception and
the action-evidence contract.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

RecoveryStrategy = Literal[
    "grasp", "spatial", "contact", "observe", "stop"
]

MAX_ATTEMPTS_PER_STRATEGY = 3
MAX_TOTAL_RECOVERY_ATTEMPTS = 6


@dataclass
class RecoveryBudget:
    """Per-episode recovery budget; counters are independent of LLM call counts."""

    total_attempts: int = 0
    by_strategy: dict[str, int] = field(default_factory=dict)

    def consume(self, strategy: RecoveryStrategy) -> bool:
        if strategy in {"observe", "stop"}:
            return True
        used = self.by_strategy.get(strategy, 0)
        if used >= MAX_ATTEMPTS_PER_STRATEGY:
            return False
        if self.total_attempts >= MAX_TOTAL_RECOVERY_ATTEMPTS:
            return False
        self.by_strategy[strategy] = used + 1
        self.total_attempts += 1
        return True

    def snapshot(self) -> dict:
        return {
            "total_attempts": self.total_attempts,
            "by_strategy": dict(self.by_strategy),
            "max_attempts_per_strategy": MAX_ATTEMPTS_PER_STRATEGY,
            "max_total_attempts": MAX_TOTAL_RECOVERY_ATTEMPTS,
        }


@dataclass(frozen=True)
class RecoveryPlan:
    """A workflow recommendation; it never authorizes physical execution."""

    strategy: RecoveryStrategy
    state_machine: tuple[str, ...]
    reason: str
    execution_authorized: bool = False


def select_recovery_plan(failure_kinds: tuple[str, ...] | list[str]) -> RecoveryPlan:
    """Route observed failures to the narrowest applicable skill workflow."""
    kinds = set(failure_kinds)
    if not kinds:
        return RecoveryPlan(
            "observe",
            ("REOBSERVE", "CLASSIFY_FAILURE"),
            "no_attributed_failure",
        )
    if kinds & {"insertion_blocked", "release_stuck", "contact_stalled"}:
        return RecoveryPlan(
            "contact",
            (
                "PRE_ALIGN",
                "VERIFY_ORIENTATION",
                "APPROACH_OPENING",
                "VERIFY_CONTACT",
                "SHORT_PUSH",
                "VERIFY_INSERTION_DEPTH",
                "RELOCALIZE_OR_STOP",
            ),
            "contact_or_insertion_failure",
        )
    if kinds & {"wrong_spatial_relation", "placement_not_satisfied", "object_collision"}:
        return RecoveryPlan(
            "spatial",
            (
                "INSPECT_CURRENT_RELATION",
                "CHECK_DIRECT_CORRECTION_FEASIBILITY",
                "CHECK_SWEPT_PATH_AND_CLEARANCE",
                "SELECT_DIRECT_CORRECTION_OR_REGRASP",
                "RELOCALIZE",
                "VERIFY_PLACEMENT",
            ),
            "spatial_relation_or_collision_failure",
        )
    if kinds & {
        "grasp_not_closed",
        "object_slipped",
        "grasp_space_insufficient",
        "object_tipped",
    }:
        return RecoveryPlan(
            "grasp",
            (
                "REOBSERVE_OBJECT_POSE",
                "CHECK_GRASP_GEOMETRY",
                "CHANGE_GRASP_POSE",
                "GRASP_ATTEMPT",
                "VERIFY_GRIPPER_CLOSURE",
                "SHORT_LIFT",
                "VERIFY_OBJECT_FOLLOWS",
            ),
            "grasp_or_object_stability_failure",
        )
    return RecoveryPlan(
        "observe",
        ("REOBSERVE", "REFINE_FAILURE_CLASSIFICATION"),
        "unmapped_failure_requires_observation",
    )


def apply_recovery_budget(
    plan: RecoveryPlan, budget: RecoveryBudget
) -> RecoveryPlan:
    """Return STOP when a strategy has exhausted its bounded attempts."""
    if budget.consume(plan.strategy):
        return plan
    return RecoveryPlan(
        "stop",
        ("STOP_REPEATED_ATTEMPTS", "REPORT_UNRESOLVED_FAILURE"),
        f"recovery_budget_exhausted:{plan.strategy}",
    )
