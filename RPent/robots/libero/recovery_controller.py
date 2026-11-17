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

from robots.libero.skill_abstraction import Skill, compatible_recovery_skills

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
    active_skill: Skill = "unknown"
    compatible_skills: tuple[str, ...] = ()


def select_recovery_plan(
    failure_kinds: tuple[str, ...] | list[str],
    active_skill: Skill = "unknown",
) -> RecoveryPlan:
    """Route by observed failure and active skill, never by task identifier."""
    kinds = set(failure_kinds)
    compatible = compatible_recovery_skills(tuple(kinds), active_skill)
    if not kinds:
        return RecoveryPlan(
            "observe",
            ("REOBSERVE", "CLASSIFY_FAILURE"),
            "no_attributed_failure",
            active_skill=active_skill,
            compatible_skills=compatible,
        )
    if kinds & {
        "insertion_blocked",
        "release_stuck",
        "contact_stalled",
        "pose_control_diverged",
        "door_close_failed",
    }:
        return RecoveryPlan(
            "contact",
            (
                "VERIFY_CURRENT_GRASP_AND_OBJECT_POSE",
                "PLAN_SMALL_POSE_TRANSITION",
                "VERIFY_GRIPPER_HOLD_AFTER_TRANSITION",
                "PRE_ALIGN_OUTSIDE_OPENING",
                "APPROACH_OPENING_IN_SHORT_SEGMENTS",
                "VERIFY_CONTACT_AND_IK_PROGRESS",
                "INSERT_IN_SHORT_SEGMENTS",
                "VERIFY_INTERIOR_CONTAINMENT",
                "RELEASE_ONLY_AFTER_SUPPORT_CHECK",
                "VERIFY_NATIVE_IN_PREDICATE",
                "CLOSE_DOOR_ONLY_AFTER_OBJECTS_SECURED",
                "VERIFY_DOOR_CLOSED",
                "RELOCALIZE_OR_STOP",
            ),
            "contact_or_insertion_failure",
            active_skill=active_skill,
            compatible_skills=compatible,
        )
    if kinds & {
        "wrong_spatial_relation",
        "placement_not_satisfied",
        "object_collision",
        "support_not_established",
        "goal_predicate_unsatisfied",
    }:
        return RecoveryPlan(
            "spatial",
            (
                "REOBSERVE_TARGET_AND_SUPPORT_SURFACE",
                "ROUTE_BDDL_PREDICATE_TO_MATCHING_VALIDATOR",
                "CHECK_SUPPORT_OR_SPATIAL_GEOMETRY",
                "CHECK_SWEPT_PATH_AND_CLEARANCE",
                "SELECT_SMALL_CORRECTION_OR_REGRASP",
                "RELOCALIZE_AFTER_EACH_CORRECTION",
                "VERIFY_NATIVE_GOAL_PREDICATE",
            ),
            "spatial_relation_or_collision_failure",
            active_skill=active_skill,
            compatible_skills=compatible,
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
            active_skill=active_skill,
            compatible_skills=compatible,
        )
    return RecoveryPlan(
        "observe",
        ("REOBSERVE", "REFINE_FAILURE_CLASSIFICATION"),
        "unmapped_failure_requires_observation",
        active_skill=active_skill,
        compatible_skills=compatible,
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
