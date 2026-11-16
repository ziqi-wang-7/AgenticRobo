# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Resolve conflicts between task goals, applicable successful memory and defaults.

This module resolves semantic relation conventions only. Historical coordinates
are never returned or reused; callers must validate a freshly localized candidate.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Axis = Literal["x", "y", "z"]
Relation = Literal["left_of", "right_of", "front_of", "behind", "above", "below"]


def extract_markdown_frontmatter(text: str) -> tuple[str, str]:
    """Extract YAML frontmatter even when an indexed entry prepends a notice.

    Returns (frontmatter, body). Only a delimiter pair on standalone lines is
    accepted; arbitrary prose containing '---' is not treated as metadata.
    """
    lines = text.splitlines()
    delimiters = [
        index for index, line in enumerate(lines)
        if line.strip() == "---"
    ]
    for opening, closing in zip(delimiters, delimiters[1:]):
        if closing <= opening:
            continue
        frontmatter = "\n".join(lines[opening + 1:closing]).strip()
        if not frontmatter:
            continue
        # Require task-family identity fields so an unrelated horizontal rule
        # in a preamble/body cannot be mistaken for the source metadata.
        if any(
            line.startswith(("id:", "scope:", "suite:", "regime:", "task_id:"))
            for line in frontmatter.splitlines()
        ):
            body = "\n".join(lines[closing + 1:]).strip()
            return frontmatter, body
    return "", text


@dataclass(frozen=True)
class TaskScope:
    suite: str
    task_id: int
    target_object: str
    reference_object: str


@dataclass(frozen=True)
class RelationEvidence:
    relation: Relation
    axis: Axis
    sign: Literal[-1, 1]
    suite: str
    task_id: int
    target_object: str
    reference_object: str
    successful: bool
    source_id: str


@dataclass(frozen=True)
class RelationResolution:
    relation: Relation
    axis: Axis
    sign: Literal[-1, 1]
    source: Literal["exact_task_success", "generic_convention", "unresolved"]
    conflict: bool
    source_id: str | None
    reason: str


def resolve_relation(
    *,
    scope: TaskScope,
    relation: Relation,
    generic: tuple[Axis, Literal[-1, 1]] | None,
    memories: list[RelationEvidence],
    require_exact_success: bool = False,
) -> RelationResolution:
    """Prefer verified successful evidence for the exact task/object relation.

    Sibling tasks, matching task IDs in another suite, unsuccessful memories,
    and memories for different target/reference objects cannot override defaults.
    """
    exact = [
        item for item in memories
        if item.successful
        and item.suite == scope.suite
        and item.task_id == scope.task_id
        and item.target_object == scope.target_object
        and item.reference_object == scope.reference_object
        and item.relation == relation
    ]
    # Conflicting exact-task successes are ambiguous; fail closed.
    directions = {(item.axis, item.sign) for item in exact}
    if len(directions) > 1:
        return RelationResolution(
            relation, "y", 1, "unresolved", True, None,
            "conflicting_exact_task_successes_require_current_scene_calibration",
        )
    if exact:
        evidence = exact[0]
        conflict = generic is not None and generic != (evidence.axis, evidence.sign)
        return RelationResolution(
            relation, evidence.axis, evidence.sign, "exact_task_success",
            conflict, evidence.source_id,
            "exact_suite_task_and_object_match; re-localize and validate current scene",
        )
    if require_exact_success:
        return RelationResolution(
            relation, "y", 1, "unresolved", False, None,
            "task_specific_success_evidence_missing; observe and calibrate before release",
        )
    if generic is not None:
        return RelationResolution(
            relation, generic[0], generic[1], "generic_convention", False, None,
            "no_applicable_exact_task_success; validate convention against current scene",
        )
    return RelationResolution(
        relation, "y", 1, "unresolved", False, None,
        "no_applicable_relation_evidence; observe and calibrate before placement",
    )


def validate_candidate_relation(
    *,
    resolution: RelationResolution,
    target_coordinate: float,
    reference_coordinate: float,
    minimum_separation: float = 0.0,
) -> bool | None:
    """Validate a freshly localized candidate against a resolved signed axis.

    Returns None for unresolved direction. This checks only the signed relation,
    not full BDDL geometry, collision clearance, support or physical stability.
    """
    if resolution.source == "unresolved":
        return None
    delta = target_coordinate - reference_coordinate
    signed_distance = delta * resolution.sign
    return signed_distance > minimum_separation
