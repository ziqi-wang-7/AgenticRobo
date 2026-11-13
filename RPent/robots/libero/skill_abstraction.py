# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Task-independent skill abstraction for manipulation and recovery routing.

The layer classifies the active operation from primitive semantics and
candidate roles, never from LIBERO task IDs, object names, or stored poses.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Skill = Literal[
    "grasp", "transport", "placement", "contact", "insertion",
    "articulation", "observe", "unknown",
]

# Primitive semantics are deliberately broad. Task-specific object names and
# coordinates must not be used as routing features.
_TOOL_SKILLS: dict[str, Skill] = {
    "pi0_pick": "grasp",
    "pi0_doubled": "contact",
    "move_to": "transport",
    "move_pose": "transport",
    "release": "placement",
    "set_gripper": "grasp",
    "rotate_wrist": "transport",
    "rotate_pitch": "transport",
    "back_project": "observe",
    "segment": "observe",
}

@dataclass(frozen=True)
class SkillContext:
    active_skill: Skill
    target_role: str | None = None
    destination_role: str | None = None
    requires_contact: bool = False

    def as_dict(self) -> dict:
        return {
            "active_skill": self.active_skill,
            "target_role": self.target_role,
            "destination_role": self.destination_role,
            "requires_contact": self.requires_contact,
        }


def infer_skill_context(
    *,
    tool: str | None = None,
    roles: dict | None = None,
    task_operation: str | None = None,
) -> SkillContext:
    """Infer a reusable operation class without looking at task identity."""
    roles = roles or {}
    normalized_operation = (task_operation or "").casefold()
    if normalized_operation in {"insert", "insertion"}:
        skill: Skill = "insertion"
    elif normalized_operation in {"articulate", "articulation", "open", "close"}:
        skill = "articulation"
    elif tool:
        skill = _TOOL_SKILLS.get(tool, "unknown")
    else:
        skill = "unknown"
    target = roles.get("target_object") or roles.get("held_object")
    destination = roles.get("destination")
    return SkillContext(
        active_skill=skill,
        target_role=str(target) if target else None,
        destination_role=str(destination) if destination else None,
        requires_contact=skill in {"contact", "insertion", "articulation"},
    )


def compatible_recovery_skills(
    failure_kinds: tuple[str, ...] | list[str],
    active_skill: Skill,
) -> tuple[str, ...]:
    """Return relevant recovery skills; preserve multiple independent causes."""
    kinds = set(failure_kinds)
    result: list[str] = []
    if kinds & {"grasp_not_closed", "object_slipped", "grasp_space_insufficient", "object_tipped"}:
        result.append("grasp_recovery")
    if kinds & {"wrong_spatial_relation", "placement_not_satisfied", "object_collision"}:
        result.append("spatial_recovery")
    if kinds & {"insertion_blocked", "release_stuck", "contact_stalled"}:
        result.append("contact_recovery")
    # The active skill is context, not a reason to discard an observed failure.
    if active_skill in {"insertion", "articulation", "contact"} and "contact_recovery" not in result:
        if kinds & {"object_collision", "grasp_space_insufficient"}:
            result.append("contact_recovery")
    return tuple(result)
