# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Route BDDL goal predicates to validators with matching semantics."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Validator = Literal["support", "containment", "spatial_relation", "unsupported"]


@dataclass(frozen=True)
class GoalPredicateRoute:
    predicate: str
    validator: Validator
    required_evidence: tuple[str, ...]
    reason: str


def route_goal_predicate(predicate: str) -> GoalPredicateRoute:
    """Select a validator by BDDL predicate semantics, never by object names."""
    normalized = predicate.strip().casefold()
    if normalized == "on":
        return GoalPredicateRoute(
            normalized,
            "support",
            ("object_base", "support_surface", "vertical_contact", "uprightness"),
            "On requires support/contact evidence; above is not an equivalent predicate",
        )
    if normalized == "in":
        return GoalPredicateRoute(
            normalized,
            "containment",
            ("object_boundary", "container_interior", "opening_clearance", "depth"),
            "In requires containment within the current container interior",
        )
    if normalized in {"left_of", "right_of", "front_of", "behind", "above", "below"}:
        return GoalPredicateRoute(
            normalized,
            "spatial_relation",
            ("current_scene_reference", "resolved_axis", "signed_distance"),
            "Directional relation requires current-scene spatial grounding",
        )
    return GoalPredicateRoute(
        normalized,
        "unsupported",
        ("native_environment_verification",),
        "No geometric validator is registered for this predicate",
    )
