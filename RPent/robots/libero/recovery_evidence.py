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

"""Current image-attributed failures and drawer progress, without pose oracles."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict

from robots.libero.action_evidence import _VISUAL_ARTIFACTS

FailureKind = Literal[
    "wrong_drawer",
    "object_slipped",
    "grasp_not_closed",
    "insertion_blocked",
    "release_stuck",
    "contact_stalled",
]


class FailureObservation(BaseModel):
    """Planner attribution to one current camera image, not verified physics."""

    model_config = ConfigDict(extra="forbid", strict=True)
    kind: FailureKind
    object_id: str
    phase: Literal[
        "approach", "grasp", "transport", "insert", "release", "articulate", "recovery"
    ]
    step: int
    artifact: str


class DrawerProgress(BaseModel):
    """Compare the intended drawer with the drawer seen moving after a primitive."""

    model_config = ConfigDict(extra="forbid", strict=True)
    previous_step: int
    step: int
    artifact: str
    target_level: Literal["top", "middle", "bottom"]
    observed_level: Literal["top", "middle", "bottom", "unknown"]
    expected_direction: Literal["outward", "inward"]
    observed_direction: Literal["outward", "inward", "none", "unknown"]


def validate_failure(failure: FailureObservation, record, objects: set[str]) -> dict:
    """Reject stale images and unknown object identities before retrieval."""
    if failure.object_id not in objects:
        raise ValueError("Failure object must use a current task/context object ID")
    if (
        failure.step != record.step_idx
        or failure.artifact not in _VISUAL_ARTIFACTS
        or failure.artifact not in record.artifacts
    ):
        raise ValueError("Failure observation requires a current camera artifact")
    return {
        "failure_kind": failure.kind,
        "failure_object": failure.object_id,
        "phase": failure.phase,
    }


def drawer_progress_failure(progress: DrawerProgress, current, previous) -> str | None:
    """Evaluate a drawer-specific predicate on attributed before/after images.

    A single explicit wrong level/direction is a failure. No motion is a reason
    to inspect contact and retrieve guidance, not proof of a collision.
    """
    if (
        progress.step != current.step_idx
        or progress.previous_step != current.step_idx - 1
        or previous.step_idx != progress.previous_step
    ):
        raise ValueError("Drawer progress requires consecutive primitive states")
    if progress.artifact not in _VISUAL_ARTIFACTS or any(
        progress.artifact not in r.artifacts for r in (previous, current)
    ):
        raise ValueError(
            "Drawer progress requires recorded before/after camera artifacts"
        )
    if (
        progress.observed_level != "unknown"
        and progress.observed_level != progress.target_level
    ):
        return "wrong_drawer"
    if progress.observed_direction == "none":
        return "contact_stalled"
    if progress.observed_direction not in {"unknown", progress.expected_direction}:
        return "contact_stalled"
    return None


def motion_alert(record) -> str | None:
    """Compare a servo's own residual with its requested tolerance, not all task motion."""
    if record.terminated or record.truncated:
        return None
    command, result = record.command or {}, record.result or {}
    if command.get("action") not in {"move_to", "move_pose"}:
        return None
    distance = result.get("final_dist_m")
    tolerance = command.get("tol", 0.012)
    if (
        isinstance(distance, (int, float))
        and distance > tolerance
        and result.get("steps_used", 0) >= command.get("max_steps", 80)
    ):
        return "contact_stalled"
    return None
