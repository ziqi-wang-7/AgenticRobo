# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""LIBERO contracts for reviewing evidence for a proposed primitive."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Collection, Mapping, Sequence
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from rpent.memory.memcodex.policy import ActionFact, ActionQuery
from rpent.tools.base import Tool

_ROLE_REQUIREMENTS = {
    "pi0_pick": ("target_object",),
    "move_to": ("held_object", "destination"),
    "release": ("held_object", "destination"),
}
_VISUAL_ARTIFACTS = frozenset(
    {"agentview.png", "agentview_high.png", "wrist.png", "wrist_high.png"}
)


class ActionCandidate(BaseModel):
    """A planner proposal. The expected effect is descriptive, not evidence."""

    model_config = ConfigDict(extra="forbid", strict=True)

    tool: str
    args: dict[str, Any]
    roles: dict[str, str]
    expected_effect: str | None = None


class ActionVisualCheck(BaseModel):
    """A planner's positive interpretation of a current camera artifact."""

    model_config = ConfigDict(extra="forbid", strict=True)

    fact: Literal["holding", "placement_ready"]
    object_id: str
    step: int
    artifact: str


def build_action_query(
    candidate: ActionCandidate,
    *,
    tools: Mapping[str, Tool],
    step: int,
    task_objects: Collection[str],
    localizations: Mapping[str, dict[str, Any]],
    artifacts: Collection[str],
    visual_checks: Sequence[ActionVisualCheck],
) -> tuple[dict[str, Any], ActionQuery]:
    """Validate a proposal and create immutable facts without executing the tool.

    Args:
        candidate: Proposed LIBERO primitive, parameters, and task-object roles.
        tools: Toolkit's registered tool definitions.
        step: Latest observation step.
        task_objects: Object IDs from the task profile.
        localizations: Already validated current-step perception results by object ID.
        artifacts: Artifacts recorded for the current step.
        visual_checks: Planner-attributed interpretations of current camera images.

    Returns:
        A normalized proposal for the audit and an immutable routing query.
    """
    required_roles = _ROLE_REQUIREMENTS.get(candidate.tool)
    primitive = tools.get(candidate.tool)
    if required_roles is None or primitive is None or primitive.readonly:
        raise ValueError(f"Unsupported action evidence primitive: {candidate.tool}")
    if set(candidate.roles) != set(required_roles):
        raise ValueError(
            f"{candidate.tool} requires roles: {', '.join(required_roles)}"
        )
    if not set(candidate.roles.values()) <= set(task_objects):
        raise ValueError("Candidate roles must use current task object IDs")
    if len(set(candidate.roles.values())) != len(required_roles):
        raise ValueError("Candidate roles require distinct task object IDs")

    parameters = primitive.args_schema.model_validate(candidate.args, strict=True)
    args = parameters.model_dump(mode="json")
    if candidate.tool == "move_to" and args["gripper"] != 1.0:
        raise ValueError("Transport move_to requires gripper=1.0")

    if candidate.tool == "pi0_pick":
        geometry_object = candidate.roles["target_object"]
        claim_keys = ()
    else:
        geometry_object = candidate.roles["destination"]
        claim_keys = (f"holding:{candidate.roles['held_object']}",)
        if candidate.tool == "release":
            claim_keys += (f"placement_ready:{geometry_object}",)

    geometry_key = f"localized:{geometry_object}"
    identity_key = f"identity:{geometry_object}"
    facts = []
    if localization := localizations.get(geometry_object):
        facts.append(
            ActionFact(
                key=geometry_key,
                status="verified",
                evidence_ref=localization["id"],
                source="perception_geometry_with_planner_semantic_binding",
            )
        )
        facts.append(
            ActionFact(
                key=identity_key,
                status="claimed",
                evidence_ref=localization["id"],
                source="planner_semantic_binding",
            )
        )
    seen_claims = set()
    for check in visual_checks:
        key = f"{check.fact}:{check.object_id}"
        if key not in claim_keys or key in seen_claims:
            raise ValueError(f"Unexpected or duplicate visual check: {key}")
        if (
            check.step != step
            or check.artifact not in _VISUAL_ARTIFACTS
            or check.artifact not in artifacts
        ):
            raise ValueError(f"Visual check requires a current camera artifact: {key}")
        seen_claims.add(key)
        facts.append(
            ActionFact(
                key=key,
                status="claimed",
                evidence_ref=f"{step}:{check.artifact}",
                source="planner_visual_claim",
            )
        )

    normalized = {**candidate.model_dump(), "args": args}
    fingerprint = hashlib.sha256(
        _canonical_json({"step": step, "candidate": normalized}).encode()
    ).hexdigest()
    return normalized, ActionQuery(
        tool=candidate.tool,
        step=step,
        required_facts=(geometry_key, identity_key, *claim_keys),
        facts=tuple(facts),
        args_json=_canonical_json(args),
        roles=tuple(sorted(candidate.roles.items())),
        expected_effect=candidate.expected_effect or "",
        fingerprint=fingerprint,
    )


def _canonical_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
