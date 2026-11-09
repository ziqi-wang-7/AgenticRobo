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

"""Pure routing policy for action-specific M2 memory reads."""

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ActionFact:
    """One current-step fact or planner claim with its evidence provenance."""

    key: str
    status: Literal["verified", "claimed"]
    evidence_ref: str
    source: str


@dataclass(frozen=True)
class ActionQuery:
    """Normalized candidate and fixed evidence contract from the robot adapter."""

    tool: str
    step: int
    required_facts: tuple[str, ...]
    facts: tuple[ActionFact, ...]
    args_json: str = "{}"
    roles: tuple[tuple[str, str], ...] = ()
    expected_effect: str = ""
    fingerprint: str = ""


@dataclass(frozen=True)
class RoutingContext:
    """Facts and coverage of the retained context after a retrieval layer."""

    layer: str
    query: ActionQuery | None
    memory_missing: tuple[str, ...]
    action_missing_facts: tuple[str, ...]
    action_unverified_facts: tuple[str, ...]
    review_entries: tuple[str, ...]
    selected_ids: tuple[str, ...]
    recovery: bool
    narrow_ids: tuple[str, ...] = ()

    @property
    def memory_read_complete(self) -> bool:
        """Return whether retained memory covers the candidate's guidance needs."""
        return not self.memory_missing

    @property
    def action_preconditions_met(self) -> bool:
        """Return whether all required current-step facts are verified."""
        return not (self.action_missing_facts or self.action_unverified_facts)


@dataclass(frozen=True)
class RoutingDecision:
    """Propose a read action without changing evidence or granting execution."""

    action: Literal["STOP", "NARROW", "DESCEND", "OBSERVE", "REVIEW"]

    def __post_init__(self) -> None:
        if self.action not in {"STOP", "NARROW", "DESCEND", "OBSERVE", "REVIEW"}:
            raise ValueError(f"Unknown memory routing action: {self.action!r}")


def manual_route(context: RoutingContext) -> RoutingDecision:
    """Read candidate guidance, then distinguish observation from review gaps."""
    if context.memory_read_complete or context.layer == "raw":
        if context.action_missing_facts:
            return RoutingDecision("OBSERVE")
        if context.action_unverified_facts:
            return RoutingDecision("REVIEW")
        if context.memory_read_complete:
            return RoutingDecision("STOP")
    if context.layer in {"skills", "raw"} and context.review_entries:
        return RoutingDecision("REVIEW")
    if context.narrow_ids:
        return RoutingDecision("NARROW")
    return RoutingDecision("DESCEND")
