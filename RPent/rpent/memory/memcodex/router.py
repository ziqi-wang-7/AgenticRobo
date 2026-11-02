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

"""Conservative, explicit rules for applicability and evidence sufficiency."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from rpent.memory.memcodex.hierarchy import (
    MemoryEntry,
    Scalar,
    conditions,
    string_tuple,
)


@dataclass(frozen=True)
class RetrievalQuery:
    """Only structured perception facts enter routing; camera arrays stay out."""

    task: str
    required_objects: tuple[str, ...]
    localized_objects: tuple[str, ...]
    objects_localized: bool | None
    required_skills: tuple[str, ...]
    available_skills: tuple[str, ...] | None
    anomaly: bool
    needs_recovery: bool
    facts: tuple[tuple[str, Scalar], ...]

    def cache_key(self) -> str:
        """Return a canonical key covering every fact used by the router."""
        from dataclasses import asdict

        return json.dumps(asdict(self), sort_keys=True, ensure_ascii=False)


@dataclass(frozen=True)
class Sufficiency:
    """A rule verdict, never a prediction of environment success."""

    sufficient: bool
    missing: tuple[str, ...]


class RetrievalRouter:
    """Match exact task IDs and explicit conditions without model calls."""

    def query(
        self,
        task: str,
        observation: Mapping[str, Any] | None,
        state: Mapping[str, Any] | None,
    ) -> RetrievalQuery:
        """Normalize the current state, keeping unknown localization unknown.

        State conditions override older observation conditions. Unrecognized
        observation fields (including RGB/depth arrays) are ignored.
        """
        if not isinstance(task, str) or not task.strip():
            raise ValueError(
                "task must be a nonempty canonical task ID or exact instruction"
            )
        if observation is not None and not isinstance(observation, Mapping):
            raise ValueError("observation must be a mapping or None")
        if state is not None and not isinstance(state, Mapping):
            raise ValueError("state must be a mapping or None")
        observation, state = observation or {}, state or {}
        for key in ("objects_localized", "anomaly", "needs_recovery"):
            if key in state and type(state[key]) is not bool:
                raise ValueError(f"state.{key} must be a boolean")
        facts = dict(conditions(observation.get("conditions", {})))
        facts.update(conditions(state.get("conditions", {})))
        return RetrievalQuery(
            task.strip(),
            string_tuple(state.get("required_objects", []), "required_objects"),
            string_tuple(observation.get("localized_objects", []), "localized_objects"),
            state.get("objects_localized"),
            string_tuple(state.get("required_skills", []), "required_skills"),
            string_tuple(state["available_skills"], "available_skills")
            if "available_skills" in state
            else None,
            state.get("anomaly", False),
            state.get("needs_recovery", False),
            tuple(sorted(facts.items())),
        )

    def applicable(self, entry: MemoryEntry, query: RetrievalQuery) -> bool:
        """Unknown or contradictory preconditions reject an entry."""
        if entry.condition_text:
            # Natural-language conditions are retained for fixed-mode planners.
            # Progressive routing must not silently treat them as unconditional.
            return False
        if query.task not in entry.task_ids and "*" not in entry.task_ids:
            return False
        facts = dict(query.facts)
        return all(
            key in facts and type(facts[key]) is type(value) and facts[key] == value
            for key, value in entry.applies_when
        )

    def assess(
        self, entries: Sequence[MemoryEntry], query: RetrievalQuery
    ) -> Sufficiency:
        """Check only applicable evidence actually retained in the context."""
        usable = [e for e in entries if self.applicable(e, query)]
        missing = []
        if not any(query.task in e.task_ids and "task" in e.provides for e in usable):
            missing.append("task_match")
        objects = set(query.required_objects)
        if not objects:
            objects.update(obj for e in usable for obj in e.objects)
        localized = query.objects_localized
        if localized is None:
            localized = bool(objects) and objects <= set(query.localized_objects)
        if not localized:
            missing.append("object_localization")
        skills = {
            skill for e in usable if "skill" in e.provides for skill in e.skill_ids
        }
        if query.available_skills is not None:
            skills &= set(query.available_skills)
        if not skills or not set(query.required_skills) <= skills:
            missing.append("applicable_skill")
        capabilities = {c for e in usable for c in e.provides}
        if query.anomaly and "anomaly" not in capabilities:
            missing.append("anomaly_guidance")
        if (query.needs_recovery or query.anomaly) and "recovery" not in capabilities:
            missing.append("recovery_guidance")
        if not usable:
            missing.append("applicable_conditions")
        return Sufficiency(not missing, tuple(missing))
