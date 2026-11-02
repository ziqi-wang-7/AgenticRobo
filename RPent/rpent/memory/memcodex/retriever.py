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

"""Deterministic per-layer retrieval over task and applicability metadata."""

from __future__ import annotations

import re
from collections.abc import Sequence

from rpent.memory.memcodex.hierarchy import MemoryEntry
from rpent.memory.memcodex.router import RetrievalQuery, RetrievalRouter


class LayerRetriever:
    """Rank eligible entries; lexical similarity never overrides task identity."""

    def __init__(self, router: RetrievalRouter, *, max_per_layer: int = 8) -> None:
        if type(max_per_layer) is not int or max_per_layer <= 0:
            raise ValueError("max_per_layer must be a positive integer")
        self.router = router
        self.max_per_layer = max_per_layer

    def retrieve(
        self,
        entries: Sequence[MemoryEntry],
        query: RetrievalQuery,
        evidence: Sequence[MemoryEntry] = (),
    ) -> tuple[MemoryEntry, ...]:
        """Use relation skill links and current needs to prioritize evidence."""
        links = {skill for e in evidence for skill in e.skill_ids} | set(
            query.required_skills
        )
        words = set(
            re.findall(
                r"\w+", " ".join((query.task, *query.required_objects)).casefold()
            )
        )

        def rank(entry: MemoryEntry) -> tuple:
            urgent = int(
                (query.needs_recovery or query.anomaly) and "recovery" in entry.provides
            )
            urgent += int(query.anomaly and "anomaly" in entry.provides)
            overlap = len(words & set(re.findall(r"\w+", entry.text.casefold())))
            return (
                -urgent,
                -int(query.task in entry.task_ids),
                -len(links & set(entry.skill_ids)),
                -overlap,
                entry.id,
            )

        eligible = [e for e in entries if self.router.applicable(e, query)]
        if query.available_skills is not None:
            eligible = [
                e
                for e in eligible
                if "skill" not in e.provides
                or set(e.skill_ids) & set(query.available_skills)
            ]
        return tuple(sorted(eligible, key=rank)[: self.max_per_layer])
