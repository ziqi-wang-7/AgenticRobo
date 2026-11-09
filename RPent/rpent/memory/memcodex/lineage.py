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

"""Source-derived descent links and deterministic failure-topic metadata."""

from __future__ import annotations

import re

from rpent.memory.memcodex.hierarchy import LAYERS

# These terms rank historical guidance; they never establish environment facts.
_ISSUES = {
    "contact": r"contact|collis|stall|obstruct|block|jam|roof|clearance",
    "grasp": r"grasp|grip|pinch|handle",
    "slip": r"slip|drop|lost|empty",
    "insertion": r"insert|cavity|entrance|mouth|seat",
    "release": r"releas|detach|withdraw|retreat",
    "localization": r"localiz|localis|perception|pixel|depth|segment",
    "articulation": r"drawer|door|hinge|pull|clos",
}


def issue_tags(text: str) -> list[str]:
    """Extract bounded topic hints, not applicability or success judgments."""
    return sorted(k for k, pattern in _ISSUES.items() if re.search(pattern, text, re.I))


def source_links(entries: dict[str, dict]) -> dict[str, dict[str, list[str]]]:
    """Link entries to deeper entries citing the same source file.

    Line ranges remain in source_refs. Sharing a file gives provenance, not
    applicability. Task postings and current conditions still bound each read.
    """
    sources: dict[str, set[str]] = {}
    for path, entry in entries.items():
        for ref in entry["source_refs"]:
            sources.setdefault(ref.split("#", 1)[0], set()).add(path)
    links = {}
    for path, entry in entries.items():
        descendants = set().union(
            *(sources[ref.split("#", 1)[0]] for ref in entry["source_refs"])
        )
        links[path] = {
            layer: sorted(p for p in descendants if entries[p]["layer"] == layer)
            for layer in LAYERS[LAYERS.index(entry["layer"]) + 1 :]
        }
    return links
