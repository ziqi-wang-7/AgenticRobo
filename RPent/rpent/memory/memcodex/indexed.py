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

"""Immutable task index with bounded, hash-checked entry reads."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

from rpent.memory.memcodex.hierarchy import LAYERS, MemoryEntry
from rpent.memory.memcodex.lineage import issue_tags, source_links


class IndexedCorpus:
    """Load metadata once; read only selected entry bodies on each request."""

    def __init__(self, root: str | Path, index_path: str | Path) -> None:
        self.root = Path(root).resolve()
        self.index_path = Path(index_path).resolve()
        payload = self.index_path.read_bytes()
        self.revision = hashlib.sha256(payload).hexdigest()
        self.index = json.loads(payload)
        if (
            self.index.get("schema_version") not in {1, 2, 3}
            or self.index.get("mode") != "m2"
        ):
            raise ValueError("Expected a schema-1, schema-2 or schema-3 M2 task index")
        self.manifest_path = self.root / "MANIFEST.json"
        self.manifest_hash = self.index["corpus_manifest_sha256"]
        if (
            hashlib.sha256(self.manifest_path.read_bytes()).hexdigest()
            != self.manifest_hash
        ):
            raise ValueError("M2 index belongs to a different corpus")
        manifest = json.loads(self.manifest_path.read_text())
        for name, entry in self.index["entries"].items():
            path = Path(name)
            if (
                path.is_absolute()
                or ".." in path.parts
                or len(path.parts) != 2
                or path.parts[0] not in LAYERS
                or path.suffix != ".json"
                or entry["path"] != name
                or entry["layer"] != path.parts[0]
                or entry["sha256"] != manifest["files"].get(name)
            ):
                raise ValueError(f"Invalid indexed entry: {name}")
        for task, profile in self.index["tasks"].items():
            for layer in LAYERS:
                for item in profile["layers"][layer]:
                    metadata = self.index["entries"].get(item["path"])
                    if (
                        metadata is None
                        or metadata["layer"] != layer
                        or item["match"] not in {"exact", "shared", "analogy"}
                    ):
                        raise ValueError(f"Invalid task posting: {task}/{layer}")
        self.source_links = source_links(self.index["entries"])
        if self.index["schema_version"] >= 2:
            if self.index.get("source_links") != self.source_links:
                raise ValueError("Index source links disagree with entry provenance")
            for entry in self.index["entries"].values():
                tags = entry.get("issue_tags")
                if not isinstance(tags, list) or any(
                    not isinstance(t, str) for t in tags
                ):
                    raise ValueError("Invalid issue tags in M2 index")
        self._fingerprints = self._stat()

    def _stat(self) -> tuple:
        return tuple(
            (p.stat().st_size, p.stat().st_mtime_ns, p.stat().st_ctime_ns)
            for p in (self.index_path, self.manifest_path)
        )

    def check_revision(self) -> None:
        """Reject changed index/manifest without scanning the corpus folders."""
        if self._stat() != self._fingerprints:
            raise RuntimeError("M2 index/manifest changed during this episode")

    def profile(self, task: str) -> dict:
        """Return the exact task profile, never an implicit wildcard fallback."""
        self.check_revision()
        if task not in self.index["tasks"]:
            raise ValueError(f"Unknown indexed task: {task}")
        return self.index["tasks"][task]

    def candidates(
        self, task: str, layer: str, skills: tuple[str, ...], recovery: bool
    ) -> list[dict]:
        """Rank task postings from metadata before opening any entry bodies."""
        items = [
            {**self.index["entries"][x["path"]], **x}
            for x in self.profile(task)["layers"][layer]
        ]
        return sorted(
            items,
            key=lambda x: (
                -int(recovery and "recovery" in x["provides"]),
                {"exact": 0, "analogy": 1, "shared": 2}[x["match"]],
                -len(set(skills) & set(x["skill_ids"])),
                x["path"],
            ),
        )

    def narrow(
        self,
        candidates: list[dict],
        *,
        parents: tuple[str, ...],
        skills: tuple[str, ...],
        recovery: bool,
        issue: str,
    ) -> tuple[list[dict], str]:
        """Prefer linked evidence and current action/failure topics before widening.

        This is a bounded read proposal. Excluded candidates stay available as
        fallback if the retained evidence is insufficient; tags confer no trust.
        """
        if not candidates:
            return candidates, "no_candidates"
        layer = candidates[0]["layer"]
        linked = {
            path
            for parent in parents
            for path in self.source_links.get(parent, {}).get(layer, ())
        }
        focused = [m for m in candidates if m["path"] in linked]
        reason = ["source_links"] if focused else []
        focused = focused or candidates
        if layer in {"skills", "raw"}:
            relevant = [
                m
                for m in focused
                if set(skills).intersection(m["skill_ids"])
                or (recovery and "recovery" in m["provides"])
            ]
            if relevant:
                focused = relevant
                reason.append("action_or_recovery")
            topics = set(issue_tags(issue))
            scores = {
                m["path"]: len(topics.intersection(m.get("issue_tags", ())))
                for m in focused
            }
            best = max(scores.values(), default=0)
            if recovery and best:
                focused = [m for m in focused if scores[m["path"]] == best]
                reason.append("failure_topic")
        return focused, "+".join(reason) or "task_candidates"

    def read(self, item: dict) -> MemoryEntry:
        """Read one bounded entry and validate its content hash and metadata."""
        path = self.root / item["path"]
        if (
            path.is_symlink()
            or path.parent.is_symlink()
            or not path.resolve().is_relative_to(self.root)
        ):
            raise ValueError("Indexed memory path escapes its corpus")
        with path.open("rb") as stream:
            payload = stream.read(1024 * 1024 + 1)
        if (
            len(payload) > 1024 * 1024
            or hashlib.sha256(payload).hexdigest() != item["sha256"]
        ):
            raise ValueError(f"Indexed entry changed: {item['path']}")
        entry = MemoryEntry.from_json(json.loads(payload), item["layer"], item["path"])
        if entry.key != item["key"]:
            raise ValueError("Indexed entry ID mismatch")
        return entry
