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

"""Read versioned JSON entries lazily from the four memory layers."""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAYERS = ("summary", "relational", "skills", "raw")
CAPABILITIES = frozenset({"task", "skill", "anomaly", "recovery"})
Scalar = str | bool | int | float | None


def string_tuple(value: Any, field: str) -> tuple[str, ...]:
    """Validate an explicit list of nonempty strings at an input boundary."""
    if not isinstance(value, (list, tuple)) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{field} must be a list of nonempty strings")
    return tuple(dict.fromkeys(item.strip() for item in value))


def conditions(value: Any) -> tuple[tuple[str, Scalar], ...]:
    """Require finite JSON scalars so applicability and cache keys agree."""
    if not isinstance(value, Mapping) or any(
        not isinstance(key, str)
        or not key
        or (item is not None and type(item) not in (str, bool, int, float))
        for key, item in value.items()
    ):
        raise ValueError("conditions must map nonempty strings to JSON scalars")
    try:
        json.dumps(dict(value), allow_nan=False)
    except ValueError as exc:
        raise ValueError("conditions must contain finite values") from exc
    return tuple(sorted(value.items()))


@dataclass(frozen=True)
class MemoryEntry:
    """One indivisible evidence block with machine-readable applicability."""

    id: str
    layer: str
    text: str
    task_ids: tuple[str, ...]
    applies_when: tuple[tuple[str, Scalar], ...]
    provides: tuple[str, ...] = ()
    skill_ids: tuple[str, ...] = ()
    objects: tuple[str, ...] = ()
    source: str = ""
    condition_text: str = ""
    source_refs: tuple[str, ...] = ()

    @property
    def key(self) -> str:
        """Layer-qualified identifier used in traces and context headers."""
        return f"{self.layer}:{self.id}"

    def render(self) -> str:
        """Keep the text and its conditions together when packing context."""
        applicability = json.dumps(
            dict(self.applies_when), ensure_ascii=False, sort_keys=True
        )
        condition = (
            f"\nApplicability (evaluate against the current scene): {self.condition_text}"
            if self.condition_text
            else ""
        )
        return f"[{self.key}]\napplies_when={applicability}{condition}\n{self.text.strip()}"

    @classmethod
    def from_json(cls, data: Any, layer: str, source: str) -> MemoryEntry:
        """Reject incomplete or unknown metadata instead of guessing semantics."""
        required = {"id", "text", "task_ids", "applies_when"}
        optional = {"provides", "skill_ids", "objects", "condition_text", "source_refs"}
        if (
            not isinstance(data, dict)
            or not required <= data.keys()
            or data.keys() - required - optional
        ):
            raise ValueError(
                f"{source}: expected {sorted(required)} and optional {sorted(optional)}"
            )
        if any(
            not isinstance(data[k], str) or not data[k].strip() for k in ("id", "text")
        ):
            raise ValueError(f"{source}: id and text must be nonempty strings")
        if any(char in data["id"] for char in "\n\r[]"):
            raise ValueError(f"{source}: id cannot contain newlines or brackets")
        tasks = string_tuple(data["task_ids"], "task_ids")
        provides = string_tuple(data.get("provides", []), "provides")
        skills = string_tuple(data.get("skill_ids", []), "skill_ids")
        if not tasks or set(provides) - CAPABILITIES:
            raise ValueError(f"{source}: empty task_ids or unknown provides capability")
        if "skill" in provides and not skills:
            raise ValueError(f"{source}: skill capability requires skill_ids")
        if not isinstance(data.get("condition_text", ""), str):
            raise ValueError(f"{source}: condition_text must be a string")
        return cls(
            data["id"].strip(),
            layer,
            data["text"].strip(),
            tasks,
            conditions(data["applies_when"]),
            provides,
            skills,
            string_tuple(data.get("objects", []), "objects"),
            source,
            data.get("condition_text", ""),
            string_tuple(data.get("source_refs", []), "source_refs"),
        )


@dataclass(frozen=True)
class CorpusSnapshot:
    """Filesystem revision without reading raw experience contents."""

    revision: str
    files: tuple[tuple[str, Path], ...]


class MemoryHierarchy:
    """Own a read-only corpus; only explicitly visited layers are decoded."""

    def __init__(
        self, memory_root: str | Path, *, max_entry_bytes: int = 1024 * 1024
    ) -> None:
        self.root = Path(memory_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"Memory corpus does not exist: {self.root}")
        if type(max_entry_bytes) is not int or max_entry_bytes <= 0:
            raise ValueError("max_entry_bytes must be a positive integer")
        self.max_entry_bytes = max_entry_bytes

    def snapshot(self) -> CorpusSnapshot:
        """Invalidate caches on additions, deletions and normal filesystem edits."""
        inventory, files = [], []
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        for layer in LAYERS:
            folder = self.root / layer
            if folder.is_symlink():
                raise ValueError(f"Memory layer cannot be a symlink: {folder}")
            if not folder.exists():
                continue
            if not folder.is_dir():
                raise ValueError(f"Memory layer must be a directory: {folder}")
            for base, dirs, names in os.walk(folder):
                dirs.sort()
                for name in dirs + sorted(names):
                    path = Path(base) / name
                    if path.is_symlink():
                        raise ValueError(f"Memory paths cannot be symlinks: {path}")
                    if path.suffix != ".json" or not path.is_file():
                        continue
                    stat = path.stat()
                    relative = path.relative_to(self.root).as_posix()
                    inventory.append(
                        (
                            relative,
                            stat.st_size,
                            stat.st_mtime_ns,
                            stat.st_ctime_ns,
                            stat.st_ino,
                        )
                    )
                    files.append((layer, path))
        revision = hashlib.sha256(json.dumps(inventory).encode()).hexdigest()
        return CorpusSnapshot(revision, tuple(files))

    def read_layer(
        self, layer: str, snapshot: CorpusSnapshot
    ) -> tuple[MemoryEntry, ...]:
        """Load bounded individual entries, reporting malformed files by name."""
        if layer not in LAYERS:
            raise ValueError(f"Unknown layer: {layer}")
        entries, ids = [], set()
        for item_layer, path in snapshot.files:
            if item_layer != layer:
                continue
            if path.is_symlink() or not path.resolve().is_relative_to(self.root):
                raise ValueError(f"Memory path escaped corpus: {path}")
            with path.open("rb") as stream:
                payload = stream.read(self.max_entry_bytes + 1)
            if len(payload) > self.max_entry_bytes:
                raise ValueError(f"Memory entry exceeds max_entry_bytes: {path}")
            try:
                data = json.loads(payload)
                entry = MemoryEntry.from_json(
                    data, layer, path.relative_to(self.root).as_posix()
                )
            except (ValueError, UnicodeError) as exc:
                raise ValueError(f"Invalid memory entry {path}: {exc}") from exc
            if entry.id in ids:
                raise ValueError(f"Duplicate memory id in {layer}: {entry.id}")
            ids.add(entry.id)
            entries.append(entry)
        return tuple(entries)
