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

"""Deterministic, source-traceable LIBERO memory conversion and validation."""

from __future__ import annotations

import hashlib
import json
import re
from collections import defaultdict
from collections.abc import Iterator
from pathlib import Path, PurePosixPath
from typing import Any

import yaml

from rpent.memory.memcodex.hierarchy import MemoryEntry

REVISION = "47f301ef33bf8d0e8c770342f16b3c39c6a0e8c1"
VERSION = "GPT_6_astra_low"
ALIASES = {
    "libero10": "libero_10",
    "liberogoal": "libero_goal",
    "liberoobject": "libero_object",
    "spatial": "libero_spatial",
}
LAYERS = {"Applicable pattern": "summary", "Re-localization per scene": "relational"}
PRIMITIVES = (
    "move_to",
    "move_pose",
    "pi0_pick",
    "pi0_doubled",
    "release",
    "set_gripper",
    "rotate_pitch",
    "rotate_wrist",
)
HISTORICAL = "Historical seed-0 experience. Check its assumptions against the current scene; re-localize coordinates and verify success with the environment."


def digest(data: bytes) -> str:
    """Return a stable content digest."""
    return hashlib.sha256(data).hexdigest()


def encoded(value: object) -> bytes:
    """Serialize corpus files deterministically."""
    return (
        json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    ).encode()


def task_key(suite: str, task: int, regime: str | None = None) -> str:
    """Normalize published suite aliases to the runtime task identity."""
    suite = ALIASES.get(suite, suite)
    if not re.fullmatch(r"libero_(spatial|object|goal|10)_(task|swap)", suite):
        suite += f"_{regime}"
    if not re.fullmatch(r"libero_(spatial|object|goal|10)_(task|swap)", suite):
        raise ValueError(f"Unknown suite/regime: {suite}")
    if type(task) is not int or not 0 <= task < 10:
        raise ValueError(f"Invalid task index: {task}")
    return f"{suite}/t{task}"


def cell_key(cell: str) -> str:
    """Resolve an official seed-0 evidence filename."""
    match = re.fullmatch(r"(spatial|object|goal|10)_(task|swap)_t(\d+)_s0", cell)
    if not match:
        raise ValueError(f"Invalid evidence cell: {cell}")
    family, regime, task = match.groups()
    return task_key(f"libero_{family}_{regime}", int(task))


def frontmatter(text: str) -> tuple[dict, str, int]:
    """Read YAML metadata and preserve the original body line numbers."""
    lines = text.splitlines(keepends=True)
    if not lines or lines[0].strip() != "---":
        raise ValueError("Expected YAML frontmatter")
    end = next(i for i in range(1, len(lines)) if lines[i].strip() == "---")
    return yaml.safe_load("".join(lines[1:end])), "".join(lines[end + 1 :]), end + 2


def sections(text: str) -> Iterator[tuple[str, str, int, int]]:
    """Yield complete Markdown sections and their inclusive source line spans."""
    _, body, offset = frontmatter(text)
    lines = body.splitlines(keepends=True)
    starts = [i for i, line in enumerate(lines) if line.startswith("## ")]
    for n, start in enumerate(starts):
        end = starts[n + 1] if n + 1 < len(starts) else len(lines)
        yield (
            lines[start][3:].strip(),
            "".join(lines[start:end]).strip(),
            offset + start,
            offset + end - 1,
        )


def recipe_without_positions(value: Any) -> Any:
    """Remove spatial argument payloads; retain original command ordering."""
    spatial = {
        "xyz",
        "target_xyz",
        "position",
        "target_position",
        "pos",
        "pixel",
        "pixels",
        "bbox",
        "bbox_xyxy",
        "region",
        "target_quat",
        "target_quaternion",
    }
    if isinstance(value, dict):
        return {
            k: recipe_without_positions(v) for k, v in value.items() if k not in spatial
        }
    if isinstance(value, list):
        return [recipe_without_positions(v) for v in value]
    return value


def build(
    files: dict[str, bytes], catalog: list[dict]
) -> tuple[dict[str, bytes], dict]:
    """Build entries using extraction, exact deduplication and static source links."""
    tasks = {task_key(c["suite"], c["task"]): c["language"] for c in catalog}
    if len(tasks) != len(catalog):
        raise ValueError("Catalog contains duplicate tasks")
    globals_ = {}
    families = {}
    for path, data in sorted(files.items()):
        if path.startswith(("global/", "task-family/")) and path.endswith(".md"):
            metadata, body, offset = frontmatter(data.decode())
            if path.startswith("global/"):
                if metadata["id"] in globals_:
                    raise ValueError("Duplicate global ID")
                globals_[metadata["id"]] = (path, metadata, body, offset)
            else:
                key = task_key(
                    metadata["suite"], metadata["task_id"], metadata["regime"]
                )
                if key not in tasks or key in families:
                    raise ValueError(f"Ambiguous/out-of-catalog family: {key}")

                def normalize(s: str) -> str:
                    return re.sub(r"\W+", " ", s).strip().casefold()

                if normalize(metadata["task_language"]) != normalize(tasks[key]):
                    raise ValueError(
                        f"Source task language disagrees with catalog: {key}"
                    )
                families[key] = (path, metadata)
    bindings = defaultdict(set)
    dangling = set()
    cross_family_refs = set()
    family_ids = {metadata["id"] for _, metadata in families.values()}
    missing = sorted(set(tasks) - set(families))
    for key in tasks:
        # All globals are an explicit static fallback when a task leaf is absent.
        pending = (
            list(families[key][1].get("related", []))
            if key in families
            else list(globals_)
        )
        for identifier, (_, meta, _, _) in globals_.items():
            if any(
                cell_key(cell) == key
                for cell in meta.get("evidence", {}).get("cells", [])
            ):
                pending.append(identifier)
        visited = set()
        while pending:
            identifier = pending.pop()
            if identifier in visited:
                continue
            visited.add(identifier)
            if identifier not in globals_:
                if identifier in family_ids:
                    cross_family_refs.add(identifier)
                else:
                    dangling.add(identifier)
                continue
            bindings[identifier].add(key)
            pending.extend(globals_[identifier][1].get("related", []))

    groups = {}
    derivations = defaultdict(list)

    def add(
        layer,
        text,
        task_ids,
        path,
        start=1,
        end=None,
        transform="extract",
        condition=HISTORICAL,
    ):
        text = text if layer == "raw" else text.strip()
        if not text:
            return
        end = end or len(files[path].decode().splitlines())
        # Exact text/condition deduplication only; no semantic merging of numbers.
        identity = digest(encoded([layer, text, condition]))[:24]
        target = f"{layer}/{identity}.json"
        provides = ["task"] if layer == "summary" else []
        skills = [name for name in PRIMITIVES if re.search(r"\b" + name + r"\b", text)]
        if layer == "skills" and skills:
            provides.append("skill")
        entry = groups.setdefault(
            target,
            {
                "id": identity,
                "text": text,
                "task_ids": [],
                "applies_when": {},
                "condition_text": condition,
                "source_refs": [],
                "provides": provides,
                "skill_ids": skills,
            },
        )
        entry["task_ids"] = sorted(set(entry["task_ids"]) | set(task_ids))
        reference = f"{path}#L{start}-L{end}"
        entry["source_refs"] = sorted(set(entry["source_refs"]) | {reference})
        derivations[target].append(
            {
                "source": path,
                "start_line": start,
                "end_line": end,
                "transform": transform,
            }
        )

    for path, data in sorted(files.items()):
        add("raw", data.decode(), ["*"], path, transform="verbatim_source_archive")
    for key, (path, metadata) in sorted(families.items()):
        for heading, text, start, end in sections(files[path].decode()):
            if heading == "Cross-refs":
                continue
            add(LAYERS.get(heading, "skills"), text, [key], path, start, end)
    for identifier, (path, metadata, body, offset) in sorted(globals_.items()):
        if not bindings[identifier]:
            continue
        layer = "relational" if metadata.get("kind") == "perception" else "skills"
        condition = str(metadata["applies_when"])
        add(layer, body, bindings[identifier], path, offset, condition=condition)
    for path, data in sorted(files.items()):
        if not path.startswith("task-specific/"):
            continue
        key = cell_key(Path(path).name.split(".")[0].removesuffix("_recipe"))
        if key not in tasks:
            raise ValueError(f"Out-of-catalog task experience: {path}")
        if path.endswith(".json"):
            audit = json.loads(data)
            if task_key(audit["suite"], audit["task_id"]) != key or audit["seed"] != 0:
                raise ValueError(f"Audit identity disagrees with its filename: {path}")
            selected = {
                field: audit[field]
                for field in (
                    "strategy_notes",
                    "limitations",
                    "protocol_deviations",
                    "outcome_limitations",
                    "reliability_limit",
                )
                if field in audit
            }
            add(
                "skills",
                "Seed-0 strategy and limitations:\n"
                + json.dumps(selected, ensure_ascii=False, indent=2),
                [key],
                path,
                transform="audit_strategy_fields",
            )
        elif path.endswith(".jsonl"):
            commands = [
                recipe_without_positions(json.loads(line))
                for line in data.decode().splitlines()
                if line.strip()
            ]
            add(
                "skills",
                "Historical command order; position arguments omitted. Re-localize before executing; retained parameters were observed on seed 0.\n"
                + "\n".join(json.dumps(c, ensure_ascii=False) for c in commands),
                [key],
                path,
                transform="recipe_drop_spatial_arguments",
            )
        else:
            raise ValueError(f"Unknown source type: {path}")
    output = {}
    for path, entry in sorted(groups.items()):
        MemoryEntry.from_json(entry, path.split("/")[0], path)
        output[path] = encoded(entry)
    manifest = {
        "schema_version": 2,
        "status": "converted",
        "converter_version": "m1-extractive-v1",
        "source_version": VERSION,
        "source_revision": REVISION,
        "sources": {name: digest(data) for name, data in sorted(files.items())},
        "catalog": catalog,
        "task_count": len(tasks),
        "entry_count": len(output),
        "files": {name: digest(data) for name, data in output.items()},
        "derivations": dict(derivations),
        "missing_task_family": missing,
        "unresolved_related_ids": sorted(dangling),
        "cross_family_refs_not_expanded": sorted(cross_family_refs),
        "selection": "Fixed task/family/global source links; all globals if task family is absent; no state routing or cache",
        "raw_policy": "Archived for source audit; never returned by fixed mode",
    }
    return output, manifest


def read_sources(source: Path, receipt: Path) -> dict[str, bytes]:
    """Require the complete pinned official file set before conversion."""
    record = json.loads(receipt.read_text())
    if record["version"] != VERSION or record["revision"] != REVISION:
        raise ValueError("Expected the pinned official GPT_6_astra_low receipt")
    files = {}
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in official source: {path}")
        if path.is_file():
            files[path.relative_to(source).as_posix()] = path.read_bytes()
    if {k: digest(v) for k, v in files.items()} != record["files"]:
        raise ValueError("Official source inventory or hash mismatch")
    return files


def validate(root: Path, source: Path, receipt: Path) -> dict:
    """Verify source identity, every transformation, hashes and complete coverage."""
    manifest = json.loads((root / "MANIFEST.json").read_text())
    files = read_sources(source, receipt)
    expected, rebuilt = build(files, manifest["catalog"])
    if manifest != rebuilt:
        raise ValueError("Manifest/provenance differs from deterministic conversion")
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in converted corpus: {path}")
        if path.is_file():
            relative = path.relative_to(root).as_posix()
            if relative == "MANIFEST.json":
                continue
            actual.add(relative)
            if relative not in expected or path.read_bytes() != expected[relative]:
                raise ValueError(f"Converted entry/provenance mismatch: {relative}")
    if actual != set(expected):
        raise ValueError("Converted corpus file inventory mismatch")
    for path in manifest["files"]:
        relative = PurePosixPath(path)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Invalid corpus path")
    return {
        "status": "passed",
        "source_revision": REVISION,
        "source_files": len(files),
        "entries": len(expected),
        "tasks": manifest["task_count"],
        "missing_task_family": manifest["missing_task_family"],
        "unresolved_related_ids": manifest["unresolved_related_ids"],
        "validation_scope": "Exact extraction/transformation and source coverage; semantic applicability still requires scene verification",
    }
