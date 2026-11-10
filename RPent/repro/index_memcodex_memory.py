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

"""Build and independently validate a task index for the frozen M1 corpus."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from repro.memcodex_corpus import HISTORICAL, cell_key, digest, encoded, validate
from rpent.memory.memcodex.hierarchy import LAYERS
from rpent.memory.memcodex.lineage import issue_tags, source_links
from rpent.memory.memcodex.recovery import recovery_cases


def task_requirements(language: str) -> dict:
    """Extract visible task entities and skill families from the LIBERO vocabulary.

    These are query requirements, not observed object poses or success labels.
    Identity of each entity must be supplied by current perception at runtime.
    """
    text = language.casefold()
    phrases = [
        ("yellow and white mug", "yellow_white_mug"),
        ("white mug", "white_mug"),
        ("red mug", "red_mug"),
        ("left plate", "left_plate"),
        ("right plate", "right_plate"),
        ("both moka pots", "left_moka_pot right_moka_pot"),
        ("left moka pot", "left_moka_pot"),
        ("moka pot", "moka_pot"),
        ("akita black bowl", "target_bowl"),
        ("black bowl", "target_bowl"),
        ("cream cheese box", "cream_cheese"),
        ("cream cheese", "cream_cheese"),
        ("chocolate pudding", "chocolate_pudding"),
        ("alphabet soup", "alphabet_soup"),
        ("tomato sauce", "tomato_sauce"),
        ("salad dressing", "salad_dressing"),
        ("orange juice", "orange_juice"),
        ("bbq sauce", "bbq_sauce"),
        ("wine bottle", "wine_bottle"),
        ("cookies box", "cookie_box"),
        ("cookie box", "cookie_box"),
        ("drawer", "drawer_handle"),
        ("wooden cabinet", "cabinet"),
        ("cabinet", "cabinet"),
        ("ramekin", "ramekin"),
        ("basket", "basket"),
        ("microwave", "microwave"),
        ("stove", "stove"),
        ("plate", "plate"),
        ("bowl", "target_bowl"),
        ("bottle", "bottle"),
        ("butter", "butter"),
        ("ketchup", "ketchup"),
        ("milk", "milk"),
        ("caddy", "caddy"),
        ("cup", "cup"),
        ("book", "book"),
        ("rack", "rack"),
        ("pan", "pan"),
    ]
    objects = []
    for phrase, names in phrases:
        if re.search(r"\b" + re.escape(phrase) + r"\b", text):
            objects.extend(names.split())
            text = re.sub(r"\b" + re.escape(phrase) + r"\b", " ", text)
    if not objects:
        raise ValueError(f"No supported task entities: {language}")
    skills = []
    if re.search(r"\b(pick|put|place)\b", language, re.I):
        skills.extend(["pi0_pick", "move_to", "release"])
    if re.search(r"\b(open|close|turn)\b", language, re.I):
        skills.append("pi0_doubled")
    if re.search(r"\bpush\b", language, re.I):
        skills.append("move_to")
    result = {
        "required_objects": sorted(set(objects)),
        "required_skills": sorted(set(skills)),
    }
    if "drawer_handle" in objects:
        match = re.search(r"\b(top|upper|middle|bottom|lower)\b", language, re.I)
        level = (
            {"upper": "top", "lower": "bottom"}.get(match[0].lower(), match[0].lower())
            if match
            else None
        )
        result["drawer_goal"] = {
            "level": level,
            "operation": "close" if re.search(r"\bclose\b", language, re.I) else "open",
        }
        result["context_objects"] = [
            f"{level}_drawer_handle" for level in ("top", "middle", "bottom")
        ]
    return result


def build_index(root: Path) -> dict:
    """Index immutable metadata and source links without changing M1 entries."""
    manifest = json.loads((root / "MANIFEST.json").read_text())
    tasks = {
        f"{c['suite']}/t{c['task']}": {
            "language": c["language"],
            **task_requirements(c["language"]),
            "layers": {layer: [] for layer in LAYERS},
        }
        for c in manifest["catalog"]
    }
    entries = {}
    source_to_tasks: dict[str, set[str]] = {}
    source_to_skills: dict[str, set[str]] = {}
    for name in manifest["files"]:
        if name.startswith("raw/"):
            continue
        data = json.loads((root / name).read_text())
        for ref in data["source_refs"]:
            source_to_tasks.setdefault(ref.split("#")[0], set()).update(
                data["task_ids"]
            )
            source_to_skills.setdefault(ref.split("#")[0], set()).update(
                data.get("skill_ids", [])
            )
    for name, sha in sorted(manifest["files"].items()):
        payload = (root / name).read_bytes()
        if digest(payload) != sha:
            raise ValueError(f"Corpus changed: {name}")
        data = json.loads(payload)
        layer = name.split("/")[0]
        source = data["source_refs"][0].split("#")[0]
        bindings = set(data["task_ids"])
        if layer == "raw":
            bindings = set(source_to_tasks.get(source, ()))
            if source.startswith("task-specific/"):
                bindings = {
                    cell_key(Path(source).name.split(".")[0].removesuffix("_recipe"))
                }
            if not source.startswith(("task-family/", "task-specific/")):
                bindings = (
                    set()
                )  # Raw global/index archives add no deeper task evidence.
        provides = set(data.get("provides", []))
        if layer == "relational":
            provides.add("localization")
        skill_ids = (
            sorted(source_to_skills.get(source, ()))
            if layer == "raw"
            else data.get("skill_ids", [])
        )
        if skill_ids:
            provides.add("skill")
        if data["text"].startswith("## Failure modes"):
            provides.update(["anomaly", "recovery"])
        if layer == "raw" and "## Failure modes" in data["text"]:
            provides.update(["anomaly", "recovery"])
        entry = {
            "path": name,
            "sha256": sha,
            "key": f"{layer}:{data['id']}",
            "layer": layer,
            "provides": sorted(provides),
            "skill_ids": skill_ids,
            "applies_when": data["applies_when"],
            "review_required": bool(
                data.get("condition_text") and data["condition_text"] != HISTORICAL
            ),
            "source_refs": data["source_refs"],
            "issue_tags": issue_tags(data["text"]),
            "recovery_cases": recovery_cases(data["text"]),
        }
        entries[name] = entry
        for task in bindings & tasks.keys():
            match = "shared" if source.startswith("global/") else "exact"
            tasks[task]["layers"][layer].append({"path": name, "match": match})
    # Only tasks without their own family get static analogy references; no online search.
    for task in manifest["missing_task_family"]:
        family = task.split("/t")[0].rsplit("_", 1)[0]
        words = set(tasks[task]["required_objects"])
        others = [
            t
            for t in tasks
            if t != task
            and t.startswith(family + "_")
            and tasks[t]["layers"]["summary"]
        ]
        others.sort(key=lambda t: (-len(words & set(tasks[t]["required_objects"])), t))
        tasks[task]["analogy_tasks"] = others[:2]
        for candidate in others[:2]:
            for layer in ("skills", "raw"):
                for item in tasks[candidate]["layers"][layer]:
                    if item["match"] == "exact":
                        tasks[task]["layers"][layer].append(
                            {
                                "path": item["path"],
                                "match": "analogy",
                                "reference_task": candidate,
                            }
                        )
    return {
        "schema_version": 3,
        "mode": "m2",
        "version": "m2-index-v3",
        "source_links": source_links(entries),
        "corpus_manifest_sha256": digest((root / "MANIFEST.json").read_bytes()),
        "source_revision": manifest["source_revision"],
        "tasks": tasks,
        "entries": entries,
    }


def validate_index(root: Path, index_path: Path) -> dict:
    """Rebuild metadata, task bindings and source hashes to detect stale indexes."""
    actual = json.loads(index_path.read_text())
    expected = build_index(root)
    if actual.get("schema_version") in {1, 2}:
        version = actual["schema_version"]
        expected.update(schema_version=version, version=f"m2-index-v{version}")
        for entry in expected["entries"].values():
            entry.pop("recovery_cases")
        for task in expected["tasks"].values():
            task.pop("drawer_goal", None)
            task.pop("context_objects", None)
    if actual.get("schema_version") == 1:
        # Preserve frozen v1 runs; the reader derives their source links in memory.
        expected.update(schema_version=1, version="m2-index-v1")
        expected.pop("source_links")
        for entry in expected["entries"].values():
            entry.pop("issue_tags")
    if actual != expected:
        raise ValueError("Index differs from deterministic metadata rebuild")
    return {
        "status": "passed",
        "tasks": len(json.loads(index_path.read_text())["tasks"]),
        "index_sha256": digest(index_path.read_bytes()),
    }


def main() -> None:
    """Build or validate an index against the official source and frozen corpus."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--receipt", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    validate(args.corpus, args.source, args.receipt)
    index = build_index(args.corpus)
    if args.output.exists() and json.loads(args.output.read_text()) != index:
        raise FileExistsError("Refusing to replace a different M2 index")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_bytes(encoded(index))
    print(json.dumps(validate_index(args.corpus, args.output), indent=2))


if __name__ == "__main__":
    main()
