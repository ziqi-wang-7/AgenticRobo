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

"""Indexed coarse-to-fine retrieval without model calls or result caching."""

from __future__ import annotations

import json
from collections.abc import Mapping
from time import perf_counter
from typing import TYPE_CHECKING, Any

from rpent.memory.memcodex.hierarchy import LAYERS

if TYPE_CHECKING:
    from rpent.memory.memcodex.controller import MemCodexController, RetrievalResult


def retrieve_indexed(
    controller: MemCodexController,
    task: str,
    observation: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
) -> RetrievalResult:
    """Stop on retained evidence; report observation gaps separately from memory gaps."""
    from rpent.memory.memcodex.controller import RetrievalResult

    started = perf_counter()
    corpus = controller.indexed
    profile = corpus.profile(task)
    observation, state = observation or {}, state or {}

    def normalize(value: object) -> str:
        return " ".join(str(value).casefold().split())

    if normalize(observation.get("task_language", "")) != normalize(
        profile["language"]
    ):
        raise ValueError("Observed task language does not match the indexed task")
    purpose = state.get("purpose", "execute")
    if purpose not in {"overview", "execute", "recover"}:
        raise ValueError("purpose must be overview, execute or recover")
    recovery = bool(
        state.get("needs_recovery") or state.get("anomaly") or purpose == "recover"
    )
    localized = set(observation.get("localized_objects", ()))
    required = set(profile["required_skills"])
    available = set(state.get("available_skills", ()))
    confirmed = set(state.get("applicable_entries", ()))
    facts = state.get("conditions", {})
    kept: list[tuple[dict, str]] = []
    visited, omitted, seen_paths = [], [], set()
    reads = candidate_tokens = 0

    def assess(items: list[tuple[dict, str]]) -> tuple[str, ...]:
        usable = [
            m
            for m, _ in items
            if not (m["review_required"] or m["match"] == "analogy")
            or m["key"] in confirmed
        ]
        missing = []
        if not any(m["layer"] == "summary" and m["match"] == "exact" for m in usable):
            # The authoritative instruction still defines the task when no exact
            # experience exists; it never claims that an analogous task was solved.
            if profile["layers"]["summary"]:
                missing.append("task_summary")
        if purpose != "overview" or recovery:
            if not set(profile["required_objects"]) <= localized:
                missing.append("object_localization")
            skills = {s for m in usable for s in m["skill_ids"]} & available
            if not required <= skills:
                missing.append("skill_guidance")
            if recovery and not any("recovery" in m["provides"] for m in usable):
                missing.append("recovery_guidance")
        if not usable:
            missing.append("applicable_memory")
        return tuple(missing)

    def render(items: list[tuple[dict, str]]) -> str:
        missing = assess(items)
        review = [
            m["key"]
            for m, _ in items
            if (m["review_required"] or m["match"] == "analogy")
            and m["key"] not in confirmed
        ]
        info = {
            "task": task,
            "task_language": profile["language"],
            "purpose": purpose,
            "required_objects": profile["required_objects"],
            "required_skills": profile["required_skills"],
            "sufficient": not missing,
            "missing": missing,
            "review_entries": review,
            "exact_summary_available": bool(profile["layers"]["summary"]),
        }
        header = (
            "CURRENT QUERY (sufficiency is evidence coverage, not task success):\n"
            + json.dumps(info, ensure_ascii=False)
        )
        return (
            header
            + "\nHistorical coordinates are NOT current targets. Verify conditions and re-localize.\n\n"
            + "\n\n".join(t for _, t in items)
        )

    def fits(text: str) -> bool:
        # ToolResult truncates at 60,000 bytes. Never let that cut a retained skill.
        wire = json.dumps({"memory_context": text}, indent=2, ensure_ascii=True)
        return (
            len(wire) <= 60000
            and controller.compressor.count(text) <= controller.token_budget
        )

    if not fits(render([])):
        raise ValueError("Memory budget cannot fit the query/status envelope")
    for layer in LAYERS:
        visited.append(layer)
        candidates = corpus.candidates(task, layer, tuple(required), recovery)
        eligible = [
            m
            for m in candidates
            if all(
                k in facts and type(facts[k]) is type(v) and facts[k] == v
                for k, v in m["applies_when"].items()
            )
        ]
        for metadata in eligible[: controller.retriever.max_per_layer]:
            if metadata["path"] in seen_paths:
                continue
            seen_paths.add(metadata["path"])
            entry = corpus.read(metadata)
            reads += 1
            text = entry.render()
            if metadata["match"] == "analogy":
                text = (
                    "ANALOGY ONLY — different task: "
                    + metadata["reference_task"]
                    + "\n"
                    + text
                )
            candidate_tokens += controller.compressor.count(text)
            proposal = [*kept, (metadata, text)]
            if fits(render(proposal)):
                kept = proposal
            else:
                omitted.append(entry.key)
            if not assess(kept):
                break
        missing = assess(kept)
        if not missing:
            break
        if layer == "summary" and purpose == "overview" and not recovery:
            break
        if layer == "relational" and "object_localization" in missing:
            break  # Reading historical raw coordinates cannot repair absent perception.
        if layer == "skills" and any(
            (m["review_required"] or m["match"] == "analogy")
            and m["key"] not in confirmed
            for m, _ in kept
        ):
            break  # Have the existing planner review candidates before escalating.
    corpus.check_revision()
    context = render(kept)
    missing = assess(kept)
    result = RetrievalResult(
        context=context,
        token_count=controller.compressor.count(context),
        token_budget=controller.token_budget,
        counter_name=controller.compressor.counter_name,
        candidate_tokens=candidate_tokens,
        sufficient=not missing,
        missing=missing,
        layers_visited=tuple(visited),
        selected_ids=tuple(m["key"] for m, _ in kept),
        omitted_ids=tuple(omitted),
        sources=tuple(sorted({s for m, _ in kept for s in m["source_refs"]})),
        corpus_revision=corpus.revision,
        files_read=reads,
        cache_hit=False,
        elapsed_ms=(perf_counter() - started) * 1000,
        files_opened=tuple(sorted(seen_paths)),
    )
    controller.last_result = result
    return result
