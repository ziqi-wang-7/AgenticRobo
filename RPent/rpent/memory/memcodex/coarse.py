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
from rpent.memory.memcodex.policy import RoutingContext, RoutingDecision, manual_route

if TYPE_CHECKING:
    from rpent.memory.memcodex.controller import MemCodexController, RetrievalResult
    from rpent.memory.memcodex.policy import ActionQuery


_ROUTING_ENVELOPES = (
    ("STOP", "evidence_covered"),
    ("DESCEND", "continue_memory_read"),
    ("DESCEND", "layers_exhausted"),
    ("OBSERVE", "missing_action_facts"),
    ("REVIEW", "unverified_action_facts"),
    ("REVIEW", "memory_entry_review"),
    *(
        (action, "fixed_evidence_contract")
        for action in ("STOP", "DESCEND", "OBSERVE", "REVIEW")
    ),
)


def retrieve_indexed(
    controller: MemCodexController,
    task: str,
    observation: Mapping[str, Any] | None,
    state: Mapping[str, Any] | None,
    *,
    action_query: ActionQuery | None = None,
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
    required_objects = profile["required_objects"]
    available = set(state.get("available_skills", ()))
    confirmed = set(state.get("applicable_entries", ()))
    rejected = set(state.get("rejected_entries", ()))
    if confirmed & rejected:
        raise ValueError("Memory entries cannot be both confirmed and rejected")
    action_missing_facts: tuple[str, ...] = ()
    action_unverified_facts: tuple[str, ...] = ()
    if action_query is not None:
        required = {action_query.tool}
        required_objects = sorted(
            {name for _, name in action_query.roles}
            | {
                key.removeprefix("localized:")
                for key in action_query.required_facts
                if key.startswith("localized:")
            }
        )
        present = {fact.key for fact in action_query.facts}
        verified = {
            fact.key for fact in action_query.facts if fact.status == "verified"
        }
        action_missing_facts = tuple(
            k for k in action_query.required_facts if k not in present
        )
        action_unverified_facts = tuple(
            k for k in action_query.required_facts if k in present and k not in verified
        )
    facts = state.get("conditions", {})
    kept: list[tuple[dict, str]] = []
    visited, omitted, seen_paths = [], [], set()
    reads = candidate_tokens = 0

    def relevant_guidance(metadata: dict) -> bool:
        return (
            "skill" in metadata["provides"]
            and bool(required.intersection(metadata["skill_ids"]))
        ) or (recovery and "recovery" in metadata["provides"])

    def review_entries(items: list[tuple[dict, str]]) -> tuple[str, ...]:
        return tuple(
            m["key"]
            for m, _ in items
            if (m["review_required"] or m["match"] == "analogy")
            and m["key"] not in confirmed
            and (action_query is None or relevant_guidance(m))
        )

    def assess(items: list[tuple[dict, str]]) -> tuple[str, ...]:
        usable = [
            m
            for m, _ in items
            if not (m["review_required"] or m["match"] == "analogy")
            or m["key"] in confirmed
        ]
        missing = []
        if action_query is None and not any(
            m["layer"] == "summary" and m["match"] == "exact" for m in usable
        ):
            # The authoritative instruction still defines the task when no exact
            # experience exists; it never claims that an analogous task was solved.
            if profile["layers"]["summary"]:
                missing.append("task_summary")
        if action_query is not None or purpose != "overview" or recovery:
            if action_query is None and not set(required_objects) <= localized:
                missing.append("object_localization")
            skills = {
                s
                for m in usable
                if action_query is None or "skill" in m["provides"]
                for s in m["skill_ids"]
            } & available
            if not required <= skills:
                missing.append("skill_guidance")
            if recovery and not any("recovery" in m["provides"] for m in usable):
                missing.append("recovery_guidance")
        if not usable:
            missing.append("applicable_memory")
        return tuple(missing)

    def combined_missing(memory_missing: tuple[str, ...]) -> tuple[str, ...]:
        if action_missing_facts or action_unverified_facts:
            return (*memory_missing, "action_preconditions")
        return memory_missing

    def render_data(items: list[tuple[dict, str]]) -> tuple[dict, str]:
        memory_missing = assess(items)
        missing = combined_missing(memory_missing)
        info = {
            "task": task,
            "task_language": profile["language"],
            "purpose": purpose,
            "required_objects": required_objects,
            "required_skills": profile["required_skills"]
            if action_query is None
            else sorted(required),
            "sufficient": not missing,
            "missing": missing,
            "review_entries": review_entries(items),
            "exact_summary_available": bool(profile["layers"]["summary"]),
        }
        if action_query is not None:
            info.update(
                candidate_tool=action_query.tool,
                candidate_step=action_query.step,
                candidate_fingerprint=action_query.fingerprint,
                memory_read_complete=not memory_missing,
                action_preconditions_met=not (
                    action_missing_facts or action_unverified_facts
                ),
                action_missing_facts=action_missing_facts,
                action_unverified_facts=action_unverified_facts,
                routing_policy=controller.route_name,
            )
        return info, "\n\n".join(t for _, t in items)

    def render(info: dict, blocks: str, routing: tuple[str, str] | None = None) -> str:
        if routing is not None:
            action, reason = routing
            info = {**info, "routing_decision": action, "routing_reason": reason}
        header = (
            "CURRENT QUERY (sufficiency is evidence coverage, not task success):\n"
            + json.dumps(info, ensure_ascii=False)
        )
        return (
            header
            + "\nHistorical coordinates are NOT current targets. Verify conditions and re-localize.\n\n"
            + blocks
        )

    def fits_text(text: str) -> bool:
        # ToolResult truncates at 60,000 bytes. Never let that cut a retained skill.
        wire = json.dumps({"memory_context": text}, indent=2, ensure_ascii=True)
        return (
            len(wire) <= 60000
            and controller.compressor.count(text) <= controller.token_budget
        )

    def fits(items: list[tuple[dict, str]]) -> bool:
        info, blocks = render_data(items)
        if action_query is None:
            return fits_text(render(info, blocks))
        # The policy only runs once per layer. Reserve every final status before
        # admitting evidence because tokenizer costs need not follow label lengths.
        return all(
            fits_text(render(info, blocks, routing)) for routing in _ROUTING_ENVELOPES
        )

    if not fits([]):
        raise ValueError("Memory budget cannot fit the query/status envelope")
    routing = None
    for layer in LAYERS:
        visited.append(layer)
        candidates = corpus.candidates(task, layer, tuple(required), recovery)
        eligible = [
            m
            for m in candidates
            if m["key"] not in rejected
            and (
                action_query is None
                or layer not in {"skills", "raw"}
                or relevant_guidance(m)
            )
            and all(
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
            if fits(proposal):
                kept = proposal
            else:
                omitted.append(entry.key)
            if not assess(kept):
                break
        missing = assess(kept)
        if action_query is not None:
            context = RoutingContext(
                layer=layer,
                query=action_query,
                memory_missing=missing,
                action_missing_facts=action_missing_facts,
                action_unverified_facts=action_unverified_facts,
                review_entries=review_entries(kept),
                selected_ids=tuple(m["key"] for m, _ in kept),
                recovery=recovery,
            )
            routing = _candidate_route(controller, context)
            if routing[0] != "DESCEND":
                break
            continue
        if not missing:
            break
        if layer == "summary" and purpose == "overview" and not recovery:
            break
        if layer == "relational" and "object_localization" in missing:
            break  # Reading historical raw coordinates cannot repair absent perception.
        if layer == "skills" and review_entries(kept):
            break  # Have the existing planner review candidates before escalating.
    corpus.check_revision()
    context = render(*render_data(kept), routing=routing)
    if routing is not None and not fits_text(context):
        raise ValueError("Memory routing status exceeded the reserved response budget")
    memory_missing = assess(kept)
    missing = combined_missing(memory_missing)
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
        memory_read_complete=not memory_missing if action_query is not None else None,
        action_preconditions_met=not (action_missing_facts or action_unverified_facts)
        if action_query is not None
        else None,
        action_missing_facts=action_missing_facts,
        action_unverified_facts=action_unverified_facts,
        routing_policy=controller.route_name if action_query is not None else None,
        routing_decision=routing[0] if routing else None,
        routing_reason=routing[1] if routing else None,
    )
    controller.last_result = result
    return result


def _candidate_route(
    controller: MemCodexController, context: RoutingContext
) -> tuple[str, str]:
    """Check a pure proposal against the fixed evidence contract."""
    proposal = controller.route(context)
    if not isinstance(proposal, RoutingDecision):
        raise TypeError("The memory routing policy must return a RoutingDecision")
    action = proposal.action
    if (
        (
            action == "STOP"
            and not (context.memory_read_complete and context.action_preconditions_met)
        )
        or (action == "OBSERVE" and not context.action_missing_facts)
        or (
            action == "REVIEW"
            and not (context.review_entries or context.action_unverified_facts)
        )
        or (
            action == "DESCEND"
            and context.memory_read_complete
            and not context.action_preconditions_met
        )
    ):
        return manual_route(context).action, "fixed_evidence_contract"
    if action == "OBSERVE":
        return action, "missing_action_facts"
    if action == "REVIEW":
        return (
            action,
            "unverified_action_facts"
            if context.action_unverified_facts
            else "memory_entry_review",
        )
    if action == "STOP":
        return action, "evidence_covered"
    return (
        action,
        "layers_exhausted" if context.layer == "raw" else "continue_memory_read",
    )
