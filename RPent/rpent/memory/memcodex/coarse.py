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
from rpent.memory.memcodex.recovery import (
    RAW_FALLBACK_FAILURES,
    covers,
    recovery_cases,
    requirements,
)

if TYPE_CHECKING:
    from rpent.memory.memcodex.controller import MemCodexController, RetrievalResult
    from rpent.memory.memcodex.policy import ActionQuery


_ROUTING_ENVELOPES = (
    ("STOP", "evidence_covered"),
    ("DESCEND", "continue_memory_read"),
    ("DESCEND", "layers_exhausted"),
    ("NARROW", "focus_next_layer"),
    ("OBSERVE", "missing_action_facts"),
    ("REVIEW", "unverified_action_facts"),
    ("REVIEW", "memory_entry_review"),
    ("EXHAUSTED", "no_searchable_recovery_match"),
    *(
        (action, "fixed_evidence_contract")
        for action in ("STOP", "NARROW", "DESCEND", "OBSERVE", "REVIEW", "EXHAUSTED")
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
    required_recovery = requirements(state) if recovery else ()
    unclassified = (
        recovery and bool(str(state.get("issue", "")).strip()) and not required_recovery
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
    state_missing = ()
    if (
        action_query is None
        and (purpose != "overview" or recovery)
        and not set(required_objects) <= localized
    ):
        state_missing = ("object_localization",)
    if action_query is not None:
        state_missing = (*action_missing_facts, *action_unverified_facts)
    facts = state.get("conditions", {})
    kept: list[tuple[dict, str]] = []
    visited, omitted, seen_paths = [], [], set()
    reads = candidate_tokens = 0
    unavailable: set[str] = {"unclassified_issue"} if unclassified else set()

    def searchable_missing(items: list[tuple[dict, str]]) -> tuple[str, ...]:
        return tuple(m for m in assess(items) if m not in unavailable)

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
            for kind in required_recovery:
                if not any(
                    covers(case, kind, state)
                    for m in usable
                    for case in m.get("recovery_cases", ())
                ):
                    missing.append(f"recovery_case:{kind}")
            if unclassified:
                missing.append("unclassified_issue")
        if not usable:
            missing.append("applicable_memory")
        return tuple(missing)

    def combined_missing(memory_missing: tuple[str, ...]) -> tuple[str, ...]:
        if action_missing_facts or action_unverified_facts:
            return (*memory_missing, "action_preconditions")
        return (*memory_missing, *state_missing)

    def render_data(items: list[tuple[dict, str]]) -> tuple[dict, str]:
        memory_missing = assess(items)
        missing = combined_missing(memory_missing)
        info = {
            "task": task,
            "task_language": profile["language"],
            "context_objects": profile.get("context_objects", []),
            "drawer_goal": profile.get("drawer_goal"),
            "recovery_requirements": list(required_recovery),
            "unavailable_recovery": sorted(unavailable),
            "search_exhausted": bool(memory_missing) and not searchable_missing(items),
            "retry_condition": "New failure details, applicability evidence or corpus revision; unchanged queries cannot fill unavailable gaps."
            if unavailable
            else None,
            "failure_context": {
                key: state[key]
                for key in ("failure_kind", "failure_object", "phase")
                if state.get(key)
            },
            "purpose": purpose,
            "required_objects": required_objects,
            "required_skills": profile["required_skills"]
            if action_query is None
            else sorted(required),
            "sufficient": not missing,
            "memory_read_complete": not memory_missing,
            "memory_missing": memory_missing,
            "state_missing": state_missing,
            "state_evidence_complete": not state_missing,
            "missing": missing,
            "review_entries": review_entries(items),
            "exact_summary_available": bool(profile["layers"]["summary"]),
        }
        if action_query is not None:
            info.update(
                candidate_tool=action_query.tool,
                candidate_step=action_query.step,
                candidate_fingerprint=action_query.fingerprint,
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

    def eligible_candidates(layer: str) -> list[dict]:
        return [
            m
            for m in corpus.candidates(task, layer, tuple(sorted(required)), recovery)
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

    # A missing schema-3 case is a retrieval gap, not a reason to dump source
    # files. Legacy indexes lack these annotations: bounded fallback remains
    # possible. Positive metadata never substitutes for a hash-checked body.
    recovery_candidates = (
        [m for layer in LAYERS for m in eligible_candidates(layer)]
        if required_recovery
        else []
    )
    for kind in required_recovery:
        has_indexed_case = any(
            covers(case, kind, state)
            for m in recovery_candidates
            for case in m.get("recovery_cases", ())
        )
        if has_indexed_case:
            continue
        # Spatial/clutter failures may be described in narrative Raw memory
        # without structured index annotations. Defer exhaustion until a
        # bounded Raw pass has checked the hash-verified source body.
        if kind in RAW_FALLBACK_FAILURES:
            continue
        if recovery_candidates and all(
            "recovery_cases" in m for m in recovery_candidates
        ):
            unavailable.add(f"recovery_case:{kind}")
        elif not recovery_candidates:
            unavailable.add(f"recovery_case:{kind}")

    if not fits([]):
        raise ValueError("Memory budget cannot fit the query/status envelope")

    routing = None
    routing_steps = []
    pending_scope: set[str] | None = None
    focus_skills = (
        (action_query.tool,)
        if action_query is not None
        else (state["last_action"],)
        if state.get("last_action") in available
        else tuple(sorted(required))
    )
    for layer_index, layer in enumerate(LAYERS):
        visited.append(layer)
        eligible = eligible_candidates(layer)
        if layer in {"skills", "raw"} and required_recovery:
            eligible.sort(
                key=lambda m: (
                    -sum(
                        covers(case, kind, state)
                        for kind in required_recovery
                        for case in m.get("recovery_cases", ())
                    )
                )
            )
        # Raw is a bounded fallback. Restrict case-only searches to possible
        # matches instead of serializing unrelated complete source files.
        if (
            layer == "raw"
            and searchable_missing(kept)
            and all(m.startswith("recovery_case:") for m in searchable_missing(kept))
        ):
            allow_narrative_fallback = bool(
                set(required_recovery) & RAW_FALLBACK_FAILURES
            )
            eligible = [
                m
                for m in eligible
                if "recovery_cases" not in m
                or allow_narrative_fallback
                or any(
                    covers(case, kind, state)
                    for kind in required_recovery
                    for case in m["recovery_cases"]
                )
            ]
        first = (
            [m for m in eligible if m["key"] in pending_scope]
            if pending_scope is not None
            else eligible
        )
        fallback = [m for m in eligible if m not in first]
        layer_reads = 0
        read_cap = (
            min(controller.retriever.max_per_layer, 2)
            if layer == "raw"
            else controller.retriever.max_per_layer
        )
        for pass_index, candidates in enumerate((first, fallback)):
            if pass_index:
                if not searchable_missing(kept) or not candidates:
                    break
                routing_steps.append(
                    {
                        "action": "WIDEN",
                        "layer": layer,
                        "reason": "focused_memory_incomplete",
                        "candidate_ids": [m["key"] for m in candidates],
                    }
                )
            for metadata in candidates:
                if layer_reads >= read_cap:
                    break
                remaining = searchable_missing(kept)
                if (
                    layer in {"skills", "raw"}
                    and remaining
                    and all(m.startswith("recovery_case:") for m in remaining)
                    and "recovery_cases" in metadata
                    and not any(
                        covers(case, m.removeprefix("recovery_case:"), state)
                        for m in remaining
                        for case in metadata["recovery_cases"]
                    )
                ):
                    continue
                if metadata["path"] in seen_paths:
                    continue
                seen_paths.add(metadata["path"])
                entry = corpus.read(metadata)
                # Derive from the actual hash-checked body even with a legacy
                # index; untrusted metadata alone cannot certify coverage.
                metadata = {**metadata, "recovery_cases": recovery_cases(entry.text)}
                reads += 1
                layer_reads += 1
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
                if not searchable_missing(kept):
                    break
        missing = assess(kept)
        unread = [m["key"] for m in eligible if m["path"] not in seen_paths]
        if layer == "raw" and required_recovery and not unread:
            # Exhaust only after bounded Raw candidates have been checked.
            # Coverage is derived from each hash-checked body, never metadata alone.
            for kind in required_recovery:
                if not any(
                    covers(case, kind, state)
                    for metadata, _ in kept
                    for case in metadata.get("recovery_cases", ())
                ):
                    unavailable.add(f"recovery_case:{kind}")
            missing = assess(kept)
        if searchable_missing(kept) and layer_reads >= read_cap and unread:
            routing_steps.append(
                {
                    "action": "LIMIT",
                    "layer": layer,
                    "reason": "per_layer_read_cap",
                    "candidate_ids": unread,
                }
            )
        narrow_ids: tuple[str, ...] = ()
        narrow_reason = ""
        next_layer = None
        if searchable_missing(kept) and layer_index + 1 < len(LAYERS):
            next_layer = LAYERS[layer_index + 1]
            next_candidates = eligible_candidates(next_layer)
            focused, narrow_reason = corpus.narrow(
                next_candidates,
                parents=tuple(m["path"] for m, _ in kept),
                skills=focus_skills,
                recovery=recovery,
                issue=str(state.get("issue", "")),
            )
            if 0 < len(focused) < len(next_candidates):
                narrow_ids = tuple(m["key"] for m in focused)
        context = RoutingContext(
            layer=layer,
            query=action_query,
            memory_missing=missing,
            action_missing_facts=action_missing_facts
            if action_query
            else state_missing,
            action_unverified_facts=action_unverified_facts,
            review_entries=tuple(
                m["key"]
                for m, _ in kept
                if m["key"] in review_entries(kept) and relevant_guidance(m)
            ),
            selected_ids=tuple(m["key"] for m, _ in kept),
            recovery=recovery,
            narrow_ids=narrow_ids,
            unavailable=tuple(sorted(unavailable)),
        )
        if action_query is not None:
            routing = _candidate_route(controller, context)
            decision, reason = routing
        else:
            decision = manual_route(context).action
            reason = "memory_coverage_and_current_evidence"
        routing_steps.append(
            {
                "action": decision,
                "layer": layer,
                "reason": narrow_reason if decision == "NARROW" else reason,
                "memory_missing": list(missing),
                "state_missing": list(state_missing),
                **(
                    {"next_layer": next_layer, "candidate_ids": list(narrow_ids)}
                    if decision == "NARROW"
                    else {}
                ),
            }
        )
        if decision not in {"DESCEND", "NARROW"}:
            break
        pending_scope = set(narrow_ids) if decision == "NARROW" else None
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
        memory_read_complete=not memory_missing,
        memory_missing=memory_missing,
        state_missing=state_missing,
        state_evidence_complete=not state_missing,
        routing_steps=tuple(routing_steps),
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
    if context.defer_for_localization and action != "OBSERVE":
        return "OBSERVE", "fixed_evidence_contract"
    if (
        (
            action == "STOP"
            and not (context.memory_read_complete and context.action_preconditions_met)
        )
        or (
            action == "OBSERVE"
            and not (
                context.action_missing_facts
                or context.action_unverified_facts
                or context.defer_for_localization
            )
        )
        or (
            action == "EXHAUSTED"
            and (not context.memory_missing or bool(context.searchable_missing))
        )
        or (
            action == "NARROW"
            and (not context.narrow_ids or context.memory_read_complete)
        )
        or (
            action == "REVIEW"
            and (
                not (context.review_entries or context.action_unverified_facts)
                or (
                    not context.memory_read_complete
                    and context.layer != "raw"
                    and not context.review_entries
                )
            )
        )
        or (
            action == "DESCEND"
            and context.memory_read_complete
            and not context.action_preconditions_met
        )
    ):
        return manual_route(context).action, "fixed_evidence_contract"
    if action == "NARROW":
        return action, "focus_next_layer"
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
    if action == "EXHAUSTED":
        return action, "no_searchable_recovery_match"
    return (
        action,
        "layers_exhausted" if context.layer == "raw" else "continue_memory_read",
    )
