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

"""Rule-driven progressive memory retrieval with a hard context budget."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path
from time import perf_counter
from typing import Any

from rpent.memory.memcodex.cache import RetrievalCache
from rpent.memory.memcodex.compressor import ContextCompressor
from rpent.memory.memcodex.hierarchy import LAYERS, MemoryEntry, MemoryHierarchy
from rpent.memory.memcodex.policy import (
    ActionQuery,
    RoutingContext,
    RoutingDecision,
    manual_route,
)
from rpent.memory.memcodex.retriever import LayerRetriever
from rpent.memory.memcodex.router import RetrievalQuery, RetrievalRouter, Sufficiency


@dataclass(frozen=True)
class RetrievalResult:
    """Context plus enough measurements to audit routing and budget decisions."""

    context: str
    token_count: int
    token_budget: int
    counter_name: str
    candidate_tokens: int
    sufficient: bool
    missing: tuple[str, ...]
    layers_visited: tuple[str, ...]
    selected_ids: tuple[str, ...]
    omitted_ids: tuple[str, ...]
    sources: tuple[str, ...]
    corpus_revision: str
    files_read: int
    cache_hit: bool
    elapsed_ms: float
    files_opened: tuple[str, ...] = ()
    memory_read_complete: bool | None = None
    action_preconditions_met: bool | None = None
    action_missing_facts: tuple[str, ...] = ()
    action_unverified_facts: tuple[str, ...] = ()
    routing_policy: str | None = None
    routing_decision: str | None = None
    routing_reason: str | None = None


class MemCodexController:
    """Read summary, relations, skills and raw evidence only as needed.

    Create one controller per episode/thread. The default counter uses UTF-8
    bytes conservatively; inject a local model tokenizer for exact token counts.
    Rule sufficiency is not a task-success prediction or a replacement for evals.

    Args:
        memory_root: Existing corpus with JSON files in the four layer folders.
        token_budget: Hard maximum for returned memory context, including headers.
        token_counter: Pure deterministic local text-to-token-count function.
        counter_name: Identifier to retain alongside profiling measurements.
        cache_size: Maximum cached results; zero disables caching.
        max_per_layer: Maximum applicable candidates retained from each layer.
            Ignored in fixed mode, which returns all task-matched entries.
        mode: ``m2`` uses a validated task index and current observation/state;
            ``progressive`` is the original standalone prototype; ``fixed`` returns
            all summary, relational and skill entries in stable order, without
            routing, early stopping or caching. Budget overflow raises an error.
        index_path: Required immutable metadata index for M2; no full-layer scans.
        route: Pure read decision for candidate-specific M2 requests.
        route_name: Stable trace label for a supplied routing policy.
    """

    def __init__(
        self,
        memory_root: str | Path,
        token_budget: int,
        *,
        token_counter: Callable[[str], int] | None = None,
        counter_name: str | None = None,
        cache_size: int = 128,
        max_per_layer: int = 8,
        mode: str = "progressive",
        index_path: str | Path | None = None,
        route: Callable[[RoutingContext], RoutingDecision] = manual_route,
        route_name: str | None = None,
    ) -> None:
        if mode not in {"progressive", "fixed", "m2"}:
            raise ValueError("mode must be progressive, fixed or m2")
        self.mode = mode
        self.memory_root = Path(memory_root).expanduser().resolve()
        self.compressor = ContextCompressor(
            token_budget, token_counter=token_counter, counter_name=counter_name
        )
        self.hierarchy = MemoryHierarchy(self.memory_root)
        self.router = RetrievalRouter()
        self.retriever = LayerRetriever(self.router, max_per_layer=max_per_layer)
        self.route = route
        self.route_name = route_name or getattr(
            route, "__qualname__", type(route).__name__
        )
        self.indexed = None
        if mode == "m2":
            from rpent.memory.memcodex.indexed import IndexedCorpus

            if index_path is None:
                raise ValueError("M2 requires index_path")
            self.indexed = IndexedCorpus(self.memory_root, index_path)
        self.cache: RetrievalCache[RetrievalResult] = RetrievalCache(
            0 if mode in {"fixed", "m2"} else cache_size
        )
        self.last_result: RetrievalResult | None = None

    @property
    def token_budget(self) -> int:
        """Maximum memory tokens configured for this controller."""
        return self.compressor.token_budget

    def is_sufficient(
        self, entries: Sequence[MemoryEntry], query: RetrievalQuery
    ) -> bool:
        """Return the conservative rule decision for already packed evidence."""
        return self.router.assess(entries, query).sufficient

    def retrieve(
        self,
        task: str,
        observation: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        *,
        action_query: ActionQuery | None = None,
    ) -> str:
        """Return memory context; inspect last_result for missing evidence."""
        return self.retrieve_with_trace(
            task, observation, state, action_query=action_query
        ).context

    def retrieve_with_trace(
        self,
        task: str,
        observation: Mapping[str, Any] | None = None,
        state: Mapping[str, Any] | None = None,
        *,
        action_query: ActionQuery | None = None,
    ) -> RetrievalResult:
        """Escalate only when the context that fits is insufficient.

        Cache hits still count against the context budget when sent to a model.
        The cache saves retrieval work, not prompt tokens.
        """
        started = perf_counter()
        if self.mode == "m2":
            from rpent.memory.memcodex.coarse import retrieve_indexed

            return retrieve_indexed(
                self, task, observation, state, action_query=action_query
            )
        if action_query is not None:
            raise ValueError("An action query requires indexed M2 retrieval")
        if self.mode == "fixed":
            return self._retrieve_fixed(task, started)
        query = self.router.query(task, observation, state)
        snapshot = self.hierarchy.snapshot()
        key = (
            snapshot.revision,
            query.cache_key(),
            self.token_budget,
            self.compressor.counter_name,
        )
        cached = self.cache.get(key)
        if cached is not None:
            self.last_result = replace(
                cached,
                cache_hit=True,
                files_read=0,
                elapsed_ms=(perf_counter() - started) * 1000,
            )
            return self.last_result
        candidates: list[MemoryEntry] = []
        visited = []
        files_read = 0
        packed = self.compressor.pack(candidates, query)
        verdict: Sufficiency = self.router.assess(packed.entries, query)
        if self.token_budget:
            for layer in LAYERS:
                visited.append(layer)
                entries = self.hierarchy.read_layer(layer, snapshot)
                files_read += len(entries)
                candidates.extend(self.retriever.retrieve(entries, query, candidates))
                packed = self.compressor.pack(candidates, query)
                verdict = self.router.assess(packed.entries, query)
                if verdict.sufficient:
                    break
        if self.hierarchy.snapshot().revision != snapshot.revision:
            raise RuntimeError(
                "Memory corpus changed during retrieval; retry against a stable corpus"
            )
        self.last_result = RetrievalResult(
            packed.text,
            packed.token_count,
            self.token_budget,
            self.compressor.counter_name,
            packed.candidate_tokens,
            verdict.sufficient,
            verdict.missing,
            tuple(visited),
            tuple(e.key for e in packed.entries),
            packed.omitted,
            tuple(e.source for e in packed.entries),
            snapshot.revision,
            files_read,
            False,
            (perf_counter() - started) * 1000,
        )
        self.cache.put(key, self.last_result)
        return self.last_result

    def _retrieve_fixed(self, task: str, started: float) -> RetrievalResult:
        """Return every statically mapped non-raw entry, without state routing.

        An undersized budget is an error, not silent truncation. Sufficiency is
        deliberately unassessed: M1 does not infer localization or task success.
        """
        query = self.router.query(task, None, None)
        snapshot = self.hierarchy.snapshot()
        selected = []
        files_read = 0
        layers = LAYERS[:3]
        for layer in layers:
            entries = self.hierarchy.read_layer(layer, snapshot)
            files_read += len(entries)
            selected.extend(
                sorted(
                    (
                        e
                        for e in entries
                        if query.task in e.task_ids or "*" in e.task_ids
                    ),
                    key=lambda e: e.id,
                )
            )
        context = "\n\n".join(entry.render() for entry in selected)
        count = self.compressor.count(context)
        if count > self.token_budget:
            raise ValueError(
                f"Fixed memory requires {count} {self.compressor.counter_name}; "
                f"budget is {self.token_budget}. No entries were truncated."
            )
        if self.hierarchy.snapshot().revision != snapshot.revision:
            raise RuntimeError("Memory corpus changed during fixed retrieval")
        self.last_result = RetrievalResult(
            context,
            count,
            self.token_budget,
            self.compressor.counter_name,
            count,
            False,
            ("not_assessed_in_fixed_mode",),
            layers,
            tuple(e.key for e in selected),
            (),
            tuple(e.source for e in selected),
            snapshot.revision,
            files_read,
            False,
            (perf_counter() - started) * 1000,
        )
        return self.last_result
