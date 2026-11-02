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

"""Pack intact evidence blocks under a measured context budget."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from rpent.memory.memcodex.hierarchy import LAYERS, MemoryEntry
from rpent.memory.memcodex.router import RetrievalQuery


def utf8_token_upper_bound(text: str) -> int:
    """Use UTF-8 bytes as a conservative bound for byte-based tokenizers.

    This is not a model token measurement. Supply a local tokenizer counter
    for exact counts. No tokenizer download or LLM call occurs by default.
    """
    return len(text.encode("utf-8"))


@dataclass(frozen=True)
class PackedContext:
    """The exact rendered context and the evidence that survived its budget."""

    text: str
    token_count: int
    candidate_tokens: int
    entries: tuple[MemoryEntry, ...]
    omitted: tuple[str, ...]


class ContextCompressor:
    """Extractive packing: never cut a condition or an instruction in half."""

    def __init__(
        self,
        token_budget: int,
        *,
        token_counter: Callable[[str], int] | None = None,
        counter_name: str | None = None,
    ) -> None:
        if type(token_budget) is not int or token_budget < 0:
            raise ValueError("token_budget must be a nonnegative integer")
        self.token_budget = token_budget
        self.token_counter = token_counter or utf8_token_upper_bound
        self.counter_name = counter_name or (
            "custom" if token_counter else "utf8_byte_upper_bound"
        )

    def count(self, text: str) -> int:
        """Validate counter results before making any budget decision."""
        count = self.token_counter(text)
        if (
            type(count) is not int
            or count < 0
            or (text and count == 0)
            or (not text and count != 0)
        ):
            raise ValueError(
                "token_counter must return zero for empty text and a positive "
                "integer for nonempty text; exclude model special tokens"
            )
        return count

    def pack(
        self, entries: Sequence[MemoryEntry], query: RetrievalQuery
    ) -> PackedContext:
        """Count headers, conditions and separators, giving recovery priority."""

        def rank(entry: MemoryEntry) -> tuple:
            urgent = int(
                (query.needs_recovery or query.anomaly) and "recovery" in entry.provides
            )
            urgent += int(query.anomaly and "anomaly" in entry.provides)
            return (-urgent, LAYERS.index(entry.layer))

        kept, omitted, seen = [], [], set()
        text = ""
        for entry in sorted(entries, key=rank):
            signature = (
                entry.text,
                entry.task_ids,
                entry.applies_when,
                entry.provides,
                entry.skill_ids,
                entry.objects,
            )
            if signature in seen:
                omitted.append(entry.key)
                continue
            seen.add(signature)
            candidate = "\n\n".join(e.render() for e in (*kept, entry))
            if self.count(candidate) <= self.token_budget:
                kept.append(entry)
                text = candidate
            else:
                omitted.append(entry.key)
        count = self.count(text)
        if count > self.token_budget:
            raise ValueError("token_counter must consistently count identical strings")
        return PackedContext(
            text,
            count,
            self.count("\n\n".join(e.render() for e in entries)),
            tuple(kept),
            tuple(omitted),
        )
