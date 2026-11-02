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

"""Small bounded LRU for immutable retrieval results."""

from __future__ import annotations

from collections import OrderedDict
from typing import Generic, TypeVar

T = TypeVar("T")


class RetrievalCache(Generic[T]):
    """Cache results by corpus revision and all routing inputs; zero disables it."""

    def __init__(self, max_entries: int = 128) -> None:
        if type(max_entries) is not int or max_entries < 0:
            raise ValueError("cache_size must be a nonnegative integer")
        self.max_entries = max_entries
        self._values: OrderedDict[tuple, T] = OrderedDict()

    def get(self, key: tuple) -> T | None:
        """Return a hit and promote it to the most recently used position."""
        if key not in self._values:
            return None
        self._values.move_to_end(key)
        return self._values[key]

    def put(self, key: tuple, value: T) -> None:
        """Bound retained results without writing to the memory corpus."""
        if not self.max_entries:
            return
        self._values[key] = value
        self._values.move_to_end(key)
        while len(self._values) > self.max_entries:
            self._values.popitem(last=False)

    def clear(self) -> None:
        """Discard results after changing an injected tokenizer's configuration."""
        self._values.clear()
