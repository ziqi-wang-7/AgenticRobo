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

"""Write controller measurements and separately observed episode outcomes."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from rpent.memory.memcodex import RetrievalResult


class MemoryProfiler:
    """Append one process's JSONL measurements without storing memory contents.

    Use a distinct variant for each budget/tokenizer/controller configuration.
    A native episode outcome is recorded once, separately from retrieval calls.
    """

    def __init__(self, path: str | Path, variant: str) -> None:
        if not isinstance(variant, str) or not variant.strip():
            raise ValueError("variant must be a nonempty string")
        self.path = Path(path)
        self.variant = variant

    def _append(self, episode_id: str, data: dict) -> None:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id must be a nonempty string")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(
                json.dumps(
                    {"variant": self.variant, "episode_id": episode_id, **data},
                    ensure_ascii=False,
                )
                + "\n"
            )

    def record(self, episode_id: str, result: RetrievalResult) -> None:
        """Record a retrieval, counting cached context if it will be injected."""
        data = asdict(result)
        data.pop("context")
        self._append(episode_id, {"event": "retrieval", **data})

    def outcome(self, episode_id: str, native_success: bool) -> None:
        """Record the environment verdict, never the controller's sufficiency."""
        if type(native_success) is not bool:
            raise ValueError("native_success must be a boolean")
        self._append(episode_id, {"event": "outcome", "native_success": native_success})
