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

"""Run-scoped file access for fixed memory supplied to a planner."""

from __future__ import annotations

from pathlib import Path

from rpent.memory.manager import MemoryManager
from rpent.utils.config import get_repo_root


class FixedMemoryManager(MemoryManager):
    """Expose the fixed view, robot guides and this run's artifacts only.

    Codex uses ``native_file_access=False`` to disable native command execution;
    MCP reads enforce the same run boundary after resolving symlinks.
    """

    native_file_access = False
    file_tool_description = (
        " This run already received its fixed memory context. "
        "File access is limited to that view, robot guides, and this run's artifacts."
    )

    def __init__(
        self, root: str | Path, *, output_dir: str | Path, guides: str | Path
    ) -> None:
        super().__init__(root)
        self.output_dir = Path(output_dir).resolve()
        self.guides = Path(guides).resolve()

    def _authorize(self, path: str | Path, *, write: bool) -> Path:
        resolved = (get_repo_root() / path).resolve()
        if resolved.is_relative_to(self.root):
            if write:
                raise PermissionError("Fixed memory is read-only")
            if resolved == self.root or resolved == self.root / "MEMORY.md":
                return resolved
            raise PermissionError("Only the prepared fixed memory view is available")
        if resolved.is_relative_to(self.output_dir):
            if write and resolved.name in {
                "m1-context.json",
                "telemetry.jsonl",
                "metrics.json",
            }:
                raise PermissionError(
                    "Evaluation measurements are read-only to the planner"
                )
            return resolved
        if not write and resolved.is_relative_to(self.guides):
            return resolved
        raise PermissionError(
            "File access is outside this run's memory, guides or artifacts"
        )
