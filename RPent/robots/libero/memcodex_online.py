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

"""LIBERO-owned M2 memory tool using current perception-tool evidence."""

from __future__ import annotations

import hashlib
import json
import math
from argparse import Namespace
from dataclasses import asdict
from functools import partial
from pathlib import Path
from threading import RLock
from typing import Any, Literal

from robots.libero.action_evidence import (
    ActionCandidate,
    ActionVisualCheck,
    ManipulationCandidate,
    ManipulationVisualCheck,
    build_action_query,
)
from robots.libero.memcodex import SOURCE_REVISION
from robots.libero.recovery_evidence import (
    DrawerProgress,
    FailureObservation,
    drawer_progress_failure,
    motion_alert,
    validate_failure,
)
from robots.libero.toolkit import LiberoToolkit
from rpent.memory.memcodex import MemCodexController
from rpent.memory.memcodex.adapter import FixedMemoryManager
from rpent.robots.robot_spec import RunConfig
from rpent.tools.base import ToolResult, tool
from rpent.utils.config import get_repo_root


def make_controller(record: dict) -> MemCodexController:
    """Count the actual serialized memory response, including its JSON envelope."""
    import tiktoken

    encoder = tiktoken.get_encoding("o200k_base")

    def count(text: str) -> int:
        if not text:
            return 0
        # Do not use ToolResult.to_text here: it truncates large responses.
        wire = json.dumps({"memory_context": text}, indent=2, ensure_ascii=True)
        return len(encoder.encode(wire, disallowed_special=()))

    return MemCodexController(
        record["corpus"],
        record["token_budget"],
        mode="m2",
        index_path=record["index"],
        token_counter=count,
        counter_name="o200k_base_tool_response_estimate",
        cache_size=0,
    )


def prepare_indexed_memory(args: Namespace, config: RunConfig) -> None:
    """Bind an immutable index without loading memory bodies into the prompt."""
    root, index = (
        Path(args.memcodex_root).resolve(),
        Path(args.memcodex_index).resolve(),
    )
    manifest = json.loads((root / "MANIFEST.json").read_text())
    if (
        manifest.get("schema_version") != 2
        or manifest.get("status") != "converted"
        or manifest.get("source_version") != "GPT_6_astra_low"
        or manifest.get("source_revision") != SOURCE_REVISION
    ):
        raise ValueError("M2 requires the converted official GPT_6_astra_low corpus")
    record = {
        "mode": "m2",
        "task": f"{args.suite}/t{args.task}",
        "corpus": str(root),
        "index": str(index),
        "token_budget": args.memory_token_budget,
        "index_sha256": hashlib.sha256(index.read_bytes()).hexdigest(),
        "corpus_manifest_sha256": hashlib.sha256(
            (root / "MANIFEST.json").read_bytes()
        ).hexdigest(),
        "source_revision": SOURCE_REVISION,
        "cache_enabled": False,
        "memory_update": False,
        "delivery": "retrieve_memory_tool",
        "action_evidence": bool(getattr(args, "memcodex_action_evidence", False)),
        "localization": "current_step_perception_evidence_with_planner_semantic_binding",
    }
    controller = make_controller(record)
    controller.indexed.profile(record["task"])
    output = Path(config.output_dir).resolve()
    view = output / "memory-view"
    view.mkdir(parents=True, exist_ok=False)
    (view / "MEMORY.md").write_text(
        "Use retrieve_memory for indexed task experience.\n"
    )
    (output / "m2-config.json").write_text(json.dumps(record, indent=2) + "\n")
    config.prompt_vars.update(
        memory_mode="m2",
        memory_dir=str(view),
        memcodex_action_evidence=record["action_evidence"],
        memcodex_drawer=bool(
            controller.indexed.profile(record["task"]).get("drawer_goal")
        ),
    )


class IndexedMemoryManager(FixedMemoryManager):
    """Expose guided tool retrieval; keep indexes and measurement logs private."""

    memory_mode = "m2"
    file_tool_description = " Use retrieve_memory for experience. File access is limited to robot guides and this run's public artifacts."

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.record = json.loads((self.output_dir / "m2-config.json").read_text())
        self.controller = make_controller(self.record)
        if self.controller.indexed.revision != self.record["index_sha256"]:
            raise ValueError("M2 index changed after preparation")

    def _authorize(self, path: str | Path, *, write: bool) -> Path:
        resolved = (get_repo_root() / path).resolve()
        if resolved.name in {"m2-config.json", "m2-retrievals.jsonl"}:
            raise PermissionError("M2 configuration and measurements are private")
        return super()._authorize(path, write=write)


def _reviewable_memory_ids(
    memory_seen: dict[tuple[int, str], set[str]],
    step: int,
    candidate_fingerprint: str,
) -> set[str]:
    """Return IDs shown in this step's overview or this exact candidate scope.

    Overview results are discovery candidates and may be reviewed for a
    candidate in the same observation step. Results from another candidate or
    an earlier step remain ineligible.
    """
    allowed = set(memory_seen.get((step, ""), set()))
    if candidate_fingerprint:
        allowed.update(memory_seen.get((step, candidate_fingerprint), set()))
    return allowed


class M2LiberoToolkit(LiberoToolkit):
    """Add indexed retrieval with auditable, step-bound localization evidence."""

    def _register_libero_tools(self) -> None:
        super()._register_libero_tools()
        self._memory_lock = RLock()
        self._localization_evidence: dict[str, dict] = {}
        self._memory_seen: dict[tuple[int, str], set[str]] = {}
        for name in ("back_project", "segment"):
            original = self._tools[name]
            self.add_tool(
                original.with_handler(partial(self._capture_localization, original)),
                replace=True,
            )
        self.add_tool(self._retrieval_tool())

    def _retrieval_tool(self):
        """Publish drawer fields only for indexed drawer tasks."""
        profile = self.memory.controller.indexed.profile(self.memory.record["task"])
        return (
            self.retrieve_memory
            if profile.get("drawer_goal")
            else self._retrieve_general
        )

    def _capture_localization(self, original: Any, **kwargs: Any) -> ToolResult:
        result = original(**kwargs)
        data = result.data
        xyz = data.get("world_xyz", data.get("center_xyz"))
        if (
            not result.is_error
            and data.get("found", True)
            and isinstance(xyz, (list, tuple))
            and len(xyz) == 3
            and all(isinstance(x, (int, float)) and math.isfinite(x) for x in xyz)
            and data.get("step") == self._state.latest_record().step_idx
        ):
            with self._memory_lock:
                evidence_id = f"loc-{len(self._localization_evidence) + 1:04d}"
                self._localization_evidence[evidence_id] = {
                    "step": data["step"],
                    "tool": original.name,
                    "xyz": list(xyz),
                    "camera": data.get("camera"),
                    "source_artifact": data.get("source_artifact")
                    or data.get("image_artifact"),
                    "image_artifact": data.get("image_artifact"),
                    "segment_artifact": data.get("segment_artifact"),
                    "arguments": kwargs,
                }
                data["memory_evidence_id"] = evidence_id
        return result

    def _execute_primitive(self, name: str, handler: Any, **kwargs: Any) -> ToolResult:
        response = super()._execute_primitive(name, handler, **kwargs)
        record = self._state.latest_record()
        if record and not record.terminated:
            alert = motion_alert(record)
            if alert:
                response.data["motion_check"] = {
                    "step": record.step_idx,
                    "target_reached": False,
                    "failure_confirmed": False,
                    "instruction": "Compare intended progress with current images; servo residual alone does not identify a failure.",
                }
        return response

    @tool(name="retrieve_memory", readonly=True)
    def _retrieve_general(
        self,
        step: int,
        purpose: Literal["overview", "execute", "recover"] = "execute",
        localizations: dict[str, str] | None = None,
        applicable_entries: list[str] | None = None,
        issue: str = "",
        candidate: ManipulationCandidate | None = None,
        visual_checks: list[ManipulationVisualCheck] | None = None,
        rejected_entries: list[str] | None = None,
        failure: FailureObservation | None = None,
    ) -> ToolResult:
        """Read indexed guidance; inspect state gaps and unavailable recovery matches.

        Args:
            step: Latest observation step.
            purpose: Summary, execution guidance, or recovery after an observed problem.
            localizations: Object IDs mapped to current memory_evidence_id values.
            applicable_entries: Applicable returned IDs for this step/candidate.
            issue: Observed problem; not a verified environment fact.
            candidate: Proposed primitive, parameters and object roles.
            visual_checks: Current image-attributed holding/placement checks.
            rejected_entries: Inapplicable returned IDs for this step/candidate.
            failure: Optional specific failure attributed to a current image.
        """
        return self.retrieve_memory(
            step=step,
            purpose=purpose,
            localizations=localizations,
            applicable_entries=applicable_entries,
            issue=issue,
            candidate=candidate.model_dump()
            if isinstance(candidate, ManipulationCandidate)
            else candidate,
            visual_checks=[
                v.model_dump() if isinstance(v, ManipulationVisualCheck) else v
                for v in visual_checks or ()
            ],
            rejected_entries=rejected_entries,
            failure=failure,
        )

    @tool(readonly=True)
    def retrieve_memory(
        self,
        step: int,
        purpose: Literal["overview", "execute", "recover"] = "execute",
        localizations: dict[str, str] | None = None,
        applicable_entries: list[str] | None = None,
        issue: str = "",
        candidate: ActionCandidate | None = None,
        visual_checks: list[ActionVisualCheck] | None = None,
        rejected_entries: list[str] | None = None,
        failure: FailureObservation | None = None,
        drawer_progress: DrawerProgress | None = None,
    ) -> ToolResult:
        """Read task experience from summary through raw evidence only as needed.

        Args:
            step: Latest observation step.
            purpose: Summary, execution guidance, or recovery after an observed problem.
            localizations: Object IDs mapped to current memory_evidence_id values.
            applicable_entries: Applicable returned IDs for this step/candidate.
            issue: Observed problem; not a verified environment fact.
            candidate: Proposed primitive, parameters and object roles.
            visual_checks: Current image-attributed manipulation/drawer checks.
            rejected_entries: Inapplicable returned IDs for this step/candidate.
            failure: Optional specific failure attributed to a current image.
            drawer_progress: Drawer progress attributed to consecutive images.
        """
        with self._memory_lock:
            record = self._state.latest_record()
            if step != record.step_idx:
                raise ValueError(
                    f"Stale memory query; latest step is {record.step_idx}"
                )
            memory = self.memory
            profile = memory.controller.indexed.profile(memory.record["task"])
            bindings = localizations or {}
            objects = set(profile["required_objects"]) | set(
                profile.get("context_objects", ())
            )
            if not set(bindings) <= objects:
                raise ValueError("Use required_objects IDs returned by overview")
            if len(set(bindings.values())) != len(bindings):
                raise ValueError(
                    "Different objects require distinct localization evidence"
                )
            evidence = {}
            for name, key in bindings.items():
                item = self._localization_evidence.get(key)
                if item is None or item["step"] != step:
                    raise ValueError(
                        f"Missing or stale perception evidence for {name}: {key}"
                    )
                evidence[name] = {"id": key, **item}
            normalized_candidate, action_query = None, None
            if candidate is not None:
                normalized_candidate, action_query = build_action_query(
                    ActionCandidate.model_validate(candidate, strict=True),
                    tools=self._tools,
                    step=step,
                    task_objects=objects,
                    drawer_goal=profile.get("drawer_goal"),
                    localizations=evidence,
                    artifacts=record.artifacts,
                    visual_checks=[
                        ActionVisualCheck.model_validate(check, strict=True)
                        for check in visual_checks or ()
                    ],
                )
            elif visual_checks:
                raise ValueError("Visual checks require an action candidate")

            review_scope = (step, action_query.fingerprint if action_query else "")
            confirmed = set(applicable_entries or [])
            rejected = set(rejected_entries or [])
            if confirmed & rejected:
                raise ValueError("An entry cannot be both applicable and rejected")
            reviewable_ids = _reviewable_memory_ids(
                self._memory_seen,
                step,
                action_query.fingerprint if action_query else "",
            )
            if not (confirmed | rejected) <= reviewable_ids:
                raise ValueError(
                    "Only entries returned by this step's overview or this exact "
                    "candidate may be reviewed"
                )
            result = record.result or {}
            action = (record.command or {}).get("action")
            failed = bool(
                result.get("error")
                or (action == "pi0_pick" and result.get("success") is False)
            )
            failure_state = {}
            if failure is not None:
                failure = FailureObservation.model_validate(failure, strict=True)
                failure_state = validate_failure(failure, record, objects)
            progress_data = None
            if drawer_progress is not None:
                if not profile.get("drawer_goal"):
                    raise ValueError("Drawer progress requires a drawer task")
                drawer_progress = DrawerProgress.model_validate(
                    drawer_progress, strict=True
                )
                kind = drawer_progress_failure(
                    drawer_progress,
                    record,
                    self._state.get(drawer_progress.previous_step),
                )
                progress_data = drawer_progress.model_dump()
                if kind:
                    if failure_state and failure_state["failure_kind"] != kind:
                        raise ValueError(
                            "Failure observation conflicts with drawer progress"
                        )
                    failure_state = {
                        "failure_kind": kind,
                        "failure_object": f"{drawer_progress.target_level}_drawer_handle",
                        "phase": "articulate",
                    }
            observation = {
                "task_language": record.extras.get("task_language"),
                "localized_objects": sorted(evidence),
            }
            state = {
                "purpose": purpose,
                "available_skills": sorted(self._tools),
                "needs_recovery": failed or bool(failure_state) or purpose == "recover",
                "anomaly": bool(issue),
                "issue": issue,
                "last_action": action,
                "applicable_entries": sorted(confirmed),
                **failure_state,
            }
            if rejected:
                state["rejected_entries"] = sorted(rejected)
            retrieved = memory.controller.retrieve_with_trace(
                memory.record["task"], observation, state, action_query=action_query
            )
            response = ToolResult(data={"memory_context": retrieved.context})
            if "[truncated]" in response.to_text():
                raise RuntimeError("M2 memory response unexpectedly truncated")
            self._memory_seen.setdefault(review_scope, set()).update(
                retrieved.selected_ids
            )
            trace = {
                **asdict(retrieved),
                "step": step,
                "observation": observation,
                "state": state,
                "localization_evidence": evidence,
                "semantic_binding_source": "planner_image_interpretation",
                "issue_report": issue,
                "issue_source": "planner",
                "last_action": action,
                "last_action_failed": failed,
                "failure_observation": failure.model_dump() if failure else None,
                "drawer_progress": progress_data,
                "terminated": record.terminated,
                "truncated": record.truncated,
                "response_sha256": hashlib.sha256(
                    response.to_text().encode()
                ).hexdigest(),
            }
            if action_query is not None:
                trace["candidate"] = normalized_candidate
                trace["action_query"] = asdict(action_query)
            with (memory.output_dir / "m2-retrievals.jsonl").open("a") as stream:
                stream.write(
                    json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n"
                )
            return response
