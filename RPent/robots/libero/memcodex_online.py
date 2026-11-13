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
from robots.libero.recovery_controller import (
    RecoveryBudget,
    apply_recovery_budget,
    select_recovery_plan,
)
from robots.libero.recovery_evidence import (
    DrawerProgress,
    FailureObservation,
    classify_grasp,
    drawer_progress_failure,
    motion_alert,
    validate_failure,
)
from robots.libero.toolkit import LiberoToolkit
from rpent.memory.memcodex import MemCodexController
from rpent.memory.memcodex.adapter import FixedMemoryManager
from rpent.memory.memcodex.recovery import requirements as recovery_requirements
from rpent.robots.robot_spec import RunConfig
from rpent.tools.base import ToolResult, tool
from rpent.utils.config import get_repo_root


MAX_REVIEWED_ENTRIES_PER_FAILURE = 4


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


def _resolve_schema_ref(root: dict[str, Any], node: dict[str, Any]) -> dict[str, Any]:
    """Resolve a local JSON Schema reference emitted for a nested tool model."""
    reference = node.get("$ref")
    if not reference:
        return node
    if not reference.startswith("#/"):
        raise RuntimeError(f"Unsupported external tool-schema reference: {reference}")
    resolved: Any = root
    for component in reference[2:].split("/"):
        component = component.replace("~1", "/").replace("~0", "~")
        resolved = resolved[component]
    if not isinstance(resolved, dict):
        raise RuntimeError(f"Invalid tool-schema reference: {reference}")
    return resolved


def _visual_check_item_schema(
    root: dict[str, Any], node: dict[str, Any]
) -> dict[str, Any] | None:
    """Find array item schema through nullable/union wrappers and local refs."""
    node = _resolve_schema_ref(root, node)
    if isinstance(node.get("items"), dict):
        return _resolve_schema_ref(root, node["items"])
    for key in ("anyOf", "oneOf", "allOf"):
        for option in node.get(key, ()):
            if isinstance(option, dict):
                found = _visual_check_item_schema(root, option)
                if found is not None:
                    return found
    return None


def validate_identity_tool_schema(declaration: Any) -> dict[str, Any]:
    """Fail fast unless runtime and published schemas expose identity checks."""
    schemas = {
        "runtime validator": declaration.args_schema.model_json_schema(),
        "published tool": declaration.input_schema,
    }
    for label, root in schemas.items():
        visual = root.get("properties", {}).get("visual_checks")
        if not isinstance(visual, dict):
            raise RuntimeError(
                f"M2.2 requires visual_checks in the {label} retrieve_memory schema"
            )
        item_schema = _visual_check_item_schema(root, visual)
        if item_schema is None:
            raise RuntimeError(
                f"M2.2 requires array item schema for visual_checks in the {label}"
            )
        properties = item_schema.get("properties", {})
        fact_schema = properties.get("fact", {})
        facts = set(fact_schema.get("enum", ()))
        if "identity" not in facts or not {"value", "status"} <= set(properties):
            raise RuntimeError(
                "M2.2 runtime tool schema is stale: visual_checks must expose "
                "fact=identity plus value and status; restart the LIBERO/MCP server "
                "with the current AgenticRobo checkout."
            )
    return {
        "tool_name": declaration.name,
        "runtime_validator_sha256": hashlib.sha256(
            json.dumps(schemas["runtime validator"], sort_keys=True).encode()
        ).hexdigest(),
        "published_schema_sha256": hashlib.sha256(
            json.dumps(schemas["published tool"], sort_keys=True).encode()
        ).hexdigest(),
        "identity_fact_supported": True,
        "visual_check_fields": ["fact", "object_id", "step", "artifact", "value", "status"],
        "multi_failure_supported": "failures" in schemas["published tool"].get("properties", {}),
        "incremental_delivery_supported": {
            "delivered_entries", "force_full_delivery"
        } <= set(schemas["published tool"].get("properties", {})),
        "skill_specific_recovery_controller": True,
        "recovery_failure_inputs": ["failure", "failures", "issue"],
    }


def _omit_repeated_memory_bodies(context: str) -> str:
    """Return an ID-only receipt for an identical query already delivered."""
    lines = context.splitlines()
    if len(lines) < 3:
        raise ValueError("Cannot compact malformed Memory retrieval context")
    info = json.loads(lines[1])
    selected = list(info.get("selected_ids", ()))
    info.update(
        new_memory_ids=[],
        already_delivered_ids=selected,
        new_memory_tokens=0,
        repeated_query=True,
    )
    historical_index = next(
        (
            index for index, line in enumerate(lines)
            if line.startswith("Historical coordinates")
        ),
        None,
    )
    if historical_index is None:
        raise ValueError("Memory context is missing its evidence preamble")
    header = "\n".join([lines[0], json.dumps(info, ensure_ascii=False)])
    preamble = lines[historical_index]
    body = "\n\n".join(
        f"[MEMORY BODY OMITTED: {memory_id} was already delivered for this "
        "unchanged query. Use force_full_delivery after context compaction.]"
        for memory_id in selected
    )
    return header + "\n" + preamble + "\n\n" + body


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
        if resolved.name in {
            "m2-config.json",
            "m2-retrievals.jsonl",
            "m2-runtime-tool-schema.json",
        }:
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
        self._review_decisions: dict[tuple[int, str, str], dict[str, str]] = {}
        self._retrieval_cache: dict[str, dict[str, Any]] = {}
        self._recovery_budget = RecoveryBudget()
        for name in ("back_project", "segment"):
            original = self._tools[name]
            self.add_tool(
                original.with_handler(partial(self._capture_localization, original)),
                replace=True,
            )
        self.add_tool(self._retrieval_tool())
        schema_audit = validate_identity_tool_schema(self._tools["retrieve_memory"])
        schema_audit["contract"] = "M2.3"
        schema_audit["source_revision"] = SOURCE_REVISION
        (self.memory.output_dir / "m2-runtime-tool-schema.json").write_text(
            json.dumps(schema_audit, indent=2) + "\n",
            encoding="utf-8",
        )

    def _retrieval_tool(self):
        """Publish the full current-step schema for every M2 task.

        The general legacy wrapper only accepts ManipulationVisualCheck and
        silently omits identity/value/status. All M2 tasks must use the unified
        ActionVisualCheck contract; drawer-only fields remain optional.
        """
        return self.retrieve_memory

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
        failures: list[FailureObservation] | None = None,
        delivered_entries: list[str] | None = None,
        force_full_delivery: bool = False,
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
            failure: Optional single failure attributed to a current image.
            failures: Multiple independently attributed causes/objects from current images.
            delivered_entries: IDs whose full bodies remain visible in active context.
            force_full_delivery: Rehydrate full bodies after compaction or uncertain retention.
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
            failures=failures,
            delivered_entries=delivered_entries,
            force_full_delivery=force_full_delivery,
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
        failures: list[FailureObservation] | None = None,
        drawer_progress: DrawerProgress | None = None,
        delivered_entries: list[str] | None = None,
        force_full_delivery: bool = False,
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
            failure: Optional single failure attributed to a current image.
            failures: Multiple independently attributed causes/objects from current images.
            delivered_entries: IDs whose full bodies remain visible in active context.
            force_full_delivery: Rehydrate full bodies after compaction or uncertain retention.
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
            normalized_visual_checks = [
                ActionVisualCheck.model_validate(check, strict=True)
                for check in visual_checks or ()
            ]
            if candidate is not None:
                normalized_candidate, action_query = build_action_query(
                    ActionCandidate.model_validate(candidate, strict=True),
                    tools=self._tools,
                    step=step,
                    task_objects=objects,
                    drawer_goal=profile.get("drawer_goal"),
                    localizations=evidence,
                    artifacts=record.artifacts,
                    visual_checks=normalized_visual_checks,
                )
            elif visual_checks:
                raise ValueError("Visual checks require an action candidate")

            candidate_fingerprint = action_query.fingerprint if action_query else ""
            review_scope = (step, candidate_fingerprint)
            failure_signature = hashlib.sha256(
                json.dumps(
                    {
                        "purpose": purpose,
                        "issue": " ".join(issue.casefold().split()),
                        "failure": (
                            failure.model_dump()
                            if isinstance(failure, FailureObservation)
                            else failure
                        ),
                        "failures": [
                            item.model_dump() if isinstance(item, FailureObservation) else item
                            for item in (failures or ())
                        ],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            decision_scope = (step, candidate_fingerprint, failure_signature)
            confirmed = set(applicable_entries or [])
            rejected = set(rejected_entries or [])
            if confirmed & rejected:
                raise ValueError("An entry cannot be both applicable and rejected")
            reviewable_ids = _reviewable_memory_ids(
                self._memory_seen, step, candidate_fingerprint
            )
            if not (confirmed | rejected) <= reviewable_ids:
                raise ValueError(
                    "Only entries returned by this step's overview or this exact "
                    "candidate may be reviewed"
                )
            previous_decisions = self._review_decisions.setdefault(decision_scope, {})
            new_review_ids = (confirmed | rejected) - set(previous_decisions)
            for entry_id in confirmed:
                previous = previous_decisions.get(entry_id)
                if previous == "rejected":
                    raise ValueError(f"Conflicting repeated Memory review: {entry_id}")
                previous_decisions[entry_id] = "applicable"
            for entry_id in rejected:
                previous = previous_decisions.get(entry_id)
                if previous == "applicable":
                    raise ValueError(f"Conflicting repeated Memory review: {entry_id}")
                previous_decisions[entry_id] = "rejected"
            confirmed.update(
                entry_id for entry_id, decision in previous_decisions.items()
                if decision == "applicable"
            )
            rejected.update(
                entry_id for entry_id, decision in previous_decisions.items()
                if decision == "rejected"
            )
            review_no_progress = bool(previous_decisions) and not new_review_ids
            review_limit_reached = (
                len(previous_decisions) >= MAX_REVIEWED_ENTRIES_PER_FAILURE
                or review_no_progress
            )
            delivered = set()
            if not force_full_delivery:
                delivered = set(delivered_entries or [])
                if not delivered <= reviewable_ids:
                    raise ValueError(
                        "Incremental delivery IDs must have been returned in this step"
                    )
            result = record.result or {}
            action = (record.command or {}).get("action")
            held_object = (
                normalized_candidate.get("roles", {}).get("held_object")
                if normalized_candidate
                else None
            )
            holding_check = next(
                (
                    check
                    for check in normalized_visual_checks
                    if check.fact == "holding" and check.object_id == held_object
                ),
                None,
            )
            grasp_status = (
                classify_grasp(
                    pi0_success=result.get("success")
                    if action == "pi0_pick"
                    else None,
                    visual_status=holding_check.status if holding_check else None,
                )
                if action == "pi0_pick" or holding_check is not None
                else None
            )
            failed = bool(result.get("error") or grasp_status == "failed")
            failure_state = (
                {
                    "failure_kind": "grasp_not_closed",
                    "failure_object": held_object,
                    "phase": "grasp",
                }
                if grasp_status == "failed" and held_object
                else {}
            )
            failure_cases = []
            if failure is not None:
                failure = FailureObservation.model_validate(failure, strict=True)
                failure_cases.append(validate_failure(failure, record, objects))
            normalized_failures = [
                FailureObservation.model_validate(item, strict=True)
                for item in (failures or ())
            ]
            failure_cases.extend(
                validate_failure(item, record, objects)
                for item in normalized_failures
            )
            if failure_cases:
                unique_cases = {
                    (
                        item["failure_kind"],
                        item["failure_object"],
                        item["phase"],
                    ): item
                    for item in failure_cases
                }
                failure_cases = list(unique_cases.values())
                if failure_state:
                    failure_cases = [
                        failure_state,
                        *[
                            item
                            for item in failure_cases
                            if (
                                item["failure_kind"],
                                item["failure_object"],
                                item["phase"],
                            )
                            != (
                                failure_state["failure_kind"],
                                failure_state["failure_object"],
                                failure_state["phase"],
                            )
                        ],
                    ]
                else:
                    failure_state = dict(failure_cases[0])
                failure_state["failure_cases"] = failure_cases
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
                "delivered_entries": sorted(delivered),
                "force_full_delivery": force_full_delivery,
                "review_limit_reached": review_limit_reached,
                "review_no_progress": review_no_progress,
                "recovery_strategy": state.get("recovery_strategy"),
                "recovery_state_machine": state.get("recovery_state_machine", []),
                "recovery_budget": state.get("recovery_budget"),
                "recovery_budget_exhausted": state.get("recovery_budget_exhausted", False),
                "recovery_execution_authorized": False,
                **failure_state,
            }
            if rejected:
                state["rejected_entries"] = sorted(rejected)
            recovery_kinds = (
                recovery_requirements(
                    {
                        "failure_kind": failure_state.get("failure_kind"),
                        "failure_object": failure_state.get("failure_object"),
                        "phase": failure_state.get("phase"),
                        "failure_cases": failure_state.get("failure_cases", ()),
                        "issue": issue,
                    }
                )
                if purpose == "recover" or state["needs_recovery"]
                else ()
            )
            recovery_plan = (
                select_recovery_plan(recovery_kinds)
                if recovery_kinds
                else None
            )
            memory.controller.indexed.check_revision()
            request_fingerprint = hashlib.sha256(
                json.dumps(
                    {
                        "corpus_revision": memory.controller.indexed.revision,
                        "step": step,
                        "purpose": purpose,
                        "candidate_fingerprint": candidate_fingerprint,
                        "action_facts": [
                            asdict(item) for item in action_query.facts
                        ] if action_query else [],
                        "issue": " ".join(issue.casefold().split()),
                        "failure_signature": failure_signature,
                        "last_action": action,
                        "pi0_success_hint": result.get("success"),
                        "drawer_progress": progress_data,
                        "applicable_entries": sorted(confirmed),
                        "rejected_entries": sorted(rejected),
                        "delivered_entries": sorted(delivered),
                        "localization_ids": sorted(bindings.items()),
                        "visual_checks": [
                            item.model_dump() for item in normalized_visual_checks
                        ],
                    },
                    sort_keys=True,
                ).encode()
            ).hexdigest()
            cached = (
                self._retrieval_cache.get(request_fingerprint)
                if not force_full_delivery
                else None
            )
            if cached is not None:
                response_context = _omit_repeated_memory_bodies(cached["context"])
                response = ToolResult(data={"memory_context": response_context})
                cached_info = json.loads(response_context.splitlines()[1])
                trace = {
                    **cached["trace"],
                    "cache_hit": True,
                    "repeated_query": True,
                    "new_review_decisions": 0,
                    "review_no_progress": True,
                    "review_limit_reached": True,
                    "incremental_delivery": {
                        "requested_delivered_ids": cached_info.get("already_delivered_ids", []),
                        "new_memory_ids": [],
                        "already_delivered_ids": cached_info.get("already_delivered_ids", []),
                        "new_memory_tokens": 0,
                        "recovery_coverage": cached_info.get("recovery_coverage", False),
                        "unresolved_failures": cached_info.get("unresolved_failures", []),
                        "force_full_delivery": False,
                    },
                    "files_read": 0,
                    "files_opened": [],
                    "elapsed_ms": 0.0,
                    "response_sha256": hashlib.sha256(
                        response.to_text().encode()
                    ).hexdigest(),
                }
                with (memory.output_dir / "m2-retrievals.jsonl").open("a") as stream:
                    stream.write(
                        json.dumps(trace, ensure_ascii=False, allow_nan=False) + "\n"
                    )
                return response

            budget_plan = (
                apply_recovery_budget(recovery_plan, self._recovery_budget)
                if recovery_plan is not None
                else None
            )
            if budget_plan is not None and budget_plan.strategy == "stop":
                state["recovery_budget_exhausted"] = True
                state["recovery_budget"] = self._recovery_budget.snapshot()
                state["recovery_strategy"] = "stop"
                state["recovery_state_machine"] = list(budget_plan.state_machine)
                state["recovery_execution_authorized"] = False
                # Do not claim retrieval coverage when stopping on budget.
                state["recovery_budget_reason"] = budget_plan.reason
            else:
                state["recovery_budget_exhausted"] = False
                state["recovery_budget"] = self._recovery_budget.snapshot()
                if budget_plan is not None:
                    state["recovery_strategy"] = budget_plan.strategy
                    state["recovery_state_machine"] = list(budget_plan.state_machine)
                    state["recovery_execution_authorized"] = False
            retrieved = memory.controller.retrieve_with_trace(
                memory.record["task"], observation, state, action_query=action_query
            )
            response = ToolResult(data={"memory_context": retrieved.context})
            if "[truncated]" in response.to_text():
                raise RuntimeError("M2 memory response unexpectedly truncated")
            self._memory_seen.setdefault(review_scope, set()).update(
                retrieved.selected_ids
            )
            response_info = json.loads(retrieved.context.splitlines()[1])
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
                "grasp_status": grasp_status,
                "pi0_success_hint": result.get("success") if action == "pi0_pick" else None,
                "failure_observation": failure.model_dump() if failure else None,
                "failure_observations": [
                    item.model_dump() for item in normalized_failures
                ],
                "review_decision_scope": failure_signature,
                "review_decisions": dict(previous_decisions),
                "reviewed_entry_count": len(previous_decisions),
                "review_limit": MAX_REVIEWED_ENTRIES_PER_FAILURE,
                "review_limit_reached": review_limit_reached,
                "review_no_progress": review_no_progress,
                "new_review_decisions": len(new_review_ids),
                "incremental_delivery": {
                    "requested_delivered_ids": sorted(delivered),
                    "new_memory_ids": response_info.get("new_memory_ids", []),
                    "already_delivered_ids": response_info.get("already_delivered_ids", []),
                    "new_memory_tokens": response_info.get("new_memory_tokens", 0),
                    "recovery_coverage": response_info.get("recovery_coverage", False),
                    "unresolved_failures": response_info.get("unresolved_failures", []),
                    "force_full_delivery": force_full_delivery,
                },
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
            if not force_full_delivery:
                self._retrieval_cache[request_fingerprint] = {
                    "context": retrieved.context,
                    "trace": trace,
                }
            return response
