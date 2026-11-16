# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from robots.libero.goal_memory_resolver import (
    RelationEvidence,
    TaskScope,
    resolve_relation,
    task_family_relation_evidence,
    validate_candidate_relation,
)


def _swap_t6_memory(sign=1):
    return RelationEvidence(
        relation="right_of",
        axis="y",
        sign=sign,
        suite="libero_10_swap",
        task_id=6,
        target_object="chocolate_pudding",
        reference_object="plate",
        successful=True,
        source_id="task-family_libero10_swap_t6",
    )


def _scope():
    return TaskScope(
        suite="libero_10_swap",
        task_id=6,
        target_object="chocolate_pudding",
        reference_object="plate",
    )


def test_exact_swap_t6_success_overrides_conflicting_generic_right():
    resolution = resolve_relation(
        scope=_scope(),
        relation="right_of",
        generic=("y", -1),
        memories=[_swap_t6_memory(+1)],
    )
    assert resolution.axis == "y"
    assert resolution.sign == 1
    assert resolution.source == "exact_task_success"
    assert resolution.conflict is True
    assert resolution.source_id == "task-family_libero10_swap_t6"


def test_sibling_suite_memory_cannot_override_generic_convention():
    sibling = RelationEvidence(
        relation="right_of",
        axis="y",
        sign=1,
        suite="libero_10_task",
        task_id=6,
        target_object="chocolate_pudding",
        reference_object="plate",
        successful=True,
        source_id="task-family_libero10_task_t6",
    )
    resolution = resolve_relation(
        scope=_scope(),
        relation="right_of",
        generic=("y", -1),
        memories=[sibling],
        require_exact_success=True,
    )
    assert resolution.source == "unresolved"
    assert resolution.source_id is None


def test_wrong_target_or_reference_cannot_override_direction():
    wrong_target = RelationEvidence(
        relation="right_of",
        axis="y",
        sign=1,
        suite="libero_10_swap",
        task_id=6,
        target_object="red_mug",
        reference_object="plate",
        successful=True,
        source_id="wrong-target",
    )
    resolution = resolve_relation(
        scope=_scope(),
        relation="right_of",
        generic=("y", -1),
        memories=[wrong_target],
    )
    assert resolution.source == "generic_convention"
    assert resolution.sign == -1


def test_conflicting_exact_task_successes_fail_closed():
    resolution = resolve_relation(
        scope=_scope(),
        relation="right_of",
        generic=("y", -1),
        memories=[_swap_t6_memory(+1), _swap_t6_memory(-1)],
    )
    assert resolution.source == "unresolved"
    assert resolution.conflict is True
    assert validate_candidate_relation(
        resolution=resolution,
        target_coordinate=0.10,
        reference_coordinate=-0.02,
    ) is None


def test_candidate_relation_uses_fresh_coordinates_not_memory_coordinates():
    resolution = resolve_relation(
        scope=_scope(),
        relation="right_of",
        generic=("y", -1),
        memories=[_swap_t6_memory(+1)],
    )
    assert validate_candidate_relation(
        resolution=resolution,
        target_coordinate=0.10,
        reference_coordinate=-0.02,
        minimum_separation=0.02,
    ) is True
    assert validate_candidate_relation(
        resolution=resolution,
        target_coordinate=-0.10,
        reference_coordinate=-0.02,
        minimum_separation=0.02,
    ) is False

def test_missing_exact_task_memory_without_generic_fallback_fails_closed():
    resolution = resolve_relation(
        scope=_scope(),
        relation="right_of",
        generic=("y", -1),
        memories=[],
        require_exact_success=True,
    )
    assert resolution.source == "unresolved"
    assert resolution.source_id is None
    assert validate_candidate_relation(
        resolution=resolution,
        target_coordinate=-0.11,
        reference_coordinate=0.02,
    ) is None


def test_unsuccessful_exact_task_memory_cannot_override_generic():
    unsuccessful = RelationEvidence(
        relation="right_of",
        axis="y",
        sign=1,
        suite="libero_10_swap",
        task_id=6,
        target_object="chocolate_pudding",
        reference_object="plate",
        successful=False,
        source_id="unverified-swap-t6",
    )
    resolution = resolve_relation(
        scope=_scope(),
        relation="right_of",
        generic=None,
        memories=[unsuccessful],
    )
    assert resolution.source == "unresolved"
    assert resolution.source_id is None

def test_indexed_raw_entry_with_preface_binds_exact_swap_t6_success():
    # Runtime extractive entries prepend a notice and strip the source
    # Markdown frontmatter. Provenance is carried by INDEX source_refs.
    body = """Historical seed-0 experience. Re-localize in the current scene.

    ## Applicable pattern
    Place the chocolate-pudding box to the right of the plate.

    ## Winning technique
    Order here was MUG FIRST, then pudding (worked cleanly).
    Carry to the +y (IMAGE-RIGHT) side of the plate.
    """
    metadata = {
        "key": "raw:90fe45a875944dc295362f15",
        "layer": "raw",
        "source_refs": [
            "task-family/task-family_libero10_swap_t6.md#L1-L97"
        ],
    }
    evidence = task_family_relation_evidence(
        metadata=metadata,
        body=body,
        suite="libero_10_swap",
        task_id=6,
        target_object="chocolate_pudding",
        reference_object="plate",
        relation="right_of",
    )
    assert evidence is not None
    assert evidence.suite == "libero10/swap"
    assert evidence.sign == 1
    assert evidence.source_id == "raw:90fe45a875944dc295362f15"


def test_sibling_task_or_non_raw_entry_cannot_bind_exact_swap_t6():
    body = (
        "Chocolate pudding to the right of the plate; "
        "worked cleanly; +y (IMAGE-RIGHT)."
    )
    metadata = {
        "key": "skills:wrong-scope",
        "layer": "skills",
        "source_refs": [
            "task-family/task-family_libero10_task_t6.md#L1-L97"
        ],
    }
    evidence = task_family_relation_evidence(
        metadata=metadata,
        body=body,
        suite="libero_10_swap",
        task_id=6,
        target_object="chocolate_pudding",
        reference_object="plate",
        relation="right_of",
    )
    assert evidence is None

