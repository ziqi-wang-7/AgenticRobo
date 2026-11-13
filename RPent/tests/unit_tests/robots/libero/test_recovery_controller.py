# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from robots.libero.recovery_controller import (
    MAX_ATTEMPTS_PER_STRATEGY,
    RecoveryBudget,
    apply_recovery_budget,
    select_recovery_plan,
)


def test_grasp_failures_route_to_grasp_state_machine():
    plan = select_recovery_plan(["grasp_not_closed", "object_tipped"])
    assert plan.strategy == "grasp"
    assert plan.state_machine == (
        "REOBSERVE_OBJECT_POSE",
        "CHECK_GRASP_GEOMETRY",
        "CHANGE_GRASP_POSE",
        "GRASP_ATTEMPT",
        "VERIFY_GRIPPER_CLOSURE",
        "SHORT_LIFT",
        "VERIFY_OBJECT_FOLLOWS",
    )
    assert plan.execution_authorized is False


def test_spatial_failures_route_to_spatial_state_machine():
    plan = select_recovery_plan(["placement_not_satisfied", "object_collision"])
    assert plan.strategy == "spatial"
    assert "CHECK_SWEPT_PATH_AND_CLEARANCE" in plan.state_machine
    assert "SELECT_DIRECT_CORRECTION_OR_REGRASP" in plan.state_machine
    assert plan.execution_authorized is False


def test_contact_failures_route_to_contact_state_machine():
    plan = select_recovery_plan(["insertion_blocked"])
    assert plan.strategy == "contact"
    assert "VERIFY_CONTACT" in plan.state_machine
    assert "SHORT_PUSH" in plan.state_machine
    assert "VERIFY_INSERTION_DEPTH" in plan.state_machine


def test_unknown_failure_routes_to_observe_not_guess():
    plan = select_recovery_plan([])
    assert plan.strategy == "observe"
    assert plan.state_machine == ("REOBSERVE", "CLASSIFY_FAILURE")
    assert plan.execution_authorized is False


def test_recovery_budget_stops_after_bounded_strategy_attempts():
    budget = RecoveryBudget()
    plans = [
        apply_recovery_budget(select_recovery_plan(["grasp_not_closed"]), budget)
        for _ in range(MAX_ATTEMPTS_PER_STRATEGY + 1)
    ]
    assert [plan.strategy for plan in plans] == [
        "grasp", "grasp", "grasp", "stop"
    ]
    assert plans[-1].reason == "recovery_budget_exhausted:grasp"
    assert budget.snapshot()["total_attempts"] == MAX_ATTEMPTS_PER_STRATEGY
