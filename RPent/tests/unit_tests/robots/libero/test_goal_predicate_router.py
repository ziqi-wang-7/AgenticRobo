# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

from robots.libero.goal_predicate_router import route_goal_predicate


def test_on_routes_to_support_not_directional_relation():
    route = route_goal_predicate("On")
    assert route.validator == "support"
    assert "vertical_contact" in route.required_evidence


def test_in_routes_to_containment_not_directional_relation():
    route = route_goal_predicate("In")
    assert route.validator == "containment"
    assert "opening_clearance" in route.required_evidence


def test_right_of_routes_to_spatial_relation():
    route = route_goal_predicate("right_of")
    assert route.validator == "spatial_relation"


def test_unknown_predicate_fails_to_native_verification():
    route = route_goal_predicate("touching")
    assert route.validator == "unsupported"
