# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

import pytest

from robots.libero.placement_geometry import (
    estimate_xy_radius,
    validate_object_path_clearance,
    validate_support_footprint,
)


def test_mug_base_footprint_fits_inside_plate_support():
    result = validate_support_footprint(
        object_base_xyz=[0.0, 0.0, 0.45],
        support_center_xyz=[0.01, 0.0, 0.44],
        object_boundary_xyz=[
            [0.025, 0.0, 0.45],
            [-0.025, 0.0, 0.45],
            [0.0, 0.025, 0.45],
            [0.0, -0.025, 0.45],
        ],
        support_boundary_xyz=[
            [0.10, 0.0, 0.44],
            [-0.10, 0.0, 0.44],
            [0.0, 0.10, 0.44],
            [0.0, -0.10, 0.44],
        ],
        edge_margin_m=0.005,
    )
    assert result["footprint_inside_support"] is True
    assert result["object_radius_m"] == pytest.approx(0.025)
    assert result["support_radius_m"] == pytest.approx(0.10)
    assert "does_not_verify_vertical_contact" in result["limitations"]


def test_mug_base_footprint_outside_plate_support_is_rejected():
    result = validate_support_footprint(
        object_base_xyz=[0.09, 0.0, 0.45],
        support_center_xyz=[0.0, 0.0, 0.44],
        object_boundary_xyz=[
            [0.12, 0.0, 0.45],
            [0.06, 0.0, 0.45],
            [0.09, 0.03, 0.45],
            [0.09, -0.03, 0.45],
        ],
        support_boundary_xyz=[
            [0.10, 0.0, 0.44],
            [-0.10, 0.0, 0.44],
            [0.0, 0.10, 0.44],
            [0.0, -0.10, 0.44],
        ],
    )
    assert result["footprint_inside_support"] is False


def test_object_path_clearance_detects_nearby_mug_collision():
    obstacle_center = [0.0, 0.0, 0.50]
    obstacle_boundary = [
        [0.04, 0.0, 0.50],
        [-0.04, 0.0, 0.50],
        [0.0, 0.04, 0.50],
        [0.0, -0.04, 0.50],
    ]
    blocked = validate_object_path_clearance(
        path_xyz=[[-0.10, 0.0, 0.50], [0.10, 0.0, 0.50]],
        obstacle_center_xyz=obstacle_center,
        obstacle_boundary_xyz=obstacle_boundary,
        moving_object_radius_m=0.025,
        clearance_margin_m=0.015,
    )
    clear = validate_object_path_clearance(
        path_xyz=[[-0.10, 0.0, 0.65], [0.10, 0.0, 0.65]],
        obstacle_center_xyz=obstacle_center,
        obstacle_boundary_xyz=obstacle_boundary,
        moving_object_radius_m=0.025,
        clearance_margin_m=0.015,
    )
    assert blocked["object_path_clear"] is False
    assert clear["object_path_clear"] is True
    assert "does_not_model_robot_housing_or_arm_links" in clear["limitations"]


def test_geometry_rejects_insufficient_boundary_samples():
    with pytest.raises(ValueError, match="at least three"):
        estimate_xy_radius([0.0, 0.0, 0.0], [[0.1, 0.0, 0.0]])
