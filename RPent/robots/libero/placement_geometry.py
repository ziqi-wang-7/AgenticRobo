# Copyright 2026 The RPent Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Current-scene placement support and object-path clearance checks.

These checks consume current-step back-projection evidence. They are geometric
screening checks, not a substitute for simulator contact/BDDL verification or
a robot-link collision model.
"""
from __future__ import annotations

import math
from collections.abc import Sequence


def _xy(point: Sequence[float]) -> tuple[float, float]:
    if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
        raise ValueError("geometry points must be finite xyz vectors")
    return float(point[0]), float(point[1])


def estimate_xy_radius(
    center_xyz: Sequence[float], boundary_xyz: Sequence[Sequence[float]]
) -> float:
    """Estimate a conservative XY radius from current-scene boundary samples."""
    cx, cy = _xy(center_xyz)
    if len(boundary_xyz) < 3:
        raise ValueError("at least three boundary samples are required")
    radii = []
    for point in boundary_xyz:
        x, y = _xy(point)
        radii.append(math.hypot(x - cx, y - cy))
    radius = max(radii)
    if radius <= 1e-4:
        raise ValueError("estimated XY radius is degenerate")
    return radius


def validate_support_footprint(
    *,
    object_base_xyz: Sequence[float],
    support_center_xyz: Sequence[float],
    object_boundary_xyz: Sequence[Sequence[float]],
    support_boundary_xyz: Sequence[Sequence[float]],
    edge_margin_m: float = 0.005,
    max_height_delta_m: float = 0.012,
) -> dict:
    """Check whether the measured object-base footprint fits inside support.

    Circular conservative approximation: center offset + object radius +
    requested margin must not exceed the support radius. This does not verify
    vertical contact, friction, tilt, or the environment's native On predicate.
    """
    if not math.isfinite(edge_margin_m) or edge_margin_m < 0:
        raise ValueError("edge_margin_m must be finite and non-negative")
    if not math.isfinite(max_height_delta_m) or max_height_delta_m <= 0:
        raise ValueError("max_height_delta_m must be positive and finite")
    ox, oy = _xy(object_base_xyz)
    sx, sy = _xy(support_center_xyz)
    object_radius = estimate_xy_radius(object_base_xyz, object_boundary_xyz)
    support_radius = estimate_xy_radius(support_center_xyz, support_boundary_xyz)
    center_offset = math.hypot(ox - sx, oy - sy)
    required = center_offset + object_radius + edge_margin_m
    height_delta = abs(float(object_base_xyz[2]) - float(support_center_xyz[2]))
    xy_valid = required <= support_radius
    height_valid = height_delta <= max_height_delta_m
    return {
        "object_base_xyz": list(map(float, object_base_xyz)),
        "support_center_xyz": list(map(float, support_center_xyz)),
        "base_to_support_z_delta_m": float(object_base_xyz[2]) - float(support_center_xyz[2]),
        "absolute_base_height_delta_m": height_delta,
        "max_height_delta_m": max_height_delta_m,
        "xy_footprint_valid": xy_valid,
        "base_height_consistent": height_valid,
        "placement_geometry_valid": xy_valid and height_valid,
        "object_radius_m": object_radius,
        "support_radius_m": support_radius,
        "center_offset_m": center_offset,
        "edge_margin_m": edge_margin_m,
        "required_support_radius_m": required,
        "footprint_inside_support": xy_valid,
        "check_type": "conservative_xy_circle",
        "limitations": [
            "height_consistency_is_only_a_screen_not_contact_proof",
            "does_not_verify_object_uprightness",
            "does_not_verify_native_bddl_predicate",
        ],
    }


def _point_segment_distance_3d(
    point: Sequence[float], start: Sequence[float], end: Sequence[float]
) -> float:
    p = tuple(map(float, point))
    a = tuple(map(float, start))
    b = tuple(map(float, end))
    ab = tuple(b[i] - a[i] for i in range(3))
    ap = tuple(p[i] - a[i] for i in range(3))
    denom = sum(v * v for v in ab)
    if denom <= 1e-12:
        return math.dist(p, a)
    t = max(0.0, min(1.0, sum(ap[i] * ab[i] for i in range(3)) / denom))
    closest = tuple(a[i] + t * ab[i] for i in range(3))
    return math.dist(p, closest)


def validate_object_path_clearance(
    *,
    path_xyz: Sequence[Sequence[float]],
    obstacle_center_xyz: Sequence[float],
    obstacle_boundary_xyz: Sequence[Sequence[float]],
    moving_object_radius_m: float,
    clearance_margin_m: float = 0.015,
) -> dict:
    """Screen a planned moving-object centerline against one localized obstacle.

    This checks the carried object's path only. It does not model the robot
    housing, wrist, elbow, gripper links, or dynamic tracking error.
    """
    if len(path_xyz) < 2:
        raise ValueError("path_xyz requires at least two waypoints")
    if not math.isfinite(moving_object_radius_m) or moving_object_radius_m <= 0:
        raise ValueError("moving_object_radius_m must be positive and finite")
    if not math.isfinite(clearance_margin_m) or clearance_margin_m < 0:
        raise ValueError("clearance_margin_m must be finite and non-negative")
    path = [list(map(float, _xy3(point))) for point in path_xyz]
    obstacle = list(_xy3(obstacle_center_xyz))
    obstacle_radius = estimate_xy_radius(obstacle_center_xyz, obstacle_boundary_xyz)
    distances = [
        _point_segment_distance_3d(obstacle, path[i], path[i + 1])
        for i in range(len(path) - 1)
    ]
    minimum_distance = min(distances)
    required = obstacle_radius + moving_object_radius_m + clearance_margin_m
    return {
        "minimum_centerline_distance_m": minimum_distance,
        "obstacle_radius_m": obstacle_radius,
        "moving_object_radius_m": moving_object_radius_m,
        "clearance_margin_m": clearance_margin_m,
        "required_distance_m": required,
        "object_path_clear": minimum_distance >= required,
        "check_type": "3d_object_centerline_to_obstacle_sphere",
        "limitations": [
            "does_not_model_robot_housing_or_arm_links",
            "does_not_verify_dynamic_collision",
            "requires_independent_robot_link_clearance_assessment",
        ],
    }


def _xy3(point: Sequence[float]) -> tuple[float, float, float]:
    if len(point) != 3 or not all(math.isfinite(float(v)) for v in point):
        raise ValueError("geometry points must be finite xyz vectors")
    return float(point[0]), float(point[1]), float(point[2])


def validate_containment_aabb(
    *,
    object_boundary_xyz: Sequence[Sequence[float]],
    interior_boundary_xyz: Sequence[Sequence[float]],
    clearance_margin_m: float = 0.003,
) -> dict:
    """Check whether current object boundary samples fit inside a container AABB.

    This is a conservative geometric screen only. It does not prove the object
    crossed the opening, is supported, or satisfies the simulator's native In().
    Interior samples must describe the actual usable cavity, not the outer shell.
    """
    if len(object_boundary_xyz) < 4 or len(interior_boundary_xyz) < 4:
        raise ValueError("containment requires at least four object and interior samples")
    if not math.isfinite(clearance_margin_m) or clearance_margin_m < 0:
        raise ValueError("clearance_margin_m must be finite and non-negative")
    obj = [_xy3(point) for point in object_boundary_xyz]
    interior = [_xy3(point) for point in interior_boundary_xyz]
    obj_min = [min(point[i] for point in obj) for i in range(3)]
    obj_max = [max(point[i] for point in obj) for i in range(3)]
    cavity_min = [min(point[i] for point in interior) for i in range(3)]
    cavity_max = [max(point[i] for point in interior) for i in range(3)]
    margins = [
        min(obj_min[i] - cavity_min[i], cavity_max[i] - obj_max[i])
        for i in range(3)
    ]
    contained = all(
        obj_min[i] >= cavity_min[i] + clearance_margin_m
        and obj_max[i] <= cavity_max[i] - clearance_margin_m
        for i in range(3)
    )
    return {
        "object_min_xyz": obj_min,
        "object_max_xyz": obj_max,
        "interior_min_xyz": cavity_min,
        "interior_max_xyz": cavity_max,
        "axis_clearance_m": margins,
        "clearance_margin_m": clearance_margin_m,
        "geometric_containment": contained,
        "check_type": "sampled_axis_aligned_interior_bounds",
        "limitations": [
            "does_not_verify_path_through_opening",
            "does_not_verify_door_state",
            "does_not_verify_native_bddl_in_predicate",
        ],
    }
