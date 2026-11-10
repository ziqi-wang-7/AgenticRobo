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

"""Frozen LIBERO reset state, including fixtures outside MuJoCo qpos.

Snapshots are private evaluation inputs, never planner tools. They are applied
at the ordinary set_init_state boundary, before RLinf's settling steps.
"""

from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np


def array_digest(value: Any) -> str:
    """Hash shape and dtype as well as array values."""
    array = np.ascontiguousarray(value)
    header = json.dumps([array.dtype.str, array.shape]).encode()
    return hashlib.sha256(header + array.tobytes()).hexdigest()


def model_arrays(model: Any) -> dict[str, np.ndarray]:
    """Capture numeric model arrays, including fixed-body poses and geometry."""
    return {
        name: value.copy()
        for name in dir(model)
        if isinstance(value := getattr(model, name), np.ndarray)
        and value.dtype.kind in "biuf"
    }


def check_arrays(actual: dict, expected: dict) -> None:
    """Fail closed when a frozen model or observation no longer agrees."""
    if actual.keys() != expected.keys():
        raise ValueError("Frozen scene array inventory changed")
    changed = [
        name
        for name in actual
        if array_digest(actual[name]) != array_digest(expected[name])
    ]
    if changed:
        raise ValueError(f"Frozen scene mismatch: {', '.join(changed[:12])}")


def seeded_scene_factory(factory, *, seed: int, path: str, mode: str, identity: dict):
    """Wrap an RLinf worker factory without modifying installed packages."""
    # LIBERO samplers use global numpy; robosuite 1.5 additionally has env.rng.
    random.seed(seed)
    np.random.seed(seed)
    env = factory()
    original_seed, original_init = env.seed, env.set_init_state

    def seed_all(value):
        random.seed(value)
        np.random.seed(value)
        return original_seed(value)

    def set_initial(init_state):
        original_init(init_state)
        import mujoco

        sim = env.env.sim
        model, data = sim.model._model, sim.data._data
        spec = mujoco.mjtState.mjSTATE_INTEGRATION
        state = np.empty(mujoco.mj_stateSize(model, spec))
        mujoco.mj_getState(model, data, state, spec)
        arrays = model_arrays(model)
        metadata = {
            "schema": 1,
            "identity": identity,
            "mujoco": mujoco.__version__,
            "input_state_sha256": array_digest(init_state),
            "state_spec": int(spec),
        }
        snapshot = Path(path)
        if mode == "record":
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            with snapshot.open("xb") as stream:
                np.savez_compressed(
                    stream,
                    metadata=json.dumps(metadata),
                    state=state,
                    **{"model_" + k: v for k, v in arrays.items()},
                )
        else:
            with np.load(snapshot, allow_pickle=False) as saved:
                if json.loads(str(saved["metadata"])) != metadata:
                    raise ValueError(
                        "Frozen scene identity, init state or simulator version changed"
                    )
                expected = {
                    k[6:]: saved[k] for k in saved.files if k.startswith("model_")
                }
                # LIBERO reset writes fixture transforms into the model; these
                # are absent from pruned_init. All other model arrays must agree.
                for name in ("body_pos", "body_quat"):
                    if arrays[name].shape != expected[name].shape:
                        raise ValueError("Frozen scene model layout changed")
                    getattr(model, name)[:] = expected[name]
                check_arrays(model_arrays(model), expected)
                if saved["state"].shape != state.shape:
                    raise ValueError("Frozen scene state layout changed")
                state = saved["state"].copy()
        # Recording and restoration must run the same forward/observable path;
        # otherwise warm-start and sensor caches can differ before settling.
        mujoco.mj_setState(model, data, state, spec)
        sim.forward()
        env._post_process()
        env._update_observables(force=True)
        return env.env._get_observations()

    env.seed = seed_all
    env.set_init_state = set_initial
    seed_all(seed)
    return env


def validate_observation(
    facade, path: str, mode: str, receipt_path: str | None
) -> dict:
    """Check settled state and both RGB-D cameras before giving control to a planner."""
    snapshot = Path(path)
    observed = {}
    camera_arrays = {}
    for name, value in facade.raw_obs().items():
        if isinstance(value, np.ndarray):
            observed["obs/" + name] = array_digest(value)
            if name.endswith("_image"):
                camera_arrays["obs/" + name] = value
    for camera in ("agentview", "robot0_eye_in_hand"):
        rgb, depth = facade.render_camera(camera, height=1024, width=1024, depth=True)
        camera_arrays[camera + "/rgb"] = rgb
        camera_arrays[camera + "/depth"] = depth
        observed[camera + "/rgb"] = array_digest(rgb)
        observed[camera + "/depth"] = array_digest(depth)
        observed[camera + "/calibration"] = hashlib.sha256(
            json.dumps(
                facade.get_camera_meta(camera, height=1024, width=1024),
                sort_keys=True,
                default=lambda a: np.asarray(a).tolist(),
            ).encode()
        ).hexdigest()
    evidence = {
        "snapshot_sha256": hashlib.sha256(snapshot.read_bytes()).hexdigest(),
        "observations": observed,
    }
    reference = snapshot.with_suffix(".observation.json")
    rgb_differences = {}
    if mode == "record":
        with snapshot.with_suffix(".cameras.npz").open("xb") as stream:
            np.savez_compressed(stream, **camera_arrays)
        with reference.open("x") as stream:
            json.dump(evidence, stream, indent=2)
    else:
        expected = json.loads(reference.read_text())
        if evidence != expected:
            changed = [
                k for k in observed if observed[k] != expected["observations"].get(k)
            ]
            if (
                evidence["snapshot_sha256"] != expected["snapshot_sha256"]
                or observed.keys() != expected["observations"].keys()
            ):
                raise ValueError(
                    "Frozen scene snapshot or observation inventory changed"
                )
            with np.load(
                snapshot.with_suffix(".cameras.npz"), allow_pickle=False
            ) as cameras:
                for key in list(changed):
                    if not (key.endswith("/rgb") or key.endswith("_image")):
                        continue
                    current, original = camera_arrays[key], cameras[key]
                    if (
                        current.shape != original.shape
                        or current.dtype != np.uint8
                        or original.dtype != np.uint8
                        or array_digest(original) != expected["observations"][key]
                    ):
                        continue
                    difference = np.abs(
                        current.astype(np.int16) - original.astype(np.int16)
                    )
                    maximum, fraction = (
                        int(difference.max()),
                        float(np.count_nonzero(difference) / difference.size),
                    )
                    rgb_differences[key] = {
                        "max_channel_error": maximum,
                        "changed_channel_fraction": fraction,
                    }
                    # EGL can round a few color channels by one uint8 level.
                    # No tolerance is granted to physics, depth or calibration.
                    if maximum <= 1 and fraction <= 0.0001:
                        changed.remove(key)
            if changed and receipt_path:
                diagnostic = Path(receipt_path).with_suffix(".mismatch.npz")
                diagnostic.parent.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(diagnostic, **camera_arrays)
            if changed:
                raise ValueError(f"Initial settled observation mismatch: {changed}")
    receipt = {
        **evidence,
        "matched": True,
        "mode": mode,
        "rgb_differences": rgb_differences,
        "rgb_tolerance": {
            "max_channel_error": 1,
            "max_changed_channel_fraction": 0.0001,
        },
        "physics_depth_calibration_exact": True,
    }
    if receipt_path:
        target = Path(receipt_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt
