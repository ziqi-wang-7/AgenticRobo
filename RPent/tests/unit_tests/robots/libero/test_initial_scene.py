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

"""Frozen reset fixtures, state, and observation verification stay fail-closed."""

import json
import sys
from types import SimpleNamespace

import numpy as np
import pytest

from robots.libero.initial_scene import seeded_scene_factory, validate_observation


@pytest.fixture
def scene(tmp_path, monkeypatch):
    def get_state(model, data, output, spec):
        output[:] = data.state

    def set_state(model, data, value, spec):
        data.state[:] = value

    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(
            __version__="fake",
            mjtState=SimpleNamespace(mjSTATE_INTEGRATION=1),
            mj_stateSize=lambda *a: 3,
            mj_getState=get_state,
            mj_setState=set_state,
        ),
    )

    def create(mode="record", fixture=1.0, geometry=2.0, identity=None):
        model = SimpleNamespace(
            body_pos=np.array([[fixture, 0.0, 0.0]]),
            body_quat=np.array([[1.0, 0.0, 0.0, 0.0]]),
            geom_size=np.array([geometry]),
        )
        data = SimpleNamespace(state=np.array([1.0, 2.0, 3.0]))
        simulator = SimpleNamespace(
            model=SimpleNamespace(_model=model),
            data=SimpleNamespace(_data=data),
            forward=lambda: None,
        )
        env = SimpleNamespace(
            env=SimpleNamespace(sim=simulator, _get_observations=lambda: {}),
            seed=lambda value: None,
            set_init_state=lambda value: None,
            _post_process=lambda: None,
            _update_observables=lambda **kwargs: None,
        )
        return (
            seeded_scene_factory(
                lambda: env,
                seed=1,
                path=str(tmp_path / "scene.npz"),
                mode=mode,
                identity=identity or {"task": 3},
            ),
            model,
            data,
        )

    return create, tmp_path / "scene.npz"


def test_restore_includes_fixed_fixture_poses_not_only_qpos(scene):
    create, _ = scene
    env, _, _ = create()
    env.set_init_state(np.array([5.0]))
    restored, model, data = create("restore", fixture=9.0)
    data.state[:] = 7
    restored.set_init_state(np.array([5.0]))
    assert model.body_pos[0, 0] == 1
    np.testing.assert_array_equal(data.state, [1, 2, 3])


@pytest.mark.parametrize(
    "kwargs,initial",
    [({"identity": {"task": 4}}, [5.0]), ({"geometry": 3.0}, [5.0]), ({}, [6.0])],
)
def test_changed_task_model_or_native_init_state_is_rejected(scene, kwargs, initial):
    create, _ = scene
    create()[0].set_init_state(np.array([5.0]))
    with pytest.raises(ValueError, match="Frozen scene"):
        create("restore", **kwargs)[0].set_init_state(np.array(initial))


def test_record_cannot_overwrite_existing_snapshot(scene):
    create, _ = scene
    create()[0].set_init_state(np.array([5.0]))
    with pytest.raises(FileExistsError):
        create()[0].set_init_state(np.array([5.0]))


def test_worker_seed_covers_global_numpy_sampler(scene):
    create, _ = scene
    env = create()[0]
    env.seed(11)
    first = np.random.uniform(size=5)
    env.seed(11)
    np.testing.assert_array_equal(first, np.random.uniform(size=5))


def test_changed_rgb_or_depth_fails_before_planner(scene, tmp_path):
    create, path = scene
    create()[0].set_init_state(np.array([5.0]))
    rgb, depth = np.zeros((2, 2, 3), dtype=np.uint8), np.zeros((2, 2))
    facade = SimpleNamespace(
        raw_obs=lambda: {"robot": np.ones(2)},
        render_camera=lambda *a, **kw: (rgb, depth),
        get_camera_meta=lambda *a, **kw: {"K": [1, 2]},
    )
    validate_observation(facade, str(path), "record", None)
    receipt = tmp_path / "validation.json"
    validate_observation(facade, str(path), "restore", str(receipt))
    assert json.loads(receipt.read_text())["matched"]
    depth[0, 0] = 0.01
    with pytest.raises(ValueError, match="observation mismatch"):
        validate_observation(facade, str(path), "restore", None)


@pytest.mark.parametrize(
    "change,count,accepted", [(1, 1, True), (2, 1, False), (1, 100, False)]
)
def test_rgb_only_allows_sparse_single_level_rounding(scene, change, count, accepted):
    create, path = scene
    create()[0].set_init_state(np.array([5.0]))
    rgb = np.zeros((100, 100, 3), dtype=np.uint8)
    depth = np.zeros((100, 100))
    facade = SimpleNamespace(
        raw_obs=lambda: {"agentview_image": rgb, "robot": np.ones(2)},
        render_camera=lambda *a, **kw: (rgb, depth),
        get_camera_meta=lambda *a, **kw: {"K": [1, 2]},
    )
    validate_observation(facade, str(path), "record", None)
    rgb.flat[:count] = change
    if accepted:
        receipt = validate_observation(facade, str(path), "restore", None)
        assert receipt["matched"] and receipt["physics_depth_calibration_exact"]
        assert receipt["rgb_differences"]["agentview/rgb"]["max_channel_error"] == 1
    else:
        with pytest.raises(ValueError, match="observation mismatch"):
            validate_observation(facade, str(path), "restore", None)


def test_runner_detects_modified_private_scene_reference(tmp_path):
    from repro.liberopro_baseline import sha, validate_scenes

    path = tmp_path / "scene.npz"
    expected = {}
    for suffix in (".npz", ".observation.json", ".cameras.npz"):
        file = path.with_suffix(suffix)
        file.write_bytes(b"frozen")
        expected[suffix] = sha(file)
    plan = {
        "episodes": [{"initial_scene": str(path), "initial_scene_sha256": expected}]
    }
    validate_scenes(plan)
    path.with_suffix(".cameras.npz").write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="Frozen initial scene changed"):
        validate_scenes(plan)
