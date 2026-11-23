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

"""Dashboard startup memory requirements before shared robot services."""

import argparse
from dataclasses import replace
from types import SimpleNamespace

import pytest

from rpent.cli import dashboard
from rpent.cli.main import _build_argparser
from rpent.dashboard.server import DashboardServer
from rpent.dashboard.state import DashboardState
from rpent.memory import MemoryManager


@pytest.mark.parametrize(
    ("robot", "profile", "sync_before_shared", "explicit_repo"),
    [
        ("robocasa", "hf", True, False),
        ("robocasa", "hf", True, True),
        ("robotwin", "hf", True, False),
        ("robotwin", "hf", True, True),
        ("robocasa", "local", False, False),
        ("libero", "hf", False, False),
        ("libero", "local", False, False),
        ("dual_franka", "local", False, False),
    ],
)
@pytest.mark.parametrize("planner", ["api", "claude_code", "codex"])
def test_shared_runtime_gets_memory_before_startup(
    robot, profile, sync_before_shared, planner, tmp_path, monkeypatch, explicit_repo
):
    root = tmp_path / "memory" / robot
    events = []

    def sync(manager, *, remote_repo):
        assert manager.root == root
        assert remote_repo == "test/memory"
        global_file = root / "global" / "GLOBAL_MEMORY.md"
        global_file.parent.mkdir(parents=True)
        global_file.write_text("Live task observations take precedence.")
        events.append("sync")
        return root

    def init_runtime(args, output_dir, state, components):
        assert components == {"vla"}
        if sync_before_shared:
            assert (root / "global" / "GLOBAL_MEMORY.md").is_file()
        events.append("shared-runtime")
        return [], {}

    def start(server):
        # Exercise actual session orchestration, then exit after service startup.
        server._state.request_shutdown()
        return "http://127.0.0.1:12345"

    def fail_session(state, error):
        pytest.fail(f"Shared service startup failed: {error}")

    monkeypatch.setattr(dashboard, "get_memory_dir", lambda name: root)
    monkeypatch.setattr(MemoryManager, "sync", sync)
    if explicit_repo:
        from rpent.memory import source

        def prepare(source_url, name):
            assert name == robot and source_url == "team/memory@release"
            sync(MemoryManager(root), remote_repo="test/memory")
            return root

        monkeypatch.setattr(source, "prepare_explicit_source", prepare)
    monkeypatch.setattr(DashboardServer, "start", start)
    monkeypatch.setattr(DashboardState, "fail_session", fail_session)
    spec = SimpleNamespace(
        name=robot,
        is_real_robot=robot == "dual_franka",
        supports_human_interactive_exploration=robot == "dual_franka",
        memory_repo_id="test/memory",
        prepare_memory=(lambda args, config: None) if robot == "libero" else None,
        dashboard={
            "task": {
                "command": "/rpent-task",
                "usage": "/rpent-task <seed>",
                "fields": ({"name": "seed", "kind": "integer", "minimum": 0},),
                "display": "seed {seed}",
                "output_slug": "s{seed}",
            },
            "runtime_components": ({"name": "vla", "scope": "shared"},),
            "primitives": (),
        },
        init_runtime=init_runtime,
    )
    args = _build_argparser().parse_args(
        [
            "--robot",
            robot,
            "--dashboard",
            "--planner",
            planner,
            "--model",
            "test",
            "--memory-profile",
            profile,
            "--output-dir",
            str(tmp_path / "run"),
        ]
    )
    if explicit_repo:
        args.memory_repo = "team/memory@release"
    assert (
        dashboard.run_dashboard_session(args, spec, parser=argparse.ArgumentParser())
        == 0
    )
    assert events == (["sync"] if sync_before_shared else []) + ["shared-runtime"]
    if explicit_repo:
        assert args.memory_dir == str(root)


@pytest.mark.parametrize("with_molmo", [False, True])
def test_flash_evolve_dashboard_forwards_session_options(
    tmp_path, monkeypatch, with_molmo
):
    from robots.libero.robot_spec import get_robot_spec
    from rpent.planner.base import PlannerResult

    spec = get_robot_spec()
    parser = _build_argparser()
    spec.add_cli_args(parser, use_dashboard=True)
    store = tmp_path / "learned"
    args = parser.parse_args(
        [
            "--robot",
            "libero",
            "--dashboard",
            "--planner",
            "flash_evolve",
            "--flash-slow-planner",
            "claude_code",
            "--no-flash-evolve",
            "--memory-profile",
            "local",
            "--memory-dir",
            str(tmp_path / "empty"),
            "--flash-memory-dir",
            str(store),
            "--output-dir",
            str(tmp_path / "run"),
            *(["--molmo-endpoint", "http://localhost:1234"] if with_molmo else []),
        ]
    )
    original_ready = DashboardState.shared_services_ready
    calls = []

    def request_one_task(state):
        original_ready(state)
        state.request_task({"suite": "libero_object_task", "task": 0, "seed": 1})

    def init_runtime(args, output_dir, state, components):
        calls.append(components)
        return [], {}

    def build(planner, **kwargs):
        calls.append(
            (
                planner,
                kwargs["flash_slow_planner"],
                kwargs["flash_evolve"],
                kwargs["flash_memory_dir"],
            )
        )
        return SimpleNamespace(solve=lambda **kw: PlannerResult())

    monkeypatch.setattr(DashboardState, "shared_services_ready", request_one_task)
    monkeypatch.setattr(DashboardServer, "start", lambda self: "http://localhost")
    monkeypatch.setattr(dashboard, "build_planner", build)
    monkeypatch.setattr(
        dashboard,
        "get_toolkit",
        lambda *a, **kw: SimpleNamespace(
            memory=None,
            solved=lambda: False,
            close=lambda: None,
            cancel_active_and_wait=lambda: None,
        ),
    )
    # Shut down after the task even when setup fails before planner construction.
    original_complete = DashboardState.complete_task

    def complete(self, **kwargs):
        self.request_shutdown()
        original_complete(self, **kwargs)

    monkeypatch.setattr(DashboardState, "complete_task", complete)
    dashboard.run_dashboard_session(
        args, replace(spec, init_runtime=init_runtime), parser=parser
    )
    assert calls == [
        {"vla", "sam3", *(["molmo"] if with_molmo else [])},
        {"env"},
        ("flash_evolve", "claude_code", False, store),
    ]
