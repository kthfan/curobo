# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for update_tool_pose_bounds across cost manager, solvers and planner (no CUDA)."""

from types import SimpleNamespace

import pytest
import torch

from curobo._src.cost.cost_eef_bound import EEFBoundCost
from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg
from curobo._src.cost.tool_pose_bound import ToolPoseBound
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.rollout.cost_manager.cost_manager_robot import RobotCostManager
from curobo._src.solver.solver_core import SolverCore
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp


@pytest.fixture(scope="module", autouse=True)
def setup_warp():
    init_warp(quiet=True)


class _RecordingRollout:
    def __init__(self, names):
        self.names = names
        self.calls = []

    def get_cost_component_names(self):
        return self.names

    def update_params_cost_managers(self, **kwargs):
        self.calls.append(kwargs)


def _stub_core(has_eef_bound: bool = True) -> SolverCore:
    core = SolverCore.__new__(SolverCore)
    names = ["tool_pose", "eef_bound"] if has_eef_bound else ["tool_pose"]
    core.metrics_rollout = _RecordingRollout(names)
    core.auxiliary_rollout = _RecordingRollout(names)
    core.auxiliary_rollout.transition_model = SimpleNamespace(
        robot_model=SimpleNamespace(tool_frames=["ee"])
    )
    core.additional_metrics_rollouts = {"extra": _RecordingRollout(names)}
    core.optimizer_rollouts = [_RecordingRollout(names), _RecordingRollout(names)]
    core._has_eef_bound_cost = has_eef_bound
    return core


def _all_rollouts(core):
    return (
        [core.metrics_rollout, core.auxiliary_rollout]
        + list(core.additional_metrics_rollouts.values())
        + core.optimizer_rollouts
    )


class TestCostManager:
    def test_update_params_sets_and_clears_bounds(self, cpu_device_cfg):
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=cpu_device_cfg, tool_frames=["ee"])
        manager = RobotCostManager.__new__(RobotCostManager)
        manager._initialized = True
        manager.costs = {"eef_bound": EEFBoundCost(cfg)}
        bound = ToolPoseBound.plane()
        manager.update_params(tool_pose_bounds={"ee": bound})
        assert manager.costs["eef_bound"].tool_pose_bounds["ee"] is bound
        manager.update_params(tool_pose_bounds={"ee": None})
        assert manager.costs["eef_bound"].tool_pose_bounds["ee"] is None

    def test_update_params_without_eef_cost_is_noop(self):
        manager = RobotCostManager.__new__(RobotCostManager)
        manager._initialized = True
        manager.costs = {}
        manager.update_params(tool_pose_bounds={"ee": ToolPoseBound.plane()})


class TestSolverCore:
    def test_propagates_to_every_rollout(self):
        core = _stub_core()
        bounds = {"ee": ToolPoseBound.plane()}
        core.update_tool_pose_bounds(bounds)
        for rollout in _all_rollouts(core):
            assert rollout.calls == [{"tool_pose_bounds": bounds}]

    def test_unknown_tool_frame_raises(self):
        with pytest.raises(ValueError, match="Unknown tool frames"):
            _stub_core().update_tool_pose_bounds({"other": ToolPoseBound.plane()})

    def test_bound_without_eef_cost_raises(self):
        with pytest.raises(ValueError, match="eef_bound"):
            _stub_core(has_eef_bound=False).update_tool_pose_bounds({"ee": ToolPoseBound.plane()})

    def test_clearing_without_eef_cost_is_allowed(self):
        core = _stub_core(has_eef_bound=False)
        core.update_tool_pose_bounds({"ee": None})
        assert core.metrics_rollout.calls == [{"tool_pose_bounds": {"ee": None}}]


class _RecordingSolver:
    def __init__(self):
        self.bounds = []
        self.criteria = []

    def update_tool_pose_bounds(self, bounds):
        self.bounds.append(bounds)

    def update_tool_pose_criteria(self, criteria):
        self.criteria.append(criteria)


def test_mpc_updates_ik_and_core():
    mpc = MPCSolver.__new__(MPCSolver)
    mpc.ik_solver = _RecordingSolver()
    mpc.core = _RecordingSolver()
    bounds = {"ee": ToolPoseBound.plane()}
    mpc.update_tool_pose_bounds(bounds)
    assert mpc.ik_solver.bounds == [bounds] and mpc.core.bounds == [bounds]


class TestMotionPlanner:
    @staticmethod
    def _planner():
        planner = MotionPlanner.__new__(MotionPlanner)
        planner.ik_solver = _RecordingSolver()
        planner.trajopt_solver = _RecordingSolver()
        planner._tool_pose_bounds = {}
        planner._destroyed = True  # nothing to release in __del__
        return planner

    def test_bounds_reach_both_solvers_and_are_tracked(self):
        planner = self._planner()
        bound = ToolPoseBound.plane()
        planner.update_tool_pose_bounds({"ee": bound})
        assert planner.ik_solver.bounds == [{"ee": bound}]
        assert planner.trajopt_solver.bounds == [{"ee": bound}]
        assert planner._tool_pose_bounds == {"ee": bound}
        planner.update_tool_pose_bounds({"ee": None})
        assert planner._tool_pose_bounds == {}

    def test_criteria_update_replaces_tracked_bounds(self, cpu_device_cfg):
        from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria

        planner = self._planner()
        planner.update_tool_pose_bounds({"ee": ToolPoseBound.plane()})
        planner.update_tool_pose_criteria({"ee": ToolPoseCriteria(device_cfg=cpu_device_cfg)})
        assert planner._tool_pose_bounds == {}


@pytest.fixture
def cpu_device_cfg():
    return DeviceCfg(device=torch.device("cpu"), dtype=torch.float32)
