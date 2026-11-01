# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the ``mesh_collision_check`` and ``mesh_self_collision_check`` solver options."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.solver import solver_core_cfg
from curobo._src.solver.solver_core_cfg import (
    MESH_COLLISION_CONSTRAINT_WEIGHT,
    enable_mesh_collision,
    enable_mesh_self_collision,
    resolve_robot_config,
    resolve_yaml_configs,
    uses_link_meshes,
    uses_mesh_collision,
    uses_mesh_self_collision,
)
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.robot import RobotCfg
from curobo._src.util_file import get_task_configs_path, join_path, load_yaml


@pytest.fixture
def metrics_base_dict():
    """Fresh copy of the default metrics rollout config."""
    return load_yaml(join_path(get_task_configs_path(), "metrics_base.yml"))


def _resolve(device_cfg, robot="franka.yml", metrics_rollout="metrics_base.yml", **kwargs):
    return resolve_yaml_configs(
        robot,
        ["trajopt/lbfgs_bspline_trajopt.yml"],
        metrics_rollout,
        "trajopt/transition_bspline_trajopt.yml",
        None,
        device_cfg,
        **kwargs,
    )


def test_metrics_base_does_not_use_mesh_collision(metrics_base_dict):
    assert not uses_mesh_collision(metrics_base_dict)


def test_enable_mesh_collision_adds_constraint(metrics_base_dict):
    enabled = enable_mesh_collision(metrics_base_dict, margin=0.002)
    mesh_cfg = enabled["rollout"]["constraint_cfg"]["mesh_collision_cfg"]
    assert mesh_cfg == {"weight": MESH_COLLISION_CONSTRAINT_WEIGHT, "margin": 0.002}
    assert uses_mesh_collision(enabled)
    # Other constraints are kept and the input is left untouched.
    assert "scene_collision_cfg" in enabled["rollout"]["constraint_cfg"]
    assert not uses_mesh_collision(metrics_base_dict)


def test_enable_mesh_collision_keeps_existing_settings(metrics_base_dict):
    metrics_base_dict["rollout"]["constraint_cfg"]["mesh_collision_cfg"] = {
        "weight": 10.0,
        "margin": 0.01,
        "check_containment": False,
    }
    enabled = enable_mesh_collision(metrics_base_dict, margin=0.0)
    mesh_cfg = enabled["rollout"]["constraint_cfg"]["mesh_collision_cfg"]
    assert mesh_cfg == {"weight": 10.0, "margin": 0.0, "check_containment": False}


def test_enable_mesh_collision_rejects_negative_margin(metrics_base_dict):
    with pytest.raises(Exception):
        enable_mesh_collision(metrics_base_dict, margin=-0.01)


class _FakeRobotCfg:
    """Stand-in for RobotCfg that records the loader options it was created with."""

    def __init__(self, load_collision_meshes: bool):
        self.load_collision_meshes = load_collision_meshes


@pytest.fixture
def fake_robot_cfg(monkeypatch):
    """Replace robot loading, which needs CUDA, with a recorder."""

    def create(data, device_cfg, load_collision_spheres=True, num_envs=1,
               load_collision_meshes=False):
        return _FakeRobotCfg(load_collision_meshes)

    monkeypatch.setattr(solver_core_cfg.RobotCfg, "create", staticmethod(create))


def test_resolve_default_skips_collision_meshes(fake_robot_cfg):
    robot_config, _, metrics_dict, _, _ = _resolve(DeviceCfg())
    assert not uses_mesh_collision(metrics_dict)
    assert not robot_config.load_collision_meshes


def test_resolve_does_not_alias_a_shared_metrics_dict(fake_robot_cfg, metrics_base_dict):
    """Solvers modify the resolved dict, so a dict shared between solvers must be copied."""
    reference = load_yaml(join_path(get_task_configs_path(), "metrics_base.yml"))
    _, _, metrics_dict, _, _ = _resolve(DeviceCfg(), metrics_rollout=metrics_base_dict)
    assert metrics_dict is not metrics_base_dict
    metrics_dict["rollout"]["constraint_cfg"] = object()
    assert metrics_base_dict == reference


def test_resolve_mesh_collision_check_loads_meshes(fake_robot_cfg):
    robot_config, _, metrics_dict, _, _ = _resolve(
        DeviceCfg(), mesh_collision_check=True, mesh_collision_margin=0.001
    )
    mesh_cfg = metrics_dict["rollout"]["constraint_cfg"]["mesh_collision_cfg"]
    assert mesh_cfg["margin"] == 0.001
    assert robot_config.load_collision_meshes


def test_resolve_task_config_with_mesh_constraint_loads_meshes(fake_robot_cfg, metrics_base_dict):
    metrics_base_dict["rollout"]["constraint_cfg"]["mesh_collision_cfg"] = {"weight": 1.0}
    robot_config, _, metrics_dict, _, _ = _resolve(
        DeviceCfg(), metrics_rollout=metrics_base_dict
    )
    assert uses_mesh_collision(metrics_dict)
    assert robot_config.load_collision_meshes


def test_enable_mesh_self_collision_adds_constraint(metrics_base_dict):
    enabled = enable_mesh_self_collision(metrics_base_dict, margin=0.003)
    self_cfg = enabled["rollout"]["constraint_cfg"]["mesh_self_collision_cfg"]
    assert self_cfg == {"weight": MESH_COLLISION_CONSTRAINT_WEIGHT, "margin": 0.003}
    assert uses_mesh_self_collision(enabled) and uses_link_meshes(enabled)
    assert not uses_mesh_collision(enabled)
    assert not uses_link_meshes(metrics_base_dict)


def test_enable_mesh_self_collision_rejects_negative_margin(metrics_base_dict):
    with pytest.raises(Exception):
        enable_mesh_self_collision(metrics_base_dict, margin=-0.01)


def test_resolve_mesh_self_collision_check_loads_meshes(fake_robot_cfg):
    robot_config, _, metrics_dict, _, _ = _resolve(
        DeviceCfg(), mesh_self_collision_check=True, mesh_self_collision_margin=0.002
    )
    self_cfg = metrics_dict["rollout"]["constraint_cfg"]["mesh_self_collision_cfg"]
    assert self_cfg["margin"] == 0.002
    assert not uses_mesh_collision(metrics_dict)
    assert robot_config.load_collision_meshes


@pytest.fixture(scope="module")
def cuda_device_cfg():
    """CUDA device configuration; loading a robot needs CUDA."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


def test_resolve_mesh_collision_check_loads_robot_meshes(cuda_device_cfg):
    robot_config, _, _, _, _ = _resolve(cuda_device_cfg, mesh_collision_check=True)
    meshes = robot_config.kinematics.kinematics_config.link_collision_meshes
    assert meshes is not None and len(meshes) > 0


def test_resolve_rejects_robot_cfg_without_meshes(cuda_device_cfg):
    robot_config = resolve_robot_config("franka.yml", cuda_device_cfg)
    assert robot_config.kinematics.kinematics_config.link_collision_meshes is None
    with pytest.raises(Exception):
        _resolve(cuda_device_cfg, robot=robot_config, mesh_collision_check=True)


def test_resolve_accepts_robot_cfg_with_meshes(cuda_device_cfg):
    robot_config = resolve_robot_config(
        "franka.yml", cuda_device_cfg, load_collision_meshes=True
    )
    assert isinstance(robot_config, RobotCfg)
    resolved_robot, _, _, _, _ = _resolve(
        cuda_device_cfg, robot=robot_config, mesh_collision_check=True
    )
    assert resolved_robot is robot_config
