# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for :attr:`KinematicsState.link_transforms`.

These assert the contract the exact-mesh collision constraint relies on: the forward
kinematics cumulative-transform buffer is a row-major ``[R | t]`` per link, indexed by
internal body index, and it agrees with ``tool_poses`` for links that are tool frames.
"""

# Third Party
import pytest
import torch

from curobo._src.geom.transform import quaternion_to_matrix
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

TOOL_FRAME = "panda_hand"


@pytest.fixture(scope="module")
def cfg():
    robot_data = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    return KinematicsCfg.from_robot_yaml_file(robot_data, [TOOL_FRAME])


def _random_joint_state(kin, batch, horizon, seed=0):
    generator = torch.Generator(device=kin.device_cfg.device).manual_seed(seed)
    q = torch.rand(
        (batch, horizon, kin.dof),
        device=kin.device_cfg.device,
        dtype=kin.device_cfg.dtype,
        generator=generator,
    )
    limits = kin.get_joint_limits().position
    q = limits[0] + q * (limits[1] - limits[0])
    return JointState(position=q)


def test_link_transforms_disabled_by_default(cfg):
    kin = Kinematics(cfg)
    state = kin.compute_kinematics(_random_joint_state(kin, 2, 1))
    assert state.link_transforms is None


def test_link_transforms_shape(cfg):
    kin = Kinematics(cfg, compute_link_transforms=True)
    batch, horizon = 3, 4
    state = kin.compute_kinematics(_random_joint_state(kin, batch, horizon))

    assert state.link_transforms is not None
    num_links = len(cfg.kinematics_config.link_name_to_idx_map)
    assert state.link_transforms.shape == (batch, horizon, num_links, 3, 4)


def test_link_transforms_match_tool_pose(cfg):
    """A link that is also a tool frame must have an identical transform in both paths."""
    kin = Kinematics(cfg, compute_link_transforms=True)
    batch, horizon = 5, 2
    state = kin.compute_kinematics(_random_joint_state(kin, batch, horizon, seed=7))

    idx = cfg.kinematics_config.link_name_to_idx_map[TOOL_FRAME]
    mat = state.link_transforms[:, :, idx]  # [b, h, 3, 4] row-major [R | t]

    pose = state.tool_poses.get_link_pose(TOOL_FRAME)
    expected_position = pose.position.view(batch, horizon, 3)
    expected_rotation = quaternion_to_matrix(pose.quaternion.view(-1, 4)).view(batch, horizon, 3, 3)

    # Translation is column 3 of each of the 3 rows.
    torch.testing.assert_close(mat[..., 3], expected_position, atol=1e-5, rtol=1e-4)
    # Rotation is the leading 3x3 block, row-major.
    torch.testing.assert_close(mat[..., :3], expected_rotation, atol=1e-5, rtol=1e-4)


def test_link_transforms_are_valid_rigid_transforms(cfg):
    """Every link transform must be a proper rotation (orthonormal, det=+1)."""
    kin = Kinematics(cfg, compute_link_transforms=True)
    state = kin.compute_kinematics(_random_joint_state(kin, 2, 1, seed=11))

    rot = state.link_transforms[..., :3].reshape(-1, 3, 3)
    eye = torch.eye(3, device=rot.device, dtype=rot.dtype).expand_as(rot)
    torch.testing.assert_close(rot @ rot.transpose(-1, -2), eye, atol=1e-4, rtol=1e-4)
    torch.testing.assert_close(
        torch.linalg.det(rot), torch.ones(rot.shape[0], device=rot.device, dtype=rot.dtype),
        atol=1e-4, rtol=1e-4,
    )


def test_link_transforms_clone_survives_next_fk_call(cfg):
    """The buffer is reused across FK calls; clone() must decouple from it."""
    kin = Kinematics(cfg, compute_link_transforms=True)
    state_a = kin.compute_kinematics(_random_joint_state(kin, 2, 1, seed=1))
    cloned = state_a.link_transforms.clone()
    aliased = state_a.link_transforms

    kin.compute_kinematics(_random_joint_state(kin, 2, 1, seed=2))

    assert not torch.allclose(cloned, aliased), "second FK call should have overwritten the buffer"
