# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for SelfCollisionKinematicsCfg.create_from_sphere_pair_distances."""

# Third Party
import torch

# CuRobo
from curobo._src.robot.types.self_collision_params import SelfCollisionKinematicsCfg


def _no_pairs(num_spheres: int) -> torch.Tensor:
    return torch.full((num_spheres, num_spheres), -torch.inf)


def test_a_single_collision_link_has_no_pairs_to_check():
    """All spheres on one link: every pair is ignored, so nothing is checked."""
    cfg = SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(
        _no_pairs(4), torch.zeros(4)
    )
    assert cfg.num_spheres == 4
    assert cfg.collision_pairs.shape == (0, 2)


def test_a_robot_without_spheres_has_no_pairs_to_check():
    cfg = SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(
        _no_pairs(0), torch.zeros(0)
    )
    assert cfg.num_spheres == 0
    assert cfg.collision_pairs.shape == (0, 2)


def test_pairs_that_are_not_ignored_are_still_checked():
    d = _no_pairs(3)
    d[0, 2] = d[2, 0] = 0.1
    cfg = SelfCollisionKinematicsCfg.create_from_sphere_pair_distances(d, torch.zeros(3))
    assert cfg.collision_pairs.tolist() == [[0, 2]]
