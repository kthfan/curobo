# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, List, Optional, Tuple

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.geom.collision.wp_robot_mesh_collision import check_robot_mesh_self_collision
from curobo._src.robot.kinematics.kinematics import KinematicsState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_robot_mesh_self_collision_cfg import (
        RobotMeshSelfCollisionConstraintCfg,
    )


class RobotMeshSelfCollisionConstraint(BaseCost):
    """Boolean self-collision constraint between the robot's exact link meshes.

    Like :class:`~curobo._src.cost.cost_robot_mesh_constraint.RobotMeshConstraint`, it returns
    the weight for configurations in which two tested links collide and 0 otherwise, with no
    gradient. The sphere self-collision cost still drives the optimizer.
    """

    def __init__(self, config: RobotMeshSelfCollisionConstraintCfg):
        """Create the constraint. CUDA devices only.

        Args:
            config: Configuration for the constraint.
        """
        super().__init__(config)
        if self.config.robot_mesh_data is None:
            log_and_raise("_robot_mesh_data must be set for RobotMeshSelfCollisionConstraintCfg")
        self._in_collision: Optional[torch.Tensor] = None
        self._contact_pair: Optional[torch.Tensor] = None
        self._pair_near: Optional[torch.Tensor] = None

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        """Allocate the per-configuration output buffers.

        Args:
            batch_size: Number of batch elements.
            horizon: Trajectory horizon.
        """
        if self._batch_size == batch_size and self._horizon == horizon:
            return
        self._batch_size = batch_size
        self._horizon = horizon
        device = self.device_cfg.device
        self._in_collision = torch.zeros(batch_size * horizon, dtype=torch.int32, device=device)
        self._contact_pair = torch.zeros(batch_size * horizon, dtype=torch.int32, device=device)
        pairs = self.config.robot_mesh_data.self_collision_pairs
        num_pairs = 0 if pairs is None else pairs.shape[0]
        self._pair_near = torch.zeros(
            batch_size * horizon * num_pairs, dtype=torch.int32, device=device
        )

    def forward(self, state: KinematicsState) -> torch.Tensor:
        """Compute the mesh self-collision constraint.

        Args:
            state: Kinematics state; must carry ``link_transforms``.

        Returns:
            Constraint value of shape ``(batch, horizon, 1)``: ``weight`` where two links
            collide, 0 otherwise.
        """
        if state.link_transforms is None:
            log_and_raise(
                "state.link_transforms is None; construct Kinematics with "
                "compute_link_transforms=True to use the mesh self-collision constraint"
            )
        link_transforms = state.link_transforms
        batch, horizon = link_transforms.shape[0], link_transforms.shape[1]
        self.setup_batch_tensors(batch, horizon)
        in_collision, _ = check_robot_mesh_self_collision(
            robot_mesh=self.config.robot_mesh_data,
            link_transforms=link_transforms,
            margin=self.config.margin,
            in_collision=self._in_collision,
            contact_pair=self._contact_pair,
            pair_near=self._pair_near,
        )
        cost = in_collision.view(batch, horizon, 1).to(dtype=link_transforms.dtype)
        return cost * self._weight[0]

    def get_contact_pair_names(self) -> List[List[Optional[Tuple[str, str]]]]:
        """Return the colliding link pairs from the most recent :meth:`forward` call.

        Returns:
            Nested list of shape ``(batch, horizon)`` holding the names of two colliding
            links, or ``None`` where the configuration is free of self-collision.
        """
        if self._contact_pair is None:
            log_and_raise("forward() must be called before get_contact_pair_names()")
        names = self.config.robot_mesh_data.self_collision_pair_names(self._contact_pair)
        horizon = self._horizon
        return [names[i * horizon : (i + 1) * horizon] for i in range(self._batch_size)]
