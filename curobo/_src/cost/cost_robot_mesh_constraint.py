# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, List, Optional

# Third Party
import torch

# CuRobo
from curobo._src.cost.cost_base import BaseCost
from curobo._src.geom.collision.wp_robot_mesh_collision import check_robot_mesh_collision
from curobo._src.robot.kinematics.kinematics import KinematicsState
from curobo._src.util.logging import log_and_raise

if TYPE_CHECKING:
    # CuRobo
    from curobo._src.cost.cost_robot_mesh_constraint_cfg import RobotMeshConstraintCfg


class RobotMeshConstraint(BaseCost):
    """Boolean collision constraint using the robot's exact triangle geometry.

    This is a *constraint*, not a differentiable cost: it returns 1.0 for configurations that
    collide and 0.0 otherwise, with no gradient. The sphere-based collision cost remains
    responsible for driving the optimizer; this term only rejects configurations whose
    sphere approximation is too coarse to catch a real collision.
    """

    def __init__(self, config: RobotMeshConstraintCfg):
        """Create an exact robot-mesh collision constraint. CUDA devices only.

        Args:
            config: Configuration for the constraint.
        """
        super().__init__(config)
        if self.config._scene_collision_checker is None:
            log_and_raise("_scene_collision_checker must be set for RobotMeshConstraintCfg")
        if self.config._robot_mesh_data is None:
            log_and_raise("_robot_mesh_data must be set for RobotMeshConstraintCfg")
        self._in_collision: Optional[torch.Tensor] = None
        self._contact_link: Optional[torch.Tensor] = None

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
        self._in_collision = torch.zeros(
            batch_size * horizon, dtype=torch.int32, device=self.device_cfg.device
        )
        self._contact_link = torch.zeros(
            batch_size * horizon, dtype=torch.int32, device=self.device_cfg.device
        )

    def forward(
        self,
        state: KinematicsState,
        idxs_env_query: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute the exact mesh collision constraint.

        Args:
            state: Kinematics state; must carry ``link_transforms``, which requires the
                kinematics model to be built with ``compute_link_transforms=True``.
            idxs_env_query: Environment index per batch element.

        Returns:
            Constraint value of shape ``(batch, horizon)``: ``weight`` where the
            configuration collides, 0 otherwise.
        """
        if state.link_transforms is None:
            log_and_raise(
                "state.link_transforms is None; construct Kinematics with "
                "compute_link_transforms=True to use the mesh collision constraint"
            )

        link_transforms = state.link_transforms
        batch, horizon = link_transforms.shape[0], link_transforms.shape[1]
        self.setup_batch_tensors(batch, horizon)

        if idxs_env_query is None:
            use_multi_env = False
            idxs_env_query = torch.zeros(
                batch, dtype=torch.int32, device=link_transforms.device
            )
        else:
            use_multi_env = True

        in_collision, _ = check_robot_mesh_collision(
            robot_mesh=self.config.robot_mesh_data,
            link_transforms=link_transforms,
            scene=self.config.scene_collision_checker.data,
            env_query_idx=idxs_env_query,
            use_multi_env=use_multi_env,
            margin=self.config.margin,
            check_containment=self.config.check_containment,
            in_collision=self._in_collision,
            contact_link=self._contact_link,
        )

        cost = in_collision.view(batch, horizon, 1).to(dtype=link_transforms.dtype)
        return cost * self._weight[0]

    def get_contact_links(self) -> torch.Tensor:
        """Return the contacting link index from the most recent :meth:`forward` call.

        The values are kinematic *body indices*, not positions in
        :attr:`~curobo._src.geom.data.data_robot_mesh.RobotMeshData.link_names`; the two
        differ whenever the kinematic tree holds links without collision meshes. Use
        :meth:`get_contact_link_names` to avoid the ambiguity.

        Returns:
            Tensor of shape ``(batch, horizon)`` holding a colliding link's body index, or
            -1 where the configuration is collision free.
        """
        if self._contact_link is None:
            log_and_raise("forward() must be called before get_contact_links()")
        return self._contact_link.view(self._batch_size, self._horizon)

    def get_contact_link_names(self) -> List[List[Optional[str]]]:
        """Return the contacting link *names* from the most recent :meth:`forward` call.

        Returns:
            Nested list of shape ``(batch, horizon)`` holding a colliding link's name, or
            ``None`` where the configuration is collision free.
        """
        contact_link = self.get_contact_links()
        names = self.config.robot_mesh_data.contact_link_names(contact_link)
        horizon = self._horizon
        return [names[i * horizon : (i + 1) * horizon] for i in range(self._batch_size)]
