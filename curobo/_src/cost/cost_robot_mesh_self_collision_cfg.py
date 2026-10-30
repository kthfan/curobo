# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration of the exact self-collision constraint between link meshes."""
from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Optional, Type

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_robot_mesh_self_collision import RobotMeshSelfCollisionConstraint
from curobo._src.geom.collision.wp_robot_mesh_collision import check_robot_mesh_self_collision
from curobo._src.geom.data.data_robot_mesh import RobotMeshData
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.state.state_joint import JointState
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.util.logging import log_warn


@dataclass
class RobotMeshSelfCollisionConstraintCfg(BaseCostCfg):
    """Configuration for the exact self-collision constraint between link meshes."""

    class_type: Type[RobotMeshSelfCollisionConstraint] = RobotMeshSelfCollisionConstraint

    #: Clearance two tested links must keep from each other, in meters.
    margin: float = 0.0

    #: Packed robot collision triangles with the tested link pairs. Set from the robot model.
    _robot_mesh_data: Optional[RobotMeshData] = None

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        """Attach the robot's shared exact-mesh geometry, see :meth:`initialize_from_kinematics`.

        Args:
            transition_model: Transition model holding the robot model.
        """
        self.initialize_from_kinematics(transition_model.robot_model)

    def initialize_from_kinematics(self, robot_model: Kinematics):
        """Attach the robot's shared exact-mesh geometry to this constraint.

        The first time the geometry is used for self-collision, the tested pairs are checked at
        the robot's default joint configuration, and a warning names a pair that collides
        there: such pairs usually touch by design and belong in ``mesh_self_collision_ignore``.

        Args:
            robot_model: Robot kinematics. The check at the default configuration runs only
                when it computes link transforms.
        """
        self._robot_mesh_data = robot_model.config.kinematics_config.get_robot_mesh_data()
        data = self._robot_mesh_data
        if getattr(data, "_self_collision_default_checked", False):
            return
        if data.self_collision_pairs is None or not robot_model.compute_link_transforms:
            return
        data._self_collision_default_checked = True
        state = robot_model.compute_kinematics(
            JointState.from_position(
                robot_model.default_joint_position.view(1, -1),
                joint_names=robot_model.joint_names,
            )
        )
        in_collision, contact_pair = check_robot_mesh_self_collision(
            data, state.link_transforms, margin=self.margin
        )
        if in_collision.item() == 1:
            link_a, link_b = data.self_collision_pair_names(contact_pair)[0]
            log_warn(
                f"link meshes {link_a} and {link_b} collide at the default joint configuration "
                "(other pairs may too). The mesh self-collision check rejects every "
                "configuration where they touch; add pairs that touch by design to "
                "mesh_self_collision_ignore."
            )

    @property
    def robot_mesh_data(self) -> RobotMeshData:
        """Packed robot collision triangle data."""
        return self._robot_mesh_data
