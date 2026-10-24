# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

# Standard Library
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Type, Union

# CuRobo
from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_robot_mesh_constraint import RobotMeshConstraint
from curobo._src.geom.collision.collision_scene import SceneCollision
from curobo._src.geom.data.data_robot_mesh import (
    DEFAULT_MAX_TRIANGLES_PER_LINK,
    RobotMeshData,
)
from curobo._src.util.logging import log_and_raise
from curobo._src.transition.robot_state_transition import RobotStateTransition
from curobo._src.geom.data.data_robot_mesh import RobotMeshData

@dataclass
class RobotMeshConstraintCfg(BaseCostCfg):
    """Configuration for the exact robot-mesh collision constraint."""

    class_type: Type[RobotMeshConstraint] = RobotMeshConstraint

    #: Clearance the robot geometry must keep from obstacles, in meters. A small positive
    #: value compensates for the shrinkage introduced by per-link mesh decimation.
    margin: float = 0.0

    #: Also flag configurations where a link lies entirely inside a mesh obstacle and so
    #: crosses none of its triangles.
    check_containment: bool = True

    #: Per-link triangle budget applied when packing the robot geometry. Query cost scales
    #: roughly linearly with the total triangle count, so keep this tight. Either one value
    #: for every link or a per-link mapping. Only used when the robot config does not set
    #: ``max_triangles_per_link`` itself; None then falls back to
    #: :data:`DEFAULT_MAX_TRIANGLES_PER_LINK`.
    max_triangles_per_link: Union[None, int, Dict[str, int]] = None

    #: Links excluded from mesh collision checking entirely, by name. Their geometry is not
    #: packed, so they can never report a collision against any obstacle. Use this for links
    #: in permanent contact with the scene by design, such as a base link bolted to its
    #: mounting plate: an exact triangle test resolves a flush, coplanar contact as a
    #: collision even though the penetration depth is zero. Only used when the robot config
    #: does not set ``mesh_collision_ignored_link_names`` itself.
    ignored_link_names: List[str] = field(default_factory=list)

    #: SceneCollision instance providing the obstacle data.
    _scene_collision_checker: Optional[SceneCollision] = None
    _num_scene_collision_checkers: int = 0

    #: Packed robot collision triangles. Built from the robot model at load time.
    _robot_mesh_data: Optional[RobotMeshData] = None

    def __post_init__(self):
        if self._scene_collision_checker is not None:
            self.scene_collision_checker = self._scene_collision_checker
        # if self._robot_mesh_data is None:
        #     log_and_raise("_robot_mesh_data must be set for RobotMeshConstraintCfg")
        return super().__post_init__()

    def initialize_from_transition_model(self, transition_model: RobotStateTransition):
        """Attach the robot's shared exact-mesh geometry to this constraint.

        The geometry is cached on the shared ``KinematicsParams``, so every solver's
        constraint sees the same buffers and a runtime
        :meth:`~curobo._src.robot.types.kinematics_params.KinematicsParams.update_link_mesh`
        reaches all of them.

        Settings declared in the robot config take precedence; the values on this config are
        used only to fill in what the robot config left unset.
        """
        kinematics_config = transition_model.robot_model.config.kinematics_config
        if kinematics_config.max_triangles_per_link is None:
            kinematics_config.max_triangles_per_link = self.max_triangles_per_link
        if not kinematics_config.mesh_collision_ignored_link_names:
            kinematics_config.mesh_collision_ignored_link_names = self.ignored_link_names
        self._robot_mesh_data = kinematics_config.get_robot_mesh_data()

    def update_num_scene_collision_checkers(self, num_scene_collision_checkers: int):
        self._num_scene_collision_checkers = num_scene_collision_checkers

    @property
    def scene_collision_checker(self) -> SceneCollision:
        """Scene collision checker holding the obstacle data."""
        return self._scene_collision_checker
        
    @scene_collision_checker.setter
    def scene_collision_checker(self, scene_collision_checker: SceneCollision):
        if not isinstance(scene_collision_checker, SceneCollision):
            log_and_raise(
                "scene_collision_checker must be an instance of SceneCollision, got {}".format(
                    type(scene_collision_checker)
                )
            )
        self._scene_collision_checker = scene_collision_checker
        self.update_num_scene_collision_checkers(
            self.scene_collision_checker.get_num_scene_collision_checkers()
        )

    @property
    def robot_mesh_data(self) -> RobotMeshData:
        """Packed robot collision triangle data."""
        return self._robot_mesh_data
