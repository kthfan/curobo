# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Via-point motion planning.

Plan one continuous trajectory from the current state through ordered Cartesian waypoints, with
:class:`~curobo.types.ToolPoseCriteria` and :class:`~curobo.types.ToolPoseBound` limits per
segment and a choice among the IK branches of every waypoint, solved by a single trajectory
optimization.

Example:
    ```python
    import torch
    from curobo.types import GoalToolPose, JointState, ToolPoseCriteria
    from curobo.via_point_planner import ViaPointMotionPlanner, ViaPointMotionPlannerCfg

    planner = ViaPointMotionPlanner(
        ViaPointMotionPlannerCfg.create(robot="franka.yml", scene_model="collision_table.yml")
    )
    planner.warmup()

    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
    )
    frame = planner.tool_frames[0]
    home = planner.compute_kinematics(start).tool_poses
    offsets = torch.tensor([[0.1, 0.0, 0.0], [0.1, 0.2, 0.0], [0.1, 0.2, -0.15]])
    position = home.position[:, 0:1] + offsets.to(home.position).view(1, 3, 1, 3)
    quaternion = home.quaternion[:, 0:1].expand(-1, 3, -1, -1)
    waypoints = GoalToolPose(
        planner.tool_frames, position.unsqueeze(3), quaternion.unsqueeze(3).contiguous()
    )
    # Free motion to the first two waypoints, then straight down to the last one.
    straight = ToolPoseCriteria.linear_motion("z")
    result = planner.plan_waypoints(waypoints, start, [None, None, {frame: straight}])
    if result.success.item():
        trajectory = result.get_interpolated_plan()
    ```
"""

from curobo._src.motion.planner_via_point import ViaPointMotionPlanner
from curobo._src.motion.planner_via_point_cfg import ViaPointMotionPlannerCfg, ViaPointPlanResult

__all__ = [
    "ViaPointMotionPlanner",
    "ViaPointMotionPlannerCfg",
    "ViaPointPlanResult",
]
