# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Multi-waypoint motion planning module.

This module plans a single trajectory through an ordered list of Cartesian
waypoints by generating de-duplicated IK solutions per waypoint, running
trajectory optimization on the edges between consecutive waypoints, pruning
failed edges, finding the cheapest surviving path, and stitching the chosen
segment trajectories together.

Example:
    ```python
    from curobo.multi_waypoint_planner import (
        MultiWaypointMotionPlanner,
        MultiWaypointMotionPlannerCfg,
    )
    from curobo.types import ToolPoseCriteria

    config = MultiWaypointMotionPlannerCfg.create(robot="franka.yml")
    planner = MultiWaypointMotionPlanner(config)

    segment_criteria = [
        {f: ToolPoseCriteria.track_position_and_orientation() for f in planner.tool_frames}
        for _ in range(len(waypoints) - 1)
    ]
    result = planner.plan_waypoints(waypoints, start_state, segment_criteria)
    ```
"""

from curobo._src.motion.multi_waypoint_planner import MultiWaypointMotionPlanner
from curobo._src.motion.multi_waypoint_planner_cfg import (
    MultiWaypointMotionPlannerCfg,
    SegmentPlanCfg,
)
from curobo._src.motion.multi_waypoint_planner_result import MultiWaypointPlanResult

__all__ = [
    "MultiWaypointMotionPlanner",
    "MultiWaypointMotionPlannerCfg",
    "SegmentPlanCfg",
    "MultiWaypointPlanResult",
]
