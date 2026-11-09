(guide_via_point_planning)=

# Plan Through Via Points

Use {py:class}`~curobo.via_point_planner.ViaPointMotionPlanner` to move the tool through an
ordered list of poses in one continuous motion, with different constraints on the way to each
pose: for example move freely above a part, then go straight down onto it while staying
upright. How the planner works is explained in {ref}`via_point_planner_note`.

## Plan a path

Create the planner once and warm it up. Waypoints are a
{py:class}`~curobo.types.GoalToolPose` of shape `[1, N, tool frames, 1, 3/4]` in the robot
base frame, without the current pose; the start state must be at rest.

```python
import torch
from curobo.types import GoalToolPose, JointState
from curobo.via_point_planner import ViaPointMotionPlanner, ViaPointMotionPlannerCfg

planner = ViaPointMotionPlanner(
    ViaPointMotionPlannerCfg.create(robot="franka.yml", scene_model="collision_table.yml")
)
planner.warmup()

start = JointState.from_position(
    planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
)
home = planner.compute_kinematics(start).tool_poses           # [1, 1, frames, 3/4]
offsets = torch.tensor([[0.15, 0.0, 0.0], [0.15, 0.2, 0.0], [0.15, 0.2, -0.2]])
position = home.position[:, 0:1] + offsets.to(home.position).view(1, 3, 1, 3)
quaternion = home.quaternion[:, 0:1].expand(-1, 3, -1, -1)
waypoints = GoalToolPose(
    planner.tool_frames, position.unsqueeze(3), quaternion.unsqueeze(3).contiguous()
)

result = planner.plan_waypoints(waypoints, start)
if result.success.item():
    trajectory = result.get_interpolated_plan()               # executable, trimmed
```

Without segment criteria, the planner reaches every waypoint's full pose within the
configured `waypoint_tolerance` (5 mm, 0.05 rad) and leaves the path between waypoints free.
`result.waypoint_position_error` and `result.waypoint_rotation_error` report how closely the
trajectory passes each waypoint.

## Constrain a segment

Pass one entry per waypoint: entry `k` constrains segment `k`, the motion that ends at waypoint
`k`. `None`, or a tool frame left out, keeps the default criteria.

```python
import math
from dataclasses import replace
from curobo.types import ToolPoseCriteria

frame = planner.tool_frames[0]
straight_down = replace(
    ToolPoseCriteria.linear_motion("z"),
    # Accept 5 mm off the line and 0.05 rad of rotation (as 2 * sin(angle / 2)).
    path_tolerance=(0.005, 2.0 * math.sin(0.05 / 2.0)),
    enforce_path_tolerance=True,
)
result = planner.plan_waypoints(waypoints, start, [None, None, {frame: straight_down}])
```

The path fields of a segment's criteria compare the tool with the straight line and SLERP
between the segment's two waypoints. Without `enforce_path_tolerance`, they only pull the
trajectory toward that line; with it, a trajectory that leaves the tolerance fails.

The terminal fields set how the segment's waypoint is reached. For example
`ToolPoseCriteria.track_position()` reaches a waypoint's position with any orientation. Set
`terminal_tolerance` to accept a different error at that waypoint (meters, and rotation as
`2 * sin(angle / 2)`); otherwise `waypoint_tolerance` applies.

## Bound a segment

{py:class}`~curobo.types.ToolPoseBound` limits in a segment's criteria hold inside that segment
only. Here the tool must stay upright while it moves to the second waypoint:

```python
from curobo.types import ToolPoseBound

upright = ToolPoseCriteria(bounds=[ToolPoseBound.axis_tilt(max_angle=0.17, anchor="base")])
result = planner.plan_waypoints(waypoints, start, [None, {frame: upright}, None])
```

A `"goal"` anchor is the segment's waypoint, and a `"start"` anchor is the previous waypoint
(the current pose for the first segment). IK solutions of a waypoint that violate the bounds of
its two segments are discarded, so a bound no solution satisfies fails before the trajectory
optimization with the status `No IK solution at waypoint k satisfies its segments' bounds.`

## Diagnose a failed plan

`result.status` names the stage that failed:

- `No IK solution at waypoint k.`: the waypoint is not reachable with its criteria; check the
  pose, the tool frame and collisions at the waypoint.
- `Trajectory optimization found no valid trajectory.`: inspect
  `result.trajopt_result.debug_info["validation"]`, which tells whether path tracking,
  collisions, or joint limits and dynamics rejected the trajectory.

Segments get native samples in proportion to their joint-space distance, at least
`min_segment_intervals` each. When a plan with many waypoints or sharp corners misses a
waypoint by a few millimeters, configure more spline knots with `num_knots` in
`ViaPointMotionPlannerCfg.create` (64 by default) or a larger `min_segment_intervals`. Fewer knots
plan slightly faster and give somewhat shorter motions, but miss sharp corners more often. One
plan holds at most `planner.max_waypoints` waypoints.
