(tool_pose_bounds_guide)=

# Constrain the Tool Frame During Motion

**Goal:** keep a tool frame inside position or orientation limits over the whole trajectory,
for example keep a cup upright, slide a tool on a table plane, or move straight down onto a
part.

Limits are described with {py:class}`~curobo.types.ToolPoseBound` and set per tool frame on
the motion planner, IK, trajectory optimization, or MPC. The cost that enforces them is part of
the default IK, trajectory optimization, and MPC configurations, so no config change is needed.

## Set bounds

```python
from curobo import MotionPlanner, MotionPlannerCfg
from curobo.types import ToolPoseBound

planner = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml"))
tool_frame = planner.tool_frames[0]

# Keep the tool at its start height and its z axis within 10 degrees of the start z axis.
planner.update_tool_pose_bounds(
    {
        tool_frame: [
            ToolPoseBound.plane((0.0, 0.0, 1.0), anchor="start", reference_axes="base"),
            ToolPoseBound.axis_tilt(max_angle=0.17, anchor="start"),
        ]
    }
)
result = planner.plan_pose(goal_tool_poses, current_state)

# Remove the bounds again.
planner.update_tool_pose_bounds({tool_frame: None})
```

A tool frame takes one bound or a list of bounds, and a pose must satisfy all of them. Up to
four bounds per tool frame are supported by default (`eef_bound_cfg.max_bounds_per_tool_frame`
in the task configs).

`update_tool_pose_bounds` replaces the bounds of the given tool frames and keeps their other
tool pose criteria. The same method exists on
{py:class}`~curobo.inverse_kinematics.InverseKinematics`,
{py:class}`~curobo.trajectory_optimizer.TrajectoryOptimizer`, and
{py:class}`~curobo.model_predictive_control.ModelPredictiveControl`. Bounds can be changed
between calls without re-creating the solver or recapturing CUDA graphs. Bounds can also be
passed through the `bounds` field of {py:class}`~curobo.types.ToolPoseCriteria` when you set
the full criteria with `update_tool_pose_criteria`.

## Use the common shapes

The factory methods take plain vectors and angles:

| Motion | Bound |
| --- | --- |
| Slide on the table plane at height 0.4 m | `ToolPoseBound.plane((0, 0, 1), point=(0, 0, 0.4), anchor="base")` |
| Stay above the table | `ToolPoseBound.plane((0, 0, 1), point=(0, 0, 0.4), max_distance=None, anchor="base")` |
| Stay within ±3 mm of the start height | `ToolPoseBound.plane((0, 0, 1), min_distance=-0.003, max_distance=0.003, anchor="start", reference_axes="base")` |
| Slide on a plane through `p0` with normal `n` | `ToolPoseBound.plane(n, point=p0, anchor="base")` |
| Move along the goal z axis, within 12 cm before the goal | `ToolPoseBound.line((0, 0, 1), min_along=-0.12, max_along=0.0)` |
| Drop vertically onto the goal | `ToolPoseBound.line((0, 0, 1), min_along=0.0, max_along=0.2, reference_axes="base")` |
| Keep a cup upright within 10° | `ToolPoseBound.axis_tilt(max_angle=0.17, anchor="base")` |
| Keep the approach axis (tool z) 20° above the horizon, any heading and roll | `ToolPoseBound.axis_tilt(min_angle=1.22, max_angle=1.22, anchor="base")` |
| Stay within ±5° roll, pitch and yaw of the goal orientation | `ToolPoseBound.orientation_box(0.087, 0.087, 0.087)` |

Angles are in radians, distances in meters, and vectors and points are expressed in the axes
selected by `reference_axes` (described below). Combine shapes in a list, for example a plane
and an `axis_tilt` to slide on a sloped surface while keeping a cup upright.

## Choose the reference frame

Every bound measures the tool pose in a reference frame `F`:

- **`anchor`** sets the origin of `F` and the axes it starts from:

  | `anchor` | Origin and axes | Use it for |
  | --- | --- | --- |
  | `"goal"` (default) | the goal pose of the tool frame | approach and insertion moves relative to the target |
  | `"start"` | the tool pose at the start state | keeping the start position or orientation |
  | `"base"` | the robot base frame | fixed workspace limits such as a table, a wall, or gravity |

- **`reference_axes="base"`** keeps the origin at the anchor but takes the axes from the robot
  base frame, for example a vertical line through the goal.
- **`reference_rotation`** (wxyz) rotates those axes. The factories set it from the vectors you
  pass.

All poses are expressed in the robot base frame. If the robot is mounted on a slope, express
world directions such as "up" in the base frame before passing them.

```{note}
In MPC the start state is the current robot state, so a `"start"` reference moves with the
robot at every step. Use `"goal"` or `"base"` when the reference must stay fixed.
```

With a goalset, `"goal"` bounds are satisfied when the pose satisfies them for any goal in the
set.

## Write limits directly

For limits the factories do not cover, set the components of
{py:class}`~curobo.types.ToolPoseBound` directly. Each takes `None` (no limit), a number
(equality), or a `(lower, upper)` pair whose ends may be `None`.

- **`x`, `y`, `z`** (m): the tool frame origin relative to the anchor, along the axes of `F`.
- **`tilt`** (rad, 0 to π): the angle between the tool axis and the reference axis.
- **`yaw`** (rad): the direction the tool axis tilts toward, measured about the reference axis
  from the x axis of `F`.
- **`roll`** (rad): the rotation of the tool about the tool axis.

The reference axis is the z axis of `F`. The tool axis is the z axis of the tool frame rotated
by `tool_rotation` (wxyz); `tool_rotation` does not move the point whose position is limited.
In other words, the orientation of the tool axis frame relative to `F` is
`Rz(yaw) · Ry(tilt) · Rz(roll)`.

```python
import math

# Approach axis 20 degrees above the horizon, roll fixed, heading free.
ToolPoseBound(anchor="base", tilt=math.radians(70), roll=0.0)

# Inside a box around the start position, with the tool z axis within 30 degrees of vertical.
ToolPoseBound(anchor="start", reference_axes="base",
              x=(-0.1, 0.1), y=(-0.1, 0.1), z=(0.0, 0.2), tilt=(None, math.radians(30)))
```

`yaw` and `roll` intervals need both ends and may cross ±π, for example `roll=(3.0, 3.4)`.
They are undefined when the tool axis is parallel to the reference axis, so they can only be
limited together with a `tilt` interval at least 10° away from 0 and π. To limit roll, pitch,
and yaw near a reference orientation, use `orientation_box`, which picks a tool axis
perpendicular to the reference axis.

## Tune the tolerance

Each bound has a `position_tolerance` (default 2 mm) and a `rotation_tolerance` (default
0.02 rad); the factories take a single `tolerance`. The optimizer and the feasibility check use
it differently:

- **Inequality limits** (an interval wider than the tolerance, or one-sided): the feasibility
  check uses your interval as given, and the optimization cost moves each side inward by half
  the tolerance, so paths stay inside instead of settling on the edge.
- **Equality limits** (a number, or an interval no wider than the tolerance, such as `plane`,
  `line`, or a fixed `tilt`): the cost pulls to the center, and the feasibility check accepts
  errors up to the tolerance. A tolerance of 0 makes an equality infeasible.

Adjust them as follows:

- **Planning fails although the path looks close to the limit**: increase the tolerance.
- **The path should stay closer to the reference**: increase `eef_bound_cfg.weight` in the IK,
  trajectory optimization, and MPC task configs (default `1e5`). A higher weight competes more
  strongly with the other costs, such as collision avoidance.

## Bound multi-waypoint plans

{py:class}`~curobo.multi_waypoint_planner.MultiWaypointMotionPlanner` plans each segment
between two waypoints separately. A `"goal"` bound refers to the waypoint that ends the segment
and a `"start"` bound to the waypoint that starts it.

- To bound one segment, set `bounds` on that segment's criteria.
- To bound every segment, call `update_tool_pose_bounds` on the multi-waypoint planner. The
  bounds apply to each segment whose criteria leave the tool frame unbounded, and stay set
  until you clear them with `None`.

```python
from curobo.types import ToolPoseBound, ToolPoseCriteria

frame = planner.tool_frames[0]
insert = ToolPoseCriteria.track_position_and_orientation()
# Along the goal z axis onto at_part.
insert.bounds = [ToolPoseBound.line((0, 0, 1), min_along=-0.12, max_along=0.0)]
result = planner.plan_waypoints(
    [above_part, at_part],
    start_state,
    segment_tool_pose_criteria=[{frame: insert}],
)
```

IK solutions at a waypoint that violate the bounds of its adjacent segments are discarded. If
none remain, planning fails with the status
`No IK solution at waypoint i satisfies the tool pose bounds of its segments.` A `"start"` bound
of the segment that ends at the waypoint depends on the previous waypoint's solution, so the
segment's trajectory optimization checks it instead.

## Check path criteria with trajectory tolerances

Tool pose criteria can also shape the path: the non-terminal weights of
{py:meth}`ToolPoseCriteria.linear_motion <curobo.types.ToolPoseCriteria.linear_motion>`
pull every step toward the line along the goal z axis. By default only the final pose is
checked for success, so a path that drifts from the line still succeeds. To also check the
steps before the last one, set a trajectory tolerance:

```python
from curobo import MotionPlanner, MotionPlannerCfg
from curobo.types import ToolPoseCriteria

planner = MotionPlanner(
    MotionPlannerCfg.create(
        robot="franka.yml",
        position_tolerance_traj=0.005,  # m
        orientation_tolerance_traj=0.05,  # rad
    )
)
frame = planner.tool_frames[0]
planner.update_tool_pose_criteria({frame: ToolPoseCriteria.linear_motion("z")})
result = planner.plan_pose(goal_tool_poses, current_state)
```

`TrajectoryOptimizerCfg.create` accepts the same two arguments. A trajectory then succeeds only
if every step stays within the tolerances on the axes the non-terminal weights constrain; 0
(the default) turns the check off.

The error of a step is weighted by the non-terminal weights. Use the tolerances with criteria
whose non-terminal weights are 0 on the axes of free motion, such as `linear_motion`.
`track_position_and_orientation` weights every axis, so its non-terminal error is the scaled
distance to the goal, and a trajectory tolerance would reject almost every path.

Bounds and trajectory tolerances can be combined. Prefer bounds for one-sided limits, limits
relative to the start or the robot base, and limits on the tool's tilt, yaw, or roll.

## Handle errors

| Error | Cause | Fix |
| --- | --- | --- |
| `yaw and roll are undefined when the tool axis is parallel ...` | `yaw` or `roll` is limited without a `tilt` interval away from 0 and π | choose `tool_rotation` so the tool axis is perpendicular to the reference axis, or use `orientation_box` |
| `Tool pose bounds are violated: ... start violates bound i: ...` | the start state is outside bound `i` | start inside the bound, or loosen it |
| `Tool pose bounds are violated: ... goal violates bound i: ...` | every goal is outside bound `i` | change the goal or loosen the bound |
| `a goal-anchored bound needs goal poses ...` | a `"goal"` bound on a solve without a tool goal | use `"base"` or `"start"` |
| `... at most 4 are supported (EEFBoundCostCfg.max_bounds_per_tool_frame)` | too many bounds for one tool frame | combine limits into fewer bounds, or raise the setting |

## Limitations

- The graph planner that seeds trajectory optimization on later attempts ignores bounds. The
  bound is enforced by the trajectory optimization that follows the seed.
- {py:meth}`~curobo.motion_planner.MotionPlanner.plan_grasp` applies bounds to the goalset and
  approach phases only. The grasp and lift phases use their own linear-motion criteria, and the
  criteria of all tool frames are reset when the call returns.
- Limits are planes, lines, boxes, and angle ranges. Curved paths, such as an arc around a door
  hinge, are not covered.
