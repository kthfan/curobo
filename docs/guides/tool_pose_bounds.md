(tool_pose_bounds_guide)=

# Constrain the Tool Frame During Motion

**Goal:** keep a tool frame inside position or orientation limits over the whole trajectory,
for example keep a cup upright, slide a tool on a table plane, or move straight down onto a
part.

Limits are described with {py:class}`~curobo.types.ToolPoseBound` and set per tool frame on
the motion planner, IK, trajectory optimization, or MPC. The cost that enforces them is part of
the default IK, trajectory optimization, and MPC configurations, so no config change is needed.

## Set a bound

```python
from curobo import MotionPlanner, MotionPlannerCfg
from curobo.types import ToolPoseBound

planner = MotionPlanner(MotionPlannerCfg.create(robot="franka.yml"))
tool_frame = planner.tool_frames[0]

# Keep the tool on the plane through its start pose, normal to the start tool z axis.
planner.update_tool_pose_bounds({tool_frame: ToolPoseBound.plane("z", anchor="start")})
result = planner.plan_pose(goal_tool_poses, current_state)

# Remove the bound again.
planner.update_tool_pose_bounds({tool_frame: None})
```

`update_tool_pose_bounds` only changes the bound; the other tool pose criteria of the frame are
kept. The same method exists on {py:class}`~curobo.inverse_kinematics.InverseKinematics`,
{py:class}`~curobo.trajectory_optimizer.TrajectoryOptimizer`, and
{py:class}`~curobo.model_predictive_control.ModelPredictiveControl`. Bounds can be changed
between calls without re-creating the solver or recapturing CUDA graphs.

A bound can also be passed through the `bound` field of
{py:class}`~curobo._src.cost.tool_pose_criteria.ToolPoseCriteria` when you set the full
criteria with `update_tool_pose_criteria`.

## Choose the reference frame

Every bound is expressed in a reference frame

```{math}
T_{ref} = T_{anchor} \cdot T_{offset}
```

`anchor` selects {math}`T_{anchor}` and `offset_position` / `offset_quaternion` (wxyz) select
{math}`T_{offset}`, expressed in the anchor frame.

| `anchor` | Reference follows | Use it for |
| --- | --- | --- |
| `"goal"` (default) | the goal pose of the tool frame | approach and insertion moves relative to the target |
| `"start"` | the tool pose at the start state | keeping the start position or orientation |
| `"base"` | the robot base frame | fixed workspace limits such as a table height |

Entries of `offset_position` left as `None` are 0, so the reference stays at the anchor along
that axis. For example, a plane at height 0.4 m in the base frame:

```python
ToolPoseBound.plane("z", anchor="base", offset_position=[None, None, 0.4])
```

```{note}
In MPC the start state is the current robot state, so a `"start"` reference moves with the
robot at every step. Use `"goal"` or `"base"` when the reference must stay fixed.
```

With a goalset, a `"goal"` bound is satisfied when the pose satisfies the bound of any goal in
the set.

## Describe the limits

Position limits apply to {math}`(x, y, z)` of the tool frame in the reference frame (m). Rotation
limits apply to the swing-twist decomposition of the tool orientation relative to the reference
about the reference z axis (rad):

- `swing_x`, `swing_y`: where the tool z axis points (tilt).
- `twist`: rotation about the tool z axis.
- `max_swing`: a cone, the largest tilt in any direction.

Each component has an optional `[lower, upper]`; `None` leaves that side unbounded. Every
interval must contain the reference (value 0). Move the reference with the offset instead of
using an interval that excludes 0.

| Motion | Bound |
| --- | --- |
| Slide on a plane | `ToolPoseBound.plane("z")` |
| Move along the goal z axis, within 12 cm before the goal | `ToolPoseBound.line("z", lower=-0.12, upper=0.0)` |
| Stay in a height band of ±3 mm | `ToolPoseBound(position_lower=[None, None, -0.003], position_upper=[None, None, 0.003])` |
| Stay above the reference | `ToolPoseBound(position_lower=[None, None, 0.0])` |
| Keep upright within 5° and limit twist to ±5° | `ToolPoseBound(max_swing=0.087, rotation_lower=[None, None, -0.087], rotation_upper=[None, None, 0.087])` |

Position and rotation limits can be combined in one bound.

## Tune margin and tolerance

The optimizer and the feasibility check read the limits differently:

- The optimization cost moves each finite side inward by `position_margin` (default 1 mm) or
  `rotation_margin` (default 0.01 rad), never past the reference. Paths then stay inside your
  interval instead of settling just outside its edge.
- The feasibility check uses your interval as given. A side the cost could not move inward by
  the full margin, such as both sides of a plane or a line, is widened by `position_tolerance`
  (default 2 mm) or `rotation_tolerance` (default 0.02 rad).

Adjust them as follows:

- **Equality limits** (`plane`, `line`, and any side within the margin of the reference) are
  accepted only within the tolerance. Pass `tolerance=` to `plane` / `line`, or set the
  tolerance fields.
  A tolerance of 0 makes an equality infeasible.
- **Planning fails although the path looks close to the edge**: increase the margin.
- **The path should stay closer to the reference**: increase `eef_bound_cfg.weight` in the IK,
  trajectory optimization, and MPC task configs (default `1e5`). A higher weight competes more
  strongly with the other costs, such as collision avoidance.

## Handle errors

| Error | Cause | Fix |
| --- | --- | --- |
| `interval [...] does not contain the reference pose (0)` | an interval excludes the reference | move the reference with `offset_position` / `offset_quaternion` |
| `Tool pose bounds are violated: ... start violates ...` | the start state is outside a bound | start inside the bound, or loosen it |
| `Tool pose bounds are violated: ... goal violates ...` | every goal is outside a bound | change the goal or loosen the bound |
| `a goal-anchored bound needs goal poses ...` | a `"goal"` bound on a solve without a tool goal | use `"base"` or `"start"` |

## Limitations

- The graph planner that seeds trajectory optimization on later attempts ignores bounds. The
  bound is enforced by the trajectory optimization that follows the seed.
- {py:meth}`~curobo.motion_planner.MotionPlanner.plan_grasp` applies bounds to the goalset and
  approach phases only. The grasp and lift phases use their own linear-motion criteria, and the
  criteria of all tool frames are reset when the call returns.
- {py:class}`~curobo.multi_waypoint_planner.MultiWaypointMotionPlanner` sets the tool pose
  criteria of each segment itself, which replaces bounds set with `update_tool_pose_bounds`. Pass
  bounds through the `bound` field of each segment's criteria instead. Interior waypoints are
  solved without bounds, and the start and goal check of `plan_pose` is not run.
