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
only. Here the gripper axis (tool z) must stay within 10° of its direction at the start while
the tool moves to the second waypoint:

```python
from curobo.types import ToolPoseBound

w, x, y, z = home.quaternion[0, 0, 0].tolist()
gripper_axis = [2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)]  # base frame
steady = ToolPoseCriteria(
    bounds=[ToolPoseBound.axis_tilt(reference_axis=gripper_axis, max_angle=0.17, anchor="base")]
)
result = planner.plan_waypoints(waypoints, start, [None, {frame: steady}, None])
```

The reference axis must match the tool's orientation at the waypoints. With the default
`reference_axis` of `+z`, the bound would ask the downward-pointing gripper to point up; and the
Franka's default gripper axis is about 17° off vertical, so even a cone around `-z` would exclude
these waypoints, which keep that orientation.

A `"goal"` anchor is the segment's waypoint, and a `"start"` anchor is the previous waypoint
(the current pose for the first segment). IK solutions of a waypoint that violate the bounds of
its two segments are discarded, so a bound no solution satisfies fails before the trajectory
optimization with the status `No IK solution at waypoint k satisfies its segments' bounds.`

## Keep the joint-space path short

The planner chooses the IK solutions of all waypoints together, so it does not turn a joint
into its limit at one waypoint and unwind it at the next. These settings of
`ViaPointMotionPlannerCfg.create` control this:

- `chain_metric` sets what "short" means:

  | Value | Length of a step `Δq` | Prefers |
  | --- | --- | --- |
  | `"euclidean"` (default) | `‖Δq‖₂` | many joints moving a little over one joint moving a lot |
  | `"total_rotation"` | `Σ_j abs(Δq_j)` | the least joint motion in total, e.g. turning only the wrist |
  | `"time"` | `max_j abs(Δq_j) / v_j` | the shortest motion at the joint velocity limits |
  | `"max_norm"` | `max_j abs(Δq_j)` | the smallest largest joint motion, ignoring velocity limits |

  `ChainMetric` from `curobo.via_point_planner` names the same values.

- `expand_revolutions` (default `True`) lets joints with more than one turn of range, such as
  those of a UR arm, reach a waypoint at `q ± 2π`.
- `chain_length_slack` (default `0.1`) limits the first attempt to IK chains at most 10 %
  longer than the shortest one; the fastest trajectory among them is returned. `None` returns
  the fastest trajectory over more varied chains, which can take a much longer way in joint
  space, for example through another arm configuration.
- `seed_bulge` (default `0.4` rad) bends the seeds that repeat a chain. Raise it in cluttered
  scenes; zero repeats the straight seed.
- `num_ik_seeds` (default `64`) sets how many IK solutions of a waypoint the planner can find.
  With fewer seeds it plans a few milliseconds faster and misses the shortest chain more often.

`result.attempts` above one means no trajectory was found along the shortest chains and the
planner fell back to exploring longer ones. Compare `result.waypoint_configs` with the start
state to see which IK solutions the trajectory passes.

## Hold an object or change the gripper

The planner's IK and trajectory solvers share one robot model. Change it in place between
plans; the warmed CUDA graphs stay valid. Do not change it while a plan runs.

- **Attach a grasped object**: `planner.attachment_manager.attach_from_scene(grasp_state,
  ["box"], world_objects_pose_offset=Pose.from_list([0, 0, 0, 1, 0, 0, 0]))` fits spheres on
  the reserved `attached_object` link and disables the box's world copy; `attach(grasp_state,
  [obstacle], ...)` takes an obstacle that is not in the scene.
  `world_objects_pose_offset` is the pose of the frame the obstacle poses are given in: the
  identity for scene obstacles, whose poses are in the base frame. Without it the obstacle pose
  is read in the tool frame, and the spheres end up far from the object.
  `planner.attachment_manager.detach()` restores the link and re-enables the world copy.
- **Payload dynamics**: attaching sets only collision geometry. Pass the combined link and payload
  inertial to `planner.update_link_inertial("attached_object", mass, com, inertia)` (or several
  links to `update_links_inertial`); this needs a robot loaded with dynamics. Restore the
  unloaded inertial on release.
- **Locked joint values**, such as a gripper opening:

  ```python
  import torch
  from curobo.kinematics import KinematicsCfg

  updated = KinematicsCfg.from_robot_yaml_file(
      "franka.yml", lock_joints={"panda_finger_joint1": 0.02, "panda_finger_joint2": 0.02}
  )
  torch.cuda.synchronize()
  planner.kinematics.update_kinematics_config(updated.kinematics_config)
  planner.reset_seed()
  ```

  The new parameters replace the attachments, collision masks and payload inertials as well:
  build them from the live model's state. The set of locked joints cannot change.
- **Contact links**: `planner.disable_link_collision(["panda_hand"])` stops scene collision
  checks of a link, for example while the gripper touches a part;
  `planner.enable_link_collision` turns them back on.

These are the same operations as on {py:class}`~curobo.motion_planner.MotionPlanner`. The
via-point planner does not plan single poses: use a separate `MotionPlanner` for those, and apply
the same model updates to both.

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
