(via_point_planner_note)=

# Via-Point Planner

The via-point planner plans one continuous trajectory from the robot's current state through
an ordered list of Cartesian waypoints. Like the {ref}`multi-waypoint planner
<multi_waypoint_planner_note>`, every segment between two waypoints can have its own
{py:class}`~curobo.types.ToolPoseCriteria` and {py:class}`~curobo.types.ToolPoseBound` limits,
and the planner chooses among the IK solutions of every waypoint. Like
`MotionPlanner.plan_pose_waypoints`, it solves the whole path with a single trajectory
optimization. To use it, see {ref}`guide_via_point_planning`.

## Overview

For waypoints {math}`W_0, \dots, W_{n-1}`, segment {math}`k` ends at {math}`W_k` and starts at
{math}`W_{k-1}`, or at the current state for {math}`k = 0`, so there are {math}`n` segments.
One call of `plan_waypoints` runs four stages:

1. **IK layers.** Solve IK at every waypoint with the terminal fields of its segment's criteria,
   keep the solutions that satisfy the bounds of the segments on both sides of the waypoint,
   and remove near-duplicates.
2. **IK chains.** A chain picks one IK solution per waypoint. Dynamic programming on joint-space
   distance finds, for every solution of the last waypoint, the shortest chain from the current
   state to it. The optimizer gets one seed per chain, starting with the shortest ones (8 chains
   by default, `num_trajopt_seeds`), so the seeds explore different IK branches of the goal.
3. **Reference.** The spline's native samples are spread over the segments in proportion to
   the joint-space distance of the shortest chain, at least `min_segment_intervals` per segment.
   Waypoint {math}`k` is placed at native sample {math}`w_k`, and a Cartesian reference is
   interpolated through the waypoints on that grid: linear in position, SLERP in rotation.
4. **One trajectory optimization.** The criteria and bounds of every segment are written into
   the optimizer's costs, and the spline is optimized once from the chain seeds. The geometry is
   optimized with a slow time step (`geometry_dt_scale` times the seed's, up to the optimizer's
   `maximum_trajectory_dt`), so the path can pass sharp corners at the waypoints precisely;
   TimeOpt then sets the fastest timing along that path within the joint limits. Validation checks the native samples and the dense,
   time-optimized samples.

The trajectory starts at rest at the current state, passes the intermediate waypoints without
stopping, and ends at rest at the last waypoint. If a stage fails, for example no IK solution
at a waypoint satisfies its bounds, the attempt reports why; the planner retries up to
`max_attempts` times with new IK solutions.

## Criteria per sample

The upstream tool-pose cost has two sets of criteria per tool frame: terminal ones for the last
sample and path ones for every other sample. The via-point planner gives every sample one of
{math}`2n` *slots* instead:

- **path slot** {math}`k`, for the samples inside segment {math}`k`, uses the path fields of
  segment {math}`k`'s criteria: path projectors, axis scales, `error_frame`, `path_tolerance`
  and `enforce_path_tolerance`. The reference at those samples is the line and SLERP between
  the segment's ends, so `ToolPoseCriteria.linear_motion` keeps the tool on the straight line
  between two waypoints. The default criteria have zero path projectors: the path is free.
- **reach slot** {math}`k`, for the sample {math}`w_k`, uses the terminal fields of segment
  {math}`k`'s criteria and always enforces `terminal_tolerance`, or the planner's waypoint
  tolerance when the criteria set none. The samples after the last waypoint, where the
  robot rests at the goal, use the last reach slot.

Dense validation samples come from the time-optimized trajectory, so they do not fall on native
samples. Each one carries its progress along the reference, which tells the segment it is in;
dense samples use the path slots, and the last one the final reach slot. Waypoints are checked
on the native samples, where the trajectory passes {math}`w_k` exactly: time optimization changes
the timing, not the path.

The slot tables have a fixed size, the spline's native reference intervals (at most 256
waypoints), and every plan only rewrites their values. The optimizer's CUDA graphs are captured
once and replayed for every plan.

## Bounds per segment

The bounds of segment {math}`k` hold on its path samples and at its waypoint. Before a plan,
every bound is turned into a base-anchored bound with the same limits:

- a goal anchor is waypoint {math}`k`;
- a start anchor is the current tool pose for segment 0 and waypoint {math}`k-1` for the
  others.

The robot passes waypoint {math}`k-1` within its reach tolerance, so a start anchor refers to
the waypoint target, not the exact pose reached. The multi-waypoint planner, which starts every
segment at an IK solution, anchors it at that solution instead.

## Compared with the other planners

| | `plan_pose_waypoints` | Multi-waypoint planner | Via-point planner |
|---|---|---|---|
| Path between waypoints | follows the interpolated reference | free unless the segment's criteria constrain it | free unless the segment's criteria constrain it |
| Criteria and bounds | one path and one final tolerance | per segment | per segment |
| IK branches | of the last waypoint | every waypoint, compared by optimizing every DAG edge | every waypoint, compared by joint-space distance |
| Trajectory optimizations per plan | one | one per edge, then one per chosen edge | one |
| Stops at waypoints | no | yes, at every waypoint | no |
| Starts at | the current state | an IK solution of the first waypoint | the current state |

The via-point planner compares IK chains by joint-space distance, without optimizing every
combination, so a chain it prefers may still collide where another one would not. Its seeds
cover the shortest chains to several IK solutions of the goal, and a plan that fails is retried
with new IK solutions.
