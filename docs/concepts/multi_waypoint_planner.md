(multi_waypoint_planner_note)=

# Multi-Waypoint Planner

The multi-waypoint planner turns an ordered list of Cartesian waypoints into one joint
trajectory. Planning each segment on its own and joining the results does not work well for a
redundant robot: the configuration chosen at one waypoint may be reachable from the previous
segment but leave no good way into the next one. The planner instead considers several IK
solutions at every waypoint and picks the combination that gives the cheapest whole trajectory.
To run it, see {ref}`tutorial_multi_waypoint_motion_planning`.

## Overview

For waypoints {math}`W_0, \dots, W_{n-1}`, segment {math}`s` is the motion from
{math}`W_s` to {math}`W_{s+1}`, so there are {math}`n - 1` segments. One call of
`plan_waypoints` runs five stages:

1. **IK layers.** For each waypoint, solve IK with `ik_return_seeds` solutions, keep the
   successful ones, remove near-duplicates, and add their full-turn variants. Each waypoint's
   solutions form a *layer* of nodes; only the `max_layer_nodes` nodes on the shortest
   joint-space chains through all waypoints are kept.
2. **Edges.** Connect nodes of consecutive layers. Each edge is one joint-space trajectory
   optimization problem from a node of layer {math}`s` to a node of layer {math}`s+1`, solved
   with the criteria of segment {math}`s`.
3. **Pruning.** Edges whose trajectory optimization fails get infinite cost.
4. **Shortest path.** Dynamic programming over the layers finds the path with the lowest total
   edge cost, one node per waypoint. It looks at the whole path: a node that is cheap to reach
   is not chosen when it makes the following segments expensive.
5. **Re-solve and stitch.** The chosen edges are solved again to obtain their trajectories,
   which are joined into `composed_trajectory`.

If a stage leaves nothing to continue with, for example no IK solution at a waypoint or no
surviving edge in a segment, the attempt fails with a status naming the waypoint or segment.
The planner resets the solver seeds and retries up to `max_attempts` times.

The trajectory starts at the chosen node of the first waypoint, not at the robot's start
state. The start state seeds the IK of every waypoint and, with `start_state_cost` (the
default), the shortest path counts the way from the start state to the first node: the time the
joints need for it at their velocity limits. When the first waypoint is the robot's current
tool pose, the trajectory then starts at the current configuration. Without it the first node
is whichever IK solution of the first waypoint gives the cheapest path, which can be a
different arm configuration.

## IK layers

Many IK solutions of one waypoint are almost the same configuration. Duplicates add edges without adding choices, so each layer keeps a solution only
if it differs from every solution already kept by at least `ik_dedup_joint_tol` in some joint
(L-infinity distance in joint space).

A waypoint inside the list belongs to two segments. Its IK problem uses the merge of both
segments' criteria (`waypoint_constraint_merge_mode="max_weights_min_tols"`):

- each axis weight is the larger of the two, so an axis constrained by either segment is
  constrained at the waypoint;
- each convergence tolerance is the smaller non-zero value of the two, so the waypoint meets
  the stricter segment.

A revolute joint whose limits span more than one turn reaches the same pose at `q` and
`q ± 2π`, but IK returns only one of them. With `expand_revolutions` (the default) every
solution's full-turn variants within the joint limits are added to the layer. Without them the
planner cannot see that a waypoint is better reached "the other way round": for example a
joint at 0° that turns to 90° and 260° and should then turn on to 360°, which is beyond its
limit, has to unwind 260° back to 0°; turning to -100° at the second waypoint instead leaves
only 100° to go.

The variants multiply the nodes of a layer (by up to 32 for a UR arm), and every pair of nodes
is one trajectory optimization. `max_layer_nodes` therefore keeps, per waypoint, the nodes with
the shortest joint-space chain through them, from the start state to the last waypoint. The
globally shortest chain is always kept. `None` keeps every node.

Tool pose bounds are not part of the waypoint IK problem, because their `"goal"` and `"start"`
anchors refer to the endpoints of a segment. Instead, IK solutions that violate the bounds of
the adjacent segments are removed from the layer afterwards. See
{ref}`tool_pose_bounds_guide`.

## Edges and their cost

Between layers with {math}`M_s` and {math}`M_{s+1}` nodes there are up to
{math}`M_s \cdot M_{s+1}` edges. Each edge is a trajectory optimization problem, solved in
batches of the trajectory optimizer's `max_batch_size`. The cost of an edge is the ranking cost
of its best seed, the same cost trajectory optimization uses to order its seeds. It is dominated
by the trajectory time step, so faster motions are preferred, with smaller terms for
acceleration, jerk, and the remaining goal error.

Two settings limit the number of edges when layers are large:

| Setting | Effect |
| --- | --- |
| `knn_edges` | connect each node only to its `knn_edges` nearest nodes (in joint space) of the next layer |
| `max_edges_per_layer_pair` | keep at most this many edges per segment, the pairs closest in joint space |

Both trade completeness for speed: a pruned edge could have been the best one. With the default
of `None` every pair is tried.

The planning time is dominated by the number of edge problems. Fewer nodes per waypoint
(`max_layer_nodes`), a larger `ik_dedup_joint_tol`, the edge limits above, and a larger
`max_batch_size` all reduce it.

## Shortest path and stitching

The layers form a directed acyclic graph, so the cheapest path is found exactly with one forward
pass of dynamic programming: for each node, keep the cheapest way to reach it from the first
layer. Edge results are not stored, so the chosen edges are solved once more to get their
trajectories. The segment trajectories are then concatenated, dropping the repeated
configuration at each junction, into one trajectory sampled at the trajectory optimizer's
`interpolation_dt`.

After the call, the tool pose criteria of all tool frames are reset to the defaults, so later
calls on the underlying solvers are not affected by the segment criteria.
