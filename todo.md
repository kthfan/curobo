# TODO: Multi-Waypoint Motion Planner

Working notes for `dev/improve_multi_waypoint`. This file lives only on this branch; remove it
before merging into `dev/main`.

Scope: `MultiWaypointMotionPlanner` (`curobo/_src/motion/multi_waypoint_planner.py`), its
config (`multi_waypoint_planner_cfg.py`), IK de-duplication (`ik_dedup.py`) and trajectory
stitching (`trajectory_stitching.py`).

Priority: **P1** fix first, **P2** next, **P3** later / research.

## 1. Performance

- [ ] **P1 – Edges are solved one at a time by default.** `_solve_segment_edges` chunks edges
  by `trajopt_solver.config.max_batch_size` (`multi_waypoint_planner.py:313`), and
  `MotionPlannerCfg.create` defaults `max_batch_size` to 1. With ~8 IK nodes per layer a
  segment means ~64 sequential trajopt solves. The example does not change it either.
  Options: give `MultiWaypointMotionPlannerCfg.create` its own batch default (e.g. 16 or
  32), or warn when `max_batch_size == 1`; document the trade-off with GPU memory.
- [ ] **P1 – Every chosen edge is solved twice.** `_resolve_path` re-solves the chosen edges
  (`:633`) because the edge trajectories are not kept. Store the best trajectory (or the
  optimized knots) per edge during `_solve_segment_edges` and reuse them.
- [ ] **P2 – Retries recompute everything.** Each attempt re-runs IK for all waypoints and all
  edges. Reuse IK layers and surviving edges across attempts; only add new IK seeds and
  re-try failed edges.
- [ ] **P2 – Waypoint IK runs sequentially** (`:516`). Waypoints with identical criteria could
  be solved in one batched IK call. Criteria are per solver, not per batch item, so group
  waypoints by criteria, or make criteria batchable.

## 2. Planning behaviour

- [ ] **P1 – The trajectory does not start at `start_state`.** `start_state` only seeds IK;
  the composed trajectory starts at an IK node of waypoint 0, and the DP starts with zero
  cost for every node of layer 0 (`:342`), so it does not prefer nodes near the current
  configuration. Add `start_state` as an extra layer 0 (a single node), so the first segment
  is planned from the robot's actual state. Keep the current behaviour behind an option
  for users who already stand at waypoint 0.
- [ ] **P1 – A failed re-solve fails the whole attempt** (`:635`). Instead, set that edge's
  cost to `inf`, re-run the DP and take the next-best path (or keep the top-k paths from
  the DP). This becomes moot once edge trajectories are cached (see Performance).
- [ ] **P2 – Edges have no graph-planner fallback.** Edges call `trajopt_solver.solve_cspace`
  directly, so a segment through a narrow passage often fails. Use
  `MotionPlanner.plan_cspace` (graph seeding on later attempts) at least for edges that
  fail, or for the re-solve of the chosen path.
- [ ] **P2 – The robot stops at every waypoint.** Each segment starts from
  `JointState.from_position` (zero velocity) and ends at rest, so the stitched trajectory
  is rest-to-rest per segment. See "Whole-trajectory refinement" in Future work.
- [ ] **P2 – Edge cost is fixed.** The cost is the trajopt ranking cost
  (`seed_cost`), dominated by `dt * 1000`. Allow a configurable edge cost: joint path length,
  tool path length, trajectory duration, distance to joint limits, or a user callback.
- [ ] **P3 – IK de-duplication is unweighted.** `dedup_joint_solutions` supports `weights`, but
  the config does not expose them; an L-infinity tolerance is not meaningful when revolute
  and prismatic joints are mixed. Add `ik_dedup_joint_weights` to the config.

## 3. Task expressiveness

- [ ] **P2 – Per-segment collision settings.** There is no way to disable collisions for some
  links in one segment (e.g. fingers during a grasp approach, as `plan_grasp` does).
  Add an optional per-segment link-collision switch to `SegmentPlanCfg`.
- [ ] **P2 – Attach / detach between segments.** Pick-and-place needs an object attached for
  the segments after the grasp waypoint. Add per-segment attachment actions to
  `SegmentPlanCfg` (and, when mesh collision is enabled, update the attached mesh too).
- [ ] **P3 – Per-segment speed scaling** (velocity / acceleration limits per segment, e.g.
  slow insertion).
- [ ] **P3 – Waypoint tolerance separate from segment criteria.** A waypoint's IK uses the
  merge of its segments' criteria. Allow a per-waypoint tolerance ("pass near") that does
  not change the segments.
- [ ] **P3 – Goalset waypoints.** A waypoint with several candidate poses
  (`GoalToolPose.num_goalset > 1`) looks supported by the code but is untested; add a test
  and document it.

## 4. Results and diagnostics

- [ ] **P2 – Per-stage diagnostics in `MultiWaypointPlanResult`:** IK solutions per waypoint
  (before and after de-duplication and bound filtering), surviving edges per segment, and
  time per stage. This makes it obvious where planning fails and what to tune.
- [ ] **P3 – Partial results on failure:** return the IK layers and the segments that did
  succeed, so callers can show or reuse them.

## 5. Tests

- [ ] **P1 – Run the GPU tests.** `curobo/tests/_src/motion/test_multi_waypoint_planner.py` is
  mostly CUDA-only and has not been run on a GPU in this environment.
- [ ] **P2 – Tests for the new behaviour:** start-state layer, next-best path after a failed
  edge, batched edge solving, goalset waypoints, and interaction with
  `mesh_collision_check=True` (edges validated against the meshes).

## Future work

- **Whole-trajectory refinement.** Use the stitched trajectory as the seed of one trajopt
  over the full path, with pose constraints at the waypoint time steps, so the robot passes
  through waypoints without stopping and the timing is optimized globally.
- **Task-level API.** Combine per-segment collision switches, attach/detach, speed scaling
  and custom edge costs so that a full pick-and-place can be described as one
  `plan_waypoints` call.
- **Batch planning.** Plan several independent multi-waypoint problems at once
  (`BatchMotionPlanner`-style), including multi-environment scenes.
- **Online execution.** Track the composed trajectory with MPC and re-plan the remaining
  segments when the scene changes.
