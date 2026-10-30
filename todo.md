# TODO: Exact Mesh Collision Constraint

Working notes for `dev/improve_mesh`. This file and the `cpu_verify/` directory live only on
this branch; remove both before merging into `dev/main`.

Scope: the exact robot-mesh collision constraint (`RobotMeshConstraint`, the Warp kernels in
`curobo/_src/geom/collision/wp_robot_mesh_kernel.py`, `RobotMeshData`, and its wiring into the
solvers through `mesh_collision_check`).

Priority: **P1** fix first, **P2** next, **P3** later / research.

## 1. Correctness

- [x] **P1 – `margin` is not a clearance for mesh obstacles.** The narrowphase only ran the
  exact intersection test, so `mesh_collision_margin` had almost no effect against mesh
  obstacles. Fixed with option A: `triangles_within` flags a triangle pair closer than the
  margin (6 vertex-face + 9 edge-edge distances); `margin == 0` keeps the old fast path.
  Tests in `test_robot_mesh_constraint.py`, passing on CPU and GPU.
- [x] **P1 – Fix the docs that rely on the margin.** With the fix above the guide and the
  `mesh_collision_margin` docstrings are correct as written; the concept page now states the
  margin per obstacle type.
- [x] **P2 – Voxel sampling cap.** Franka's decimated links have triangles with 18–29 cm
  edges, so on a 1 cm grid nearly every large triangle hits the cap of 16. Measured on a T4
  (4096 configs, table + box ESDF vs the exact cuboid check): misses were identical for caps
  16 / 32 / 64 / 128 and came from the table extending past the grid and from contacts within
  one voxel (gone with margin + 2 cm), while a cap of 64 cost 2.5–4x on a 1 cm grid. Decision:
  keep the cap (now `wp.constant` in the kernel module) and, where it applies, widen the
  threshold by `(spacing - voxel/2) / sqrt(3)`, the extra reach of the coarse lattice; the
  voxel culling threshold adds the matching bound `2 r / (16 sqrt(3))`. Capped triangles now
  catch what half-voxel sampling would, at the price of collisions up to a few mm beyond the
  margin (7.5 mm for a 29 cm edge on 1 cm). Test:
  `test_voxel_capped_sampling_still_catches_small_obstacles` (fails without the widening).
  Re-measured on the same benchmark: time unchanged; on the 1 cm grid misses drop from 9 to 8
  (margin 0, random) and from 18 to 11 (margin 1 cm, sphere-free), with 5–7 extra collisions
  per 4096 random configs; on the 2 cm grid almost no change. GPU tests pass (71).
- [x] **P2 – Containment depends on the obstacle SDF sign.** Verified on the CPU with
  `cpu_verify/verify_open_mesh_sign.py`. The sign comes from Warp's `mesh_query_point`: rays
  along +x, +y, +z, inside if at least two first hit a back face. The sphere collision cost uses
  the same SDF, and containment agreed with it at every test point, so containment adds no
  false collisions of its own. Closed meshes are reliable, also with one flipped face. Open
  meshes are not: a bin without wall thickness reads solid (its cavity is inside), and a
  single-sided sheet has an inside region behind it when its normal has two or more positive
  components (none for axis-facing sheets). Decision: document it (concept page "Inside and
  outside of mesh obstacles", guide limitations); no code change. Skipping containment for open
  meshes would not help while the spheres see the same regions. A winding-number sign is
  listed under "Future work".
- [ ] **P3 – Obstacles fully inside a link are not detected.** Documented. A cheap partial
  fix: test each obstacle's AABB centre / a few obstacle vertices against the link's convex
  hull.

## 2. Coverage

- [x] **P2 – No check between optimized steps.** The mesh constraint now also runs on the
  interpolated trajectory (every `interpolation_dt`), computed with the metrics transition
  model so the link transforms exist. Cost 1–6 % per `plan_pose`. The gap was smaller than
  assumed: the optimized B-spline trajectory is evaluated about every 14 ms (hand steps of at
  most 1–2 cm in measured plans), and a 5 mm plate on the path was already rejected before the
  change. Both checks sample the motion, so thinner obstacles than one step can still be missed;
  see the swept check under "Future work".
- [ ] **P2 – Graph planner ignores meshes.** Its feasibility rollout (`graph_planner_rollout`,
  `metrics_base.yml`) never gets the mesh constraint, so node and edge checks are sphere-only.
  Its paths only seed TrajOpt, which applies the mesh check, so this costs attempts, not
  correctness. The graph planner needs no gradient, so it should check with the exact geometry,
  but mesh-only paths may squeeze through gaps the inflated optimizer spheres push away from;
  that push-away problem is left to the gradient mesh check (Future work).
  **Benchmark first**, scheduled after the "inflated spheres + exact meshes" decision below:
  add a narrow-passage scene to `benchmark/mesh_collision_benchmark.py` and, for plans that
  reach the graph attempts, compare graph rollout = spheres / mesh / spheres + mesh on graph
  query time, `plan_pose` success, total time and the share of graph seeds TrajOpt accepts.
- [ ] **P2 – Mesh self-collision** (raised from P3 by request). Self-collision is sphere-only.
  A mesh self-collision check needs a per-robot list of link pairs to test (reuse the sphere
  self-collision ignore matrix) and a triangle-triangle kernel between robot links.
- [ ] **P2 – MPC executes infeasible actions.** `MPCSolverResult.success` is the feasibility of
  the whole horizon (mesh check included), but `optimize_next_action` / `optimize_action_sequence`
  return the optimized actions regardless. A safe-deceleration fallback for the first
  `2 * interpolation_steps` steps exists but is disabled (`if ... and False:` in
  `MPCSolver._solve_impl`, since the initial release), for every constraint, not only meshes.
  With inflated cost spheres mesh-infeasible steps should be rare (goal pulling into an
  obstacle, fast dynamic obstacles, coverage gaps), and with exact mesh metrics they are rarely
  false alarms, so acting on them becomes practical. To decide: split near-term feasibility
  (act now) from far-horizon feasibility (warning); reaction (decelerate with the existing,
  untested code / stop / report only); all constraints or scene collision only.
  **Handled on another branch**; recorded here for reference only.

## 3. Integration

- [x] **P1 – AttachmentManager does not update meshes.** `attach` now writes the object mesh
  into the reserved `extra_collision_meshes` slots with the same offset as the spheres, and
  `detach` clears it with the new `clear_link_mesh` (a plain disable would be restored by
  `enable_link_collision`). With several envs the shared link meshes need the same offset in
  every env; otherwise it warns and leaves the object to the spheres. Without reserved slots
  it warns too. franka.yml now reserves 256 slots for `attached_object`.
- [x] **P1 – `plan_grasp` keeps the gripper meshes active.** `disable_link_collision` only
  disabled spheres, so the fingers touching the object failed the mesh check. The planners'
  `disable_link_collision` / `enable_link_collision` now toggle the link meshes too, with the
  new `enable_link_mesh` restoring the geometry last written.
- [x] **P1 – `plan_grasp` overwrote its own grasp poses.** Found by the `plan_grasp` test.
  `GoalManager.create_goal_buffer` stored the caller's goal tensors as the reusable buffer, so
  the next solve's `copy_` wrote into them: planning to the approach pose replaced the grasp
  poses, the grasp phase never moved in, and the lift target ended up 30 cm above the grasp.
  The buffer now clones its inputs.
- [x] **P1 – `plan_grasp` approach-to-grasp and lift phases always failed.** Was hidden by
  the aliasing bug above, and no test ran these phases (all set `plan_approach_to_grasp=False`
  and `plan_grasp_to_lift=False`). Root cause: the start states of these phases come from the
  end of the previous trajectory and carried its trajectory dt, which IK reads as "current
  state observed dt ago" and so clamps every joint to within max velocity * dt of the start
  (`CSpacePositionCost`). `GoalManager` also kept a previous solve's dt when the next current
  state had none. Fixed in both planners and in `GoalManager`; `test_plan_grasp_moves_in_and_lifts`
  covers all phases.
- [x] **P2 – Report which link collided.** On-demand: `get_mesh_collision_links(joint_state)`
  on `SolverCore`, delegated by the IK, TrajOpt and MPC solvers and both motion planners. It
  re-runs the mesh check on the auxiliary rollout (not graph-captured, so captured buffers are
  left alone) and returns nested lists of link names (or `None`) with the leading shape of the
  joint positions, e.g. `[batch][seed][step]` for `interpolated_trajectory`. Chosen over
  storing the links in the results (needs plumbing through metrics, seed selection and CUDA
  graphs, and costs every solve). Guide section "Find the link that collides".
- [ ] **P2 – "Inflated spheres for gradients, exact meshes for checking" mode.** Option to drop
  the sphere scene-collision constraint from the metrics rollouts when the mesh check is on,
  so success is decided by the exact geometry, while the optimizers keep sphere costs inflated
  by `optimizer_collision_activation_distance` (cost-only: `radius + eta`; do not use
  `collision_sphere_buffer`, which also inflates metrics and self-collision spheres). Needs a
  positive `mesh_collision_margin` to absorb decimation shrinkage. Measure success in tight
  spaces and goal accuracy near obstacles for several activation distances.
  **Decide whether to add it only after evaluating the gradient mesh check** (Future work).
  - Coverage: links with spheres but no mesh (no collision geometry, reserved slots missing,
    `mesh_collision_ignored_link_names`) would go unchecked. Warn about them when the mode is
    turned on; if a warning turns out hard to implement, document the limitation instead.
  - Self-collision keeps the sphere check until mesh self-collision exists.
- [ ] **P2 – No gradient in mesh-only collisions.** When the spheres are clear but a mesh
  collides, the optimizer has nothing to follow and only re-seeding helps. Prefer improving
  the spheres (see "Future work") over adding a mesh gradient; at minimum, document the
  symptom (many failed seeds in tight spaces) in the guide.
- [x] **P3 – Make `fast_simplification` a declared dependency.** Now a required dependency in
  `pyproject.toml` (decided over an extra: Franka's links 1–5 exceed the default budget of 256,
  so the default setup needs it). Without it trimesh's quadric decimation fails and
  `decimate_mesh` fell back to the convex hull, which fills concave gaps. The fallback stays for
  a failing decimation, with a warning that names the link and the package.
  `test_decimation_keeps_the_gap_between_fingers` fails with the convex-hull fallback, and
  `test_decimation_respects_budget` no longer skips.

## 4. Performance

- [x] **P2 – Benchmark.** `benchmark/mesh_collision_benchmark.py`; results (Tesla T4) in the
  "Cost" section of `docs/concepts/mesh_collision.md`, updated after the link culling below.
  Before culling, IK with 100 problems cost +3 % (cuboid), +11 % (mesh), +50 % (voxel); after,
  +2 %, +3 %, +8 %. Single IK / `plan_pose` / MPC step show no measurable overhead. The solver
  rows come from few samples (2 batches of 100 for IK, 10 plans).
- [x] **P2 – Link-level broadphase / skip links far from obstacles.** Done as one feature: a
  pre-pass tests each (configuration, link) bounding sphere against the obstacles with one SDF
  query, and the per-triangle kernels skip links that are clear, with per-type conservative
  thresholds (see `link_clear_of_obstacle`). The sphere collision cost could not be reused: it
  outputs no distance for free spheres. Benchmark, 4096 sphere-free configs: cuboid 1.9x,
  mesh 6.5x, dense mesh 12x, voxel 8.5x faster; identical collision results. Small batches
  (32 configs) pay about 0.1–0.2 ms for the extra kernel launch.
- [ ] **P3 – Cheaper margin test on dense meshes.** With a 1 cm margin against a 192k-triangle
  mesh the query is about 2.7x slower (2.7 → 7.3 ms for 4096 sphere-free configs, 32 → 88 ms
  for random ones, after link culling): the inflated broadphase passes many more pairs to the
  15 exact distance tests. Try a per-pair reject first (e.g. distance between triangle
  bounding spheres > margin), and measure with the benchmark.
- [ ] **P3 – Cache decimated meshes.** Decimation runs at every start-up. Cache the packed
  triangle soup per (URDF hash, budget) under the runtime cache dir.

## 5. Tests

- [x] **P1 – End-to-end solver test on GPU.**
  `curobo/tests/_src/motion/test_mesh_collision_end_to_end.py`: with the gripper spheres
  disabled and a box pushed into the palm, IK and `plan_pose` fail with the mesh check and
  succeed without it; also covers `plan_grasp` with the mesh check and an attached object's
  mesh. Scene checked on the CPU by `cpu_verify/verify_mesh_collision_scene.py`. Passing on a
  GPU (Kaggle, 2x T4).
- [x] **P1 – Run the CUDA-only tests added for `mesh_collision_check`**
  (`curobo/tests/_src/solver/test_solver_core_cfg_mesh_collision.py`,
  `curobo/tests/_src/motion/test_motion_planner_cfg.py`). Passing on a GPU after fixing the
  shared metrics rollout dict in `resolve_yaml_configs`.
- [x] **P2 – CUDA graph test.** `test_robot_mesh_constraint_cuda_graph.py` captures the check
  once, then applies 14 in-place updates (cuboid move / add / disable, mesh load / replace /
  move, voxel load / replace, with mesh and voxel empty at capture; link mesh write / disable /
  enable / clear; new link transforms) and compares every replay with an eager call, for
  margin 0 and 2 cm. `cpu_verify/verify_cuda_graph_steps.py` checks that every step changes
  the result. The IK solver and `plan_pose` also run with CUDA graphs through `update_world`
  and attach / detach (`test_mesh_collision_end_to_end.py`). Passing on a GPU; a probe that
  reallocates the cuboid poses instead of writing in place makes the test fail. No code fix
  was needed.
- [x] **P2 – Mesh margin test** (see Correctness). Passing on a GPU.

## Future work

- **Use the meshes to improve the spheres.** A validation tool that samples many
  configurations, finds where the mesh collides but the spheres do not (or where spheres
  stick out far beyond the mesh), and suggests or adds spheres. This attacks the root
  cause of mesh-only collisions and keeps the optimizer gradient-based.
- **Distance output.** Return a signed clearance per configuration (not just a flag), for
  diagnostics and as an optional soft metric.
- **Mesh check with gradient.** A differentiable mesh distance cost for the optimizers, so
  trajectories follow the exact geometry instead of inflated spheres. It is meant to solve the
  inflated spheres pushing TrajOpt away from narrow passages that graph or mesh-only metrics
  would allow; evaluate it before deciding on the "inflated spheres + exact meshes" mode.
- **Swept / continuous collision checking** between trajectory steps: test the volume a
  triangle sweeps between two samples (continuous triangle collision, or conservative
  advancement with the link bounding spheres used by the culling) so that no obstacle is
  missed between samples.
- **Winding-number sign for mesh obstacles.** Warp's `mesh_query_point_sign_winding_number`
  classifies open meshes more robustly than the ray-parity sign. It would change the scene SDF
  for the whole library (sphere cost included) and its cost, so it needs its own evaluation.
- **Mesh self-collision** (see Coverage). Until it exists, self-collision stays sphere-based,
  also in the "inflated spheres + exact meshes" mode.
