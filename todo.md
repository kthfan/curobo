# TODO: Exact Mesh Collision Constraint

Working notes for `dev/improve_mesh`. This file and the `cpu_verify/` directory live only on
this branch; remove both before merging into `dev/main`.

Scope: the exact robot-mesh collision constraint (`RobotMeshConstraint`, the Warp kernels in
`curobo/_src/geom/collision/wp_robot_mesh_kernel.py`, `RobotMeshData`, and its wiring into the
solvers through `mesh_collision_check`).

Priority: **P1** fix first, **P2** next, **P3** later / research.

## 1. Correctness

- [x] **P1 – `margin` is not a clearance for mesh obstacles.** Done with option A
  (`triangles_within` in the narrowphase). Tests added in `test_robot_mesh_constraint.py`
  and passing on the CPU (`cpu_verify/`) and on a GPU.
  In `robot_mesh_collision_kernel`, `margin` only inflates the broadphase AABB
  (`wp_robot_mesh_kernel.py:174`). The narrowphase is an exact `wp.intersect_tri_tri`
  (`:183`) that ignores it, and the containment pass compares the SDF of a single vertex
  against it (`:191`). So `mesh_collision_margin > 0` has almost no effect against mesh
  obstacles, while it works for cuboids (inflated half extents) and voxels (`sdf <= margin`).
  - Option A: in the narrowphase, flag a pair when the triangle-triangle distance is below
    `margin` (segment-triangle / vertex-triangle distance), keeping the exact predicate as a
    fast path for `margin == 0`.
  - Option B: keep the exact predicate and also flag when any vertex or edge sample of the
    robot triangle has `compute_local_sdf < margin` (cheaper, not exact).
  - Add a test like `test_cuboid_margin_expands_collision` for mesh obstacles.
- [x] **P1 – Fix the docs that rely on the margin.** With option A the guide and the
  docstrings are correct as written; the concept page now states the margin per obstacle
  type. `docs/guides/mesh_collision.md` ("Keep a clearance") recommends a positive margin to compensate for decimation shrinkage.
  That is wrong for mesh obstacles until the item above is fixed. Update the guide, the
  concept page, and the `mesh_collision_margin` docstrings in the solver configs.
- [ ] **P2 – Voxel sampling cap.** `MAX_VOXEL_SAMPLES_PER_EDGE = 16`
  (`wp_robot_mesh_collision.py:33`) caps the lattice, so a large triangle on a fine grid is
  sampled coarser than voxel/2 (e.g. a 10 cm edge on a 1 cm grid gives ~6 mm spacing).
  Options: subdivide large robot triangles at load time so no edge exceeds a target length,
  or make the cap depend on the triangle size and voxel size, with a warning when clamped.
- [ ] **P2 – Containment depends on the obstacle SDF sign.** The mesh containment pass uses
  `compute_local_sdf(...) < margin` on one vertex. Verify the sign for non-watertight
  obstacle meshes (e.g. meshes from the mapper or scans). If it is unreliable, skip
  containment for open meshes or document that `check_containment` needs closed meshes.
- [ ] **P3 – Obstacles fully inside a link are not detected.** Documented. A cheap partial
  fix: test each obstacle's AABB centre / a few obstacle vertices against the link's convex
  hull.

## 2. Coverage

- [ ] **P2 – No check between optimized steps.** `solver_trajopt.py:61` disables the mesh
  constraint on the interpolated rollout, so a fast motion can pass through a thin obstacle
  between two optimized steps. Options:
  - run the mesh check on the interpolated trajectory with a stride (every k-th step);
  - add a swept check between consecutive steps (continuous collision for the triangles,
    or conservative advancement using the sphere distances);
  - expose a config switch and measure the cost.
- [ ] **P2 – Graph planner ignores meshes.** Graph seeds can be mesh-infeasible, wasting
  trajopt attempts. Add an option to run the mesh constraint in the graph planner's edge
  validation rollout (`graph_planner_rollout`).
- [ ] **P3 – Mesh self-collision.** Self-collision is sphere-only. A mesh self-collision
  check needs a per-robot list of link pairs to test (reuse the sphere self-collision
  ignore matrix) and a triangle-triangle kernel between robot links.
- [ ] **P2 – Verify MPC behaviour.** The constraint is part of the MPC metrics rollout.
  Check what MPC does with a mesh-infeasible step (does it still execute the action, does it
  report it) and document or change it.

## 3. Integration

- [x] **P1 – AttachmentManager does not update meshes.** Done: `attach` writes the object
  mesh into the reserved slots with the sphere offset, `detach` clears it with the new
  `clear_link_mesh` (a plain disable would be restored by `enable_link_collision`). With several
  envs the shared link meshes need the same offset in every env; otherwise it warns and leaves
  the object to the spheres. franka.yml still reserves no mesh slots by default. Attaching an object updates the
  collision spheres only; users must call `KinematicsParams.update_link_mesh` themselves,
  which is easy to forget in pick-and-place. Make the attachment path also write the
  object's mesh into the reserved `extra_collision_meshes` slots (and clear it on detach),
  when mesh collision is enabled.
- [x] **P1 – `plan_grasp` keeps the gripper meshes active.** Fixed: the planners'
  `disable_link_collision` / `enable_link_collision` toggle the meshes too, and
  `enable_link_mesh` restores the geometry last written. During the approach and grasp
  phases `MotionPlanner.plan_grasp` calls `disable_link_collision` on
  `grasp_contact_link_names` (`motion_planner.py:462-597`), which only disables the spheres.
  With `mesh_collision_check=True` the fingers touching the object still fail the mesh
  constraint, so the grasp phase most likely always fails. Make `disable_link_collision` /
  `enable_link_collision` also toggle the link meshes when mesh collision is enabled.
  `KinematicsParams.disable_link_mesh` exists but there is no way to re-enable a link mesh
  yet, so an `enable_link_mesh` (restore from the loaded geometry) is needed too. Add a
  `plan_grasp` test with the mesh check on.
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
- [ ] **P2 – Report which link collided.** `RobotMeshConstraint.get_contact_link_names()` is
  only reachable on the internal cost object. Surface the colliding link and time step in
  the solver result or status, so a failed plan says "mesh collision: panda_hand at step 12"
  instead of a generic failure.
- [ ] **P2 – No gradient in mesh-only collisions.** When the spheres are clear but a mesh
  collides, the optimizer has nothing to follow and only re-seeding helps. Prefer improving
  the spheres (see "Future work") over adding a mesh gradient; at minimum, document the
  symptom (many failed seeds in tight spaces) in the guide.
- [ ] **P3 – Make `fast_simplification` a declared dependency** (or an extra), since the
  convex-hull fallback in `decimate_mesh` (`data_robot_mesh.py:161`) closes gripper gaps and
  causes false collisions.

## 4. Performance

- [x] **P2 – Benchmark.** `benchmark/mesh_collision_benchmark.py`; results (Tesla T4) in the
  "Cost" section of `docs/concepts/mesh_collision.md`. Single IK / `plan_pose` / MPC step show
  no measurable overhead; IK with 100 problems costs +3 % (cuboid), +11 % (mesh), +50 % (voxel).
  The voxel kernel is the slowest at large batches (49 ms for 4096 configs vs 1.4 ms cuboid).
  The solver rows come from few samples (2 batches of 100 for IK, 10 plans); rerun with more
  goals before quoting small differences.
- [ ] **P2 – Link-level broadphase.** Each thread tests its triangle against every obstacle
  in the environment. Add a per-link AABB (from the transformed link bounds) vs obstacle
  AABB cull before the per-triangle work. Most useful for voxels, the slowest case in the
  benchmark (every triangle is sampled against the ESDF even far from obstacles).
- [ ] **P2 – Skip links far from obstacles using the sphere distances.** The sphere collision
  cost already computes distances per sphere. A link whose spheres are all farther than a
  threshold (sphere coverage error + margin) cannot touch an obstacle and can be skipped.
- [ ] **P3 – Cheaper margin test on dense meshes.** With a 1 cm margin against a 192k-triangle
  mesh, a 4096-config query goes from 61 to 138 ms for random configs (+10 % for sphere-free
  ones): the inflated broadphase passes many more pairs to the 15 exact distance tests. Try a
  per-pair reject first (e.g. distance between triangle bounding spheres > margin), and
  measure with the benchmark.
- [ ] **P3 – Cache decimated meshes.** Decimation runs at every start-up. Cache the packed
  triangle soup per (URDF hash, budget) under the runtime cache dir.

## 5. Tests

- [x] **P1 – End-to-end solver test on GPU.** Written in
  `curobo/tests/_src/motion/test_mesh_collision_end_to_end.py` (gripper spheres disabled, box
  pushed into the palm; scene checked on the CPU by `cpu_verify/verify_mesh_collision_scene.py`).
  Passing on a GPU (Kaggle, 2x T4). No test checks that a solver actually rejects a
  mesh-colliding solution. Add a scene where the spheres are clear but a link mesh
  collides (e.g. a thin plate between the fingers), and assert that IK / `plan_pose` fails
  with `mesh_collision_check=True` and succeeds without it.
- [x] **P1 – Run the CUDA-only tests added for `mesh_collision_check`**
  (`curobo/tests/_src/solver/test_solver_core_cfg_mesh_collision.py`,
  `curobo/tests/_src/motion/test_motion_planner_cfg.py`). Passing on a GPU after fixing the
  shared metrics rollout dict in `resolve_yaml_configs`.
- [ ] **P2 – CUDA graph test.** Capture a graph with the mesh constraint, then
  `update_world` (move / replace obstacles) and `update_link_mesh`, and check the replayed
  results match eager execution. The benchmark ran IK, `plan_pose` and MPC with CUDA graphs and
  the mesh check without errors, but did not compare against eager results.
- [x] **P2 – Mesh margin test** (see Correctness). Passing on a GPU.

## Future work

- **Use the meshes to improve the spheres.** A validation tool that samples many
  configurations, finds where the mesh collides but the spheres do not (or where spheres
  stick out far beyond the mesh), and suggests or adds spheres. This attacks the root
  cause of mesh-only collisions and keeps the optimizer gradient-based.
- **Distance output.** Return a signed clearance per configuration (not just a flag), for
  diagnostics and as an optional soft metric.
- **Swept / continuous collision checking** between trajectory steps (see Coverage).
- **Mesh self-collision** (see Coverage).
