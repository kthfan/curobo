# TODO: Exact Mesh Collision Constraint

Working notes for `dev/improve_mesh`. This file lives only on this branch; remove it before
merging into `dev/main`.

Scope: the exact robot-mesh collision constraint (`RobotMeshConstraint`, the Warp kernels in
`curobo/_src/geom/collision/wp_robot_mesh_kernel.py`, `RobotMeshData`, and its wiring into the
solvers through `mesh_collision_check`).

Priority: **P1** fix first, **P2** next, **P3** later / research.

## 1. Correctness

- [ ] **P1 – `margin` is not a clearance for mesh obstacles.**
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
- [ ] **P1 – Fix the docs that rely on the margin.** `docs/guides/mesh_collision.md`
  ("Keep a clearance") recommends a positive margin to compensate for decimation shrinkage.
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

- [ ] **P1 – AttachmentManager does not update meshes.** Attaching an object updates the
  collision spheres only; users must call `KinematicsParams.update_link_mesh` themselves,
  which is easy to forget in pick-and-place. Make the attachment path also write the
  object's mesh into the reserved `extra_collision_meshes` slots (and clear it on detach),
  when mesh collision is enabled.
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

- [ ] **P2 – Benchmark.** No numbers exist yet. Measure IK, TrajOpt and MPC solve time with
  and without `mesh_collision_check` on Franka with cuboid, mesh and voxel scenes, and with
  different `max_triangles_per_link`. Add the results to the docs.
- [ ] **P2 – Link-level broadphase.** Each thread tests its triangle against every obstacle
  in the environment. Add a per-link AABB (from the transformed link bounds) vs obstacle
  AABB cull before the per-triangle work.
- [ ] **P2 – Skip links far from obstacles using the sphere distances.** The sphere collision
  cost already computes distances per sphere. A link whose spheres are all farther than a
  threshold (sphere coverage error + margin) cannot touch an obstacle and can be skipped.
- [ ] **P3 – Cache decimated meshes.** Decimation runs at every start-up. Cache the packed
  triangle soup per (URDF hash, budget) under the runtime cache dir.

## 5. Tests

- [ ] **P1 – End-to-end solver test on GPU.** No test checks that a solver actually rejects a
  mesh-colliding solution. Add a scene where the spheres are clear but a link mesh
  collides (e.g. a thin plate between the fingers), and assert that IK / `plan_pose` fails
  with `mesh_collision_check=True` and succeeds without it.
- [ ] **P1 – Run the CUDA-only tests added for `mesh_collision_check`**
  (`curobo/tests/_src/solver/test_solver_core_cfg_mesh_collision.py`,
  `curobo/tests/_src/motion/test_motion_planner_cfg.py`); they were never run on a GPU.
- [ ] **P2 – CUDA graph test.** Capture a graph with the mesh constraint, then
  `update_world` (move / replace obstacles) and `update_link_mesh`, and check the replayed
  results match eager execution.
- [ ] **P2 – Mesh margin test** (see Correctness).

## Future work

- **Use the meshes to improve the spheres.** A validation tool that samples many
  configurations, finds where the mesh collides but the spheres do not (or where spheres
  stick out far beyond the mesh), and suggests or adds spheres. This attacks the root
  cause of mesh-only collisions and keeps the optimizer gradient-based.
- **Distance output.** Return a signed clearance per configuration (not just a flag), for
  diagnostics and as an optional soft metric.
- **Swept / continuous collision checking** between trajectory steps (see Coverage).
- **Mesh self-collision** (see Coverage).
