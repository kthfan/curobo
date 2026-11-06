# TODO: Exact mesh collision – follow-ups

Handoff notes for the next session. This file and `cpu_verify/` live only on development
branches (`dev/mesh_followups`); **remove both before merging into `dev/main`**.

Scope: the exact robot-mesh collision checks (scene and self-collision) in
`curobo/_src/geom/collision/wp_robot_mesh_*`, `RobotMeshData`, their constraints and solver
wiring (`mesh_collision_check`, `mesh_self_collision_check`), and the test workflow.

## Working rules (from the user)

- Reply in Traditional Chinese. Ask when a decision is ambiguous.
- GPU-only tests run on Kaggle (skill `kaggle-curobo-test`); do not make them pass on the CPU.
- CPU-only helper tools go in `cpu_verify/`, which never reaches `dev/main`.
- Merge into `dev/main` by fast-forward after a full GPU run of `curobo/tests`; check that
  `dev/main` has not moved first. Other people push to `dev/main` too.
- Never put a Kaggle token in a file, a commit or the chat. The session reads
  `KAGGLE_API_TOKEN` (and `KAGGLE_USERNAME`) from the environment's variables.

## Test workflow

- Use the user skill `kaggle-curobo-test` (it builds on `kaggle-gpu-test`). Targeted runs while
  developing, the full suite (`run_split_suite.sh`, both T4s) before merging: about 12 min.
- The warm environment `kgt-env-curobo-warm` (venv plus compiled Warp kernels, 8.5 GB) lives in
  the Kaggle account `followjohn`; `--env curobo-warm` restores it. A token of another account
  cannot use it: rebuild it there (about 35 min, see the skill).
- CPU checks: `PYTHONPATH=. pytest -p cpu_verify.cpu_device_plugin -n0 <tests>` runs tests that
  only need torch and Warp on the CPU (see `cpu_verify/README.md`).
- Kaggle `--ref` needs the full commit SHA.

## Done (all on `dev/main`)

- Exact mesh scene check: margin as exact clearance, link-level culling, interpolated trajectory
  check, CUDA graph test, AttachmentManager meshes, `plan_grasp` fixes, `get_mesh_collision_links`,
  voxel threshold widening, open-mesh sign documented, `fast_simplification` required.
- Mesh self-collision (`mesh_self_collision_check`), its own ignore list
  `mesh_self_collision_ignore`, `RobotBuilder.compute_mesh_collision_matrix`,
  `get_mesh_self_collision_pairs`.
- Decimation: combined strategy (merge, close, keep borders, pick the closest of several
  settings), deviation logged with a warning above 2 mm. Franka links within about 1 mm.
- `mesh_collision_ignored_link_names` only skips the scene check; ignored links stay in the
  self-collision check.
- Tests: `pytest_ignore_collect` returns None (so `--ignore` works), multi-waypoint planner tests
  capped at 16 edges per segment.
- Decimated meshes are cached under `runtime.cache_dir/decimated_meshes`, keyed by the mesh,
  the budget and a fingerprint of the decimation code and of `fast_simplification`/`trimesh`;
  `runtime.cache_decimated_meshes = False` disables it. Only thorough (load-time) decimation
  is cached; runtime geometry is not. Writes are atomic; unreadable files are recomputed.
  CPU: Franka 0.80 s -> 3 ms, UR10e 1.13 s -> 3 ms; the deviation warning is still logged.

## Done and merged into `dev/main` on 2026-11-06

- **Scene distance** (user's choices: negative estimate, scene only, cap parameter 0.1 m,
  solver/planner method). `compute_robot_mesh_distance` in `wp_robot_mesh_collision.py`
  (kernels in `wp_robot_mesh_distance_kernel.py`) returns the signed clearance per
  (configuration, link), capped at `max_distance`; links farther away, without geometry or
  ignored report the cap. Positive: exact for cuboids (vertex-box, corner-face, 36 edge-edge)
  and meshes (BVH query inflated by the running best, 6 vertex-face + 9 edge-edge); voxels
  sample like the collision kernel and subtract the sampling widening. Crossing: deepest
  vertex SDF, at most 0 (0 for meshes when `check_containment` is off). Culling reuses
  `robot_link_near_kernel` with `max_distance` as margin. `RobotMeshConstraint.
  compute_link_distances` and `get_mesh_collision_distance` on the solver core, IK, TrajOpt,
  MPC and both planners return `(distance, closest_link)`. Guide: "Measure the clearance to the
  scene" in `docs/guides/mesh_collision.md`.
- **Clearance query speed.** The first version cost up to 9700x the check (dense mesh, 0.3 m
  cap, batch 4096: 26 s); commit `c856809` wrongly said "about as much as the check".
  `robot_vertex_bound_kernel` now seeds each link with its vertices' distance; the exact
  kernels skip triangles whose centroid (or, for cuboids, bounding-box) bound is no closer, and
  the cuboid kernel skips far box corners and edges (`f909f03`, `f7b8763`). The benchmark has a
  per-row time budget (`--row_budget_s`); the whole `--part distance` sweep takes ~6 min.
  T4, budget 256, cap 0.1 m, 4096 sphere-free configs: cuboid 3.0 ms (check 0.72), mesh 5.6
  (1.9), dense mesh 73 (2.8), voxel 22 (6.5). Costs are in the concept doc's Cost section.
- **Triangle intersection without `wp.intersect_tri_tri`.** Warp's Möller test zeroes plane
  distances below `1e-6` computed with an unnormalized normal, so a vertex within
  `1e-6 / (2 * area)` of a small triangle's plane counted as on it (2-5 mm for 2 cm
  triangles): gaps of 0.7-7 mm were reported as crossings. `triangles_intersect` in
  `wp_robot_mesh_kernel.py` (an edge of one triangle touching the other, signs of signed
  volumes only, 2D test for in-plane segments) replaces it in the scene check, the
  self-collision check and the clearance query. Only reachable where a link triangle's box
  straddles the obstacle triangle's plane, so tessellated curved or tilted surfaces, not
  axis-aligned faces.
  Benchmark (T4, margin 0): mesh and dense-mesh colliding rates now equal the cuboid scene's
  in every row (before: e.g. 17.11 % vs 16.97 %); with a sign-only plane reject first, the
  dense-mesh check is as fast as before or faster (batch 4096, budget 256: 31.7 -> 30.0 ms).

- **Cover the link meshes with spheres** (user's choice: report and YAML). `SphereCoverage`
  in `curobo/_src/robot/builder/coverage_sphere.py`, public in `curobo.robot_builder`;
  example `curobo/examples/guides/cover_link_meshes_with_spheres.py`; guide section
  "Cover the link meshes with spheres". `analyze()`: per link, how far the mesh sticks out of
  its spheres; per mesh self-collision pair and per configuration, missed (meshes collide,
  spheres do not) and false alarms. `cover_meshes()` keeps the spheres and greedily adds
  spheres inside the mesh (signed distance: center at least `r - tolerance` deep), default
  tolerance 1 cm and 16 per link, skips links no joint moves. `suggested_checks` /
  `remove_self_collision_ignore()` / `--check_suggested_pairs` hand ignored pairs back to the
  sphere check. `save()` replaces only the spheres (and the edited ignore list).
  Franka (2000 configs): missed 8.2 % -> 8.1 % covered, 0.0 % with the four `panda_link5`
  pairs checked; false alarms 3.7 -> 4.0 / 8.5 %; spheres 61 -> 111; every moving link
  within 1 cm (was up to 90 mm). UR10e: missed 1.9 -> 0.2 %, false alarms 4.0 -> 8.4 %,
  spheres 20 -> 90; `upper_arm_link` and `forearm_link` stay at p95 24 / 16 mm with 16 new
  spheres each.
  Solver benchmark (`benchmark/sphere_coverage_benchmark.py`: table + 40 random boxes, 300
  goals whose meshes are free but within 3 cm of the scene, mesh checks on; T4):
  Franka 61 -> 111 spheres: IK 95.0 -> 93.0 %, plan 96.7 -> 90.0 % (83.3 % with the four
  pairs checked), sphere-feasible IK solutions colliding as meshes 6.3 -> 3.9 % (0.0 %
  checked), IK batch 100 604 -> 1169 ms, MPC step 64 -> 109 ms. UR10e 20 -> 94 spheres: IK
  88.3 -> 85.7 %, plan 80.0 -> 76.7 %, colliding 5.6 -> 0.0 %, IK 245 -> 945 ms, MPC 35 -> 88
  ms. Covering does what it measures but, with many seeds, the mesh check only cost 0.3-0.7
  points of IK success before; the larger spheres cost success near obstacles and time.
  Found on the way and fixed (user's decision): `ur10e.yml` and `dual_ur10e.yml`
  `self_collision_ignore` misspelled `forearm_link` as `forarm_link`, so the joint-connected
  forearm_link - wrist_1_link pair was sphere-checked. `SphereCoverage` warns about such names.
  The covered YAMLs are not adopted (user's decision: keep the tool as a diagnostic).

## Open items

### Small, ready to do

None right now.

### Postponed (user's decision)

- [ ] **UR10e decimation.** `upper_arm_link` and `forearm_link` still move by 7-17 mm at the
  default budget of 256 and cannot reach 128. Raise their `max_triangles_per_link` in
  `ur10e.yml`, or evaluate voxel remeshing as a fallback (needs scikit-image; 2-17 s per link).
  Measured on `dev/mesh_followups`: `upper_arm_link` 16.7 mm, `forearm_link` 6.7 mm at 256.

### Handed off

- [x] **Multi-waypoint planner with `max_batch_size > 1` crashes.** Belongs to another
  session; noted as a P1 item in the todo of `dev/improve_multi_waypoint` (`7663527`), ahead
  of raising its batch default.

### Blocked on decisions

- [ ] **"Inflated spheres for gradients, exact meshes for checking" mode.** Drop the sphere
  scene check (and possibly the sphere self-collision check, now that mesh self-collision
  exists) from the metrics rollouts. Decide only after evaluating the gradient mesh check.
  Warn about links with spheres but no mesh, or document it.
- [ ] **Graph planner ignores meshes** (benchmark). Scheduled after the mode above.
- [ ] **MPC executes infeasible actions.** Handled on another branch; reference only.

### Later (P3)

- [ ] **Clearance against dense meshes with a large cap** is still slow: 192k-triangle mesh,
  0.3 m cap, 4096 configs: 2.7 s (check 2.8 ms). Each surviving link triangle traverses every
  obstacle triangle in its box inflated by the running best (~0.2 m). Ideas: split long link
  triangles before the query, or a tighter per-node bound in a custom BVH traversal.

- [ ] Cheaper margin test on dense meshes (192k-triangle mesh with 1 cm margin: 2.7x slower).
- [ ] Obstacles fully inside a link, and a link fully inside another link, are not detected
  (documented).

## Future work

- **Self-collision distance**: the link-to-link counterpart of the scene distance below
  (deferred by the user when the scene distance was built).
- **Mesh check with gradient**: lets the optimizers follow the exact geometry; unblocks the mode
  and the graph planner benchmark above.
- **Sphere coverage follow-ups** (the tool exists, see Done): cheaper variants were proposed
  but not chosen yet: hand Franka's four `panda_link5` pairs back to the sphere check without
  adding spheres, or cover with a 2 cm tolerance and few spheres; then rerun
  `benchmark/sphere_coverage_benchmark.py`.
- **Swept / continuous collision checking** between trajectory steps.
- **Winding-number sign for mesh obstacles** (affects the whole library's scene SDF).

## Branches

- `dev/main`: everything above is merged (last: `20eca58`, full GPU suite on `0eaf0c0`:
  4484 passed, 38 skipped).
- `dev/improve_mesh`, `dev/mesh_self_collision`, `dev/improve_decimation`,
  `dev/split_mesh_ignore`, `dev/faster_tests`: fully merged into `dev/main`; can be deleted.
- `dev/mesh_followups`: this handoff branch (`todo.md`, `cpu_verify/`). Start follow-up work
  from it, or from `dev/main` on a new branch that copies these two.
