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

## Done on `dev/mesh_followups` (not merged yet)

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

## Open items

### Small, ready to do

- [ ] **`wp.intersect_tri_tri` false crossings on small triangles.** Warp's Möller test
  (`warp/native/intersect_tri.h`) zeroes plane distances below `EPSILON = 1e-6` computed with
  an *unnormalized* normal, so a vertex within `1e-6 / (2 * area)` of the other triangle's
  plane counts as on it: 2.5-5 mm for 2 cm triangles, more for smaller ones. Both the
  collision check and the clearance query report such pairs as crossing (found on a box
  subdivided to 2 cm: gaps of 0.7-7 mm flagged). Fix: own scale-invariant triangle-triangle
  test (edge-pierces-triangle with signed volumes, plus the coplanar case) in
  `wp_robot_mesh_kernel.py`. `test_dense_mesh_agrees_with_the_cuboid` tolerates it below 1 cm.

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
- **Use the meshes to improve the spheres**: a tool that finds where meshes collide but spheres
  do not, and suggests spheres. Cheaper than a gradient mesh check.
- **Swept / continuous collision checking** between trajectory steps.
- **Winding-number sign for mesh obstacles** (affects the whole library's scene SDF).

## Branches

- `dev/main`: everything above is merged (last: `311c36c`, full GPU suite: 4441 passed).
- `dev/improve_mesh`, `dev/mesh_self_collision`, `dev/improve_decimation`,
  `dev/split_mesh_ignore`, `dev/faster_tests`: fully merged into `dev/main`; can be deleted.
- `dev/mesh_followups`: this handoff branch (`todo.md`, `cpu_verify/`). Start follow-up work
  from it, or from `dev/main` on a new branch that copies these two.
