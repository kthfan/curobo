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

## Open items

### Small, ready to do

- [ ] **UR10e decimation.** `upper_arm_link` and `forearm_link` still move by 7-17 mm at the
  default budget of 256 and cannot reach 128. Raise their `max_triangles_per_link` in
  `ur10e.yml`, or evaluate voxel remeshing as a fallback (needs scikit-image; 2-17 s per link).
- [ ] **Multi-waypoint planner with `max_batch_size > 1` crashes.** `_solve_segment_edges`
  chunks edges by `max_batch_size`; a last chunk smaller than the batch raises a tensor size
  mismatch in trajectory optimization. Found while speeding up the tests; not ours, check
  `dev/improve_multi_waypoint` first, it may already touch this.
- [ ] **Cache decimated meshes** (P3, more useful now): decimation costs 0.1-0.3 s per decimated
  link at every start-up (about 1.5 s for Franka). Cache the packed triangles per (mesh hash,
  budget) under the runtime cache dir.

### Blocked on decisions

- [ ] **"Inflated spheres for gradients, exact meshes for checking" mode.** Drop the sphere
  scene check (and possibly the sphere self-collision check, now that mesh self-collision
  exists) from the metrics rollouts. Decide only after evaluating the gradient mesh check.
  Warn about links with spheres but no mesh, or document it.
- [ ] **Graph planner ignores meshes** (benchmark). Scheduled after the mode above.
- [ ] **MPC executes infeasible actions.** Handled on another branch; reference only.

### Later (P3)

- [ ] Cheaper margin test on dense meshes (192k-triangle mesh with 1 cm margin: 2.7x slower).
- [ ] Obstacles fully inside a link, and a link fully inside another link, are not detected
  (documented).

## Future work

- **Distance output**: signed clearance per configuration; a first step towards the next item.
- **Mesh check with gradient**: lets the optimizers follow the exact geometry; unblocks the mode
  and the graph planner benchmark above.
- **Use the meshes to improve the spheres**: a tool that finds where meshes collide but spheres
  do not, and suggests spheres. Cheaper than a gradient mesh check.
- **Swept / continuous collision checking** between trajectory steps.
- **Winding-number sign for mesh obstacles** (affects the whole library's scene SDF).

## Branches

- `dev/main`: everything above is merged (last: `dda49df`).
- `dev/improve_mesh`, `dev/mesh_self_collision`, `dev/improve_decimation`,
  `dev/split_mesh_ignore`, `dev/faster_tests`: fully merged into `dev/main`; can be deleted.
- `dev/mesh_followups`: this handoff branch (`todo.md`, `cpu_verify/`). Start follow-up work
  from it, or from `dev/main` on a new branch that copies these two.
