# Sync dev/main with upstream cuRobo v0.9.0

Branch `dev/sync-v0.9.0` merges `upstream/main` (`5e4f23d`, cuRoboV2-Fast / v0.9.0 and
#738–#743) into `dev/main`. Merge base: `78fd485` (#723). Remove this file before merging the
branch back into `dev/main`.

## Preparation

- [x] Fast-forward the fork's `main` to `upstream/main`.
- [x] Create `dev/sync-v0.9.0` from `dev/main`.
- [ ] `git merge upstream/main`.

## A. Mechanical conflicts (keep both sides)

- [ ] `curobo/_src/rollout/goal_registry.py`: keep `link_start_poses` next to
      `tool_relative_transform_target`.
- [ ] `curobo/_src/motion/motion_planner.py`: keep `ik_seed_config` and the tool pose bound
      checks next to `tool_relative_transform_target`, `finetune_attempts`, `outer_iterations`.
- [ ] `curobo/_src/motion/motion_planner_batch.py`, `motion_planner_cfg.py`.
- [ ] `curobo/_src/solver/solver_core.py`, `solver_ik.py`, `solver_mpc.py`, `solver_trajopt.py`,
      `solver_trajopt_cfg.py`, `solver_trajopt_result.py`, `manager_goal.py`: keep
      `update_tool_pose_bounds` and the mesh collision wiring next to the relative transform and
      velocity limit criteria.
- [ ] `curobo/_src/rollout/cost_manager/cost_manager_robot.py`, `cost_manager_robot_cfg.py`:
      register the `eef_bound`, mesh collision and mesh self-collision costs next to the new
      upstream costs.
- [ ] `curobo/_src/robot/types/kinematics_params.py`, `curobo/_src/transition/robot_state_transition.py`.
- [ ] `curobo/_src/cost/wp_tool_pose.py`.
- [ ] `curobo/types.py`: keep the `ToolPoseBound` exports.
- [ ] Task configs under `curobo/content/configs/task/{ik,mpc}/*.yml`.
- [ ] Tests: `test_cost_tool_pose.py`, `test_motion_planner.py`, `test_motion_planner_cfg.py`,
      `test_manager_goal.py`, `test_solver_trajopt_result.py`.
- [ ] `.gitignore`, `CHANGELOG.md`.
- [ ] Docs: `docs/getting-started/index.rst`, `docs/guides/index.rst`, `docs/news.rst`;
      `docs/concepts/index.rst` was deleted upstream: move the mesh collision and
      multi-waypoint concept pages into the new upstream docs index.

## B. Ports that need design work

- [ ] `ToolPoseCriteria.bounds`: port the `ToolPoseBound` field into the rewritten upstream
      `ToolPoseCriteria` (`__post_init__`, `clone`, `copy_`, `to`; decide whether
      `copy_path_constraint_` / `as_path_constraint` carry the bounds).
- [ ] TrajOpt path tolerances: `TrajOptSolverCfg.position_tolerance_traj` /
      `orientation_tolerance_traj` overlap the upstream per-link `path_tolerance` /
      `enforce_path_tolerance`. Keep them working through the merge; later decide whether to map
      them onto the upstream mechanism and drop them.
- [ ] Mesh check on the interpolated trajectory in `solver_trajopt.py`: keep computing the
      augmented state with `metrics_transition_model` so the link transforms reach the mesh
      collision check.
- [ ] `MultiWaypointMotionPlanner` (to be refactored later; only keep it importable and
      working): adapt it to the upstream `MotionPlanner` / `TrajOptSolverResult` /
      `ToolPoseCriteria` APIs; note the difference from the upstream
      `MotionPlanner.plan_pose_waypoints()` (dense Cartesian path following) in its docs.

## C. Silent breakage (no text conflict)

- [ ] Delete `curobo/_src/curobolib/backends/pybind/kinematics_backward_jacobian_kernel_launch.cu`
      and `curobo/_src/curobolib/kernels/kinematics/kinematics_jacobian_backward_kernel.cuh`:
      upstream removed the pybind backend and the headers they include; nothing references them.
- [ ] Imports of modules upstream moved or removed (`curobo._src.util.trajectory*`,
      `warp_interpolation`, `lbfgs_jit_helpers`, `lsr1`) in files only `dev/main` touched.
- [ ] Unreferenced files from the 3fd54dc sync: mapper kernels `wp_hash`, `wp_coord`,
      `wp_raycast`, `wp_raycast_common`, `wp_integrate_sort_filter`,
      `perception/mapper/kernel/wp_esdf_seed`, `esdf/kernel/wp_resample`, and
      `util/warp_tile_mlp.py`. Decide whether to delete them.
- [ ] `pyproject.toml`: keep the `dev/main` dependencies (`fast_simplification`) after upstream
      removed `setup.py`; drop the pybind package data.
- [ ] `KinematicsState.link_transforms` reads `_buffers["batch_cumul_mat"]`; upstream changed the
      forward kinematics kernel, so verify the buffer layout on GPU.

## Verification

- [ ] CPU: `python -c "import curobo"`, import every module `dev/main` added, `ruff check`.
- [ ] CPU: `pytest --collect-only curobo/tests` collects without import errors.
- [ ] GPU (Kaggle): mesh collision tests (`curobo/tests/_src/collision/test_robot_mesh_*`,
      `test_mesh_collision_end_to_end.py`), `test_tool_pose_bound.py`, `test_cost_eef_bound.py`,
      `test_update_tool_pose_bounds.py`, `test_coverage_sphere.py`,
      `test_multi_waypoint_planner.py`, and the upstream motion planner tests.
- [ ] Merge `dev/sync-v0.9.0` back into `dev/main` and remove this file.
