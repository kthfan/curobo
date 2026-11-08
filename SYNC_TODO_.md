# Sync dev/main with upstream cuRobo v0.9.0

Branch `dev/sync-v0.9.0` merges `upstream/main` (`5e4f23d`, cuRoboV2-Fast / v0.9.0 and
#738–#743) into `dev/main`. Merge base: `78fd485` (#723). Remove this file before merging the
branch back into `dev/main`.

## Preparation

- [x] Fast-forward the fork's `main` to `upstream/main`.
- [x] Create `dev/sync-v0.9.0` from `dev/main`.
- [x] `git merge upstream/main` (34 conflicted files, all resolved in the merge commit).

## A. Mechanical conflicts (keep both sides)

- [x] `curobo/_src/rollout/goal_registry.py`: keep `link_start_poses` next to
      `tool_relative_transform_target`.
- [x] `curobo/_src/motion/motion_planner.py`: keep `ik_seed_config` (now the last keyword) and
      the tool pose bound checks next to `tool_relative_transform_target`, `finetune_attempts`,
      `outer_iterations`. `plan_grasp` takes upstream's `GraspPhasePlanner`, which toggles link
      meshes through the planner's `disable_link_collision` / `enable_link_collision`.
- [x] `curobo/_src/motion/motion_planner_batch.py`, `motion_planner_cfg.py` (robot resolved
      through `resolve_robot_config` so the collision meshes load, plus upstream's multi-env
      capacity check).
- [x] `curobo/_src/solver/solver_core.py`, `solver_ik.py`, `solver_mpc.py`, `solver_trajopt.py`,
      `solver_trajopt_cfg.py`, `manager_goal.py`: keep `update_tool_pose_bounds` and the mesh
      collision wiring next to the relative transform and velocity limit criteria.
- [x] `curobo/_src/rollout/cost_manager/cost_manager_robot.py`, `cost_manager_robot_cfg.py`:
      register `eef_bound`, mesh collision and mesh self-collision next to the new upstream costs.
- [x] `curobo/_src/robot/types/kinematics_params.py`,
      `curobo/_src/transition/robot_state_transition.py` (`compute_link_transforms` kept next to
      upstream's `compute_jacobian`).
- [x] `curobo/_src/cost/wp_tool_pose.py`: take upstream's rewrite; it replaces the rotation
      gradient scale fix of `dev/main`.
- [x] `curobo/types.py`: keep the `ToolPoseBound` exports next to `PoseErrorFrame`.
- [x] Task configs under `curobo/content/configs/task/{ik,mpc}/*.yml`: keep `eef_bound_cfg`,
      take upstream's removal of the legacy keys.
- [x] Tests: `test_motion_planner.py`, `test_motion_planner_cfg.py`, `test_manager_goal.py` keep
      both sides. `test_cost_tool_pose.py` and `test_solver_trajopt_result.py` take upstream (the
      `dev/main` tests covered code upstream replaced).
- [x] `.gitignore`, `CHANGELOG.md` (fork entries moved to "Latest Commit").
- [x] Docs: `docs/getting-started/index.rst`, `docs/guides/index.rst`, `docs/news.rst`.
      Upstream deleted `docs/concepts/`; `docs/concepts/index.rst` now lists only the mesh
      collision and multi-waypoint pages and is linked again from `docs/index.rst`.

## B. Ports that need design work

- [x] `ToolPoseCriteria.bounds`: ported into the rewritten upstream `ToolPoseCriteria`
      (`__post_init__`, `clone`, `to`, `copy_`, `copy_path_constraint_`, `as_path_constraint`
      all carry the bounds).
- [x] TrajOpt path tolerances: removed `position_tolerance_traj` / `orientation_tolerance_traj`
      / `check_traj`. Upstream's TrajOpt builds results in `processor_trajopt_result.py` and no
      longer calls the code that checked them; the upstream replacement is
      `ToolPoseCriteria(path_tolerance=..., enforce_path_tolerance=True)`. Guide and changelog
      updated.
- [x] Mesh check on the interpolated trajectory in `solver_trajopt.py`: still computed with
      `metrics_transition_model` so the link transforms reach the mesh collision check.
- [x] `MultiWaypointMotionPlanner` (minimal, to be refactored later): `_merge_two_criteria`
      rewritten for the canonical projector / axis-scale / tolerance fields; its tests updated.
- [ ] `MultiWaypointMotionPlanner`: note the difference from upstream's
      `MotionPlanner.plan_pose_waypoints()` (dense Cartesian path following) in its docs. Left
      for the planned refactor.

## C. Silent breakage (no text conflict)

- [x] **Quaternion gradient convention.** Upstream changed the kinematics backward from
      `ω = 0.5·vec(q⁻¹ ⊗ dq)` to the left perturbation `ω = 0.5·vec(dq ⊗ q⁻¹)`.
      `wp_eef_bound.py` now writes `dq = (0, 2ω) ⊗ q`; the reference in
      `test_cost_eef_bound.py` was updated. The mesh constraints write no pose gradients.
- [x] Deleted `backends/pybind/kinematics_backward_jacobian_kernel_launch.cu` and
      `kernels/kinematics/kinematics_jacobian_backward_kernel.cuh` (pybind backend removed
      upstream; upstream ships its own `kinematics_jacobian_backward_kernel`).
- [x] Imports of modules upstream moved or removed: none left.
- [ ] Unreferenced files from the 3fd54dc sync: mapper kernels `wp_hash`, `wp_coord`,
      `wp_raycast`, `wp_raycast_common`, `wp_integrate_sort_filter`,
      `perception/mapper/kernel/wp_esdf_seed`, `esdf/kernel/wp_resample`, and
      `util/warp_tile_mlp.py`. Nothing imports them; decide whether to delete them.
- [x] `pyproject.toml`: `fast_simplification` kept; upstream already dropped the pybind data.
- [ ] `KinematicsState.link_transforms` reads `_buffers["batch_cumul_mat"]`; upstream changed the
      forward kinematics kernel, so verify the buffer layout on GPU.
- [ ] `ToolPoseCriteria.path_tolerance` rotation units: upstream's waypoint planner converts an
      angle to the chord `2·sin(θ/2)` before storing it; the guide follows that. Confirm on GPU.

## Verification

- [x] CPU: `import curobo` and all 240 modules `dev/main` added or changed import (needs
      `cuda-core` and `viser` installed). `ruff check --select F` shows no new errors from the
      merge; imports in the touched files are sorted.
- [x] CPU: `pytest --collect-only curobo/tests` collects 8012 tests without errors.
- [x] CPU: the merged areas' tests (`test_multi_waypoint_planner`, `test_tool_pose_bound`,
      `test_cost_eef_bound`, `test_manager_goal`, `test_solver_trajopt_result`,
      `test_cost_tool_pose`, `test_self_collision_params`, `test_update_tool_pose_bounds`,
      `test_motion_planner_cfg`): 239 passed, 84 skipped for lack of CUDA.
- [ ] GPU (Kaggle): mesh collision tests (`curobo/tests/_src/collision/test_robot_mesh_*`,
      `test_mesh_collision_end_to_end.py`), `test_tool_pose_bound.py`, `test_cost_eef_bound.py`
      (gradient convention), `test_update_tool_pose_bounds.py`, `test_coverage_sphere.py`,
      `test_multi_waypoint_planner.py`, `test_link_transforms.py`, and the upstream motion
      planner tests.
- [ ] Merge `dev/sync-v0.9.0` back into `dev/main` and remove this file.
