# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""End-to-end tests: solvers reject solutions whose collision meshes hit the scene.

The scene is built so that the spheres are clear but a link mesh collides, independently of
how well the spheres cover the meshes: the gripper spheres are disabled, and a box pushes into
the palm of a top-down grasp pose. Every solution for that pose puts the hand mesh into the
box, while the closest remaining sphere (on ``panda_link7``) stays 2.7 cm away.
"""

# Standard Library
from typing import Optional

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose

#: Links whose spheres are disabled, leaving the mesh check as their only collision test.
GRIPPER_LINKS = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]

#: Top-down grasp pose of ``panda_hand``: (x, y, z, qw, qx, qy, qz).
GOAL_POSE = [0.5, 0.0, 0.3, 0.0, 1.0, 0.0, 0.0]


def palm_box(offset=(0.0, 0.0, 0.0)) -> SceneCfg:
    """Box spanning 3.5 cm to 15 cm below the hand frame, across the palm and fingers."""
    x, y, z = offset
    return SceneCfg(
        cuboid=[
            Cuboid(
                name="palm_box",
                pose=[0.5 + x, 0.0 + y, 0.3 - 0.0925 + z, 1.0, 0.0, 0.0, 0.0],
                dims=[0.1, 0.24, 0.115],
            )
        ]
    )


@pytest.fixture(scope="module")
def cuda_device_cfg():
    """CUDA device configuration."""
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


def _disable_gripper_spheres(kinematics) -> None:
    """Disable only the gripper spheres; ``disable_link_collision`` would disable the meshes."""
    for link_name in GRIPPER_LINKS:
        kinematics.config.kinematics_config.disable_link_spheres(link_name)


def _ik_solver(
    device_cfg: DeviceCfg,
    mesh_collision_check: bool,
    scene_model: Optional[SceneCfg] = None,
    use_cuda_graph: bool = False,
) -> IKSolver:
    config = IKSolverCfg.create(
        robot="franka.yml",
        scene_model=scene_model if scene_model is not None else palm_box(),
        mesh_collision_check=mesh_collision_check,
        device_cfg=device_cfg,
        num_seeds=32,
        use_cuda_graph=use_cuda_graph,
    )
    solver = IKSolver(config)
    _disable_gripper_spheres(solver.kinematics)
    return solver


@pytest.fixture(scope="module")
def sphere_ik_solver(cuda_device_cfg) -> IKSolver:
    return _ik_solver(cuda_device_cfg, mesh_collision_check=False)


@pytest.fixture(scope="module")
def mesh_ik_solver(cuda_device_cfg) -> IKSolver:
    return _ik_solver(cuda_device_cfg, mesh_collision_check=True)


def _goal(tool_frames, device_cfg: DeviceCfg) -> GoalToolPose:
    pose = Pose.from_list(GOAL_POSE, device_cfg=device_cfg)
    return GoalToolPose.from_poses(
        {tool_frames[0]: pose.unsqueeze(1)}, ordered_tool_frames=tool_frames
    )


def _solve(solver: IKSolver, device_cfg: DeviceCfg):
    goal = _goal(solver.kinematics.tool_frames, device_cfg)
    return solver.solve_pose(goal_tool_poses=goal, return_seeds=1)


# =============================================================================
# IK solver
# =============================================================================


def test_sphere_only_ik_accepts_mesh_colliding_solution(sphere_ik_solver, cuda_device_cfg):
    """Without the mesh check the box is invisible to the gripper, so IK succeeds."""
    result = _solve(sphere_ik_solver, cuda_device_cfg)
    assert result.success.any().item()


def test_mesh_collision_check_rejects_solution(mesh_ik_solver, cuda_device_cfg):
    """With the mesh check every solution is infeasible, so IK fails."""
    result = _solve(mesh_ik_solver, cuda_device_cfg)
    assert not result.success.any().item()


def test_mesh_collision_check_accepts_clear_scene(mesh_ik_solver, cuda_device_cfg):
    """The same solver succeeds once the box is moved away, so the failure is the box."""
    checker = mesh_ik_solver.scene_collision_checker
    try:
        checker.load_collision_model(palm_box(offset=(0.0, 1.5, 0.0)))
        result = _solve(mesh_ik_solver, cuda_device_cfg)
        assert result.success.any().item()
    finally:
        checker.load_collision_model(palm_box())


# =============================================================================
# Motion planner
# =============================================================================


def _planner(
    device_cfg: DeviceCfg,
    mesh_collision_check: bool,
    disable_gripper_spheres: bool = True,
    scene_model: Optional[SceneCfg] = None,
    use_cuda_graph: bool = False,
) -> MotionPlanner:
    config = MotionPlannerCfg.create(
        robot="franka.yml",
        scene_model=scene_model if scene_model is not None else palm_box(),
        mesh_collision_check=mesh_collision_check,
        device_cfg=device_cfg,
        num_ik_seeds=32,
        num_trajopt_seeds=4,
        use_cuda_graph=use_cuda_graph,
    )
    planner = MotionPlanner(config)
    if disable_gripper_spheres:
        _disable_gripper_spheres(planner.kinematics)
    return planner


def _start(planner: MotionPlanner) -> JointState:
    return JointState.from_position(
        planner.default_joint_state.position.clone().unsqueeze(0),
        joint_names=planner.joint_names,
    )


def _plan(planner: MotionPlanner, device_cfg: DeviceCfg):
    goal = _goal(planner.kinematics.tool_frames, device_cfg)
    return planner.plan_pose(goal_tool_poses=goal, current_state=_start(planner), max_attempts=3)


def test_sphere_only_plan_pose_reaches_mesh_colliding_goal(cuda_device_cfg):
    """Without the mesh check the planner reaches the goal through the box."""
    result = _plan(_planner(cuda_device_cfg, mesh_collision_check=False), cuda_device_cfg)
    assert result is not None
    assert result.success.any().item()


def test_mesh_collision_check_plan_pose_fails(cuda_device_cfg):
    """With the mesh check no goal configuration is feasible, so planning fails."""
    result = _plan(_planner(cuda_device_cfg, mesh_collision_check=True), cuda_device_cfg)
    assert result is None or not result.success.any().item()


def test_trajopt_checks_the_interpolated_trajectory(cuda_device_cfg):
    """The interpolated trajectory, which is sent to the robot, is checked as well."""
    planner = _planner(cuda_device_cfg, mesh_collision_check=True)
    rollout = planner.trajopt_solver.additional_metrics_rollouts["interpolated_rollout"]
    assert rollout.metrics_constraint_manager.get_cost("mesh_collision").enabled
    assert rollout.metrics_transition_model.robot_model.compute_link_transforms


def test_mesh_collision_check_plan_grasp_lets_gripper_touch_object(cuda_device_cfg):
    """plan_grasp disables collision of the grasp contact links on their meshes as well.

    The grasp pose pushes the palm into the box, which plan_grasp allows for the gripper while
    it plans to the grasp, moves in and lifts. With the mesh check on, the gripper meshes must
    be disabled during those phases and restored afterwards.
    """
    planner = _planner(cuda_device_cfg, mesh_collision_check=True, disable_gripper_spheres=False)
    grasp = _goal(planner.kinematics.tool_frames, cuda_device_cfg)
    result = planner.plan_grasp(grasp_poses=grasp, current_state=_start(planner))
    assert result.success.any().item(), result.status
    assert result.grasp_success.any().item()
    assert result.lift_success.any().item()

    # The gripper meshes are active again: with only the spheres off, the box still blocks it.
    _disable_gripper_spheres(planner.kinematics)
    result = _plan(planner, cuda_device_cfg)
    assert result is None or not result.success.any().item()


# =============================================================================
# Attached objects
# =============================================================================


def _attach_rod(solver: IKSolver, result, device_cfg: DeviceCfg) -> None:
    """Attach a rod to the hand at an IK solution, with the rod's spheres disabled.

    The rod reaches from the fingertips (z = 0.19) down to z = 0.0, into the top of the box
    placed 25 cm below the palm box position (top at z = 0.015).
    """
    rod = Cuboid(name="rod", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.02, 0.02, 0.19])
    rod_pose = Pose.from_list([0.5, 0.0, 0.095, 1.0, 0.0, 0.0, 0.0], device_cfg=device_cfg)
    solution = solver.kinematics.get_active_js(result.js_solution)
    grasp_state = JointState.from_position(
        solution.position.reshape(-1, solution.position.shape[-1])[0:1]
    )
    solver.core.attachment_manager.attach(
        grasp_state,
        [rod],
        num_spheres=4,
        sphere_fit_type=SphereFitType.VOXEL,
        world_objects_pose_offset=rod_pose,
    )
    solver.kinematics.config.kinematics_config.disable_link_spheres("attached_object")


def test_attached_object_mesh_is_checked(cuda_device_cfg):
    """An attached object's mesh takes part in the mesh check until it is detached.

    A box sits 25 cm below the palm box position, clear of the robot. A rod attached to the
    hand reaches into it. The rod's spheres are disabled, so only its mesh can see the box.
    """
    # franka.yml reserves triangle slots for attached_object.
    config = IKSolverCfg.create(
        robot="franka.yml",
        scene_model=palm_box(offset=(0.0, 0.0, -0.25)),
        mesh_collision_check=True,
        device_cfg=cuda_device_cfg,
        num_seeds=32,
        use_cuda_graph=False,
    )
    solver = IKSolver(config)
    result = _solve(solver, cuda_device_cfg)
    assert result.success.any().item()

    _attach_rod(solver, result, cuda_device_cfg)
    assert not _solve(solver, cuda_device_cfg).success.any().item()

    solver.core.attachment_manager.detach()
    assert _solve(solver, cuda_device_cfg).success.any().item()


# =============================================================================
# CUDA graphs
# =============================================================================


def test_captured_ik_solver_follows_scene_and_attachments(cuda_device_cfg):
    """With CUDA graphs, the mesh check sees scenes and attached objects changed after capture.

    The first solve captures the graphs with the box clear of the robot. The box is then
    moved into the palm and back, and the rod of ``test_attached_object_mesh_is_checked`` is
    attached and detached. The gripper spheres are disabled, so only the meshes see the box.
    """
    clear_scene = palm_box(offset=(0.0, 0.0, -0.25))
    solver = _ik_solver(
        cuda_device_cfg, mesh_collision_check=True, scene_model=clear_scene, use_cuda_graph=True
    )
    result = _solve(solver, cuda_device_cfg)
    assert result.success.any().item()

    solver.update_world(palm_box())
    assert not _solve(solver, cuda_device_cfg).success.any().item()

    solver.update_world(clear_scene)
    result = _solve(solver, cuda_device_cfg)
    assert result.success.any().item()

    _attach_rod(solver, result, cuda_device_cfg)
    assert not _solve(solver, cuda_device_cfg).success.any().item()

    solver.core.attachment_manager.detach()
    assert _solve(solver, cuda_device_cfg).success.any().item()


def test_captured_planner_follows_scene_updates(cuda_device_cfg):
    """With CUDA graphs, plan_pose rejects a goal once a box is moved into the palm."""
    planner = _planner(
        cuda_device_cfg,
        mesh_collision_check=True,
        scene_model=palm_box(offset=(0.0, 1.5, 0.0)),
        use_cuda_graph=True,
    )
    result = _plan(planner, cuda_device_cfg)
    assert result is not None
    assert result.success.any().item()

    planner.update_world(palm_box())
    result = _plan(planner, cuda_device_cfg)
    assert result is None or not result.success.any().item()
