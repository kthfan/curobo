# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the exact robot-mesh self-collision check."""

# Third Party
import numpy as np
import pytest
import torch
import trimesh

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_collision import check_robot_mesh_self_collision
from curobo._src.geom.data.data_robot_mesh import RobotMeshData, mesh_self_collision_pairs
from curobo._src.geom.types import Mesh
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

DEVICE_CFG = DeviceCfg()


@pytest.fixture(scope="module", autouse=True)
def _warp():
    init_warp(device_cfg=DEVICE_CFG)


# =============================================================================
# Pair selection
# =============================================================================

#: Chain base(0) -r- a(1) -r- b(2) -fixed- c(3) -r- d(4), plus e(5) on a revolute joint from c.
NAMES = {"base": 0, "a": 1, "b": 2, "c": 3, "d": 4, "e": 5}
PARENTS = [0, 0, 1, 2, 3, 3]
FIXED = [False, False, False, True, False, False]


def _pairs(links, ignore=None):
    pairs = mesh_self_collision_pairs(links, NAMES, PARENTS, FIXED, ignore)
    inverse = {v: k for k, v in NAMES.items()}
    return {(inverse[a], inverse[b]) for a, b in pairs}


def test_pairs_skip_links_one_joint_apart():
    assert _pairs(["base", "a", "b"]) == {("base", "b")}


def test_pairs_treat_fixed_joints_as_one_body():
    """Link c is fixed to b, so it is one joint from a and d, and in one body with b."""
    pairs = _pairs(["a", "b", "c", "d", "e"])
    assert ("b", "c") not in pairs
    assert ("a", "c") not in pairs and ("c", "d") not in pairs
    # d and e hang off the same body on two joints.
    assert pairs == {("a", "d"), ("a", "e"), ("d", "e")}


def test_pairs_honour_the_ignore_list_in_either_direction():
    assert _pairs(["a", "d", "e"], ignore={"e": ["a"]}) == {("a", "d"), ("d", "e")}


def test_pairs_skip_links_without_geometry():
    assert _pairs(["a", "e"]) == {("a", "e")}


# =============================================================================
# Kernel
# =============================================================================


def _box(name: str, extents) -> Mesh:
    box = trimesh.creation.box(extents=extents)
    return Mesh(
        name=name,
        vertices=np.asarray(box.vertices),
        faces=np.asarray(box.faces),
        pose=[0, 0, 0, 1, 0, 0, 0],
    )


def _robot(obj_slots: int = 0) -> RobotMeshData:
    """Two 0.1 m cubes ``a`` (body 0) and ``b`` (body 1), and ``obj`` (body 2) with slots."""
    data = RobotMeshData.from_link_meshes(
        link_meshes={"a": _box("a", (0.1, 0.1, 0.1)), "b": _box("b", (0.1, 0.1, 0.1))},
        link_name_to_idx_map={"a": 0, "b": 1, "obj": 2},
        num_links=3,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
        extra_link_triangles={"obj": obj_slots} if obj_slots else None,
    )
    pairs = [(0, 1), (0, 2), (1, 2)] if obj_slots else [(0, 1)]
    data.set_self_collision_pairs(pairs)
    return data


def _transforms(*positions_per_config) -> torch.Tensor:
    """Identity-rotation transforms ``[batch, 1, 3, 3, 4]`` from link positions.

    Links left out of a configuration are placed far away.
    """
    batch = len(positions_per_config)
    tf = torch.zeros(batch, 1, 3, 3, 4, device=DEVICE_CFG.device)
    tf[..., 0, 0] = tf[..., 1, 1] = tf[..., 2, 2] = 1.0
    for i, positions in enumerate(positions_per_config):
        padded = list(positions) + [[5.0, 5.0, 5.0]] * (3 - len(positions))
        tf[i, 0, :, :, 3] = torch.tensor(padded, dtype=torch.float32)
    return tf


def _check(robot, tf, margin=0.0):
    coll, pair = check_robot_mesh_self_collision(robot, tf, margin=margin)
    return coll.tolist(), pair.tolist()


def test_overlapping_links_collide():
    robot = _robot()
    tf = _transforms([[0, 0, 0], [0.09, 0, 0]], [[0, 0, 0], [0.11, 0, 0]])
    assert _check(robot, tf) == ([1, 0], [0, -1])


def test_rotated_links_collide_by_their_geometry():
    """Cube b turned 45 degrees about z reaches 0.0707 m: it hits a at 0.115, not at 0.125."""
    robot = _robot()
    tf = _transforms([[0, 0, 0], [0.115, 0, 0]], [[0, 0, 0], [0.125, 0, 0]])
    c = np.sqrt(0.5)
    tf[:, 0, 1, :, :3] = torch.tensor([[c, -c, 0], [c, c, 0], [0, 0, 1]], dtype=torch.float32)
    assert _check(robot, tf)[0] == [1, 0]


def test_margin_is_a_clearance_between_links():
    robot = _robot()
    tf = _transforms([[0, 0, 0], [0.11, 0, 0]])
    assert _check(robot, tf, margin=0.02)[0] == [1]
    assert _check(robot, tf, margin=0.005)[0] == [0]


def test_disabled_link_skips_its_pairs():
    robot = _robot()
    tf = _transforms([[0, 0, 0], [0.05, 0, 0]])
    robot.disable_link_mesh("a")
    assert _check(robot, tf)[0] == [0]
    robot.enable_link_mesh("a")
    robot.disable_link_mesh("b")
    assert _check(robot, tf)[0] == [0]
    robot.enable_link_mesh("b")
    assert _check(robot, tf)[0] == [1]


def test_runtime_geometry_is_checked_until_cleared():
    """The object link has more slots than a cube, so its BVH is queried and must be refit."""
    robot = _robot(obj_slots=64)
    obj_pair = robot.self_collision_pairs.tolist().index([0, 2])
    tf = _transforms([[0, 0, 0], [1.0, 0, 0], [0.0, 0.2, 0]])
    assert _check(robot, tf)[0] == [0]

    rod = trimesh.creation.box(extents=(0.02, 0.2, 0.02))
    robot.update_link_mesh("obj", rod)
    assert _check(robot, tf) == ([0], [-1])
    tf = _transforms([[0, 0, 0], [1.0, 0, 0], [0.0, 0.1, 0]])
    assert _check(robot, tf) == ([1], [obj_pair])

    robot.clear_link_mesh("obj")
    assert _check(robot, tf)[0] == [0]


def test_pair_culling_matches_unculled_results():
    """Skipping pairs whose bounding spheres are apart never changes a result."""
    generator = torch.Generator().manual_seed(0)
    q = torch.randn(512 * 2, 4, generator=generator)
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
    rot = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(-1, 3, 3)
    pos = -0.1 + 0.2 * torch.rand(512 * 2, 3, 1, generator=generator)
    tf = _transforms(*([[]] * 512))
    tf[:, :, :2] = torch.cat([rot, pos], dim=-1).view(512, 1, 2, 3, 4).to(DEVICE_CFG.device)
    for margin in (0.0, 0.02):
        robot = _robot()
        culled = _check(robot, tf, margin)
        robot.link_bounds[:, 3] = 1.0e3
        reference = _check(robot, tf, margin)
        assert 0 < sum(reference[0]) < 512
        assert culled == reference


# =============================================================================
# Franka
# =============================================================================


@pytest.fixture(scope="module")
def franka_mesh_data():
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    robot = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    robot["robot_cfg"]["kinematics"]["load_collision_meshes"] = True
    config = KinematicsCfg.from_robot_yaml_file(robot, device_cfg=DeviceCfg())
    data = config.kinematics_config.get_robot_mesh_data()
    lookup = data.body_idx_to_link_name
    pairs = {tuple(sorted((lookup[a], lookup[b]))) for a, b in data.self_collision_pairs.tolist()}
    return data, pairs


def test_franka_checks_pairs_the_spheres_ignore(franka_mesh_data):
    """link5-link7 and link5-hand are ignored by the spheres but collide as meshes."""
    _, pairs = franka_mesh_data
    assert ("panda_link5", "panda_link7") in pairs
    assert ("panda_hand", "panda_link5") in pairs


def test_franka_skips_adjacent_links(franka_mesh_data):
    _, pairs = franka_mesh_data
    for pair in [
        ("panda_link1", "panda_link2"),
        ("panda_link5", "panda_link6"),
        ("panda_hand", "panda_link7"),
        ("panda_hand", "panda_leftfinger"),
        ("panda_leftfinger", "panda_rightfinger"),
        ("attached_object", "panda_leftfinger"),
    ]:
        assert pair not in pairs


# =============================================================================
# Constraint, collision checker and solvers on Franka
# =============================================================================

#: Franka configuration in which the hand mesh folds into link5. link5-hand is a pair the
#: sphere check ignores, and no pair it checks collides here. Found by sampling with
#: cpu_verify/verify_franka_self_collision.py; it holds under +-0.05 rad joint perturbations.
Q_FOLDED = [-0.731, 0.449, -0.621, -1.891, 0.02, 0.228, -0.034]


@pytest.fixture(scope="module")
def cuda_device_cfg():
    if torch.cuda.is_available():
        return DeviceCfg(device=torch.device("cuda:0"), dtype=torch.float32)
    pytest.skip("CUDA not available")


def _folded_and_default(device_cfg, kinematics) -> torch.Tensor:
    default = kinematics.default_joint_position.view(-1)[:7].tolist()
    return torch.tensor([[default], [Q_FOLDED]], **device_cfg.as_torch_dict())


def test_checker_rejects_mesh_self_collision(cuda_device_cfg):
    """The folded configuration passes the sphere checks and fails the mesh self-check."""
    from curobo._src.collision.collision_robot_scene import RobotSceneCollision
    from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg

    valid = {}
    for use_mesh in (False, True):
        checker = RobotSceneCollision(
            RobotSceneCollisionCfg.load_from_config(
                robot_config="franka.yml",
                device_cfg=cuda_device_cfg,
                use_mesh_self_collision_constraint=use_mesh,
            )
        )
        q = _folded_and_default(cuda_device_cfg, checker.kinematics)
        valid[use_mesh] = checker.validate(q).view(-1).tolist()
    assert valid == {False: [True, True], True: [True, False]}


def test_checker_validates_with_the_scene_mesh_constraint(cuda_device_cfg):
    """The checker combines the scene mesh constraint with the other terms in validate."""
    from curobo._src.collision.collision_robot_scene import RobotSceneCollision
    from curobo._src.collision.collision_robot_scene_cfg import RobotSceneCollisionCfg
    from curobo._src.geom.types import Cuboid, SceneCfg

    checker = RobotSceneCollision(
        RobotSceneCollisionCfg.load_from_config(
            robot_config="franka.yml",
            scene_model=SceneCfg(
                cuboid=[Cuboid(name="far", pose=[3, 0, 0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1])]
            ),
            device_cfg=cuda_device_cfg,
            use_mesh_constraint=True,
            mesh_constraint_ignored_links=["panda_link0"],
            use_mesh_self_collision_constraint=True,
        )
    )
    q = _folded_and_default(cuda_device_cfg, checker.kinematics)
    assert checker.validate(q).view(-1).tolist() == [True, False]


def test_default_configuration_collision_is_reported(cuda_device_cfg, caplog):
    """A robot whose default configuration self-collides gets a warning naming a pair."""
    from curobo._src.cost.cost_robot_mesh_self_collision_cfg import (
        RobotMeshSelfCollisionConstraintCfg,
    )
    from curobo._src.robot.kinematics.kinematics import Kinematics

    robot = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    kinematics_dict = robot["robot_cfg"]["kinematics"]
    kinematics_dict["load_collision_meshes"] = True
    kinematics_dict["cspace"]["default_joint_position"] = Q_FOLDED + [0.04, 0.04]
    kinematics = Kinematics(
        KinematicsCfg.from_robot_yaml_file(robot, device_cfg=cuda_device_cfg),
        compute_link_transforms=True,
    )
    cfg = RobotMeshSelfCollisionConstraintCfg(
        weight=cuda_device_cfg.to_device([1.0]), device_cfg=cuda_device_cfg
    )
    with caplog.at_level("WARNING"):
        cfg.initialize_from_kinematics(kinematics)
    assert "panda_hand" in caplog.text and "panda_link5" in caplog.text


def _ik_solver(device_cfg, mesh_self_collision_check: bool, use_cuda_graph: bool = False):
    from curobo._src.solver.solver_ik import IKSolver
    from curobo._src.solver.solver_ik_cfg import IKSolverCfg

    return IKSolver(
        IKSolverCfg.create(
            robot="franka.yml",
            mesh_self_collision_check=mesh_self_collision_check,
            device_cfg=device_cfg,
            num_seeds=16,
            use_cuda_graph=use_cuda_graph,
        )
    )


def test_solver_names_the_colliding_pair(cuda_device_cfg):
    from curobo._src.state.state_joint import JointState

    solver = _ik_solver(cuda_device_cfg, mesh_self_collision_check=True)
    rollout = solver.metrics_rollout
    assert rollout.metrics_constraint_manager.get_cost("mesh_self_collision").enabled
    q = _folded_and_default(cuda_device_cfg, solver.kinematics).view(2, 7)
    pairs = solver.get_mesh_self_collision_pairs(JointState.from_position(q))
    assert pairs[0] is None
    assert set(pairs[1]) == {"panda_hand", "panda_link5"}


def test_solver_needs_the_option_to_name_pairs(cuda_device_cfg):
    from curobo._src.state.state_joint import JointState

    solver = _ik_solver(cuda_device_cfg, mesh_self_collision_check=False)
    q = _folded_and_default(cuda_device_cfg, solver.kinematics).view(2, 7)
    with pytest.raises(ValueError, match="mesh_self_collision_check"):
        solver.get_mesh_self_collision_pairs(JointState.from_position(q))


@pytest.mark.parametrize("use_cuda_graph", [False, True])
def test_planner_rejects_a_self_colliding_goal(cuda_device_cfg, use_cuda_graph):
    """plan_cspace reaches the folded configuration only without the mesh self-check."""
    from curobo._src.motion.motion_planner import MotionPlanner
    from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
    from curobo._src.state.state_joint import JointState

    success = {}
    for check in (False, True):
        planner = MotionPlanner(
            MotionPlannerCfg.create(
                robot="franka.yml",
                mesh_self_collision_check=check,
                device_cfg=cuda_device_cfg,
                use_cuda_graph=use_cuda_graph,
            )
        )
        start = JointState.from_position(
            planner.default_joint_state.position.clone().view(1, -1),
            joint_names=planner.joint_names,
        )
        goal = JointState.from_position(
            torch.tensor([Q_FOLDED], **cuda_device_cfg.as_torch_dict()),
            joint_names=planner.joint_names,
        )
        result = planner.plan_cspace(goal_state=goal, current_state=start, max_attempts=3)
        success[check] = result is not None and bool(result.success.any().item())
    assert success == {False: True, True: False}


def test_replay_follows_link_mesh_updates():
    """A captured self-collision check sees link meshes written and toggled after capture."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA graphs need CUDA")
    from curobo._src.util.cuda_graph_util import GraphExecutor

    robot = _robot(obj_slots=64)
    tf = _transforms(
        [[0, 0, 0], [0.05, 0, 0], [0.0, 0.1, 0]], [[0, 0, 0], [1.0, 0, 0], [0.0, 0.1, 0]]
    )
    n = tf.shape[0]
    in_collision = torch.zeros(n, dtype=torch.int32, device=DEVICE_CFG.device)
    contact = torch.zeros(n, dtype=torch.int32, device=DEVICE_CFG.device)
    near = torch.zeros(n * 3, dtype=torch.int32, device=DEVICE_CFG.device)

    def check(t):
        return check_robot_mesh_self_collision(
            robot, t, in_collision=in_collision, contact_pair=contact, pair_near=near
        )

    executor = GraphExecutor(check, DEVICE_CFG.device, use_cuda_graph=True, clone_outputs=True)
    rod = trimesh.creation.box(extents=(0.02, 0.2, 0.02))
    steps = [
        lambda: None,
        lambda: robot.update_link_mesh("obj", rod),
        lambda: robot.disable_link_mesh("a"),
        lambda: robot.enable_link_mesh("a"),
        lambda: robot.clear_link_mesh("obj"),
    ]
    results = []
    for update in steps:
        update()
        replayed = executor(tf)
        eager = check_robot_mesh_self_collision(robot, tf)
        assert torch.equal(replayed[0], eager[0]) and torch.equal(replayed[1], eager[1])
        results.append(eager[0].tolist())
    # Without a, only b and the rod overlap, in the first configuration.
    assert results == [[1, 0], [1, 1], [1, 0], [1, 1], [1, 0]]
