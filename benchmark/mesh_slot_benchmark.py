# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Plan into slots whose walls come within a few millimeters of the gripper meshes.

Spheres are larger than the meshes they cover, so in a slot that the meshes clear by a few
millimeters, the spheres overlap the walls and reject the goal. This benchmark measures how
often IK and ``plan_pose`` reach such goals with the spheres deciding success and with
``mesh_only_success``, at a set of mesh clearances.

Goals are top-down tool poses above a table, with a random yaw. For each goal, plates are
placed around the links rigidly attached to the tool frame (the *end links*: Franka's hand and
fingers, UR10e's ``wrist_3_link``), parallel to the tool axes, so that the end-link meshes
clear them by ``clearance``:

- ``x``: two plates across the tool x axis (Franka: the hand's narrow side),
- ``y``: two plates across the tool y axis (Franka: the finger direction),
- ``pocket``: all four.

The plates cover the end links from ``clearance`` below their top (toward the arm) to past
their tip. The true mesh clearance of the goal configuration is measured with the exact mesh
distance query; goals where another link comes closer than ``clearance`` are dropped. Plans
start from a configuration ``--approach`` meters straight above the goal, chosen as the IK
solution closest to the goal configuration.

Both exact mesh checks are on in both modes, with margin ``--margin``. The graph planner is
PRM, which validates with the meshes under ``mesh_only_success``.

Example::

    python benchmark/mesh_slot_benchmark.py --robot_files franka.yml ur10e.yml --save_path out
"""

# Standard Library
import argparse
import copy
import os
import time
from typing import Dict, List, Tuple

# Third Party
import numpy as np
import torch
from scipy.spatial.transform import Rotation

# CuRobo
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util.warp import init_warp
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml, write_yaml

#: Base link (left out of the mesh scene check: it sits on the table) and end links per robot.
ROBOTS = {
    "franka.yml": ("panda_link0", ["panda_hand", "panda_leftfinger", "panda_rightfinger"]),
    "ur10e.yml": ("base_link", ["wrist_3_link"]),
}
SLOT_KINDS = ("x", "y", "pocket")
PLATE_THICKNESS = 0.02
#: How far the plates reach past the end links along the slot and past their tip.
PLATE_OVERHANG = 0.03
TABLE = Cuboid(name="table", pose=[0.0, 0.0, -0.1, 1, 0, 0, 0], dims=[2.0, 2.0, 0.2])


def load_robot(robot_file: str, base_link: str) -> Dict:
    """Robot dict with its base link left out of the scene mesh check."""
    robot = load_yaml(join_path(get_robot_configs_path(), robot_file))
    robot["robot_cfg"]["kinematics"]["mesh_collision_ignored_link_names"] = [base_link]
    return robot


def end_link_vertices(robot: Dict, end_links: List[str], tool_frame: str) -> np.ndarray:
    """Vertices of the end-link collision meshes in the tool frame, ``[n, 3]``.

    The end links are rigid with the tool frame, so any configuration gives the same result.
    """
    robot = copy.deepcopy(robot)
    robot["robot_cfg"]["kinematics"]["load_collision_meshes"] = True
    kinematics = Kinematics(
        KinematicsCfg.from_robot_yaml_file(robot, device_cfg=DeviceCfg()),
        compute_link_transforms=True,
    )
    kparams = kinematics.config.kinematics_config
    state = kinematics.compute_kinematics(kinematics.default_joint_state)
    transforms = state.link_transforms.view(-1, kparams.num_links, 3, 4)[0].cpu().numpy()
    tool = transforms[kparams.link_name_to_idx_map[tool_frame]]
    meshes = kinematics.get_robot_collision_link_meshes()
    points = []
    for name in end_links:
        vertices = np.asarray(meshes[name].get_trimesh_mesh(transform_with_pose=True).vertices)
        tf = transforms[kparams.link_name_to_idx_map[name]]
        world = vertices @ tf[:, :3].T + tf[:, 3]
        points.append((world - tool[:, 3]) @ tool[:, :3])
    return np.concatenate(points, axis=0)


def slot_plates(
    position: np.ndarray, rotation: np.ndarray, extents: np.ndarray, kind: str, clearance: float
) -> List[Cuboid]:
    """Plates around the end links of a tool at ``position``, ``rotation`` (world frame).

    Args:
        position: Tool position ``[3]``.
        rotation: Tool rotation matrix ``[3, 3]``.
        extents: End-link bounds in the tool frame, ``[[x, y, z] min, [x, y, z] max]``.
        kind: ``"x"``, ``"y"`` or ``"pocket"``.
        clearance: Gap between the end-link meshes and the plates, in meters.
    """
    low, high = extents
    t = PLATE_THICKNESS
    z_low = low[2] + clearance
    z_high = high[2] + PLATE_OVERHANG
    z_center, z_dim = 0.5 * (z_low + z_high), z_high - z_low
    center = 0.5 * (low + high)
    size = high - low
    plates = []
    if kind in ("x", "pocket"):
        y_dim = size[1] + 2.0 * (clearance + t)
        for sign, bound in ((1.0, high[0]), (-1.0, low[0])):
            x = bound + sign * (clearance + 0.5 * t)
            plates.append(([x, center[1], z_center], [t, y_dim, z_dim]))
    if kind in ("y", "pocket"):
        x_dim = size[0] + 2.0 * (clearance + t)
        for sign, bound in ((1.0, high[1]), (-1.0, low[1])):
            y = bound + sign * (clearance + 0.5 * t)
            plates.append(([center[0], y, z_center], [x_dim, t, z_dim]))
    quat = Rotation.from_matrix(rotation).as_quat()  # xyzw
    wxyz = [quat[3], quat[0], quat[1], quat[2]]
    cuboids = []
    for i, (local, dims) in enumerate(plates):
        world = position + rotation @ np.asarray(local)
        cuboids.append(
            Cuboid(name=f"plate_{i}", pose=[*world.tolist(), *wxyz], dims=list(map(float, dims)))
        )
    return cuboids


def sphere_box_clearance(spheres: torch.Tensor, cuboids: List[Cuboid]) -> float:
    """Smallest signed distance between active robot spheres ``[n, 4]`` and oriented boxes."""
    spheres = spheres[spheres[:, 3] > 0.0]
    best = float("inf")
    for cuboid in cuboids:
        pose = torch.as_tensor(cuboid.pose, device=spheres.device, dtype=spheres.dtype)
        w, x, y, z = pose[3:].tolist()
        matrix = Rotation.from_quat([x, y, z, w]).as_matrix()
        rotation = torch.as_tensor(matrix, device=spheres.device, dtype=spheres.dtype)
        local = (spheres[:, :3] - pose[:3]) @ rotation
        half = 0.5 * torch.as_tensor(cuboid.dims, device=spheres.device, dtype=spheres.dtype)
        q = local.abs() - half
        outside = q.clamp(min=0.0).norm(dim=-1)
        inside = q.max(dim=-1).values.clamp(max=0.0)
        best = min(best, float((outside + inside - spheres[:, 3]).min()))
    return best


def top_down_rotation(yaw: float) -> np.ndarray:
    """Tool rotation with its z axis pointing down and the given yaw about the world z axis."""
    return Rotation.from_euler("z", yaw).as_matrix() @ Rotation.from_euler("x", np.pi).as_matrix()


def goal_from(position: np.ndarray, rotation: np.ndarray, planner: MotionPlanner) -> GoalToolPose:
    """Single-goal tool pose."""
    quat = Rotation.from_matrix(rotation).as_quat()
    pose = Pose.from_list(
        [*position.tolist(), quat[3], quat[0], quat[1], quat[2]], device_cfg=DeviceCfg()
    )
    frames = planner.kinematics.tool_frames
    return GoalToolPose.from_poses({frames[0]: pose.unsqueeze(1)}, ordered_tool_frames=frames)


def solve_ik(planner: MotionPlanner, goal: GoalToolPose, near: torch.Tensor = None):
    """Active joint positions of an IK solution, the one closest to ``near`` if given."""
    solver = planner.ik_solver
    solver.reset_seed()
    result = solver.solve_pose(goal_tool_poses=goal, return_seeds=solver.config.num_seeds)
    success = result.success.view(-1)
    if not bool(success.any()):
        return None
    js = planner.kinematics.get_active_js(result.js_solution)
    q = js.position.view(success.shape[0], -1)[success]
    if near is None:
        return q[0:1].clone()
    return q[(q - near).norm(dim=-1).argmin()].view(1, -1).clone()


def mesh_clearance(planner: MotionPlanner, q: torch.Tensor) -> float:
    """Smallest exact mesh clearance of configuration ``q`` to the scene."""
    js = JointState.from_position(q.view(1, -1), joint_names=planner.joint_names)
    distance, _ = planner.ik_solver.get_mesh_collision_distance(js)
    return float(distance.min())


def sample_goals(
    planner: MotionPlanner, extents: np.ndarray, args, rng: np.random.Generator
) -> List[Dict]:
    """Reachable top-down goals with a start configuration ``--approach`` above each."""
    planner.update_world(SceneCfg(cuboid=[TABLE]))
    goals = []
    tip = extents[1][2]
    attempts = 0
    while len(goals) < args.num_goals and attempts < 20 * args.num_goals:
        attempts += 1
        position = rng.uniform([0.35, -0.35, tip + 0.05], [0.7, 0.35, tip + 0.3])
        rotation = top_down_rotation(rng.uniform(-np.pi, np.pi))
        q_goal = solve_ik(planner, goal_from(position, rotation, planner))
        if q_goal is None:
            continue
        above = position + np.array([0.0, 0.0, args.approach])
        q_start = solve_ik(planner, goal_from(above, rotation, planner), near=q_goal)
        if q_start is None:
            continue
        goals.append(
            {"position": position, "rotation": rotation, "q_goal": q_goal, "q_start": q_start}
        )
    return goals


def make_planner(robot: Dict, mesh_only_success: bool, args) -> MotionPlanner:
    """Planner with both mesh checks, PRM, and room for the table and four plates."""
    config = MotionPlannerCfg.create(
        robot=robot,
        scene_model=SceneCfg(cuboid=[TABLE]),
        collision_cache={"cuboid": 8},
        mesh_collision_check=True,
        mesh_self_collision_check=True,
        mesh_collision_margin=args.margin,
        mesh_self_collision_margin=args.margin,
        mesh_only_success=mesh_only_success,
        graph_planner_algorithm="prm",
        device_cfg=DeviceCfg(),
        use_cuda_graph=bool(args.cuda_graph),
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    return planner


def run_case(
    planners: Dict[bool, MotionPlanner], goal: Dict, scene: SceneCfg, args
) -> Dict[bool, Tuple[bool, bool, float]]:
    """IK success, plan success and plan time per mode for one goal in one scene."""
    out = {}
    for mode, planner in planners.items():
        planner.update_world(scene)
        target = goal_from(goal["position"], goal["rotation"], planner)
        ik = solve_ik(planner, target) is not None
        start = JointState.from_position(goal["q_start"].clone(), joint_names=planner.joint_names)
        t0 = time.perf_counter()
        result = planner.plan_pose(
            goal_tool_poses=target, current_state=start, max_attempts=args.max_attempts
        )
        torch.cuda.synchronize()
        elapsed = 1000.0 * (time.perf_counter() - t0)
        plan = result is not None and bool(result.success.any())
        out[mode] = (ik, plan, elapsed)
    return out


def bench_robot(robot_file: str, args, rng: np.random.Generator) -> Tuple[List[Dict], List[str]]:
    """All slot kinds and clearances for one robot."""
    base_link, end_links = ROBOTS[robot_file]
    robot = load_robot(robot_file, base_link)
    planners = {mode: make_planner(robot, mode, args) for mode in (False, True)}
    reference = planners[False]
    tool_frame = reference.kinematics.tool_frames[0]
    vertices = end_link_vertices(robot, end_links, tool_frame)
    extents = np.stack([vertices.min(axis=0), vertices.max(axis=0)])
    goals = sample_goals(reference, extents, args, rng)
    print(
        f"{robot_file}: end-link bounds in {tool_frame} (mm) "
        f"{np.round(1000 * extents, 1).tolist()}, {len(goals)} goals",
        flush=True,
    )

    rows = []
    for kind in args.kinds:
        for clearance in args.clearances:
            stats = {"kind": kind, "clearance_mm": 1000.0 * clearance}
            sphere, mesh = [], []
            results = {False: [], True: []}
            dropped = 0
            for goal in goals:
                plates = slot_plates(
                    goal["position"], goal["rotation"], extents, kind, clearance
                )
                scene = SceneCfg(cuboid=[TABLE, *plates])
                reference.update_world(scene)
                actual = mesh_clearance(reference, goal["q_goal"])
                if actual < clearance - args.clearance_tolerance:
                    dropped += 1
                    continue
                spheres = reference.kinematics.compute_kinematics(
                    JointState.from_position(goal["q_goal"], joint_names=reference.joint_names)
                ).robot_spheres.view(-1, 4)
                sphere.append(sphere_box_clearance(spheres, plates))
                mesh.append(actual)
                for mode, value in run_case(planners, goal, scene, args).items():
                    results[mode].append(value)
            n = len(sphere)
            stats["goals"] = n
            stats["dropped"] = dropped
            if n == 0:
                rows.append(stats)
                continue
            sphere = np.asarray(sphere)
            stats["mesh_clearance_mm"] = 1000.0 * float(np.median(mesh))
            stats["sphere_clearance_mm"] = 1000.0 * float(np.median(sphere))
            stats["sphere_overlap_%"] = 100.0 * float(np.mean(sphere < 0.0))
            for mode, values in results.items():
                tag = "meshes" if mode else "spheres"
                values = np.asarray(values, dtype=np.float64)
                stats[f"ik_{tag}_%"] = 100.0 * float(values[:, 0].mean())
                stats[f"plan_{tag}_%"] = 100.0 * float(values[:, 1].mean())
                stats[f"plan_{tag}_ms"] = float(np.median(values[:, 2]))
            rows.append(stats)
            print(f"{robot_file}: {stats}", flush=True)
    for planner in planners.values():
        planner.destroy()
    torch.cuda.empty_cache()

    lines = [
        f"### {robot_file}",
        "",
        f"{len(goals)} goals; end links {end_links}; mesh margin {1000 * args.margin:.0f} mm; "
        f"start {100 * args.approach:.0f} cm above the goal; `max_attempts={args.max_attempts}`.",
        "",
        "| slot | mesh clearance (mm) | goals | median sphere clearance (mm) | spheres overlap "
        "(%) | IK, spheres decide (%) | IK, meshes decide (%) | plan, spheres decide (%) | "
        "plan, meshes decide (%) | plan ms, spheres | plan ms, meshes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        if r["goals"] == 0:
            lines.append(f"| {r['kind']} | {r['clearance_mm']:.0f} | 0 | | | | | | | | |")
            continue
        lines.append(
            f"| {r['kind']} | {r['clearance_mm']:.0f} | {r['goals']} | "
            f"{r['sphere_clearance_mm']:.1f} | {r['sphere_overlap_%']:.0f} | "
            f"{r['ik_spheres_%']:.0f} | {r['ik_meshes_%']:.0f} | {r['plan_spheres_%']:.0f} | "
            f"{r['plan_meshes_%']:.0f} | {r['plan_spheres_ms']:.0f} | {r['plan_meshes_ms']:.0f} |"
        )
    lines.append("")
    return rows, lines


def main() -> None:
    """Run every robot, slot kind and clearance, then print and save the tables."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot_files", nargs="+", default=["franka.yml"], choices=list(ROBOTS))
    parser.add_argument("--kinds", nargs="+", default=list(SLOT_KINDS), choices=SLOT_KINDS)
    parser.add_argument(
        "--clearances", type=float, nargs="+", default=[0.005, 0.01, 0.02, 0.03]
    )
    parser.add_argument("--num_goals", type=int, default=20)
    parser.add_argument("--approach", type=float, default=0.2)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument(
        "--clearance_tolerance",
        type=float,
        default=0.0005,
        help="Drop goals whose measured mesh clearance falls this far below the target.",
    )
    parser.add_argument("--seed", type=int, default=3)
    parser.add_argument("--cuda_graph", type=int, default=1)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    setup_curobo_logger("error")
    torch.manual_seed(2)
    init_warp(device_cfg=DeviceCfg())
    rng = np.random.default_rng(args.seed)
    rows, lines = [], []
    for robot_file in args.robot_files:
        robot_rows, robot_lines = bench_robot(robot_file, args, rng)
        rows.append({"robot": robot_file, "cases": robot_rows})
        lines += robot_lines
    text = f"GPU: {torch.cuda.get_device_name(0)}.\n\n" + "\n".join(lines)
    print(text)
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        with open(join_path(args.save_path, "mesh_slot_benchmark.md"), "w") as f:
            f.write(text + "\n")
        write_yaml({"rows": rows}, join_path(args.save_path, "mesh_slot_benchmark.yml"))


if __name__ == "__main__":
    main()
