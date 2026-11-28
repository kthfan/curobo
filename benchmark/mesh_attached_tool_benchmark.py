# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Plan a flat attached tool a few millimeters from obstacles, fitted with few or many spheres.

Franka holds a flat tool (``--tool_dims``, default 50 x 10 x 200 mm) below its fingers, along
the hand's z axis. The tool is attached with
:meth:`~curobo._src.collision.attachment_manager.AttachmentManager.attach`, which fits
``--num_spheres`` spheres and writes the tool mesh for the exact mesh check. Few spheres are
cheap but much larger than the tool; many small spheres follow it closely.

Scenarios, at each mesh clearance ``c`` (the gap between the tool mesh and the obstacles):

- ``insert``: a slab with a hole ``c`` wider than the tool on every side. The tool starts
  ``--start_gap`` above the slab and moves straight down (``plan_pose_waypoints``) until its
  tip is ``--depth`` into the hole.
- ``beside``: the tool starts and ends next to a wall parallel to its broad face, at ``c``
  from it. Free path (``plan_pose``).

The goals are top-down tool poses with a random yaw. Configurations whose measured mesh
clearance falls below ``c`` (another link closer than the tool) are dropped. Both mesh checks
are on in both modes; the graph planner is PRM.

Example::

    python benchmark/mesh_attached_tool_benchmark.py --save_path out
"""

# Standard Library
import argparse
import math
import os
import time
from typing import Dict, List, Tuple

# Third Party
import numpy as np
import torch

# Shared helpers: Python puts this script's folder on the import path.
from mesh_slot_benchmark import (
    TABLE,
    end_link_vertices,
    goal_from,
    load_robot,
    mesh_clearance,
    solve_ik,
    sphere_box_clearance,
    top_down_rotation,
)
from scipy.spatial.transform import Rotation

# CuRobo
from curobo._src.geom.sphere_fit.types import SphereFitType
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.pose import Pose
from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util.warp import init_warp
from curobo._src.util_file import join_path, write_yaml

ROBOT = "franka.yml"
BASE_LINK = "panda_link0"
GRIPPER_LINKS = ["panda_hand", "panda_leftfinger", "panda_rightfinger"]
SCENARIOS = ("insert", "beside")
PLATE_THICKNESS = 0.02
#: Half-size of the slab around the hole, and of the wall beside the tool, in meters.
SLAB_HALF = 0.15
WALL_HALF = 0.1


def tool_robot(max_spheres: int) -> Dict:
    """Franka with room for ``max_spheres`` tool spheres; the tool clears the right finger."""
    robot = load_robot(ROBOT, BASE_LINK)
    kinematics = robot["robot_cfg"]["kinematics"]
    kinematics["extra_collision_spheres"]["attached_object"] = max_spheres
    ignore = kinematics["self_collision_ignore"]
    ignore.setdefault("panda_rightfinger", [])
    if "attached_object" not in ignore["panda_rightfinger"]:
        ignore["panda_rightfinger"].append("attached_object")
    return robot


def make_planner(robot: Dict, mesh_only_success: bool, args) -> MotionPlanner:
    """Planner with both mesh checks, PRM, and room for the table and the plates."""
    config = MotionPlannerCfg.create(
        robot=robot,
        scene_model=SceneCfg(cuboid=[TABLE]),
        collision_cache={"cuboid": 12},
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


def attach_tool(planner: MotionPlanner, num_spheres: int, args) -> torch.Tensor:
    """Attach the tool with ``num_spheres`` spheres; returns them in the hand frame ``[n, 4]``."""
    manager = planner.attachment_manager
    manager.detach()
    planner.update_world(SceneCfg(cuboid=[TABLE]))
    state = JointState.from_position(
        planner.default_joint_state.position.view(1, -1).clone(), joint_names=planner.joint_names
    )
    hand = planner.kinematics.compute_kinematics(state).tool_poses.get_link_pose("panda_hand")
    hand = Pose(position=hand.position.view(1, 3), quaternion=hand.quaternion.view(1, 4))
    local = Pose.from_list([0.0, 0.0, args.tool_center, 1.0, 0.0, 0.0, 0.0], DeviceCfg())
    tool = Cuboid(name="tool", pose=[0, 0, 0, 1, 0, 0, 0], dims=list(args.tool_dims))
    manager.attach(
        state,
        [tool],
        num_spheres=num_spheres,
        sphere_fit_type=SphereFitType.MORPHIT,
        world_objects_pose_offset=hand.multiply(local),
    )
    kparams = planner.kinematics.config.kinematics_config
    index = kparams.get_sphere_index_from_link_name("attached_object")
    spheres = kparams.link_spheres[0, index]
    return spheres[spheres[:, 3] > 0.0].clone()


def tool_cuboid_local(args) -> Cuboid:
    """The tool in the hand frame."""
    return Cuboid(
        name="tool", pose=[0.0, 0.0, args.tool_center, 1, 0, 0, 0], dims=list(args.tool_dims)
    )


def world_cuboid(
    name: str, position: np.ndarray, rotation: np.ndarray, center: List[float], dims: List[float]
) -> Cuboid:
    """Cuboid given in the tool frame at ``position``, ``rotation``."""
    quat = Rotation.from_matrix(rotation).as_quat()
    world = position + rotation @ np.asarray(center)
    return Cuboid(
        name=name,
        pose=[*world.tolist(), quat[3], quat[0], quat[1], quat[2]],
        dims=[float(d) for d in dims],
    )


def slab_with_hole(
    position: np.ndarray, rotation: np.ndarray, clearance: float, args
) -> List[Cuboid]:
    """Slab around a hole ``clearance`` wider than the tool, for the tool at its goal pose."""
    ax = 0.5 * args.tool_dims[0] + clearance
    ay = 0.5 * args.tool_dims[1] + clearance
    tip = args.tool_center + 0.5 * args.tool_dims[2]
    top = tip - args.depth
    z, dz = top + 0.5 * args.wall, args.wall
    side = SLAB_HALF - ax
    blocks = [
        ([ax + 0.5 * side, 0.0, z], [side, 2 * SLAB_HALF, dz]),
        ([-ax - 0.5 * side, 0.0, z], [side, 2 * SLAB_HALF, dz]),
        ([0.0, ay + 0.5 * (SLAB_HALF - ay), z], [2 * ax, SLAB_HALF - ay, dz]),
        ([0.0, -ay - 0.5 * (SLAB_HALF - ay), z], [2 * ax, SLAB_HALF - ay, dz]),
    ]
    return [
        world_cuboid(f"slab_{i}", position, rotation, center, dims)
        for i, (center, dims) in enumerate(blocks)
    ]


def wall_beside(
    name: str,
    position: np.ndarray,
    rotation: np.ndarray,
    clearance: float,
    gripper_tip: float,
    args,
) -> Cuboid:
    """Wall ``clearance`` from the tool's broad face, below the fingers."""
    y = 0.5 * args.tool_dims[1] + clearance + 0.5 * PLATE_THICKNESS
    z_low = gripper_tip + clearance
    z_high = args.tool_center + 0.5 * args.tool_dims[2] + 0.03
    return world_cuboid(
        name,
        position,
        rotation,
        [0.0, y, 0.5 * (z_low + z_high)],
        [2 * WALL_HALF, PLATE_THICKNESS, z_high - z_low],
    )


def sample_cases(
    planner: MotionPlanner, scenario: str, args, rng: np.random.Generator
) -> List[Dict]:
    """Start and goal tool poses and configurations, solved in the free scene."""
    planner.update_world(SceneCfg(cuboid=[TABLE]))
    tip = args.tool_center + 0.5 * args.tool_dims[2]
    low_z = tip + args.wall + 0.08 if scenario == "insert" else tip + 0.05
    cases = []
    attempts = 0
    while len(cases) < args.num_goals and attempts < 30 * args.num_goals:
        attempts += 1
        position = rng.uniform([0.35, -0.35, low_z], [0.65, 0.35, low_z + 0.2])
        rotation = top_down_rotation(rng.uniform(-np.pi, np.pi))
        if scenario == "insert":
            start_position = position + np.array([0.0, 0.0, args.depth + args.start_gap])
            start_rotation = rotation
        else:
            start_position = rng.uniform([0.35, -0.35, low_z], [0.65, 0.35, low_z + 0.2])
            if np.linalg.norm(start_position[:2] - position[:2]) < 0.25:
                continue
            start_rotation = top_down_rotation(rng.uniform(-np.pi, np.pi))
        q_start = solve_ik(planner, goal_from(start_position, start_rotation, planner))
        if q_start is None:
            continue
        q_goal = solve_ik(planner, goal_from(position, rotation, planner), near=q_start)
        if q_goal is None:
            continue
        cases.append(
            {
                "position": position,
                "rotation": rotation,
                "start_position": start_position,
                "start_rotation": start_rotation,
                "q_start": q_start,
                "q_goal": q_goal,
            }
        )
    return cases


def case_scene(case: Dict, scenario: str, clearance: float, gripper_tip: float, args):
    """Obstacles of one case: the table and the slab, or a wall at the start and the goal."""
    if scenario == "insert":
        return [TABLE, *slab_with_hole(case["position"], case["rotation"], clearance, args)]
    return [
        TABLE,
        wall_beside(
            "wall_start",
            case["start_position"],
            case["start_rotation"],
            clearance,
            gripper_tip,
            args,
        ),
        wall_beside(
            "wall_goal", case["position"], case["rotation"], clearance, gripper_tip, args
        ),
    ]


def plan(planner: MotionPlanner, case: Dict, scenario: str, args) -> Tuple[bool, float]:
    """Plan one case; returns success and time in ms."""
    start = JointState.from_position(case["q_start"].clone(), joint_names=planner.joint_names)
    goal = goal_from(case["position"], case["rotation"], planner)
    t0 = time.perf_counter()
    if scenario == "insert":
        result = planner.plan_pose_waypoints(
            goal,
            start,
            path_tolerance=(args.path_tolerance, math.radians(2.0)),
            final_tolerance=(0.002, math.radians(2.0)),
            max_attempts=args.max_attempts,
        )
    else:
        result = planner.plan_pose(
            goal_tool_poses=goal, current_state=start, max_attempts=args.max_attempts
        )
    torch.cuda.synchronize()
    elapsed = 1000.0 * (time.perf_counter() - t0)
    return result is not None and bool(result.success.any()), elapsed


def tool_sphere_stats(spheres: torch.Tensor, args) -> Dict[str, float]:
    """Count, largest radius and largest protrusion beyond the tool box, in mm."""
    box = tool_cuboid_local(args)
    center = torch.as_tensor(box.pose[:3], device=spheres.device)
    half = 0.5 * torch.as_tensor(box.dims, device=spheres.device)
    q = (spheres[:, :3] - center).abs() - half
    sdf = q.clamp(min=0.0).norm(dim=-1) + q.max(dim=-1).values.clamp(max=0.0)
    protrusion = sdf + spheres[:, 3]
    return {
        "num_spheres": int(spheres.shape[0]),
        "max_radius_mm": 1000.0 * float(spheres[:, 3].max()),
        "max_protrusion_mm": 1000.0 * float(protrusion.max()),
    }


def main() -> None:
    """Run both scenarios for each sphere count and clearance, in both modes."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=SCENARIOS)
    parser.add_argument("--num_spheres", type=int, nargs="+", default=[3, 30])
    parser.add_argument("--clearances", type=float, nargs="+", default=[0.005, 0.01, 0.02])
    parser.add_argument("--tool_dims", type=float, nargs=3, default=[0.05, 0.01, 0.2])
    parser.add_argument(
        "--tool_center", type=float, default=0.19, help="Tool center on the hand z axis (m)."
    )
    parser.add_argument("--depth", type=float, default=0.05, help="Insertion depth (m).")
    parser.add_argument("--wall", type=float, default=0.03, help="Slab thickness (m).")
    parser.add_argument("--start_gap", type=float, default=0.01)
    parser.add_argument("--path_tolerance", type=float, default=0.002)
    parser.add_argument("--num_goals", type=int, default=20)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument("--max_attempts", type=int, default=4)
    parser.add_argument("--clearance_tolerance", type=float, default=0.0005)
    parser.add_argument("--seed", type=int, default=5)
    parser.add_argument("--cuda_graph", type=int, default=1)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    setup_curobo_logger("error")
    torch.manual_seed(2)
    init_warp(device_cfg=DeviceCfg())
    rng = np.random.default_rng(args.seed)

    robot = tool_robot(max(args.num_spheres))
    planners = {mode: make_planner(robot, mode, args) for mode in (False, True)}
    reference = planners[False]
    gripper_tip = float(end_link_vertices(robot, GRIPPER_LINKS, "panda_hand")[:, 2].max())

    # The same cases for every sphere count, solved with the many-sphere tool.
    attach_tool(reference, max(args.num_spheres), args)
    all_cases = {s: sample_cases(reference, s, args, rng) for s in args.scenarios}
    print({s: len(c) for s, c in all_cases.items()}, "cases", flush=True)

    rows = []
    for num_spheres in args.num_spheres:
        spheres = {mode: attach_tool(p, num_spheres, args) for mode, p in planners.items()}
        fit = tool_sphere_stats(spheres[False], args)
        print(f"{num_spheres} requested spheres: {fit}", flush=True)
        for scenario in args.scenarios:
            cases = all_cases[scenario]
            for clearance in args.clearances:
                row = {
                    "scenario": scenario,
                    "requested_spheres": num_spheres,
                    **fit,
                    "clearance_mm": 1000.0 * clearance,
                }
                sphere_clear, results, dropped = [], {False: [], True: []}, 0
                for case in cases:
                    obstacles = case_scene(case, scenario, clearance, gripper_tip, args)
                    scene = SceneCfg(cuboid=obstacles)
                    reference.update_world(scene)
                    actual = min(
                        mesh_clearance(reference, case["q_start"]),
                        mesh_clearance(reference, case["q_goal"]),
                    )
                    if actual < clearance - args.clearance_tolerance:
                        dropped += 1
                        continue
                    nearest = []
                    for key in ("q_start", "q_goal"):
                        js = JointState.from_position(case[key], joint_names=reference.joint_names)
                        robot_spheres = reference.kinematics.compute_kinematics(js).robot_spheres
                        nearest.append(
                            sphere_box_clearance(robot_spheres.view(-1, 4), obstacles[1:])
                        )
                    sphere_clear.append(min(nearest))
                    for mode, planner in planners.items():
                        planner.update_world(scene)
                        results[mode].append(plan(planner, case, scenario, args))
                row["goals"] = len(sphere_clear)
                row["dropped"] = dropped
                if sphere_clear:
                    row["sphere_clearance_mm"] = 1000.0 * float(np.median(sphere_clear))
                    for mode, values in results.items():
                        tag = "meshes" if mode else "spheres"
                        values = np.asarray(values, dtype=np.float64)
                        row[f"plan_{tag}_%"] = 100.0 * float(values[:, 0].mean())
                        row[f"plan_{tag}_ms"] = float(np.median(values[:, 1]))
                rows.append(row)
                print(row, flush=True)

    lines = [
        f"GPU: {torch.cuda.get_device_name(0)}. Franka with a "
        f"{' x '.join(f'{1000 * d:.0f}' for d in args.tool_dims)} mm tool; both mesh checks, "
        f"margin {1000 * args.margin:.0f} mm; `max_attempts={args.max_attempts}`; insertion "
        f"{1000 * args.depth:.0f} mm deep from {1000 * args.start_gap:.0f} mm above, path "
        f"tolerance {1000 * args.path_tolerance:.0f} mm.",
        "",
        "| scenario | tool spheres | largest radius (mm) | protrusion (mm) | mesh clearance (mm) "
        "| cases | sphere clearance (mm) | plan, spheres decide (%) | plan, meshes decide (%) "
        "| ms, spheres | ms, meshes |",
        "|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in rows:
        prefix = (
            f"| {r['scenario']} | {r['num_spheres']} | {r['max_radius_mm']:.0f} | "
            f"{r['max_protrusion_mm']:.0f} | {r['clearance_mm']:.0f} | {r['goals']} | "
        )
        if not r["goals"]:
            lines.append(prefix + " | | | | |")
            continue
        lines.append(
            prefix + f"{r['sphere_clearance_mm']:.1f} | {r['plan_spheres_%']:.0f} | "
            f"{r['plan_meshes_%']:.0f} | {r['plan_spheres_ms']:.0f} | {r['plan_meshes_ms']:.0f} |"
        )
    text = "\n".join(lines)
    print(text)
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        with open(join_path(args.save_path, "mesh_attached_tool_benchmark.md"), "w") as f:
            f.write(text + "\n")
        write_yaml({"rows": rows}, join_path(args.save_path, "mesh_attached_tool_benchmark.yml"))
    for planner in planners.values():
        planner.destroy()


if __name__ == "__main__":
    main()
