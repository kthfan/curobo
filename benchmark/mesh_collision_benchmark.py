# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Benchmark the exact mesh collision check on Franka.

Three parts, on equivalent cuboid, mesh, dense mesh and voxel (ESDF) scenes made of a table
and a box:

- ``kernel``: time one mesh collision query for a batch of configurations, for several batch
  sizes, per-link triangle budgets and margins. Configurations are either free of sphere
  collisions (what solvers mostly evaluate) or uniformly random (many collide early).
- ``solver``: time IK (single and batched), motion planning (``plan_pose``) and an MPC step,
  with and without ``mesh_collision_check``.
- ``self``: time the mesh self-collision query on the same configurations, and batched IK and
  ``plan_pose`` with and without ``mesh_self_collision_check`` (cuboid scene).

Results are printed as Markdown tables and written to ``--save_path`` as Markdown and YAML.

Example::

    python benchmark/mesh_collision_benchmark.py --part all --save_path benchmark/log
"""

# Standard Library
import argparse
import os
import time
from typing import Callable, Dict, List, Optional

import curobo._src.runtime as runtime

runtime.enable_cuda_event_timer = True

# Third Party
import numpy as np
import torch
import trimesh

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_collision import (
    check_robot_mesh_collision,
    check_robot_mesh_self_collision,
)
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, VoxelGrid
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util.warp import init_warp
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml, write_yaml

TABLE = Cuboid(name="table", pose=[0.0, 0.0, -0.1, 1, 0, 0, 0], dims=[2.0, 2.0, 0.2])
BOX = Cuboid(name="box", pose=[0.4, -0.3, 0.2, 1, 0, 0, 0], dims=[0.3, 0.1, 0.5])
VOXEL_SIZE = 0.02
DENSE_MESH_EDGE = 0.02


def franka_robot(max_triangles_per_link: Optional[int] = None) -> Dict:
    """Franka robot dict with the base link left out of the mesh check (it sits on the table)."""
    robot = load_yaml(join_path(get_robot_configs_path(), "franka.yml"))
    kinematics = robot["robot_cfg"]["kinematics"]
    kinematics["mesh_collision_ignored_link_names"] = ["panda_link0"]
    if max_triangles_per_link is not None:
        kinematics["max_triangles_per_link"] = max_triangles_per_link
    return robot


def box_mesh(cuboid: Cuboid, max_edge: Optional[float] = None) -> Mesh:
    """Mesh of a cuboid, optionally subdivided so no edge is longer than ``max_edge``."""
    tri = trimesh.creation.box(extents=cuboid.dims)
    if max_edge is not None:
        vertices, faces = trimesh.remesh.subdivide_to_size(tri.vertices, tri.faces, max_edge)
        tri = trimesh.Trimesh(vertices=vertices, faces=faces)
    return Mesh(
        name=cuboid.name + "_mesh",
        vertices=np.asarray(tri.vertices),
        faces=np.asarray(tri.faces),
        pose=list(cuboid.pose),
    )


def box_esdf(cuboids: List[Cuboid], device_cfg: DeviceCfg) -> VoxelGrid:
    """Exact signed distance to a union of axis-aligned boxes on a regular grid."""
    dims = (1.6, 1.6, 1.0)
    center = (0.2, 0.0, 0.4)
    n = [round(d / VOXEL_SIZE) for d in dims]
    axes = [
        center[i] + (torch.arange(n[i], device=device_cfg.device) - (n[i] - 1) / 2.0) * VOXEL_SIZE
        for i in range(3)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    sdf = None
    for c in cuboids:
        d = (grid - torch.tensor(c.pose[:3], device=grid.device)).abs() - torch.tensor(
            c.dims, device=grid.device
        ) / 2.0
        outside = d.clamp(min=0).norm(dim=-1)
        inside = d.max(dim=-1).values.clamp(max=0)
        box_sdf = outside + inside
        sdf = box_sdf if sdf is None else torch.minimum(sdf, box_sdf)
    return VoxelGrid(
        name="esdf",
        pose=[center[0], center[1], center[2], 1, 0, 0, 0],
        dims=list(dims),
        voxel_size=VOXEL_SIZE,
        feature_tensor=sdf.to(torch.float16),
        feature_dtype=torch.float16,
    )


def make_scenes(device_cfg: DeviceCfg) -> Dict[str, SceneCfg]:
    """Equivalent table-and-box scenes as cuboids, meshes, a dense mesh and an ESDF."""
    cuboids = [TABLE, BOX]
    return {
        "cuboid": SceneCfg(cuboid=cuboids),
        "mesh": SceneCfg(mesh=[box_mesh(c) for c in cuboids]),
        "dense_mesh": SceneCfg(mesh=[box_mesh(c, DENSE_MESH_EDGE) for c in cuboids]),
        "voxel": SceneCfg(voxel=[box_esdf(cuboids, device_cfg)]),
    }


def scene_triangles(scene: SceneCfg) -> int:
    """Number of obstacle triangles in a mesh scene."""
    return sum(len(m.faces) for m in (scene.mesh or []))


def time_ms(fn: Callable[[], object], repeats: int, warmup: int = 3) -> float:
    """Median time of ``fn`` in milliseconds, measured with CUDA events."""
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(repeats):
        timer = CudaEventTimer().start()
        fn()
        times.append(timer.stop() * 1000.0)
    return float(np.median(times))


def to_markdown(rows: List[Dict], columns: List[str]) -> str:
    """Format result rows as a Markdown table."""

    def fmt(v: object) -> str:
        return f"{v:.3f}" if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    lines += ["| " + " | ".join(fmt(r[c]) for c in columns) + " |" for r in rows]
    return "\n".join(lines)


# =============================================================================
# Kernel benchmark
# =============================================================================


def sphere_ik_solver(device_cfg: DeviceCfg, scene: SceneCfg) -> IKSolver:
    """IK solver without the mesh check, used to sample configurations and goals."""
    config = IKSolverCfg.create(
        robot=franka_robot(), scene_model=scene, device_cfg=device_cfg, use_cuda_graph=False
    )
    return IKSolver(config)


def sphere_free_configs(solver: IKSolver, n: int) -> torch.Tensor:
    """Sample ``n`` joint configurations free of sphere collisions in the solver's scene."""
    samples = []
    count = 0
    while count < n:
        q = solver.sample_configs(n, rejection_ratio=20)
        samples.append(q)
        count += q.shape[0]
    return torch.cat(samples)[:n]


def kernel_benchmark(
    args: argparse.Namespace, device_cfg: DeviceCfg, scenes: Dict[str, SceneCfg]
) -> List[Dict]:
    """Time one mesh collision query per batch size, triangle budget, scene and margin."""
    batch_sizes = args.kernel_batch_sizes
    n = max(batch_sizes)
    q_free = sphere_free_configs(sphere_ik_solver(device_cfg, scenes["cuboid"]), n)
    rows = []
    for budget in args.triangle_budgets:
        robot = franka_robot(budget)
        robot["robot_cfg"]["kinematics"]["load_collision_meshes"] = True
        kin = Kinematics(
            KinematicsCfg.from_robot_yaml_file(robot, device_cfg=device_cfg),
            compute_link_transforms=True,
        )
        robot_mesh = kin.config.kinematics_config.get_robot_mesh_data()
        limits = kin.config.kinematics_config.joint_limits.position
        q_random = limits[0] + (limits[1] - limits[0]) * torch.rand(
            n, limits.shape[-1], **device_cfg.as_torch_dict()
        )
        scene_data = {
            name: SceneData.from_scene_cfg(scene, device_cfg=device_cfg)
            for name, scene in scenes.items()
        }
        for q_name, q_all in [("sphere-free", q_free), ("random", q_random)]:
            for batch in batch_sizes:
                state = kin.compute_kinematics(JointState.from_position(q_all[:batch]))
                link_tf = state.link_transforms.clone()
                env = torch.zeros(batch, dtype=torch.int32, device=device_cfg.device)
                in_collision = torch.zeros(batch, dtype=torch.int32, device=device_cfg.device)
                contact = torch.zeros(batch, dtype=torch.int32, device=device_cfg.device)
                for scene_name, data in scene_data.items():
                    for margin in args.margins:

                        def query() -> None:
                            check_robot_mesh_collision(
                                robot_mesh,
                                link_tf,
                                data,
                                env,
                                False,
                                margin=margin,
                                in_collision=in_collision,
                                contact_link=contact,
                            )

                        t = time_ms(query, args.repeats)
                        rows.append(
                            {
                                "configs": q_name,
                                "batch": batch,
                                "tri_budget": budget,
                                "robot_tris": robot_mesh.total_tris,
                                "scene": scene_name,
                                "margin": margin,
                                "time_ms": t,
                                "us_per_config": 1000.0 * t / batch,
                                "colliding_%": 100.0 * in_collision.float().mean().item(),
                            }
                        )
                        print(rows[-1], flush=True)
        del kin
        torch.cuda.empty_cache()
    return rows


# =============================================================================
# Solver benchmark
# =============================================================================


def solver_goals(device_cfg: DeviceCfg, scene: SceneCfg, n: int) -> GoalToolPose:
    """Reachable goal poses of sphere-collision-free configurations."""
    solver = sphere_ik_solver(device_cfg, scene)
    q = sphere_free_configs(solver, n)
    return solver.compute_kinematics(JointState.from_position(q)).tool_poses.as_goal()


def goal_slice(goals: GoalToolPose, start: int, stop: int) -> GoalToolPose:
    """Goal poses ``start:stop`` along the batch dimension."""
    return GoalToolPose(
        tool_frames=goals.tool_frames,
        position=goals.position[start:stop].clone(),
        quaternion=goals.quaternion[start:stop].clone(),
    )


def bench_ik(
    device_cfg: DeviceCfg,
    scene: SceneCfg,
    mesh: bool,
    goals: GoalToolPose,
    batch: int,
    args: argparse.Namespace,
    mesh_self: bool = False,
) -> Dict:
    """Mean IK solve time and success rate over the goals, ``batch`` goals per solve."""
    config = IKSolverCfg.create(
        robot=franka_robot(),
        scene_model=scene,
        mesh_collision_check=mesh,
        mesh_self_collision_check=mesh_self,
        device_cfg=device_cfg,
        use_cuda_graph=args.cuda_graph,
        max_batch_size=batch,
    )
    solver = IKSolver(config)
    num_problems = goals.position.shape[0] // batch
    times, success = [], []
    for i in range(args.warmup + num_problems):
        goal = goal_slice(goals, (i % num_problems) * batch, (i % num_problems + 1) * batch)
        solver.reset_seed()
        timer = CudaEventTimer().start()
        result = solver.solve_pose(goal_tool_poses=goal)
        t = timer.stop() * 1000.0
        if i >= args.warmup:
            times.append(t)
            success.append(100.0 * result.success.view(batch, -1).any(-1).float().mean().item())
    return {"time_ms": float(np.mean(times)), "success_%": float(np.mean(success))}


def bench_plan(
    device_cfg: DeviceCfg,
    scene: SceneCfg,
    mesh: bool,
    goals: GoalToolPose,
    args: argparse.Namespace,
    mesh_self: bool = False,
) -> Dict:
    """Mean ``plan_pose`` wall time and success rate from the default configuration."""
    config = MotionPlannerCfg.create(
        robot=franka_robot(),
        scene_model=scene,
        mesh_collision_check=mesh,
        mesh_self_collision_check=mesh_self,
        device_cfg=device_cfg,
        use_cuda_graph=args.cuda_graph,
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    start = JointState.from_position(
        planner.default_joint_state.position.clone().unsqueeze(0),
        joint_names=planner.joint_names,
    )
    times, success = [], []
    for i in range(goals.position.shape[0]):
        goal = goal_slice(goals, i, i + 1)
        t0 = time.perf_counter()
        result = planner.plan_pose(goal_tool_poses=goal, current_state=start)
        torch.cuda.synchronize()
        times.append(1000.0 * (time.perf_counter() - t0))
        success.append(100.0 * float(result is not None and result.success.any().item()))
    return {"time_ms": float(np.mean(times)), "success_%": float(np.mean(success))}


def bench_mpc(
    device_cfg: DeviceCfg,
    scene: SceneCfg,
    mesh: bool,
    goals: GoalToolPose,
    args: argparse.Namespace,
) -> Dict:
    """Median time of one MPC ``optimize_action_sequence`` call."""
    config = MPCSolverCfg.create(
        robot=franka_robot(),
        scene_model=scene,
        mesh_collision_check=mesh,
        device_cfg=device_cfg,
        use_cuda_graph=args.cuda_graph,
    )
    solver = MPCSolver(config)
    state = JointState.from_position(
        solver.default_joint_state.position.clone().unsqueeze(0), joint_names=solver.joint_names
    )
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    solver.setup(state)
    solver.update_goal_tool_poses(goal_slice(goals, 0, 1))
    t = time_ms(lambda: solver.optimize_action_sequence(state), args.repeats, warmup=args.warmup)
    return {"time_ms": t, "success_%": float("nan")}


def solver_benchmark(
    args: argparse.Namespace, device_cfg: DeviceCfg, scenes: Dict[str, SceneCfg]
) -> List[Dict]:
    """Time IK, motion planning and MPC with and without the mesh check."""
    rows = []
    for scene_name in args.solver_scenes:
        scene = scenes[scene_name]
        goals = solver_goals(device_cfg, scene, args.num_goals)
        for mesh in [False, True]:
            runs = {
                "IK (batch 1)": lambda: bench_ik(device_cfg, scene, mesh, goals, 1, args),
                f"IK (batch {args.ik_batch})": lambda: bench_ik(
                    device_cfg, scene, mesh, goals, args.ik_batch, args
                ),
                "plan_pose": lambda: bench_plan(
                    device_cfg, scene, mesh, goal_slice(goals, 0, args.num_plans), args
                ),
                "MPC step": lambda: bench_mpc(device_cfg, scene, mesh, goals, args),
            }
            for solver_name, run in runs.items():
                try:
                    out = run()
                except Exception as e:  # report and continue with the other solvers
                    out = {"time_ms": float("nan"), "success_%": float("nan")}
                    print(f"{solver_name} on {scene_name} (mesh={mesh}) failed: {e!r}", flush=True)
                rows.append(
                    {"solver": solver_name, "scene": scene_name, "mesh_check": mesh, **out}
                )
                print(rows[-1], flush=True)
                torch.cuda.empty_cache()
    return rows


def self_collision_benchmark(
    args: argparse.Namespace, device_cfg: DeviceCfg, scenes: Dict[str, SceneCfg]
) -> List[Dict]:
    """Time the mesh self-collision query, and IK and ``plan_pose`` with and without it."""
    batch_sizes = args.kernel_batch_sizes
    n = max(batch_sizes)
    q_free = sphere_free_configs(sphere_ik_solver(device_cfg, scenes["cuboid"]), n)
    rows = []
    for budget in args.triangle_budgets:
        robot = franka_robot(budget)
        robot["robot_cfg"]["kinematics"]["load_collision_meshes"] = True
        kin = Kinematics(
            KinematicsCfg.from_robot_yaml_file(robot, device_cfg=device_cfg),
            compute_link_transforms=True,
        )
        robot_mesh = kin.config.kinematics_config.get_robot_mesh_data()
        num_pairs = robot_mesh.self_collision_pairs.shape[0]
        limits = kin.config.kinematics_config.joint_limits.position
        q_random = limits[0] + (limits[1] - limits[0]) * torch.rand(
            n, limits.shape[-1], **device_cfg.as_torch_dict()
        )
        for q_name, q_all in [("sphere-free", q_free), ("random", q_random)]:
            for batch in batch_sizes:
                state = kin.compute_kinematics(JointState.from_position(q_all[:batch]))
                link_tf = state.link_transforms.clone()
                in_collision = torch.zeros(batch, dtype=torch.int32, device=device_cfg.device)
                contact = torch.zeros(batch, dtype=torch.int32, device=device_cfg.device)
                near = torch.zeros(batch * num_pairs, dtype=torch.int32, device=device_cfg.device)
                for margin in args.margins:

                    def query() -> None:
                        check_robot_mesh_self_collision(
                            robot_mesh,
                            link_tf,
                            margin=margin,
                            in_collision=in_collision,
                            contact_pair=contact,
                            pair_near=near,
                        )

                    t = time_ms(query, args.repeats)
                    rows.append(
                        {
                            "configs": q_name,
                            "batch": batch,
                            "tri_budget": budget,
                            "pairs": num_pairs,
                            "margin": margin,
                            "time_ms": t,
                            "us_per_config": 1000.0 * t / batch,
                            "self_colliding_%": 100.0 * in_collision.float().mean().item(),
                        }
                    )
                    print(rows[-1], flush=True)
        del kin
        torch.cuda.empty_cache()

    scene = scenes["cuboid"]
    goals = solver_goals(device_cfg, scene, args.num_goals)
    for mesh_self in [False, True]:
        runs = {
            f"IK (batch {args.ik_batch})": lambda: bench_ik(
                device_cfg, scene, False, goals, args.ik_batch, args, mesh_self=mesh_self
            ),
            "plan_pose": lambda: bench_plan(
                device_cfg,
                scene,
                False,
                goal_slice(goals, 0, args.num_plans),
                args,
                mesh_self=mesh_self,
            ),
        }
        for solver_name, run in runs.items():
            out = run()
            rows.append({"solver": solver_name, "mesh_self_check": mesh_self, **out})
            print(rows[-1], flush=True)
            torch.cuda.empty_cache()
    return rows


def main() -> None:
    """Parse arguments, run the selected parts and write the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--part", choices=["kernel", "solver", "self", "all"], default="all")
    parser.add_argument("--save_path", type=str, default=None)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--kernel_batch_sizes", type=int, nargs="+", default=[32, 512, 4096])
    parser.add_argument("--triangle_budgets", type=int, nargs="+", default=[64, 256, 1024])
    parser.add_argument("--margins", type=float, nargs="+", default=[0.0, 0.01])
    parser.add_argument("--solver_scenes", nargs="+", default=["cuboid", "mesh", "voxel"])
    parser.add_argument("--num_goals", type=int, default=200)
    parser.add_argument("--ik_batch", type=int, default=100)
    parser.add_argument("--num_plans", type=int, default=10)
    parser.add_argument("--cuda_graph", type=int, default=1)
    args = parser.parse_args()
    args.cuda_graph = bool(args.cuda_graph)

    setup_curobo_logger("error")
    torch.manual_seed(2)
    np.random.seed(2)
    device_cfg = DeviceCfg()
    init_warp(device_cfg=device_cfg)
    scenes = make_scenes(device_cfg)
    print(
        "scene triangles:",
        {k: scene_triangles(v) for k, v in scenes.items() if v.mesh},
        "| GPU:",
        torch.cuda.get_device_name(0),
        flush=True,
    )

    report = []
    results = {}
    if args.part in ("kernel", "all"):
        rows = kernel_benchmark(args, device_cfg, scenes)
        results["kernel"] = rows
        report.append("## Kernel\n\n" + to_markdown(rows, list(rows[0].keys())))
    if args.part in ("solver", "all"):
        rows = solver_benchmark(args, device_cfg, scenes)
        results["solver"] = rows
        report.append("## Solvers\n\n" + to_markdown(rows, list(rows[0].keys())))
    if args.part in ("self", "all"):
        rows = self_collision_benchmark(args, device_cfg, scenes)
        results["self"] = rows
        kernel_rows = [r for r in rows if "configs" in r]
        solver_rows = [r for r in rows if "solver" in r]
        report.append(
            "## Self-collision kernel\n\n"
            + to_markdown(kernel_rows, list(kernel_rows[0].keys()))
            + "\n\n## Self-collision solvers\n\n"
            + to_markdown(solver_rows, list(solver_rows[0].keys()))
        )

    text = "\n\n".join(report)
    print(text)
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        with open(join_path(args.save_path, "mesh_collision_benchmark.md"), "w") as f:
            f.write(f"GPU: {torch.cuda.get_device_name(0)}\n\n" + text + "\n")
        write_yaml(results, join_path(args.save_path, "mesh_collision_benchmark.yml"))


if __name__ == "__main__":
    main()
