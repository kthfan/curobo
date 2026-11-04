# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
r"""Compare robot configuration files, e.g. before and after covering the meshes with spheres.

Every robot file solves the same goals in the same cluttered scene: a table and random boxes.
The goals are tool poses of joint configurations whose exact link meshes are free of the scene
and of each other, but closer than ``--max_clearance`` to an obstacle, so that the parts of the
meshes outside the spheres matter. For each file the script measures:

- IK and ``plan_pose`` success and time, with the exact mesh checks (scene and
  self-collision) and without them;
- how many IK solutions that pass the sphere checks collide as meshes: the solutions the
  mesh check rejects because the optimizer did not see the collision;
- the time of one MPC step, which grows with the number of spheres.

Example::

    python benchmark/sphere_coverage_benchmark.py \\
        --robot_files franka.yml franka_covered.yml --save_path bench_spheres
"""

# Standard Library
import argparse
import os
import time
from typing import Dict, List

# Third Party
import numpy as np
import torch

# CuRobo
from curobo._src.geom.types import Cuboid, SceneCfg
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.solver.solver_mpc import MPCSolver
from curobo._src.solver.solver_mpc_cfg import MPCSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.cuda_event_timer import CudaEventTimer
from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util.warp import init_warp
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml, write_yaml


def load_robot(robot_file: str, base_link: str) -> Dict:
    """Robot dict with its base link left out of the scene mesh check (it sits on the table)."""
    if not os.path.exists(robot_file):
        robot_file = join_path(get_robot_configs_path(), robot_file)
    robot = load_yaml(robot_file)
    robot["robot_cfg"]["kinematics"]["mesh_collision_ignored_link_names"] = [base_link]
    return robot


def clutter_scene(num_boxes: int, seed: int) -> SceneCfg:
    """A table and ``num_boxes`` random boxes in front of the robot, clear of its base."""
    rng = np.random.default_rng(seed)
    cuboids = [Cuboid(name="table", pose=[0.0, 0.0, -0.1, 1, 0, 0, 0], dims=[2.0, 2.0, 0.2])]
    while len(cuboids) < num_boxes + 1:
        center = rng.uniform([-0.2, -0.7, 0.05], [0.8, 0.7, 0.9])
        if np.linalg.norm(center[:2]) < 0.3:
            continue
        q = rng.normal(size=4)
        q /= np.linalg.norm(q)
        dims = rng.uniform(0.04, 0.15, size=3)
        cuboids.append(Cuboid(name=f"box_{len(cuboids)}", pose=[*center, *q], dims=dims.tolist()))
    return SceneCfg(cuboid=cuboids)


def make_ik(robot: Dict, scene: SceneCfg, mesh: bool, batch: int, args) -> IKSolver:
    """IK solver for ``robot`` in ``scene``, with or without both exact mesh checks."""
    config = IKSolverCfg.create(
        robot=robot,
        scene_model=scene,
        mesh_collision_check=mesh,
        mesh_self_collision_check=mesh,
        device_cfg=DeviceCfg(),
        use_cuda_graph=bool(args.cuda_graph),
        max_batch_size=batch,
    )
    return IKSolver(config)


def tight_goals(checker: IKSolver, args) -> Dict[str, torch.Tensor]:
    """Joint configurations free as meshes but within ``max_clearance`` of the scene.

    Returns:
        ``q`` and ``clearance`` of ``num_goals`` configurations.
    """
    limits = checker.kinematics.get_joint_limits()
    sampler = SampleBuffer.create_halton_sample_buffer(
        len(checker.joint_names),
        device_cfg=DeviceCfg(),
        low_bounds=limits.position_lower_limits,
        up_bounds=limits.position_upper_limits,
        seed=args.seed,
        store_buffer=None,
    )
    kept_q, kept_d = [], []
    count = 0
    while count < args.num_goals:
        q = sampler.get_samples(4096, bounded=True)
        js = JointState.from_position(q, joint_names=checker.joint_names)
        distance, _ = checker.get_mesh_collision_distance(js, max_distance=args.max_clearance)
        pairs = checker.get_mesh_self_collision_pairs(js)
        self_free = torch.tensor([p is None for p in pairs], device=q.device)
        keep = (distance > 0.0) & (distance < args.max_clearance) & self_free
        kept_q.append(q[keep])
        kept_d.append(distance[keep])
        count += int(keep.sum())
    return {
        "q": torch.cat(kept_q)[: args.num_goals],
        "clearance": torch.cat(kept_d)[: args.num_goals],
    }


def goal_poses(solver: IKSolver, q: torch.Tensor) -> GoalToolPose:
    """Tool poses of joint configurations ``q``, as goals."""
    return solver.compute_kinematics(JointState.from_position(q)).tool_poses.as_goal()


def goal_slice(goals: GoalToolPose, start: int, stop: int) -> GoalToolPose:
    """Goals ``start:stop`` along the batch dimension."""
    return GoalToolPose(
        tool_frames=goals.tool_frames,
        position=goals.position[start:stop].clone(),
        quaternion=goals.quaternion[start:stop].clone(),
    )


def bench_ik(
    solver: IKSolver, goals: GoalToolPose, batch: int, checker: IKSolver, args
) -> Dict[str, float]:
    """IK success and time over the goals.

    ``mesh_colliding_%`` is the share of successes whose meshes collide, measured with
    ``checker``.
    """
    n = goals.position.shape[0] // batch
    times, success, colliding = [], [], []
    for i in range(args.warmup + n):
        goal = goal_slice(goals, (i % n) * batch, (i % n + 1) * batch)
        solver.reset_seed()
        timer = CudaEventTimer().start()
        result = solver.solve_pose(goal_tool_poses=goal)
        t = 1000.0 * timer.stop()
        if i < args.warmup:
            continue
        times.append(t)
        ok = result.success.view(batch, -1)[:, 0]
        success.append(ok.float().mean().item())
        q = result.js_solution.position.view(batch, -1, result.js_solution.position.shape[-1])[
            :, 0
        ]
        js = JointState.from_position(q, joint_names=result.js_solution.joint_names)
        distance, _ = checker.get_mesh_collision_distance(js, max_distance=0.01)
        pairs = checker.get_mesh_self_collision_pairs(js)
        self_hit = torch.tensor([p is not None for p in pairs], device=q.device)
        hit = (distance <= 0.0) | self_hit
        colliding.append((hit & ok).float().sum().item() / max(ok.float().sum().item(), 1.0))
    return {
        "time_ms": float(np.mean(times)),
        "success_%": 100.0 * float(np.mean(success)),
        "mesh_colliding_%": 100.0 * float(np.mean(colliding)),
    }


def bench_plan(robot: Dict, scene: SceneCfg, mesh: bool, goals: GoalToolPose, args) -> Dict:
    """Mean ``plan_pose`` wall time and success rate from the default configuration."""
    config = MotionPlannerCfg.create(
        robot=robot,
        scene_model=scene,
        mesh_collision_check=mesh,
        mesh_self_collision_check=mesh,
        device_cfg=DeviceCfg(),
        use_cuda_graph=bool(args.cuda_graph),
    )
    planner = MotionPlanner(config)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    start = JointState.from_position(
        planner.default_joint_state.position.clone().unsqueeze(0),
        joint_names=planner.joint_names,
    )
    times, success = [], []
    for i in range(args.num_plans):
        t0 = time.perf_counter()
        result = planner.plan_pose(
            goal_tool_poses=goal_slice(goals, i, i + 1), current_state=start
        )
        torch.cuda.synchronize()
        times.append(1000.0 * (time.perf_counter() - t0))
        success.append(float(result is not None and result.success.any().item()))
    return {"time_ms": float(np.mean(times)), "success_%": 100.0 * float(np.mean(success))}


def bench_mpc(robot: Dict, scene: SceneCfg, goals: GoalToolPose, args) -> Dict:
    """Median time of one MPC ``optimize_action_sequence`` call."""
    config = MPCSolverCfg.create(
        robot=robot,
        scene_model=scene,
        device_cfg=DeviceCfg(),
        use_cuda_graph=bool(args.cuda_graph),
    )
    solver = MPCSolver(config)
    state = JointState.from_position(
        solver.default_joint_state.position.clone().unsqueeze(0), joint_names=solver.joint_names
    )
    state.velocity = torch.zeros_like(state.position)
    state.acceleration = torch.zeros_like(state.position)
    solver.setup(state)
    solver.update_goal_tool_poses(goal_slice(goals, 0, 1))
    for _ in range(args.warmup):
        solver.optimize_action_sequence(state)
    times = []
    for _ in range(args.repeats):
        timer = CudaEventTimer().start()
        solver.optimize_action_sequence(state)
        times.append(1000.0 * timer.stop())
    return {"time_ms": float(np.median(times))}


def to_markdown(rows: List[Dict]) -> str:
    """Format result rows as a Markdown table."""
    columns = list(rows[0].keys())
    lines = ["| " + " | ".join(columns) + " |", "|" + "---|" * len(columns)]
    for r in rows:
        cells = [f"{r[c]:.1f}" if isinstance(r[c], float) else str(r[c]) for c in columns]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> None:
    """Build the scene and goals, then benchmark every robot file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot_files", nargs="+", default=["franka.yml"])
    parser.add_argument("--base_link", default="panda_link0")
    parser.add_argument("--num_boxes", type=int, default=40)
    parser.add_argument("--num_goals", type=int, default=300)
    parser.add_argument("--max_clearance", type=float, default=0.03)
    parser.add_argument("--ik_batch", type=int, default=100)
    parser.add_argument("--num_plans", type=int, default=30)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cuda_graph", type=int, default=1)
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    setup_curobo_logger("error")
    torch.manual_seed(2)
    np.random.seed(2)
    init_warp(device_cfg=DeviceCfg())
    scene = clutter_scene(args.num_boxes, args.seed)

    # Goals come from the exact meshes, which all robot files share.
    reference = load_robot(args.robot_files[0], args.base_link)
    checker = make_ik(reference, scene, mesh=True, batch=args.ik_batch, args=args)
    sampled = tight_goals(checker, args)
    goals = goal_poses(checker, sampled["q"])
    clearance = sampled["clearance"]
    print(
        f"{goals.position.shape[0]} goals, clearance {1000 * clearance.min():.1f} to "
        f"{1000 * clearance.max():.1f} mm, median {1000 * clearance.median():.1f} mm",
        flush=True,
    )

    rows = []
    for robot_file in args.robot_files:
        robot = load_robot(robot_file, args.base_link)
        spheres = sum(
            1
            for link in robot["robot_cfg"]["kinematics"]["collision_spheres"].values()
            for s in link
            if s["radius"] > 0.0
        )
        name = os.path.basename(robot_file)
        for mesh in (False, True):
            solver = make_ik(robot, scene, mesh, args.ik_batch, args)
            ik = bench_ik(solver, goals, args.ik_batch, checker, args)
            plan = bench_plan(robot, scene, mesh, goals, args)
            rows.append(
                {
                    "robot": name,
                    "spheres": spheres,
                    "mesh_checks": mesh,
                    "ik_success_%": ik["success_%"],
                    "ik_mesh_colliding_%": ik["mesh_colliding_%"],
                    "ik_ms": ik["time_ms"],
                    "plan_success_%": plan["success_%"],
                    "plan_ms": plan["time_ms"],
                }
            )
            print(rows[-1], flush=True)
            torch.cuda.empty_cache()
        mpc = bench_mpc(robot, scene, goals, args)
        for row in rows[-2:]:
            row["mpc_step_ms"] = mpc["time_ms"]
        print(f"{name}: MPC step {mpc['time_ms']:.1f} ms", flush=True)

    text = (
        f"GPU: {torch.cuda.get_device_name(0)}. Scene: table and {args.num_boxes} boxes. "
        f"{goals.position.shape[0]} goals whose meshes are free and within "
        f"{1000 * args.max_clearance:.0f} mm of the scene (median "
        f"{1000 * clearance.median():.1f} mm).\n\n"
        "`ik_mesh_colliding_%`: share of IK successes whose meshes collide (scene or self).\n\n"
        + to_markdown(rows)
    )
    print(text)
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        with open(join_path(args.save_path, "sphere_coverage_benchmark.md"), "w") as f:
            f.write(text + "\n")
        write_yaml({"rows": rows}, join_path(args.save_path, "sphere_coverage_benchmark.yml"))


if __name__ == "__main__":
    main()
