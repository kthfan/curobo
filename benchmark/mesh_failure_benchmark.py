# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Break IK and motion planning failures down by what rejects them, spheres or meshes.

The optimizers steer with collision spheres; success is checked with the spheres and, with the
mesh checks on, also with the exact link meshes. This script asks how many failures would turn
into successes if success were checked with the meshes alone (the spheres still steering), and
how many fail for other reasons. It uses the cluttered scene and the tight goals of
``sphere_coverage_benchmark.py``.

IK (``--ik_batch`` goals per solve, every seed kept): for each goal, a seed *converged* when it
reaches the pose tolerances. A goal succeeds today when a converged seed passes the sphere
checks and its meshes are free; it would succeed *with mesh-based success* when a converged seed
is within the joint limits and its meshes are free. Each failure today is one of:

- ``mesh_success``: a converged seed is mesh-free, only the spheres reject it;
- ``mesh_rejected``: converged seeds pass the spheres, but their meshes collide;
- ``not_converged``: no seed reaches the pose;
- ``colliding``: converged seeds collide as meshes (and as spheres).

``plan_pose`` (mesh checks on): failures are split into the IK stage (no IK solution in any
attempt) and the trajectory stage. For a trajectory failure, ``mesh_success`` means a seed
reaches the goal and violates no constraint but the sphere collision ones; the constraints the
best seed violates are counted too.

Example::

    python benchmark/mesh_failure_benchmark.py --robot_files franka.yml --save_path bench_fail
"""

# Standard Library
import argparse
import os
import time
from collections import Counter
from typing import Dict, List

# Third Party
import numpy as np
import torch

# Shared scene and goals: Python puts this script's folder on the import path.
from sphere_coverage_benchmark import (
    clutter_scene,
    goal_poses,
    goal_slice,
    load_robot,
    make_ik,
    tight_goals,
)

# CuRobo
from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.solver.solver_ik import IKSolver
from curobo._src.solver.solver_ik_cfg import IKSolverCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose
from curobo._src.util.logging import setup_curobo_logger
from curobo._src.util.warp import init_warp
from curobo._src.util_file import join_path, write_yaml

SPHERE_CONSTRAINTS = ("scene_collision", "self_collision")
IK_CATEGORIES = ("mesh_success", "mesh_rejected", "not_converged", "colliding")


def mesh_free(checker: IKSolver, q: torch.Tensor) -> torch.Tensor:
    """True where the configurations ``q`` ``[n, dof]`` are free as meshes (scene and self)."""
    js = JointState.from_position(q, joint_names=checker.joint_names)
    distance, _ = checker.get_mesh_collision_distance(js, max_distance=0.01)
    pairs = checker.get_mesh_self_collision_pairs(js)
    self_free = torch.tensor([p is None for p in pairs], device=q.device)
    return (distance > 0.0) & self_free


def classify_ik(
    solver: IKSolver, checker: IKSolver, goals: GoalToolPose, batch: int
) -> Dict[str, float]:
    """Success today and with mesh-based success, and the failure categories, in percent."""
    limits = checker.kinematics.get_joint_limits()
    low = limits.position_lower_limits - 1.0e-4
    high = limits.position_upper_limits + 1.0e-4
    counts = Counter()
    total = 0
    num_seeds = solver.config.num_seeds
    for start in range(0, goals.position.shape[0] - batch + 1, batch):
        solver.reset_seed()
        result = solver.solve_pose(
            goal_tool_poses=goal_slice(goals, start, start + batch), return_seeds=num_seeds
        )
        converged = (result.position_error < solver.config.position_tolerance) & (
            result.rotation_error < solver.config.orientation_tolerance
        )
        sphere_ok = result.feasible.bool()
        q = result.solution.view(batch * num_seeds, -1)
        free = mesh_free(checker, q).view(batch, num_seeds)
        within = ((q >= low) & (q <= high)).all(dim=-1).view(batch, num_seeds)

        today = (converged & sphere_ok & free).any(dim=-1)
        with_meshes = (converged & within & free).any(dim=-1)
        failed = ~today
        mesh_rejected = failed & ~with_meshes & (converged & sphere_ok).any(dim=-1)
        not_converged = failed & ~converged.any(dim=-1)
        counts["success_today"] += int(today.sum())
        counts["success_with_meshes"] += int(with_meshes.sum())
        counts["mesh_success"] += int((failed & with_meshes).sum())
        counts["mesh_rejected"] += int(mesh_rejected.sum())
        counts["not_converged"] += int(not_converged.sum())
        counts["colliding"] += int((failed & ~with_meshes & ~mesh_rejected & ~not_converged).sum())
        total += batch
    return {k: 100.0 * v / total for k, v in counts.items()} | {"goals": total}


def trajectory_failure(result: object) -> Dict[str, object]:
    """Whether a failed trajectory stage had a seed that only the spheres reject.

    Returns:
        ``mesh_success`` and the names of the constraints the best seed violates.
    """
    metrics = result.metrics
    # Metrics hold every seed of the single problem, also after the result kept the best ones.
    num_seeds = metrics.convergence.values[0].shape[0]
    converged = torch.ones(num_seeds, dtype=torch.bool, device=result.success.device)
    for name, values in zip(metrics.convergence.names, metrics.convergence.values):
        last = values[:, -1].reshape(num_seeds, -1)
        if "position_tolerance" in name:
            converged &= (last < result.position_tolerance).all(dim=-1)
        elif "orientation_tolerance" in name:
            converged &= (last < result.orientation_tolerance).all(dim=-1)
    constraints = metrics.costs_and_constraints.constraints
    other_ok = torch.ones_like(converged)
    violated = {}
    for name, values in zip(constraints.names, constraints.values):
        bad = values.reshape(num_seeds, -1).sum(dim=-1) > 0.0
        violated[name] = bad
        if name not in SPHERE_CONSTRAINTS:
            other_ok &= ~bad
    best = int(result.seed_rank.view(-1)[0])
    return {
        "mesh_success": bool((converged & other_ok).any()),
        "best_converged": bool(converged[best]),
        "best_violates": sorted(n for n, bad in violated.items() if bool(bad[best])),
    }


def mode_options(mesh_only_success: bool, margin: float) -> Dict[str, object]:
    """Solver options with both exact mesh checks at ``margin``."""
    return {
        "mesh_collision_check": True,
        "mesh_self_collision_check": True,
        "mesh_collision_margin": margin,
        "mesh_self_collision_margin": margin,
        "mesh_only_success": mesh_only_success,
    }


def make_planner(robot: Dict, scene, args, **options) -> MotionPlanner:
    """Motion planner with the given mesh options."""
    config = MotionPlannerCfg.create(
        robot=robot,
        scene_model=scene,
        device_cfg=DeviceCfg(),
        use_cuda_graph=bool(args.cuda_graph),
        **options,
    )
    return MotionPlanner(config)


def classify_plans(robot: Dict, scene, goals: GoalToolPose, args) -> Dict[str, object]:
    """``plan_pose`` success and failure stages over ``--num_plans`` goals, in percent."""
    planner = make_planner(robot, scene, args, **mode_options(False, 0.0))
    # Keep every trajectory seed, so the result still holds the metrics of all of them; the
    # planner succeeds when any seed does, as with the single best one.
    trajopt = planner.trajopt_solver
    solve_pose = trajopt.solve_pose

    def solve_pose_all_seeds(*solve_args, **solve_kwargs):
        solve_kwargs.setdefault("return_seeds", trajopt.config.num_seeds)
        return solve_pose(*solve_args, **solve_kwargs)

    trajopt.solve_pose = solve_pose_all_seeds
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    start = JointState.from_position(
        planner.default_joint_state.position.clone().unsqueeze(0),
        joint_names=planner.joint_names,
    )
    counts = Counter()
    violations = Counter()
    times = []
    for i in range(args.num_plans):
        t0 = time.perf_counter()
        result = planner.plan_pose(
            goal_tool_poses=goal_slice(goals, i, i + 1), current_state=start
        )
        torch.cuda.synchronize()
        times.append(1000.0 * (time.perf_counter() - t0))
        if result is not None and bool(result.success.any()):
            counts["success"] += 1
        elif result is None:
            counts["ik_stage"] += 1
        else:
            counts["trajectory_stage"] += 1
            info = trajectory_failure(result)
            counts["trajectory_mesh_success"] += int(info["mesh_success"])
            counts["trajectory_best_converged"] += int(info["best_converged"])
            violations.update(info["best_violates"])
    out = {k: 100.0 * v / args.num_plans for k, v in counts.items()}
    out["plans"] = args.num_plans
    out["time_ms"] = float(np.mean(times))
    out["best_seed_violations"] = dict(violations)
    return out


def bench_mode(
    robot: Dict, scene, goals: GoalToolPose, mesh_only_success: bool, margin: float, args
) -> Dict[str, float]:
    """IK and ``plan_pose`` success and time with the mesh checks at ``margin``."""
    options = mode_options(mesh_only_success, margin)
    config = IKSolverCfg.create(
        robot=robot,
        scene_model=scene,
        device_cfg=DeviceCfg(),
        use_cuda_graph=bool(args.cuda_graph),
        max_batch_size=args.ik_batch,
        **options,
    )
    solver = IKSolver(config)
    success, times = [], []
    for start in range(0, goals.position.shape[0] - args.ik_batch + 1, args.ik_batch):
        solver.reset_seed()
        t0 = time.perf_counter()
        result = solver.solve_pose(goal_tool_poses=goal_slice(goals, start, start + args.ik_batch))
        torch.cuda.synchronize()
        times.append(1000.0 * (time.perf_counter() - t0))
        success.append(result.success.view(args.ik_batch, -1).any(dim=-1).float().mean().item())
    del solver
    torch.cuda.empty_cache()
    planner = make_planner(robot, scene, args, **options)
    planner.warmup(enable_graph=True, num_warmup_iterations=2)
    start_state = JointState.from_position(
        planner.default_joint_state.position.clone().unsqueeze(0),
        joint_names=planner.joint_names,
    )
    plan_success, plan_times = [], []
    for i in range(args.num_plans):
        t0 = time.perf_counter()
        result = planner.plan_pose(
            goal_tool_poses=goal_slice(goals, i, i + 1), current_state=start_state
        )
        torch.cuda.synchronize()
        plan_times.append(1000.0 * (time.perf_counter() - t0))
        plan_success.append(float(result is not None and result.success.any().item()))
    del planner
    torch.cuda.empty_cache()
    return {
        "mesh_only_success": mesh_only_success,
        "margin_mm": 1000.0 * margin,
        "ik_success_%": 100.0 * float(np.mean(success)),
        "ik_ms": float(np.mean(times[1:] if len(times) > 1 else times)),
        "plan_success_%": 100.0 * float(np.mean(plan_success)),
        "plan_ms": float(np.mean(plan_times)),
    }


def main() -> None:
    """Sample the goals, then break down IK and plan failures for every robot file."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot_files", nargs="+", default=["franka.yml"])
    parser.add_argument("--base_link", default="panda_link0")
    parser.add_argument("--num_boxes", type=int, default=40)
    parser.add_argument("--num_goals", type=int, default=300)
    parser.add_argument("--max_clearance", type=float, default=0.03)
    parser.add_argument("--ik_batch", type=int, default=100)
    parser.add_argument("--num_plans", type=int, default=60)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--cuda_graph", type=int, default=1)
    parser.add_argument(
        "--margins",
        type=float,
        nargs="+",
        default=[0.0, 0.005, 0.01],
        help="Mesh margins at which IK and plan_pose run with mesh_only_success off and on.",
    )
    parser.add_argument("--skip_breakdown", action="store_true")
    parser.add_argument("--save_path", default=None)
    args = parser.parse_args()

    setup_curobo_logger("error")
    torch.manual_seed(2)
    np.random.seed(2)
    init_warp(device_cfg=DeviceCfg())
    scene = clutter_scene(args.num_boxes, args.seed)

    rows: List[Dict] = []
    lines = []
    for robot_file in args.robot_files:
        robot = load_robot(robot_file, args.base_link)
        checker = make_ik(robot, scene, mesh=True, batch=args.ik_batch, args=args)
        sampled = tight_goals(checker, args)
        goals = goal_poses(checker, sampled["q"])
        clearance = sampled["clearance"]
        name = os.path.basename(robot_file)
        print(
            f"{name}: {goals.position.shape[0]} goals, median clearance "
            f"{1000 * clearance.median():.1f} mm",
            flush=True,
        )
        modes = []
        for margin in args.margins:
            for mesh_only_success in (False, True):
                modes.append(bench_mode(robot, scene, goals, mesh_only_success, margin, args))
                print(f"{name} mode: {modes[-1]}", flush=True)
        lines += [
            f"### {name}: mesh_only_success",
            "",
            f"Both mesh checks on; {goals.position.shape[0]} IK goals, {args.num_plans} plans.",
            "",
            "| mesh_only_success | margin (mm) | IK success (%) | IK (ms) | plan success (%) | "
            "plan (ms) |",
            "|---|---|---|---|---|---|",
        ]
        lines += [
            f"| {m['mesh_only_success']} | {m['margin_mm']:.0f} | {m['ik_success_%']:.1f} | "
            f"{m['ik_ms']:.0f} | {m['plan_success_%']:.1f} | {m['plan_ms']:.0f} |"
            for m in modes
        ]
        lines.append("")
        if args.skip_breakdown:
            rows.append({"robot": name, "modes": modes})
            continue
        solver = make_ik(robot, scene, mesh=False, batch=args.ik_batch, args=args)
        ik = classify_ik(solver, checker, goals, args.ik_batch)
        print(f"{name} IK: {ik}", flush=True)
        del solver
        torch.cuda.empty_cache()
        plan = classify_plans(robot, scene, goals, args)
        print(f"{name} plan_pose: {plan}", flush=True)
        rows.append({"robot": name, "ik": ik, "plan": plan, "modes": modes})
        lines += [
            f"### {name}",
            "",
            f"{ik['goals']} goals within {1000 * args.max_clearance:.0f} mm of the scene "
            f"(median {1000 * clearance.median():.1f} mm).",
            "",
            "| IK | % of goals |",
            "|---|---|",
            f"| success today (spheres and meshes) | {ik['success_today']:.1f} |",
            f"| success with mesh-based success | {ik['success_with_meshes']:.1f} |",
        ]
        lines += [f"| failure: {c} | {ik.get(c, 0.0):.1f} |" for c in IK_CATEGORIES]
        lines += [
            "",
            "| plan_pose | % of goals |",
            "|---|---|",
            f"| success | {plan.get('success', 0.0):.1f} |",
            f"| failure in the IK stage | {plan.get('ik_stage', 0.0):.1f} |",
            f"| failure in the trajectory stage | {plan.get('trajectory_stage', 0.0):.1f} |",
            "| of which a seed only the spheres reject | "
            f"{plan.get('trajectory_mesh_success', 0.0):.1f} |",
            "| of which the best seed reaches the goal | "
            f"{plan.get('trajectory_best_converged', 0.0):.1f} |",
            "",
            f"Constraints the best failed trajectory violates: {plan['best_seed_violations']}.",
            "",
        ]

    text = f"GPU: {torch.cuda.get_device_name(0)}. Scene: table and {args.num_boxes} boxes.\n\n"
    text += "\n".join(lines)
    print(text)
    if args.save_path is not None:
        os.makedirs(args.save_path, exist_ok=True)
        with open(join_path(args.save_path, "mesh_failure_benchmark.md"), "w") as f:
            f.write(text + "\n")
        write_yaml({"rows": rows}, join_path(args.save_path, "mesh_failure_benchmark.yml"))


if __name__ == "__main__":
    main()
