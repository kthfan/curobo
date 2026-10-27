# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep the ``eef_bound`` cost weight on a GPU and report planning quality per weight.

NOT for main: see ``cpu_verify/README.md``. Needs a CUDA GPU and a curobo build.

The weight of the optimization cost (task configs, ``metric: False``) is changed in place
on the solvers' optimizer rollouts, so the planner and its CUDA graphs are built once. The
metrics constraint is binary and does not depend on the weight.

Run from the repository root::

    python cpu_verify/eef_bound/gpu_weight_sweep.py
    python cpu_verify/eef_bound/gpu_weight_sweep.py --weights 0 1e3 1e4 1e5 --goals 16
    python cpu_verify/eef_bound/gpu_weight_sweep.py --mpc   # also sweep the MPC weight
    python cpu_verify/eef_bound/gpu_weight_sweep.py --weights 1e5 --margins 0 0.0005 0.001 0.002

Writes ``sweep_raw.csv``, ``sweep_summary.csv`` and ``sweep.png`` to ``--out`` and prints the
best weight and margin per scenario: highest success rate, then lowest median planning time.
Violations are measured against the user's bounds (no margin, no tolerance).
"""

from __future__ import annotations

import argparse
import dataclasses
import csv
import math
import os
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

from curobo._src.cost.tool_pose_bound import _quat_rotate, tool_pose_bound_violation
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, ToolPoseBound
from gpu_check import active_positions

Scenario = Callable[["Sweep", torch.Generator], Tuple[ToolPoseBound, torch.Tensor, torch.Tensor]]


def optimizer_bound_costs(solver) -> List:
    """``eef_bound`` optimization costs of a solver (and its seed IK solver, if any)."""
    cores = [solver.core]
    seed_core = getattr(getattr(solver, "seed_ik_solver", None), "core", None)
    if seed_core is not None:
        cores.append(seed_core)
    costs = []
    for core in cores:
        for rollout in core.optimizer_rollouts:
            for cost in rollout.get_cost_component_by_name("eef_bound"):
                if not cost.config.metric:
                    costs.append(cost)
    return costs


def set_weight(costs: List, weight: float) -> None:
    """Set the weight in place so captured CUDA graphs see the new value."""
    for cost in costs:
        cost._weight.fill_(weight)
        cost.config.weight.fill_(weight)


class Sweep:
    """Planner, start pose and helpers shared by the scenarios."""

    def __init__(self, robot: str, scene: Optional[str], use_cuda_graph: bool, target: str):
        cfg = MotionPlannerCfg.create(
            robot=robot, scene_model=scene, use_cuda_graph=use_cuda_graph
        )
        self.planner = MotionPlanner(cfg)
        self.planner.warmup(enable_graph=use_cuda_graph)
        self.ee = self.planner.tool_frames[0]
        solvers = {"ik": [self.planner.ik_solver], "trajopt": [self.planner.trajopt_solver]}
        solvers["both"] = solvers["ik"] + solvers["trajopt"]
        self.costs = [c for s in solvers[target] for c in optimizer_bound_costs(s)]
        if not self.costs:
            raise SystemExit("no eef_bound optimization cost found; check the task configs")
        self.start = JointState.from_position(
            self.planner.default_joint_state.position.unsqueeze(0),
            joint_names=self.planner.joint_names,
        )
        self.device = self.start.position.device
        p, q = self.fk(self.start.position)
        self.start_p, self.start_q = p[0], q[0]
        self.interpolation_dt = self.planner.trajopt_solver.config.interpolation_dt

    def fk(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tool = self.planner.compute_kinematics(JointState.from_position(positions)).tool_poses
        return tool.position[:, 0, 0], tool.quaternion[:, 0, 0]

    def uniform(self, gen: torch.Generator, low: float, high: float, n: int = 1) -> torch.Tensor:
        return (torch.rand(n, generator=gen) * (high - low) + low).to(self.device)

    def plan(
        self, bound: ToolPoseBound, goal_p: torch.Tensor, goal_q: torch.Tensor
    ) -> Dict[str, float]:
        self.planner.update_tool_pose_bounds({self.ee: bound})
        goal = GoalToolPose(
            tool_frames=[self.ee],
            position=goal_p.view(1, 1, 1, 1, 3).clone(),
            quaternion=goal_q.view(1, 1, 1, 1, 4).clone(),
        )
        row = {
            "success": 0.0,
            "error": "",
            "time_ms": math.nan,
            "duration_s": math.nan,
            "path_rad": math.nan,
            "pos_violation_mm": math.nan,
            "rms_pos_violation_mm": math.nan,
            "rot_violation_deg": math.nan,
        }
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        try:
            result = self.planner.plan_pose(goal, self.start.clone())
        except ValueError as e:  # start/goal check of the planner
            row["error"] = str(e).splitlines()[-1].strip()
            return row
        torch.cuda.synchronize()
        row["time_ms"] = 1e3 * (time.perf_counter() - t0)
        if result is None or not bool(result.success.any()):
            return row
        traj = active_positions(result.get_interpolated_plan(), self.planner.joint_names)
        p, q = self.fk(traj)
        anchor_p, anchor_q = self.anchor(bound, goal_p, goal_q)
        # Violation of the user's bounds themselves (no margin, no tolerance).
        raw = dataclasses.replace(
            bound,
            position_margin=0.0,
            rotation_margin=0.0,
            position_tolerance=0.0,
            rotation_tolerance=0.0,
        )
        per_step = tool_pose_bound_violation(
            raw, p, q, anchor_p.expand_as(p), anchor_q.expand_as(q), metric=True
        )
        violation = per_step.amax(dim=0)
        # RMS over the trajectory of the distance outside the cost interval; for an equality
        # axis this is the distance from the reference (how centred the path stays).
        rms_pos = per_step[:, :3].norm(dim=-1).pow(2).mean().sqrt()
        row.update(
            success=1.0,
            duration_s=traj.shape[0] * self.interpolation_dt,
            path_rad=(traj[1:] - traj[:-1]).abs().sum().item(),
            pos_violation_mm=1e3 * violation[:3].max().item(),
            rms_pos_violation_mm=1e3 * rms_pos.item(),
            rot_violation_deg=math.degrees(violation[3:].max().item()),
        )
        return row

    def anchor(self, bound: ToolPoseBound, goal_p, goal_q) -> Tuple[torch.Tensor, torch.Tensor]:
        name = bound.anchor.name
        if name == "GOAL":
            return goal_p, goal_q
        if name == "START":
            return self.start_p, self.start_q
        return torch.zeros(3, device=self.device), torch.tensor(
            [1.0, 0.0, 0.0, 0.0], device=self.device
        )


def plane_scenario(s: Sweep, gen: torch.Generator):
    """Stay on the horizontal plane through the start; goal elsewhere on that plane."""
    z0 = s.start_p[2].item()
    bound = ToolPoseBound.plane(
        "z", anchor="base", offset_position=[None, None, z0], tolerance=0.002
    )
    offset = torch.cat([s.uniform(gen, -0.25, 0.25, 2), torch.zeros(1, device=s.device)])
    return bound, s.start_p + offset, s.start_q


def band_scenario(s: Sweep, gen: torch.Generator):
    """Stay within ±3 mm of the start height (an inequality, so the margin applies)."""
    z0 = s.start_p[2].item()
    bound = ToolPoseBound(
        anchor="base",
        offset_position=[None, None, z0],
        position_lower=[None, None, -0.003],
        position_upper=[None, None, 0.003],
    )
    offset = torch.cat([s.uniform(gen, -0.25, 0.25, 2), torch.zeros(1, device=s.device)])
    return bound, s.start_p + offset, s.start_q


def line_scenario(s: Sweep, gen: torch.Generator):
    """Approach the goal along its tool z axis from the start."""
    d = s.uniform(gen, 0.05, 0.15).item()
    goal_p = s.start_p + _quat_rotate(s.start_q, torch.tensor([0.0, 0.0, d], device=s.device))
    bound = ToolPoseBound.line("z", lower=-(d + 0.02), upper=0.0, tolerance=0.003)
    return bound, goal_p, s.start_q


def orientation_scenario(s: Sweep, gen: torch.Generator):
    """Translate while keeping the tool axis within 5 degrees and the twist within 5 degrees."""
    lim = math.radians(5)
    bound = ToolPoseBound(
        max_swing=lim,
        rotation_lower=[None, None, -lim],
        rotation_upper=[None, None, lim],
        rotation_tolerance=math.radians(1),
    )
    return bound, s.start_p + s.uniform(gen, -0.15, 0.15, 3), s.start_q


SCENARIOS: Dict[str, Scenario] = {
    "plane": plane_scenario,
    "band": band_scenario,
    "line": line_scenario,
    "orientation": orientation_scenario,
}


def summarize(rows: List[Dict]) -> List[Dict]:
    summary = []
    keys = sorted({(r["scenario"], r["margin"], r["weight"]) for r in rows})
    for scenario, margin, weight in keys:
        group = [
            r
            for r in rows
            if r["scenario"] == scenario and r["margin"] == margin and r["weight"] == weight
        ]
        ok = [r for r in group if r["success"] == 1.0]

        def median(key):
            values = sorted(r[key] for r in ok)
            return values[len(values) // 2] if values else math.nan

        summary.append(
            {
                "scenario": scenario,
                "margin": margin,
                "weight": weight,
                "success_rate": len(ok) / len(group),
                "errors": sum(1 for r in group if r["error"]),
                "median_time_ms": median("time_ms"),
                "median_duration_s": median("duration_s"),
                "median_path_rad": median("path_rad"),
                "max_pos_violation_mm": max((r["pos_violation_mm"] for r in ok), default=math.nan),
                "max_rot_violation_deg": max(
                    (r["rot_violation_deg"] for r in ok), default=math.nan
                ),
                "median_rms_pos_violation_mm": median("rms_pos_violation_mm"),
            }
        )
    return summary


def best_weights(summary: List[Dict]) -> Dict[str, Dict]:
    best = {}
    for row in summary:
        key = (-row["success_rate"], row["median_time_ms"] if row["success_rate"] else math.inf)
        current = best.get(row["scenario"])
        if current is None or key < current[0]:
            best[row["scenario"]] = (key, row)
    return {k: v[1] for k, v in best.items()}


def write_csv(path: str, rows: List[Dict]) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def plot(summary: List[Dict], path: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    colors = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300"]
    styles = ["o-", "s--", "^:", "d-."]
    panels = [
        ("success_rate", "success rate"),
        ("median_time_ms", "median planning time [ms]"),
        ("max_pos_violation_mm", "max position violation [mm]"),
        ("max_rot_violation_deg", "max rotation violation [deg]"),
    ]
    fig, axes = plt.subplots(1, len(panels), figsize=(18, 4))
    scenarios = sorted({r["scenario"] for r in summary})
    margins = sorted({r["margin"] for r in summary})
    for ax, (key, label) in zip(axes, panels):
        for (color, scenario), (style, margin) in (
            (c, m) for c in zip(colors, scenarios) for m in zip(styles, margins)
        ):
            rows = [r for r in summary if r["scenario"] == scenario and r["margin"] == margin]
            x = [max(r["weight"], 1e-1) for r in rows]  # weight 0 drawn at 0.1 on the log axis
            name = scenario if len(margins) == 1 else f"{scenario} m={margin * 1e3:g}mm"
            ax.plot(x, [r[key] for r in rows], style, color=color, label=name)
        ax.set_xscale("log")
        ax.set_xlabel("eef_bound weight (0 shown at 0.1)")
        ax.set_title(label, fontsize=10)
        ax.grid(True, color="#e2e1dc")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def sweep_mpc(args, weights: List[float]) -> List[Dict]:
    """Closed-loop MPC on the plane scenario for each weight."""
    from curobo.model_predictive_control import ModelPredictiveControl, ModelPredictiveControlCfg

    scene = None if args.scene == "none" else args.scene
    cfg = ModelPredictiveControlCfg.create(
        robot=args.robot,
        scene_model=scene,
        use_cuda_graph=not args.no_cuda_graph,
        optimization_dt=0.025,
        interpolation_steps=4,
    )
    mpc = ModelPredictiveControl(cfg)
    ee = mpc.tool_frames[0]
    start = JointState.from_position(
        mpc.default_joint_position.clone().unsqueeze(0), joint_names=mpc.joint_names
    )
    start.velocity = torch.zeros_like(start.position)
    start.acceleration = torch.zeros_like(start.position)
    mpc.setup(start.clone())

    def fk(positions):
        tool = mpc.compute_kinematics(JointState.from_position(positions)).tool_poses
        return tool.position[:, 0, 0]

    start_p = fk(start.position)[0]
    start_q = mpc.compute_kinematics(JointState.from_position(start.position)).tool_poses
    start_q = start_q.quaternion[0, 0, 0]
    costs = optimizer_bound_costs(mpc) + optimizer_bound_costs(mpc.ik_solver)
    bound = ToolPoseBound.plane(
        "z", anchor="base", offset_position=[None, None, start_p[2].item()], tolerance=0.002
    )
    mpc.update_tool_pose_bounds({ee: bound})
    goal_p = start_p + torch.tensor([0.0, 0.2, 0.0], device=start_p.device)
    goal = GoalToolPose(
        tool_frames=[ee],
        position=goal_p.view(1, 1, 1, 1, 3).clone(),
        quaternion=start_q.view(1, 1, 1, 1, 4).clone(),
    )
    rows = []
    for weight in weights:
        set_weight(costs, weight)
        state = start.clone()
        mpc.reset_robot(state.clone())
        # MPC's goal IK uses a single seed; fall back to Cartesian tracking when it fails.
        run_ik = bool(mpc._solve_ik_for_goal(goal).success.all())
        mpc.update_goal_tool_poses(goal, run_ik=run_ik)
        positions, success, times = [state.position.clone()], [], []
        for _ in range(args.mpc_steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = mpc.optimize_action_sequence(state)
            torch.cuda.synchronize()
            times.append(1e3 * (time.perf_counter() - t0))
            success.append(bool(result.success.view(-1)[0]))
            actions = result.action_sequence
            state = JointState.from_position(actions.position[:, -1].clone(), mpc.joint_names)
            state.velocity = actions.velocity[:, -1].clone()
            state.acceleration = actions.acceleration[:, -1].clone()
            positions.append(state.position.clone())
        p = fk(torch.cat(positions))
        rows.append(
            {
                "weight": weight,
                "success_rate": sum(success) / len(success),
                "max_dz_mm": 1e3 * (p[:, 2] - start_p[2]).abs().max().item(),
                "final_error_mm": 1e3 * (p[-1] - goal_p).norm().item(),
                "median_step_ms": sorted(times)[len(times) // 2],
                "goal_ik": run_ik,
            }
        )
        print("mpc", rows[-1])
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--scene", default="collision_test.yml", help="'none' for no scene")
    parser.add_argument("--weights", nargs="+", type=float, default=[0, 1e2, 1e3, 1e4, 1e5, 1e6])
    parser.add_argument("--goals", type=int, default=8, help="random goals per scenario")
    parser.add_argument(
        "--margins",
        nargs="+",
        type=float,
        default=[0.001],
        help="position margins (m) to sweep; the rotation margin is 10x the value in rad",
    )
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--target", default="both", choices=["both", "ik", "trajopt"])
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--mpc", action="store_true", help="also sweep the MPC weight")
    parser.add_argument("--mpc-steps", type=int, default=120)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.makedirs(args.out, exist_ok=True)
    scene = None if args.scene == "none" else args.scene

    sweep = Sweep(args.robot, scene, not args.no_cuda_graph, args.target)
    print(f"{len(sweep.costs)} eef_bound optimization costs, start ee {sweep.start_p.tolist()}")
    rows = []
    for name in args.scenarios:
        gen = torch.Generator().manual_seed(args.seed)
        problems = [SCENARIOS[name](sweep, gen) for _ in range(args.goals)]
        for margin in args.margins:
            for weight in args.weights:
                set_weight(sweep.costs, weight)
                done = []
                for i, (bound, goal_p, goal_q) in enumerate(problems):
                    bound = dataclasses.replace(
                        bound, position_margin=margin, rotation_margin=10.0 * margin
                    )
                    row = {"scenario": name, "margin": margin, "weight": weight, "goal": i}
                    row.update(sweep.plan(bound, goal_p, goal_q))
                    rows.append(row)
                    done.append(row)
                rate = sum(r["success"] for r in done) / len(done)
                print(f"{name:12s} margin={margin:<8g} weight={weight:<10g} success={rate:.0%}")
    sweep.planner.update_tool_pose_bounds({sweep.ee: None})

    summary = summarize(rows)
    write_csv(os.path.join(args.out, "sweep_raw.csv"), rows)
    write_csv(os.path.join(args.out, "sweep_summary.csv"), summary)
    plot(summary, os.path.join(args.out, "sweep.png"))
    print(
        "\nscenario     margin[mm]  weight      success  time[ms]  pos viol[mm]  rms pos[mm]"
        "  rot viol[deg]"
    )
    for r in summary:
        print(
            f"{r['scenario']:12s} {r['margin'] * 1e3:10g}  {r['weight']:<10g}  "
            f"{r['success_rate']:6.0%}  "
            f"{r['median_time_ms']:8.0f}  {r['max_pos_violation_mm']:12.2f}  "
            f"{r['median_rms_pos_violation_mm']:11.2f}  {r['max_rot_violation_deg']:13.2f}"
        )
    print("\nbest setting per scenario (success rate, then planning time):")
    for scenario, r in best_weights(summary).items():
        print(f"  {scenario:12s} weight {r['weight']:g}, margin {r['margin'] * 1e3:g} mm")

    if args.mpc:
        del sweep
        torch.cuda.empty_cache()
        mpc_rows = sweep_mpc(args, args.weights)
        write_csv(os.path.join(args.out, "sweep_mpc.csv"), mpc_rows)
    print(f"\nwrote results to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
