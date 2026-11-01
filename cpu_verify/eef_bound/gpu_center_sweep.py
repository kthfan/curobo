# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Sweep ``ToolPoseBound.center_weight`` on a GPU and report how centred the planned paths stay.

NOT for main: see ``cpu_verify/README.md``. Needs a CUDA GPU and a curobo build.

Each scenario uses limits with two ends, so the pull toward their middle applies. For every
center weight the script plans the same random goals and reports the success rate, the planning
time, the RMS and largest distance from the middle of each limit over the interpolated path
(excluding the last 10 percent, where the path must reach the goal), the violation of the
limits themselves, and the final position error. The bound cost weight is the task config
default (``eef_bound_cfg.weight``).

Run from the repository root::

    python cpu_verify/eef_bound/gpu_center_sweep.py
    python cpu_verify/eef_bound/gpu_center_sweep.py --center-weights 0 0.01 0.1 1 10 --goals 8
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import math
import os
import sys
import time
from typing import Callable, Dict, List, Optional, Tuple

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from curobo._src.cost.cost_eef_bound_eval import (  # noqa: E402
    _quat_mul,
    eef_bound_limits,
    tool_pose_bound_components,
    tool_pose_bound_violation,
)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg  # noqa: E402
from curobo.types import GoalToolPose, JointState, ToolPoseBound  # noqa: E402
from gpu_check import active_positions  # noqa: E402

DEG = math.pi / 180.0
WRAPPED = (False, False, False, False, True, True)

Problem = Tuple[ToolPoseBound, torch.Tensor, torch.Tensor]


def axis_angle(axis: List[float], angle: float, device) -> torch.Tensor:
    a = torch.tensor(axis, device=device, dtype=torch.float32)
    a = a / a.norm()
    half = torch.tensor([math.cos(angle / 2)], device=device)
    return torch.cat([half, math.sin(angle / 2) * a])


class Setup:
    def __init__(self, robot: str, scene: Optional[str], use_cuda_graph: bool):
        cfg = MotionPlannerCfg.create(robot=robot, scene_model=scene, use_cuda_graph=use_cuda_graph)
        self.planner = MotionPlanner(cfg)
        self.planner.warmup(enable_graph=use_cuda_graph)
        self.ee = self.planner.tool_frames[0]
        self.start = JointState.from_position(
            self.planner.default_joint_state.position.unsqueeze(0),
            joint_names=self.planner.joint_names,
        )
        self.device = self.start.position.device
        p, q = self.fk(self.start.position)
        self.start_p, self.start_q = p[0], q[0]

    def fk(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tool = self.planner.compute_kinematics(JointState.from_position(positions)).tool_poses
        return tool.position[:, 0, 0], tool.quaternion[:, 0, 0]

    def uniform(self, gen: torch.Generator, low: float, high: float, n: int = 1) -> torch.Tensor:
        return (torch.rand(n, generator=gen) * (high - low) + low).to(self.device)

    def start_angle_from_down(self) -> float:
        z = self.fk(self.start.position)[1]
        z_axis = _rotate_z(z[0])
        return math.acos(max(-1.0, min(1.0, -z_axis[2].item())))


def _rotate_z(q: torch.Tensor) -> torch.Tensor:
    w, x, y, z = q.tolist()
    return torch.tensor([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def slab(s: Setup, gen: torch.Generator) -> Problem:
    """±20 mm about the start height; goal at the start height, elsewhere in the slab plane."""
    z0 = s.start_p[2].item()
    bound = ToolPoseBound(anchor="base", z=(z0 - 0.02, z0 + 0.02))
    offset = torch.cat([s.uniform(gen, -0.25, 0.25, 2), torch.zeros(1, device=s.device)])
    return bound, s.start_p + offset, s.start_q


def slab_offset_goal(s: Setup, gen: torch.Generator) -> Problem:
    """Same slab, goal 15 mm above the middle: the path should end there despite the pull."""
    bound, goal_p, goal_q = slab(s, gen)
    return bound, goal_p + torch.tensor([0.0, 0.0, 0.015], device=s.device), goal_q


def tilt_band(s: Setup, gen: torch.Generator) -> Problem:
    """Tool z within ±10 deg of its start angle from straight down, while yawing."""
    a = s.start_angle_from_down()
    bound = ToolPoseBound.axis_tilt(
        reference_axis=(0.0, 0.0, -1.0), min_angle=a - 10 * DEG, max_angle=a + 10 * DEG,
        anchor="base",
    )
    yaw = axis_angle([0.0, 0.0, 1.0], s.uniform(gen, -1.0, 1.0).item(), s.device)
    offset = s.uniform(gen, -0.15, 0.15, 3)
    return bound, s.start_p + offset, _quat_mul(yaw, s.start_q)


def orientation_box(s: Setup, gen: torch.Generator) -> Problem:
    """±15 deg roll, pitch and yaw about the goal orientation, which is 8 deg from the start."""
    bound = ToolPoseBound.orientation_box(15 * DEG, 15 * DEG, 15 * DEG)
    axis = torch.randn(3, generator=gen).tolist()
    goal_q = _quat_mul(s.start_q, axis_angle(axis, 8 * DEG, s.device))
    return bound, s.start_p + s.uniform(gen, -0.15, 0.15, 3), goal_q


SCENARIOS: Dict[str, Callable[[Setup, torch.Generator], Problem]] = {
    "slab": slab,
    "slab_offset_goal": slab_offset_goal,
    "tilt_band": tilt_band,
    "orientation_box": orientation_box,
}


def middle_distance(bound: ToolPoseBound, components: torch.Tensor) -> torch.Tensor:
    """``(T, 6)`` distance of each component from the middle of its two-ended limit (else 0)."""
    center, _, _ = eef_bound_limits(bound, metric=True)
    out = torch.zeros_like(components)
    for i, name in enumerate(("x", "y", "z", "tilt", "yaw", "roll")):
        limit = getattr(bound, name)
        if limit is None or limit[0] is None or limit[1] is None:
            continue
        if WRAPPED[i]:
            d = components[:, i] - center[i]
            d = torch.remainder(d + math.pi, 2 * math.pi) - math.pi
        else:
            d = components[:, i] - 0.5 * (limit[0] + limit[1])
        out[:, i] = d.abs()
    return out


def run(s: Setup, bound: ToolPoseBound, goal_p: torch.Tensor, goal_q: torch.Tensor) -> Dict:
    s.planner.update_tool_pose_bounds({s.ee: bound})
    goal = GoalToolPose(
        tool_frames=[s.ee],
        position=goal_p.view(1, 1, 1, 1, 3).clone(),
        quaternion=goal_q.view(1, 1, 1, 1, 4).clone(),
    )
    row = {"success": 0.0, "time_ms": math.nan, "rms_mid_mm": math.nan, "max_mid_mm": math.nan,
           "rms_mid_deg": math.nan, "max_mid_deg": math.nan, "violation": math.nan,
           "final_err_mm": math.nan, "error": ""}
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    try:
        result = s.planner.plan_pose(goal, s.start.clone())
    except ValueError as e:
        row["error"] = str(e).splitlines()[-1].strip()
        return row
    torch.cuda.synchronize()
    row["time_ms"] = 1e3 * (time.perf_counter() - t0)
    if result is None or not bool(result.success.any()):
        return row
    traj = active_positions(result.get_interpolated_plan(), s.planner.joint_names)
    p, q = s.fk(traj)
    if bound.anchor.name == "GOAL":
        anchor_p, anchor_q = goal_p, goal_q
    elif bound.anchor.name == "START":
        anchor_p, anchor_q = s.start_p, s.start_q
    else:
        anchor_p = torch.zeros(3, device=s.device)
        anchor_q = torch.tensor([1.0, 0.0, 0.0, 0.0], device=s.device)
    ap, aq = anchor_p.expand_as(p), anchor_q.expand_as(q)
    components, _ = tool_pose_bound_components(bound, p, q, ap, aq)
    body = components[: max(1, int(0.9 * components.shape[0]))]
    mid = middle_distance(bound, body)
    raw = dataclasses.replace(bound, position_tolerance=0.0, rotation_tolerance=0.0)
    violation = tool_pose_bound_violation(raw, p, q, ap, aq, metric=True).max().item()
    row.update(
        success=1.0,
        rms_mid_mm=1e3 * mid[:, :3].norm(dim=-1).pow(2).mean().sqrt().item(),
        max_mid_mm=1e3 * mid[:, :3].max().item(),
        rms_mid_deg=mid[:, 3:].norm(dim=-1).pow(2).mean().sqrt().item() / DEG,
        max_mid_deg=mid[:, 3:].max().item() / DEG,
        violation=violation,
        final_err_mm=1e3 * (p[-1] - goal_p).norm().item(),
    )
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--scene", default="collision_test.yml", help="'none' for no scene")
    parser.add_argument("--center-weights", nargs="+", type=float, default=[0, 0.01, 0.1, 1, 10])
    parser.add_argument("--goals", type=int, default=8)
    parser.add_argument("--scenarios", nargs="+", default=list(SCENARIOS), choices=list(SCENARIOS))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    )
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    os.makedirs(args.out, exist_ok=True)
    s = Setup(args.robot, None if args.scene == "none" else args.scene, use_cuda_graph=True)
    rows = []
    for name in args.scenarios:
        gen = torch.Generator().manual_seed(args.seed)
        problems = [SCENARIOS[name](s, gen) for _ in range(args.goals)]
        for cw in args.center_weights:
            for i, (bound, goal_p, goal_q) in enumerate(problems):
                row = {"scenario": name, "center_weight": cw, "goal": i}
                row.update(run(s, dataclasses.replace(bound, center_weight=cw), goal_p, goal_q))
                rows.append(row)
    s.planner.update_tool_pose_bounds({s.ee: None})

    with open(os.path.join(args.out, "center_sweep_raw.csv"), "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    def med(values: List[float]) -> float:
        values = sorted(v for v in values if not math.isnan(v))
        return values[len(values) // 2] if values else math.nan

    print(
        "\nscenario          c_w    success  time[ms]  rms_mid[mm]  max_mid[mm]  rms_mid[deg]"
        "  max_mid[deg]  viol    final_err[mm]  errors"
    )
    for name in args.scenarios:
        for cw in args.center_weights:
            group = [r for r in rows if r["scenario"] == name and r["center_weight"] == cw]
            ok = [r for r in group if r["success"] > 0]
            errors = sorted({r["error"] for r in group if r["error"]})
            print(
                f"{name:16s} {cw:<6g} {len(ok) / len(group):7.0%}  {med([r['time_ms'] for r in group]):8.0f}"
                f"  {med([r['rms_mid_mm'] for r in ok]):11.2f}  {med([r['max_mid_mm'] for r in ok]):11.2f}"
                f"  {med([r['rms_mid_deg'] for r in ok]):12.2f}  {med([r['max_mid_deg'] for r in ok]):12.2f}"
                f"  {max([r['violation'] for r in ok], default=math.nan):.1e}"
                f"  {med([r['final_err_mm'] for r in ok]):13.2f}  {'; '.join(errors)[:60]}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
