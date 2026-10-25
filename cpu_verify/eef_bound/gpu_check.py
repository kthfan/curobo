# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU checks for tool pose bounds with the motion planner.

NOT for main: see ``cpu_verify/README.md``. Needs a CUDA GPU and a curobo build.

Run from the repository root::

    python cpu_verify/eef_bound/gpu_check.py                 # CUDA graphs on (default)
    python cpu_verify/eef_bound/gpu_check.py --no-cuda-graph # eager, for comparison

Each check prints PASS/FAIL with numbers; the exit code is the number of failed checks.
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import torch

from curobo._src.cost.tool_pose_bound import _quat_rotate, tool_pose_bound_violation
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, ToolPoseBound, ToolPoseCriteria

RESULTS: List[Tuple[str, bool, str]] = []


class Setup:
    """Planner, start state, and helpers shared by the checks."""

    def __init__(self, robot: str, scene: Optional[str], use_cuda_graph: bool):
        cfg = MotionPlannerCfg.create(
            robot=robot, scene_model=scene, use_cuda_graph=use_cuda_graph
        )
        self.planner = MotionPlanner(cfg)
        self.planner.warmup(enable_graph=use_cuda_graph)
        if len(self.planner.tool_frames) != 1:
            raise SystemExit(f"this script expects one tool frame, got {self.planner.tool_frames}")
        self.ee = self.planner.tool_frames[0]
        self.device = self.planner.default_joint_state.position.device
        self.start = JointState.from_position(
            self.planner.default_joint_state.position.unsqueeze(0),
            joint_names=self.planner.joint_names,
        )
        start_pose = self.fk(self.start.position)
        self.start_p = start_pose[0][0]
        self.start_q = start_pose[1][0]

    def fk(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Tool positions ``(T, 3)`` and quaternions ``(T, 4)`` for joint positions ``(T, dof)``."""
        tool = self.planner.compute_kinematics(JointState.from_position(positions)).tool_poses
        return tool.position[:, 0, 0], tool.quaternion[:, 0, 0]

    def goal(self, position: torch.Tensor, quaternion: torch.Tensor) -> GoalToolPose:
        return GoalToolPose(
            tool_frames=[self.ee],
            position=position.view(1, 1, 1, 1, 3).clone(),
            quaternion=quaternion.view(1, 1, 1, 1, 4).clone(),
        )

    def set_bound(self, bound: Optional[ToolPoseBound]):
        criteria = ToolPoseCriteria(bound=bound, device_cfg=self.planner.device_cfg)
        self.planner.update_tool_pose_criteria({self.ee: criteria})

    def plan(self, goal: GoalToolPose) -> Tuple[bool, Optional[torch.Tensor], float]:
        """Return success, interpolated joint positions ``(T, dof)`` and wall time."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = self.planner.plan_pose(goal, self.start.clone())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if result is None or not bool(result.success.any()):
            return False, None, elapsed
        traj = result.get_interpolated_plan().position
        return True, traj.reshape(-1, traj.shape[-1]), elapsed

    def max_violation(
        self,
        positions: torch.Tensor,
        bound: ToolPoseBound,
        anchor_p: torch.Tensor,
        anchor_q: torch.Tensor,
    ) -> torch.Tensor:
        """Largest per-component violation (with tolerance) over a trajectory, shape ``(7,)``."""
        p, q = self.fk(positions)
        violation = tool_pose_bound_violation(
            bound, p, q, anchor_p.expand_as(p), anchor_q.expand_as(q), use_tolerance=True
        )
        return violation.amax(dim=0)


def check(name: str) -> Callable:
    def register(fn: Callable) -> Callable:
        def run(setup: Setup):
            try:
                ok, detail = fn(setup)
            except Exception:  # report and continue with the next check
                ok, detail = False, traceback.format_exc(limit=3)
            finally:
                setup.set_bound(None)
            RESULTS.append((name, ok, detail))
            print(f"[{'PASS' if ok else 'FAIL'}] {name}: {detail}")

        run.check_name = name
        return run

    return register


def _identity(device) -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)


def _plane_goal(s: Setup) -> GoalToolPose:
    return s.goal(s.start_p + torch.tensor([0.0, 0.2, 0.0], device=s.device), s.start_q)


@check("G1 baseline plan without bounds")
def g1_baseline(s: Setup):
    ok, traj, t = s.plan(_plane_goal(s))
    if not ok:
        return False, "planning failed without any bound; pick another goal"
    dz = (s.fk(traj)[0][:, 2] - s.start_p[2]).abs().max().item()
    return True, f"{t * 1e3:.0f} ms, max |dz| from start height {dz * 1e3:.1f} mm"


@check("G2 plane bound (base anchor) keeps the start height")
def g2_plane(s: Setup):
    z0 = s.start_p[2].item()
    bound = ToolPoseBound.plane(
        "z", anchor="base", offset_position=[None, None, z0], tolerance=0.002
    )
    s.set_bound(bound)
    ok, traj, t = s.plan(_plane_goal(s))
    if not ok:
        return False, "planning failed with the plane bound"
    v = s.max_violation(traj, bound, torch.zeros(3, device=s.device), _identity(s.device))
    dz = (s.fk(traj)[0][:, 2] - z0).abs().max().item()
    return v.max().item() == 0.0, f"{t * 1e3:.0f} ms, max |dz| {dz * 1e3:.2f} mm (tol 2 mm)"


@check("G3 line bound along the goal z axis (approach)")
def g3_line(s: Setup):
    goal_p = s.start_p + _quat_rotate(s.start_q, torch.tensor([0.0, 0.0, 0.1], device=s.device))
    bound = ToolPoseBound.line("z", lower=-0.12, upper=0.0, tolerance=0.003)
    s.set_bound(bound)
    ok, traj, t = s.plan(s.goal(goal_p, s.start_q))
    if not ok:
        return False, "planning failed with the line bound"
    v = s.max_violation(traj, bound, goal_p, s.start_q)
    return v.max().item() == 0.0, f"{t * 1e3:.0f} ms, max violation {v.max().item() * 1e3:.2f} mm"


@check("G4 orientation cone and twist limit (goal anchor)")
def g4_orientation(s: Setup):
    goal_p = s.start_p + torch.tensor([0.15, 0.15, -0.1], device=s.device)
    lim = math.radians(5)
    bound = ToolPoseBound(
        max_swing=lim,
        rotation_lower=[None, None, -lim],
        rotation_upper=[None, None, lim],
        rotation_tolerance=math.radians(1),
    )
    s.set_bound(bound)
    ok, traj, t = s.plan(s.goal(goal_p, s.start_q))
    if not ok:
        return False, "planning failed with the orientation bound"
    v = s.max_violation(traj, bound, goal_p, s.start_q)
    return v.max().item() == 0.0, (
        f"{t * 1e3:.0f} ms, max swing/twist violation {math.degrees(v[3:].max().item()):.2f} deg"
    )


@check("G5 start off the plane raises")
def g5_start_violation(s: Setup):
    z = s.start_p[2].item() + 0.05
    s.set_bound(ToolPoseBound.plane("z", anchor="base", offset_position=[None, None, z]))
    try:
        s.plan(_plane_goal(s))
    except ValueError as e:
        return "start violates" in str(e), str(e).splitlines()[-1].strip()
    return False, "plan_pose did not raise"


@check("G6 goal off the plane raises")
def g6_goal_violation(s: Setup):
    z0 = s.start_p[2].item()
    s.set_bound(ToolPoseBound.plane("z", anchor="base", offset_position=[None, None, z0]))
    goal = s.goal(s.start_p + torch.tensor([0.0, 0.2, 0.1], device=s.device), s.start_q)
    try:
        s.plan(goal)
    except ValueError as e:
        return "goal violates" in str(e), str(e).splitlines()[-1].strip()
    return False, "plan_pose did not raise"


@check("G7 start anchor after CUDA graph capture")
def g7_start_anchor(s: Setup):
    # Graphs were captured with no bounds; switch to a start-anchored plane now.
    bound = ToolPoseBound.plane("z", anchor="start", tolerance=0.002)
    s.set_bound(bound)
    # Goal on the plane through the start, normal to the start tool z axis.
    in_plane = _quat_rotate(s.start_q, torch.tensor([0.0, 0.2, 0.0], device=s.device))
    ok, traj, t = s.plan(s.goal(s.start_p + in_plane, s.start_q))
    if not ok:
        return False, "planning failed with the start-anchored plane"
    v = s.max_violation(traj, bound, s.start_p, s.start_q)
    return v.max().item() == 0.0, f"{t * 1e3:.0f} ms, max violation {v.max().item() * 1e3:.2f} mm"


@check("G8 clearing the bound restores the baseline")
def g8_clear(s: Setup):
    s.set_bound(ToolPoseBound.plane("z", anchor="start"))
    s.set_bound(None)
    goal = s.goal(s.start_p + torch.tensor([0.0, 0.2, 0.1], device=s.device), s.start_q)
    ok, traj, t = s.plan(goal)
    return ok, f"{t * 1e3:.0f} ms (would raise if the plane were still active)"


@check("G9 IK respects a plane bound")
def g9_ik(s: Setup):
    z0 = s.start_p[2].item()
    bound = ToolPoseBound.plane(
        "z", anchor="base", offset_position=[None, None, z0], tolerance=0.002
    )
    s.set_bound(bound)
    result = s.planner.ik_solver.solve_pose(_plane_goal(s), current_state=s.start.clone())
    if not bool(result.success.any()):
        return False, "IK failed"
    solution = result.solution.reshape(-1, result.solution.shape[-1])[result.success.view(-1)]
    v = s.max_violation(solution, bound, torch.zeros(3, device=s.device), _identity(s.device))
    return (
        v.max().item() == 0.0,
        f"{solution.shape[0]} solution(s), max violation {v.max().item():.2e}",
    )


@check("G10 planning time with and without a bound")
def g10_timing(s: Setup):
    goal = _plane_goal(s)
    z0 = s.start_p[2].item()
    times: Dict[str, List[float]] = {"none": [], "plane": []}
    for _ in range(5):
        s.set_bound(None)
        times["none"].append(s.plan(goal)[2])
        s.set_bound(ToolPoseBound.plane("z", anchor="base", offset_position=[None, None, z0]))
        times["plane"].append(s.plan(goal)[2])
    med = {k: sorted(v)[len(v) // 2] * 1e3 for k, v in times.items()}
    return True, f"median {med['none']:.0f} ms without, {med['plane']:.0f} ms with plane"


CHECKS = [
    g1_baseline,
    g2_plane,
    g3_line,
    g4_orientation,
    g5_start_violation,
    g6_goal_violation,
    g7_start_anchor,
    g8_clear,
    g9_ik,
    g10_timing,
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--scene", default="collision_test.yml", help="'none' for no scene")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--only", nargs="*", help="check ids to run, e.g. G2 G7")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    scene = None if args.scene == "none" else args.scene
    setup = Setup(args.robot, scene, use_cuda_graph=not args.no_cuda_graph)
    print(f"robot={args.robot} scene={scene} cuda_graph={not args.no_cuda_graph} ee={setup.ee}")
    print(f"start ee position {setup.start_p.tolist()}")
    for run in CHECKS:
        if args.only and run.check_name.split()[0] not in args.only:
            continue
        run(setup)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print(f"  failed: {name}")
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
