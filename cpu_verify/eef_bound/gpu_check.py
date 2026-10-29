# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-to-end GPU checks for tool pose bounds with the motion planner and MPC.

NOT for main: see ``cpu_verify/README.md``. Needs a CUDA GPU and a curobo build.

Run from the repository root::

    python cpu_verify/eef_bound/gpu_check.py                 # CUDA graphs on (default)
    python cpu_verify/eef_bound/gpu_check.py --no-cuda-graph # eager, for comparison
    python cpu_verify/eef_bound/gpu_check.py --only M2 M3    # selected checks

G1-G10 use the motion planner; M1-M4 run MPC in closed loop (the plan is executed perfectly).

Each check prints PASS/FAIL with numbers; the exit code is the number of failed checks.
"""

from __future__ import annotations

import argparse
import math
import time
import traceback
from typing import Callable, Dict, List, Optional, Tuple

import torch

from curobo._src.cost.tool_pose_bound import (
    _quat_mul,
    _quat_rotate,
    tool_pose_bound_violation,
)
from curobo.motion_planner import MotionPlanner, MotionPlannerCfg
from curobo.types import GoalToolPose, JointState, ToolPoseBound

RESULTS: List[Tuple[str, bool, str]] = []


def active_positions(joint_state: JointState, joint_names: List[str]) -> torch.Tensor:
    """``(T, dof)`` positions of ``joint_names`` from a trajectory that may hold locked joints.

    Planner trajectories carry every articulated joint (e.g. Franka fingers); forward kinematics
    takes the active joints only.
    """
    position = joint_state.position.reshape(-1, joint_state.position.shape[-1])
    names = joint_state.joint_names
    if names is None:
        if position.shape[-1] != len(joint_names):
            raise ValueError(f"cannot pick {joint_names} from unnamed {position.shape[-1]} joints")
        return position
    return position[:, [list(names).index(n) for n in joint_names]]


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
        self.planner.update_tool_pose_bounds({self.ee: bound})

    def plan(self, goal: GoalToolPose) -> Tuple[bool, Optional[torch.Tensor], float]:
        """Return success, interpolated joint positions ``(T, dof)`` and wall time."""
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        result = self.planner.plan_pose(goal, self.start.clone())
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        if result is None or not bool(result.success.any()):
            return False, None, elapsed
        traj = active_positions(result.get_interpolated_plan(), self.planner.joint_names)
        return True, traj, elapsed

    def max_violation(
        self,
        positions: torch.Tensor,
        bound: ToolPoseBound,
        anchor_p: torch.Tensor,
        anchor_q: torch.Tensor,
    ) -> torch.Tensor:
        """Largest per-component violation (with tolerance) over a trajectory, shape ``(6,)``."""
        p, q = self.fk(positions)
        violation = tool_pose_bound_violation(
            bound, p, q, anchor_p.expand_as(p), anchor_q.expand_as(q), metric=True
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


def _axis_angle_quat(axis: List[float], angle: float, device) -> torch.Tensor:
    a = torch.tensor(axis, device=device)
    a = a / a.norm()
    half = torch.tensor([math.cos(angle / 2)], device=device)
    return torch.cat([half, math.sin(angle / 2) * a])


def _identity(device) -> torch.Tensor:
    return torch.tensor([1.0, 0.0, 0.0, 0.0], device=device)


def _height_plane(z: float, tolerance: Optional[float] = 0.002) -> ToolPoseBound:
    """Horizontal plane at base height ``z``."""
    return ToolPoseBound.plane(
        (0.0, 0.0, 1.0), point=(0.0, 0.0, z), anchor="base", tolerance=tolerance
    )


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
    bound = _height_plane(z0)
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
    bound = ToolPoseBound.line(
        (0.0, 0.0, 1.0), min_along=-0.12, max_along=0.0, tolerance=0.003
    )
    s.set_bound(bound)
    ok, traj, t = s.plan(s.goal(goal_p, s.start_q))
    if not ok:
        return False, "planning failed with the line bound"
    v = s.max_violation(traj, bound, goal_p, s.start_q)
    return v.max().item() == 0.0, f"{t * 1e3:.0f} ms, max violation {v.max().item() * 1e3:.2f} mm"


@check("G4 orientation box of 5 deg roll/pitch/yaw (goal anchor)")
def g4_orientation(s: Setup):
    goal_p = s.start_p + torch.tensor([0.15, 0.15, -0.1], device=s.device)
    lim = math.radians(5)
    bound = ToolPoseBound.orientation_box(lim, lim, lim, tolerance=math.radians(1))
    s.set_bound(bound)
    ok, traj, t = s.plan(s.goal(goal_p, s.start_q))
    if not ok:
        return False, "planning failed with the orientation bound"
    v = s.max_violation(traj, bound, goal_p, s.start_q)
    return v.max().item() == 0.0, (
        f"{t * 1e3:.0f} ms, max tilt/yaw/roll violation "
        f"{math.degrees(v[3:].max().item()):.2f} deg"
    )


@check("G5 start off the plane raises")
def g5_start_violation(s: Setup):
    z = s.start_p[2].item() + 0.05
    s.set_bound(_height_plane(z, tolerance=None))
    try:
        s.plan(_plane_goal(s))
    except ValueError as e:
        return "start violates" in str(e), str(e).splitlines()[-1].strip()
    return False, "plan_pose did not raise"


@check("G6 goal off the plane raises")
def g6_goal_violation(s: Setup):
    z0 = s.start_p[2].item()
    s.set_bound(_height_plane(z0, tolerance=None))
    goal = s.goal(s.start_p + torch.tensor([0.0, 0.2, 0.1], device=s.device), s.start_q)
    try:
        s.plan(goal)
    except ValueError as e:
        return "goal violates" in str(e), str(e).splitlines()[-1].strip()
    return False, "plan_pose did not raise"


@check("G7 start anchor after CUDA graph capture")
def g7_start_anchor(s: Setup):
    # Graphs were captured with no bounds; switch to a start-anchored plane now.
    bound = ToolPoseBound.plane(anchor="start", tolerance=0.002)
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
    s.set_bound(ToolPoseBound.plane(anchor="start"))
    s.set_bound(None)
    goal = s.goal(s.start_p + torch.tensor([0.0, 0.2, 0.1], device=s.device), s.start_q)
    ok, traj, t = s.plan(goal)
    return ok, f"{t * 1e3:.0f} ms (would raise if the plane were still active)"


@check("G9 IK respects a plane bound")
def g9_ik(s: Setup):
    z0 = s.start_p[2].item()
    bound = _height_plane(z0)
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
    # With zero tolerance an equality bound is never feasible, so every attempt fails and the
    # planner retries; use a realistic tolerance here.
    plane = _height_plane(z0)
    times: Dict[str, List[float]] = {"none": [], "plane": []}
    successes: Dict[str, int] = {"none": 0, "plane": 0}
    for _ in range(5):
        for key, bound in (("none", None), ("plane", plane)):
            s.set_bound(bound)
            ok, _, elapsed = s.plan(goal)
            times[key].append(elapsed)
            successes[key] += int(ok)
    med = {k: sorted(v)[len(v) // 2] * 1e3 for k, v in times.items()}
    return successes["plane"] == 5, (
        f"median {med['none']:.0f} ms without ({successes['none']}/5 ok), "
        f"{med['plane']:.0f} ms with plane ({successes['plane']}/5 ok)"
    )


@check("G11 equality bound with zero tolerance (informational)")
def g11_zero_tolerance(s: Setup):
    z0 = s.start_p[2].item()
    s.set_bound(_height_plane(z0, tolerance=0.0))
    ok, _, elapsed = s.plan(_plane_goal(s))
    return True, f"success={ok} in {elapsed * 1e3:.0f} ms (expected to fail: no slack)"


def _band(z0: float, half_width: float, tolerance: float) -> ToolPoseBound:
    return ToolPoseBound(
        anchor="base", z=(z0 - half_width, z0 + half_width), position_tolerance=tolerance
    )


@check("G12 band bound is met strictly (cost kept tolerance/2 = 1 mm inside)")
def g12_band(s: Setup):
    z0 = s.start_p[2].item()
    s.set_bound(_band(z0, 0.003, tolerance=0.002))
    ok, traj, elapsed = s.plan(_plane_goal(s))
    if not ok:
        return False, "planning failed with a ±3 mm band and 2 mm tolerance"
    dz = (s.fk(traj)[0][:, 2] - z0).abs().max().item()
    return dz <= 0.003 + 1e-5, f"{elapsed * 1e3:.0f} ms, max |dz| {dz * 1e3:.2f} mm (band 3 mm)"


@check("G13 band bound with zero tolerance (informational)")
def g13_band_no_margin(s: Setup):
    z0 = s.start_p[2].item()
    s.set_bound(_band(z0, 0.003, tolerance=0.0))
    ok, traj, elapsed = s.plan(_plane_goal(s))
    dz = math.nan if traj is None else (s.fk(traj)[0][:, 2] - z0).abs().max().item()
    return True, f"success={ok} in {elapsed * 1e3:.0f} ms, max |dz| {dz * 1e3:.2f} mm"


@check("G14 tool z within 10 deg of straight down while yawing 60 deg (base anchor)")
def g14_cone_down(s: Setup):
    yaw = _axis_angle_quat([0.0, 0.0, 1.0], math.radians(60), s.device)
    goal_q = _quat_mul(yaw, s.start_q)
    goal_p = s.start_p + torch.tensor([0.1, -0.1, 0.0], device=s.device)
    bound = ToolPoseBound.axis_tilt(
        reference_axis=(0.0, 0.0, -1.0), max_angle=math.radians(10), anchor="base"
    )
    s.set_bound(bound)
    ok, traj, t = s.plan(s.goal(goal_p, goal_q))
    if not ok:
        return False, "planning failed with the cone bound"
    v = s.max_violation(traj, bound, torch.zeros(3, device=s.device), _identity(s.device))
    p, q = s.fk(traj)
    z_axis = _quat_rotate(q, torch.tensor([0.0, 0.0, 1.0], device=s.device).expand_as(p))
    tilt = torch.rad2deg(torch.acos((-z_axis[:, 2]).clamp(-1.0, 1.0))).max().item()
    return v.max().item() == 0.0, f"{t * 1e3:.0f} ms, max angle from down {tilt:.2f} deg"


@check("G15 two bounds on one frame: height plane and cone (base anchor)")
def g15_two_bounds(s: Setup):
    z0 = s.start_p[2].item()
    bounds = [
        _height_plane(z0),
        ToolPoseBound.axis_tilt(
            reference_axis=(0.0, 0.0, -1.0), max_angle=math.radians(10), anchor="base"
        ),
    ]
    s.planner.update_tool_pose_bounds({s.ee: bounds})
    ok, traj, t = s.plan(_plane_goal(s))
    if not ok:
        return False, "planning failed with plane + cone"
    zero, ident = torch.zeros(3, device=s.device), _identity(s.device)
    v = torch.stack([s.max_violation(traj, b, zero, ident) for b in bounds])
    dz = (s.fk(traj)[0][:, 2] - z0).abs().max().item()
    return v.max().item() == 0.0, f"{t * 1e3:.0f} ms, max |dz| {dz * 1e3:.2f} mm"


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
    g11_zero_tolerance,
    g12_band,
    g13_band_no_margin,
    g14_cone_down,
    g15_two_bounds,
]


class MPCSetup:
    """MPC solver, start state, and a closed-loop runner (the plan is executed perfectly)."""

    def __init__(self, robot: str, scene: Optional[str], use_cuda_graph: bool, steps: int):
        from curobo.model_predictive_control import (
            ModelPredictiveControl,
            ModelPredictiveControlCfg,
        )

        cfg = ModelPredictiveControlCfg.create(
            robot=robot,
            scene_model=scene,
            use_cuda_graph=use_cuda_graph,
            optimization_dt=0.025,
            interpolation_steps=4,
        )
        self.mpc = ModelPredictiveControl(cfg)
        self.steps = steps
        self.ee = self.mpc.tool_frames[0]
        self.start = JointState.from_position(
            self.mpc.default_joint_position.clone().unsqueeze(0), joint_names=self.mpc.joint_names
        )
        self.start.velocity = torch.zeros_like(self.start.position)
        self.start.acceleration = torch.zeros_like(self.start.position)
        self.mpc.setup(self.start.clone())  # CUDA graphs are captured here, without bounds
        self.device = self.start.position.device
        self.start_p, self.start_q = self.fk(self.start.position)
        self.start_p, self.start_q = self.start_p[0], self.start_q[0]

    def fk(self, positions: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        tool = self.mpc.compute_kinematics(JointState.from_position(positions)).tool_poses
        return tool.position[:, 0, 0], tool.quaternion[:, 0, 0]

    def set_bound(self, bound: Optional[ToolPoseBound]):
        self.mpc.update_tool_pose_bounds({self.ee: bound})

    def set_goal(self, goal: GoalToolPose) -> str:
        """Set the goal with IK when MPC's single-seed IK succeeds, else track it in Cartesian space.

        Returns a short description of what was used.
        """
        ik = self.mpc._solve_ik_for_goal(goal)
        if bool(ik.success.all()):
            self.mpc.update_goal_tool_poses(goal, run_ik=True)
            return "ik goal"
        self.mpc.update_goal_tool_poses(goal, run_ik=False)
        pos = ik.position_error.max().item() * 1e3 if ik.position_error is not None else math.nan
        rot = ik.rotation_error.max().item() if ik.rotation_error is not None else math.nan
        return f"cartesian goal (MPC IK failed: pos err {pos:.1f} mm, rot err {rot:.3f})"

    def run(self, goal_p: torch.Tensor, goal_q: torch.Tensor) -> Dict[str, object]:
        """Track the goal in closed loop; return executed positions, success flags and timing."""
        state = self.start.clone()
        self.mpc.reset_robot(state.clone())
        goal = GoalToolPose(
            tool_frames=[self.ee],
            position=goal_p.view(1, 1, 1, 1, 3).clone(),
            quaternion=goal_q.view(1, 1, 1, 1, 4).clone(),
        )
        goal_mode = self.set_goal(goal)
        executed, success, times = [state.position.clone()], [], []
        for _ in range(self.steps):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            result = self.mpc.optimize_action_sequence(state)
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
            success.append(bool(result.success.view(-1)[0]))
            actions = result.action_sequence
            state = JointState.from_position(
                actions.position[:, -1, :].clone(), joint_names=self.mpc.joint_names
            )
            state.velocity = actions.velocity[:, -1, :].clone()
            state.acceleration = actions.acceleration[:, -1, :].clone()
            executed.append(state.position.clone())
        positions = torch.cat(executed, dim=0)
        final_p = self.fk(positions[-1:])[0][0]
        return {
            "positions": positions,
            "success_rate": sum(success) / len(success),
            "step_ms": 1e3 * sorted(times)[len(times) // 2],
            "final_error": (final_p - goal_p).norm().item(),
            "goal_mode": goal_mode,
        }


@check("M1 MPC baseline without bounds")
def m1_baseline(s: MPCSetup):
    goal_p = s.start_p + torch.tensor([0.0, 0.2, 0.0], device=s.device)
    r = s.run(goal_p, s.start_q)
    dz = (s.fk(r["positions"])[0][:, 2] - s.start_p[2]).abs().max().item()
    return r["final_error"] < 0.02, (
        f"final error {r['final_error'] * 1e3:.1f} mm, max |dz| {dz * 1e3:.1f} mm, "
        f"success {r['success_rate']:.0%}, step {r['step_ms']:.1f} ms, {r['goal_mode']}"
    )


@check("M2 MPC plane bound set after graph capture")
def m2_plane(s: MPCSetup):
    z0 = s.start_p[2].item()
    bound = _height_plane(z0)
    s.set_bound(bound)
    goal_p = s.start_p + torch.tensor([0.0, 0.2, 0.0], device=s.device)
    r = s.run(goal_p, s.start_q)
    dz = (s.fk(r["positions"])[0][:, 2] - z0).abs().max().item()
    ok = dz <= 0.0025 and r["final_error"] < 0.02 and r["success_rate"] >= 0.9
    return ok, (
        f"max |dz| {dz * 1e3:.2f} mm (tol 2 mm), final error {r['final_error'] * 1e3:.1f} mm, "
        f"success {r['success_rate']:.0%}, step {r['step_ms']:.1f} ms, {r['goal_mode']}"
    )


@check("M3 MPC orientation cone (goal anchor)")
def m3_orientation(s: MPCSetup):
    lim = math.radians(5)
    bound = ToolPoseBound.axis_tilt(max_angle=lim, tolerance=math.radians(1))
    s.set_bound(bound)
    goal_p = s.start_p + torch.tensor([0.1, 0.1, -0.05], device=s.device)
    r = s.run(goal_p, s.start_q)
    p, q = s.fk(r["positions"])
    v = tool_pose_bound_violation(bound, p, q, goal_p.expand_as(p), s.start_q.expand_as(q))
    ok = v.max().item() == 0.0 and r["final_error"] < 0.02
    return ok, (
        f"max tilt violation {math.degrees(v[:, 3].max().item()):.2f} deg, "
        f"final error {r['final_error'] * 1e3:.1f} mm, success {r['success_rate']:.0%}, "
        f"{r['goal_mode']}"
    )


@check("M4 MPC start anchor follows the robot (informational)")
def m4_start_anchor(s: MPCSetup):
    # The start pose is refreshed from the current state every step, so the plane moves
    # with the robot; this reports how far the height drifts over the run.
    s.set_bound(ToolPoseBound.plane(anchor="start", tolerance=0.002))
    in_plane = _quat_rotate(s.start_q, torch.tensor([0.0, 0.2, 0.0], device=s.device))
    r = s.run(s.start_p + in_plane, s.start_q)
    p = s.fk(r["positions"])[0]
    drift = (p - s.start_p) @ _quat_rotate(
        s.start_q, torch.tensor([0.0, 0.0, 1.0], device=s.device)
    )
    return True, (
        f"max drift along the start normal {drift.abs().max().item() * 1e3:.2f} mm, "
        f"success {r['success_rate']:.0%}, {r['goal_mode']}"
    )


MPC_CHECKS = [m1_baseline, m2_plane, m3_orientation, m4_start_anchor]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="franka.yml")
    parser.add_argument("--scene", default="collision_test.yml", help="'none' for no scene")
    parser.add_argument("--no-cuda-graph", action="store_true")
    parser.add_argument("--only", nargs="*", help="check ids to run, e.g. G2 G7 M2")
    parser.add_argument("--mpc-steps", type=int, default=120)
    parser.add_argument("--skip-mpc", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required")
    scene = None if args.scene == "none" else args.scene

    def selected(run) -> bool:
        return not args.only or run.check_name.split()[0] in args.only

    if any(selected(run) for run in CHECKS):
        setup = Setup(args.robot, scene, use_cuda_graph=not args.no_cuda_graph)
        print(
            f"robot={args.robot} scene={scene} cuda_graph={not args.no_cuda_graph} ee={setup.ee}"
        )
        print(f"start ee position {setup.start_p.tolist()}")
        for run in CHECKS:
            if selected(run):
                run(setup)
        del setup
        torch.cuda.empty_cache()
    if not args.skip_mpc and any(selected(run) for run in MPC_CHECKS):
        mpc_setup = MPCSetup(args.robot, scene, not args.no_cuda_graph, args.mpc_steps)
        for run in MPC_CHECKS:
            if selected(run):
                run(mpc_setup)
    failed = [name for name, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(failed)}/{len(RESULTS)} checks passed")
    for name in failed:
        print(f"  failed: {name}")
    return len(failed)


if __name__ == "__main__":
    raise SystemExit(main())
