# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Plan through via points from moving start states.

Every problem plans a first path from the home state, takes the state at ``--fraction`` of that
trajectory as the start of a second path, and plans the second path from it with:

- ``rest``: the same position, at rest;
- ``velocity``: the position and velocity, zero acceleration;
- ``velocity+acceleration``: the position, velocity and acceleration;
- ``synthetic s``: the home position with a velocity of ``s`` times the velocity limit in a
  random direction.

Both :class:`ViaPointMotionPlanner` and :meth:`MotionPlanner.plan_pose_waypoints` (with a loose
path tolerance, so the path between waypoints is nearly free) plan every case. The script
reports success, warmed planning time and motion time per planner and case.

Example::

    python benchmark/via_point_moving_start_benchmark.py --robot ur10e.yml --num_problems 12
"""

import argparse
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np
import torch
from via_point_benchmark import random_paths, time_call

from curobo._src.motion.motion_planner import MotionPlanner
from curobo._src.motion.motion_planner_cfg import MotionPlannerCfg
from curobo._src.motion.planner_via_point import ViaPointMotionPlanner
from curobo._src.motion.planner_via_point_cfg import ViaPointMotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose


def state_at(trajectory: JointState, joint_names: List[str], fraction: float) -> JointState:
    """Return the state ``[1, dof]`` at ``fraction`` of a trajectory, in ``joint_names`` order."""
    index = [trajectory.joint_names.index(name) for name in joint_names]
    samples = trajectory.position.reshape(-1, trajectory.position.shape[-1]).shape[0]
    k = int(round(fraction * (samples - 1)))

    def pick(value: torch.Tensor) -> torch.Tensor:
        return value.reshape(-1, value.shape[-1])[k, index].view(1, -1).contiguous()

    return JointState(
        position=pick(trajectory.position),
        velocity=pick(trajectory.velocity),
        acceleration=pick(trajectory.acceleration),
        joint_names=list(joint_names),
    )


def motion_time(trajectory: JointState) -> float:
    """Duration in seconds of an executable trajectory."""
    dt = float(trajectory.dt.reshape(-1)[0])
    return (trajectory.position.reshape(-1, trajectory.position.shape[-1]).shape[0] - 1) * dt


def plan_via(planner: ViaPointMotionPlanner, path: GoalToolPose, state: JointState) -> Dict:
    """Plan with the via-point planner."""
    result, elapsed = time_call(lambda: planner.plan_waypoints(path, state))
    ok = bool(result.success.item())
    return {
        "success": ok,
        "time": elapsed,
        "motion": motion_time(result.get_interpolated_plan()) if ok else np.nan,
    }


def plan_upstream(planner: MotionPlanner, path: GoalToolPose, state: JointState) -> Dict:
    """Plan with ``MotionPlanner.plan_pose_waypoints`` and a nearly free path."""
    result, elapsed = time_call(
        lambda: planner.plan_pose_waypoints(
            path,
            state,
            path_tolerance=(0.5, 3.0),
            final_tolerance=(0.005, 0.05),
            max_attempts=3,
        )
    )
    ok = result is not None and bool(result.success.any())
    return {
        "success": ok,
        "time": elapsed,
        "motion": motion_time(result.get_interpolated_plan()) if ok else np.nan,
    }


def summarize(name: str, rows: List[Dict]) -> None:
    """Print success, mean planning time and mean motion time of one planner and case."""
    times = np.array([r["time"] for r in rows])
    motion = np.array([r["motion"] for r in rows if r["success"]] or [np.nan])
    print(
        f"{name:>40}: success {sum(r['success'] for r in rows)}/{len(rows)}, planning time "
        f"mean {1000 * times.mean():.0f} ms, motion time mean {np.nanmean(motion):.2f} s"
    )


def main() -> None:
    """Run the benchmark."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--robot", type=str, default="franka.yml")
    parser.add_argument("--num_problems", type=int, default=12)
    parser.add_argument("--num_waypoints", type=int, default=3)
    parser.add_argument("--fraction", type=float, default=0.3)
    parser.add_argument("--speeds", type=float, nargs="*", default=[0.1, 0.3, 0.5])
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    common = dict(
        robot=args.robot,
        scene_model="collision_table.yml",
        collision_cache={"cuboid": 8, "mesh": 2},
        use_cuda_graph=True,
    )
    via = ViaPointMotionPlanner(ViaPointMotionPlannerCfg.create(**common))
    via.warmup()
    upstream = MotionPlanner(MotionPlannerCfg.create(**common))
    upstream.warmup(enable_graph=False)

    home = JointState.from_position(
        via.default_joint_state.position.unsqueeze(0), joint_names=via.joint_names
    )
    home_pose = via.compute_kinematics(home).tool_poses
    first_paths = random_paths(home_pose, args.num_problems, args.num_waypoints, args.seed)
    paths = random_paths(home_pose, args.num_problems, args.num_waypoints, args.seed + 1)
    velocity_limit = via.kinematics.get_joint_limits().velocity[1]
    generator = torch.Generator().manual_seed(args.seed)

    starts: List[Tuple[str, int, JointState]] = []
    for i in range(args.num_problems):
        first = via.plan_waypoints(first_paths[i], home)
        if bool(first.success.item()):
            moving = state_at(first.get_interpolated_plan(), via.joint_names, args.fraction)
            rest = JointState.from_position(moving.position.clone(), joint_names=via.joint_names)
            velocity = moving.clone()
            velocity.acceleration = torch.zeros_like(moving.position)
            starts += [
                ("rest", i, rest),
                ("velocity", i, velocity),
                ("velocity+acceleration", i, moving),
            ]
        direction = torch.randn(velocity_limit.shape[0], generator=generator).to(velocity_limit)
        direction = direction / direction.abs().max()
        for speed in args.speeds:
            synthetic = home.clone()
            synthetic.velocity = (speed * velocity_limit * direction).view(1, -1).contiguous()
            synthetic.acceleration = torch.zeros_like(home.position)
            starts.append((f"synthetic {speed}", i, synthetic))

    rows = defaultdict(list)
    for name, i, state in starts:
        rows[("ViaPoint", name)].append(plan_via(via, paths[i], state))
        rows[("plan_pose_waypoints", name)].append(plan_upstream(upstream, paths[i], state))

    print(
        f"robot={args.robot} problems={args.num_problems} waypoints={args.num_waypoints} "
        f"fraction={args.fraction}"
    )
    for planner_name, case in sorted(rows):
        summarize(f"{planner_name} / {case}", rows[(planner_name, case)])
    via.destroy()
    upstream.destroy()


if __name__ == "__main__":
    main()
