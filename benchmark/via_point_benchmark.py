# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Compare ViaPointMotionPlanner with MultiWaypointMotionPlanner on the same via-point paths.

Both planners get the same random paths of ``--num_waypoints`` Cartesian waypoints around the
Franka's home tool pose, above the table of ``collision_table.yml``, with the default criteria
(reach every waypoint's full pose, free path between waypoints). The multi-waypoint planner
starts at an IK solution of its first waypoint, so it gets the current tool pose as an extra
first waypoint. The script reports success, warmed planning time and motion time per planner.

Example::

    python benchmark/via_point_benchmark.py --num_problems 20 --num_waypoints 4
"""

import argparse
import time
from collections import Counter
from typing import Dict, List

import numpy as np
import torch

from curobo._src.motion.multi_waypoint_planner import MultiWaypointMotionPlanner
from curobo._src.motion.multi_waypoint_planner_cfg import MultiWaypointMotionPlannerCfg
from curobo._src.motion.planner_via_point import ViaPointMotionPlanner
from curobo._src.motion.planner_via_point_cfg import ViaPointMotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose


def random_paths(home, num_problems: int, num_waypoints: int, seed: int) -> List[GoalToolPose]:
    """Paths of waypoints within 25 cm of the home tool position, with its orientation."""
    generator = torch.Generator().manual_seed(seed)
    paths = []
    for _ in range(num_problems):
        offsets = (torch.rand(num_waypoints, 3, generator=generator) - 0.5) * 0.5
        offsets[:, 0] = offsets[:, 0].abs() * 0.6
        offsets[:, 2] = offsets[:, 2].clamp(min=-0.2)
        offsets = offsets.to(home.position)
        position = home.position[:, 0:1] + offsets.view(1, num_waypoints, 1, 3)
        quaternion = home.quaternion[:, 0:1].expand(-1, num_waypoints, -1, -1)
        paths.append(
            GoalToolPose(
                home.tool_frames,
                position.unsqueeze(3).contiguous(),
                quaternion.unsqueeze(3).contiguous(),
            )
        )
    return paths


def time_call(fn):
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, time.perf_counter() - start


def summarize(name: str, rows: List[Dict[str, float]]) -> None:
    success = [r for r in rows if r["success"]]
    times = np.array([r["time"] for r in rows])
    motion = np.array([r["motion"] for r in success]) if success else np.array([np.nan])
    print(
        f"{name:>16}: success {len(success)}/{len(rows)}, planning time mean "
        f"{1000 * times.mean():.1f} ms (median {1000 * np.median(times):.1f}, max "
        f"{1000 * times.max():.1f}), motion time mean {np.nanmean(motion):.2f} s"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_problems", type=int, default=20)
    parser.add_argument("--num_waypoints", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--geometry_dt_scale", type=float, default=10.0)
    parser.add_argument("--num_knots", type=int, default=64)
    parser.add_argument("--min_segment_intervals", type=int, default=8)
    parser.add_argument("--skip_multi", action="store_true", help="only run the via-point planner")
    args = parser.parse_args()

    common = dict(robot="franka.yml", scene_model="collision_table.yml", use_cuda_graph=True)
    via = ViaPointMotionPlanner(
        ViaPointMotionPlannerCfg.create(
            geometry_dt_scale=args.geometry_dt_scale,
            num_knots=args.num_knots,
            min_segment_intervals=args.min_segment_intervals,
            **common,
        )
    )
    via.warmup()
    multi = None
    if not args.skip_multi:
        multi = MultiWaypointMotionPlanner(MultiWaypointMotionPlannerCfg.create(**common))
        multi.motion_planner.warmup(enable_graph=False)

    start = JointState.from_position(
        via.default_joint_state.position.unsqueeze(0), joint_names=via.joint_names
    )
    home = via.compute_kinematics(start).tool_poses
    paths = random_paths(home, args.num_problems, args.num_waypoints, args.seed)

    via_rows, multi_rows, via_failures = [], [], Counter()
    for path in paths:
        result, elapsed = time_call(lambda: via.plan_waypoints(path, start))
        ok = bool(result.success.item())
        motion = float(result.trajopt_result.motion_time().view(-1)[0]) if ok else np.nan
        via_rows.append({"success": ok, "time": elapsed, "motion": motion})
        if not ok:
            reason = result.status
            if result.trajopt_result is not None:
                validation = result.trajopt_result.debug_info.get("validation", {})
                failed = [name for name, valid in validation.items() if not bool(valid.all())]
                error = 1000 * result.waypoint_position_error.max().item()
                reason += f" failed={failed} max_waypoint_error={error:.1f}mm"
            via_failures[reason] += 1
        if multi is None:
            continue

        targets = [home.as_goal()] + [
            GoalToolPose(
                path.tool_frames,
                path.position[:, j : j + 1].contiguous(),
                path.quaternion[:, j : j + 1].contiguous(),
            )
            for j in range(path.horizon)
        ]
        criteria = [{} for _ in range(path.horizon)]
        result, elapsed = time_call(lambda: multi.plan_waypoints(targets, start, criteria))
        ok = bool(result.success.item())
        motion = np.nan
        if ok:
            trajectory = result.composed_trajectory
            dt = float(trajectory.dt.reshape(-1)[0])
            motion = (trajectory.position.shape[-2] - 1) * dt
        multi_rows.append({"success": ok, "time": elapsed, "motion": motion})

    print(
        f"{args.num_problems} paths of {args.num_waypoints} waypoints, "
        f"geometry_dt_scale={args.geometry_dt_scale}, num_knots={args.num_knots}, "
        f"min_segment_intervals={args.min_segment_intervals}"
    )
    summarize("ViaPoint", via_rows)
    if multi is not None:
        summarize("MultiWaypoint", multi_rows)
    for reason, count in via_failures.most_common():
        print(f"  ViaPoint failure x{count}: {reason}")
    via.destroy()
    if multi is not None:
        multi.destroy()


if __name__ == "__main__":
    main()
