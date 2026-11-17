# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Measure how close waypoint plans are to the shortest joint-space path, per chain metric.

A trajectory through Cartesian waypoints passes one IK solution of every waypoint, so it is at
least as long as the shortest *IK chain*: the choice of one IK solution per waypoint with the
smallest sum of joint-space steps. How a step is measured is the
:class:`~curobo._src.motion.ik_chain.ChainMetric`: Euclidean distance, total joint rotation,
time at the joint velocity limits, or the largest motion of one joint.

For every metric this script builds a reference, the shortest chain in that metric through a
dense set of IK solutions per waypoint (many IK rounds, plus the full-turn variants ``q ± 2π``
of joints whose limits span more than one turn), and runs :class:`ViaPointMotionPlanner` and
:class:`MultiWaypointMotionPlanner` with ``chain_metric`` set to that metric. Every planner's
chain is then measured in all four metrics, so the table shows both whether a planner finds the
shortest chain in its own metric (the diagonal) and what that costs in the others.

The paths are built to punish a planner that only looks one waypoint ahead:

- ``wrap_tool`` / ``wrap_pan``: a joint with more than one turn of range keeps turning the same
  way until the next waypoint lies beyond its limit.
- ``double_wrap``: two such joints with different step patterns.
- ``random_reach``: waypoints at the tool poses of widely spread configurations, from a start
  near a joint limit.
- ``pillar``: sweeps from one side of a pillar to the other; the shortest IK chain ignores it.

``greedy_<metric>`` is the chain of a planner that always moves to the nearest IK solution of
the next waypoint in that metric, from the same dense IK sets.

Example::

    python benchmark/waypoint_optimality_benchmark.py --robot ur10e.yml --num_problems 8
"""

import argparse
import json
import math
import time
from typing import Dict, List, Optional

import torch

from curobo._src.motion.ik_chain import (
    ChainMetric,
    chain_step_lengths,
    pairwise_joint_distance,
    select_ik_chains,
)
from curobo._src.motion.ik_revolution import (
    dedup_joint_revolutions,
    expand_joint_revolutions,
    prefer_within_limits,
    revolute_joint_mask,
)
from curobo._src.motion.multi_waypoint_planner import MultiWaypointMotionPlanner
from curobo._src.motion.multi_waypoint_planner_cfg import MultiWaypointMotionPlannerCfg
from curobo._src.motion.planner_via_point import ViaPointMotionPlanner
from curobo._src.motion.planner_via_point_cfg import ViaPointMotionPlannerCfg
from curobo._src.state.state_joint import JointState
from curobo._src.types.tool_pose import GoalToolPose
from curobo.scene import Cuboid, Scene

TURN = 2.0 * math.pi
DEG = math.pi / 180.0

#: Joints that drive the wrap paths (first is the tool roll, the second the base pan) and the
#: spread of the random configurations, per robot.
ROBOTS = {
    "ur10e.yml": {
        "wrap_joints": [5, 0],
        "spread": [2.5, 0.5, 0.5, 1.0, 1.0, 3.0],
        "pan_joint": 0,
        "pillar_sweep": (0.45, 0.75),
    },
    "franka.yml": {
        "wrap_joints": [6, 0],
        "spread": [1.2, 0.4, 1.2, 0.4, 1.2, 0.6, 2.0],
        "pan_joint": 0,
        "pillar_sweep": (0.7, 1.0),
        # A pose with the tool in front of the robot; the default one is folded onto the base.
        "pillar_config": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
    },
}

#: Metrics compared, with short names for the tables.
METRICS = {
    ChainMetric.EUCLIDEAN: "l2",
    ChainMetric.TOTAL_ROTATION: "rot",
    ChainMetric.TIME: "time",
    ChainMetric.MAX_NORM: "max",
}


def table_scene(pillar: Optional[List[float]] = None) -> Scene:
    """The table of ``collision_table.yml``, with an optional pillar ``[x, y, top z]``."""
    cuboids = [Cuboid(name="table", dims=[4.0, 4.0, 0.2], pose=[0.0, 0.0, -0.2, 1, 0, 0, 0])]
    if pillar is not None:
        x, y, top = pillar
        cuboids.append(
            Cuboid(
                name="pillar",
                dims=[0.12, 0.12, top + 0.1],
                pose=[x, y, (top - 0.1) / 2.0, 1, 0, 0, 0],
            )
        )
    return Scene(cuboid=cuboids)


# -- Chains --


def shortest_chain(
    start: torch.Tensor, layers: List[torch.Tensor], metric: ChainMetric, vmax: torch.Tensor
) -> torch.Tensor:
    """Globally shortest IK chain ``[N, dof]`` in ``metric``, by dynamic programming."""
    chains, _, _ = select_ik_chains(start, layers, 1, 0.0, metric, vmax)
    return chains[0]


def greedy_chain(
    start: torch.Tensor, layers: List[torch.Tensor], metric: ChainMetric, vmax: torch.Tensor
) -> torch.Tensor:
    """IK chain ``[N, dof]`` that always moves to the nearest solution of the next waypoint."""
    chain, current = [], start
    for layer in layers:
        step = pairwise_joint_distance(current.view(1, -1), layer, metric, vmax)[0]
        current = layer[step.argmin()]
        chain.append(current)
    return torch.stack(chain)


def chain_lengths(start: torch.Tensor, chain: torch.Tensor, vmax: torch.Tensor) -> Dict:
    """Length of a chain from ``start`` in every metric, by short name."""
    configs = torch.cat((start.view(1, -1), chain))
    return {
        short: float(chain_step_lengths(configs, metric, vmax).sum())
        for metric, short in METRICS.items()
    }


def arc_length(position: torch.Tensor) -> float:
    """Joint-space L2 arc length of a trajectory ``[T, dof]``."""
    return float((position[1:] - position[:-1]).norm(dim=-1).sum())


# -- Problems --


def make_waypoints(planner: ViaPointMotionPlanner, configs: torch.Tensor) -> GoalToolPose:
    """Waypoints ``[1, N, links, 1, 3/4]`` at the tool poses of ``configs`` ``[N, dof]``.

    Forward kinematics is periodic, so configurations beyond the joint limits are fine here.
    """
    poses = planner.compute_kinematics(
        JointState.from_position(configs, joint_names=planner.joint_names)
    ).tool_poses
    return GoalToolPose(
        planner.tool_frames,
        poses.position.view(1, configs.shape[0], -1, 1, 3).contiguous(),
        poses.quaternion.view(1, configs.shape[0], -1, 1, 4).contiguous(),
    )


def wrap_problems(q0, lower, upper, joints, count, generator) -> List[Dict]:
    """Paths that keep turning ``joints`` the same way until a waypoint is beyond a limit."""
    problems = []
    for index in range(count):
        start = q0.clone()
        steps = int(torch.randint(3, 6, (1,), generator=generator))
        configs = start.repeat(steps, 1)
        for joint in joints:
            limit = float(min(upper[joint], -lower[joint]))
            sign = 1.0 if float(torch.rand(1, generator=generator)) < 0.5 else -1.0
            if index == 0 and len(joints) == 1:
                # A joint at 0 turns to 90, 270 and 360 degrees.
                first, sizes = 0.0, torch.tensor([90.0, 180.0, 90.0]) * DEG
                configs = start.repeat(3, 1)
            else:
                first = sign * limit * float(0.2 + 0.7 * torch.rand(1, generator=generator))
                sizes = (20.0 + 150.0 * torch.rand(configs.shape[0], generator=generator)) * DEG
            start[joint] = first
            configs[:, joint] = first + sign * torch.cumsum(sizes, dim=0)
        for joint in range(q0.numel()):
            if joint not in joints:
                configs[:, joint] = start[joint]
        problems.append({"start": start, "configs": configs, "pillar": None})
    return problems


def random_reach_problems(q0, lower, upper, spec, count, generator) -> List[Dict]:
    """Waypoints at the tool poses of widely spread configurations, from a wrapped start."""
    spread = torch.tensor(spec["spread"])
    problems = []
    for _ in range(count):
        start = q0.clone()
        for joint in spec["wrap_joints"]:
            limit = float(min(upper[joint], -lower[joint]))
            start[joint] = limit * 0.9 * float(2.0 * torch.rand(1, generator=generator) - 1.0)
        offsets = (2.0 * torch.rand(4, q0.numel(), generator=generator) - 1.0) * spread
        configs = torch.maximum(torch.minimum(q0 + offsets, upper - 0.05), lower + 0.05)
        problems.append({"start": start, "configs": configs, "pillar": None})
    return problems


def pillar_problems(planner, q0, spec, count, generator) -> List[Dict]:
    """Sweeps of the base joint from one side of a pillar to the other and back."""
    pan = spec["pan_joint"]
    if "pillar_config" in spec:
        q0 = torch.tensor(spec["pillar_config"], device=q0.device, dtype=q0.dtype)
    problems = []
    for _ in range(count):
        low, high = spec["pillar_sweep"]
        half = float(low + (high - low) * torch.rand(1, generator=generator))
        start = q0.clone()
        start[pan] = q0[pan] - half
        sides = torch.tensor([half, -half, half], device=q0.device)
        configs = q0.repeat(3, 1)
        configs[:, pan] = q0[pan] + sides
        middle = planner.compute_kinematics(
            JointState.from_position(q0.view(1, -1), joint_names=planner.joint_names)
        ).tool_poses.position.view(-1)[:3]
        # x, y and the height of the tool there; the pillar's top is set when it is placed.
        pillar = [float(middle[0]), float(middle[1]), float(middle[2])]
        problems.append({"start": start, "configs": configs, "pillar": pillar})
    return problems


# -- Reference IK --


def reference_limits(planner: ViaPointMotionPlanner):
    """Joint limits with the margin that the trajectory optimizer keeps from them."""
    lower, upper = planner._joint_limits[0], planner._joint_limits[1]
    return lower + 0.01 * (upper - lower), upper - 0.01 * (upper - lower)


def reachable(planner: ViaPointMotionPlanner, waypoints: GoalToolPose, state: JointState) -> bool:
    """Whether the start state and an IK solution of every waypoint are collision-free."""
    ik_solver = planner._motion_planner.ik_solver
    # IK of the start pose from the start state returns the start state when it is valid.
    start_pose = planner.compute_kinematics(state).tool_poses.as_goal()
    ik = ik_solver.solve_pose(
        start_pose, current_state=state, return_seeds=ik_solver.config.num_seeds
    )
    solutions = ik.solution.reshape(-1, state.position.shape[-1])[ik.success.view(-1)]
    if solutions.shape[0] == 0 or float((solutions - state.position).abs().amax(-1).min()) > 0.02:
        return False
    for j in range(waypoints.horizon):
        target = GoalToolPose(
            waypoints.tool_frames,
            waypoints.position[:, j : j + 1].contiguous(),
            waypoints.quaternion[:, j : j + 1].contiguous(),
        )
        found = False
        for _ in range(3):
            ik = ik_solver.solve_pose(
                target, current_state=state, return_seeds=ik_solver.config.num_seeds
            )
            found = found or bool(ik.success.any())
        if not found:
            return False
    return True


def dense_ik_layers(
    planner: ViaPointMotionPlanner,
    waypoints: GoalToolPose,
    state: JointState,
    rounds: int,
    extra: List[List[torch.Tensor]],
) -> Optional[List[torch.Tensor]]:
    """Dense IK solutions per waypoint with their full-turn variants, or None if one has none.

    Args:
        planner: Planner whose IK solver and limits are used.
        waypoints: Targets ``[1, N, links, 1, 3/4]``.
        state: Current joint state, the first IK seed of every round.
        rounds: IK calls per waypoint; every call uses new random seeds.
        extra: Per waypoint, configurations to add (the planners' own waypoint configurations),
            so the reference is never longer than a planner's chain.
    """
    ik_solver = planner._motion_planner.ik_solver
    lower, upper = reference_limits(planner)
    layers = []
    for j in range(waypoints.horizon):
        target = GoalToolPose(
            waypoints.tool_frames,
            waypoints.position[:, j : j + 1].contiguous(),
            waypoints.quaternion[:, j : j + 1].contiguous(),
        )
        found = list(extra[j])
        for _ in range(rounds):
            ik = ik_solver.solve_pose(
                target, current_state=state, return_seeds=ik_solver.config.num_seeds
            )
            found.append(ik.solution.reshape(-1, ik.solution.shape[-1])[ik.success.view(-1)])
        solutions = torch.cat([f.view(-1, lower.numel()) for f in found])
        if solutions.shape[0] == 0:
            return None
        revolute = planner._revolute_joints
        solutions = prefer_within_limits(solutions, lower, upper)
        solutions = dedup_joint_revolutions(solutions, 0.02, lower, upper, revolute)
        layers.append(expand_joint_revolutions(solutions, lower, upper, revolute))
    return layers


# -- Planner runs --


def run_via(planner: ViaPointMotionPlanner, waypoints, state, metric: ChainMetric) -> Dict:
    planner.config.chain_metric = metric
    torch.cuda.synchronize()
    started = time.perf_counter()
    result = planner.plan_waypoints(waypoints, state)
    torch.cuda.synchronize()
    row = {"success": bool(result.success.item()), "plan_time": time.perf_counter() - started}
    row["status"] = result.status
    row["attempts"] = result.attempts
    if row["success"]:
        row["chain"] = result.waypoint_configs
        trajectory = result.get_interpolated_plan()
        row["arc"] = arc_length(trajectory.position.view(-1, trajectory.position.shape[-1]))
        row["motion_time"] = float(result.trajopt_result.motion_time().view(-1)[0])
    return row


def run_multi(
    planner: MultiWaypointMotionPlanner, start_pose, waypoints, state, metric: ChainMetric
) -> Dict:
    planner.config.chain_metric = metric
    targets = [start_pose.as_goal()] + [
        GoalToolPose(
            waypoints.tool_frames,
            waypoints.position[:, j : j + 1].contiguous(),
            waypoints.quaternion[:, j : j + 1].contiguous(),
        )
        for j in range(waypoints.horizon)
    ]
    captured = {}
    generate = planner._generate_ik_layers

    def capture(*args, **kwargs):
        captured["layers"], status = generate(*args, **kwargs)
        return captured["layers"], status

    planner._generate_ik_layers = capture
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        result = planner.plan_waypoints(targets, state, [{} for _ in range(waypoints.horizon)])
    finally:
        planner._generate_ik_layers = generate
    torch.cuda.synchronize()
    row = {"success": bool(result.success.item()), "plan_time": time.perf_counter() - started}
    row["status"] = result.status
    if row["success"]:
        # The chain starts at the planner's IK solution of the start pose; measured from the
        # current state it includes the way to that solution.
        row["chain"] = torch.stack(
            [layer[i] for layer, i in zip(captured["layers"], result.chosen_ik_indices)]
        )
        position = result.composed_trajectory.position
        row["arc"] = arc_length(position.view(-1, position.shape[-1]))
        row["motion_time"] = float(
            sum(float(r.motion_time().view(-1)[0]) for r in result.segment_results)
        )
    return row


def summarize(family: str, rows: List[Dict], planners: List[str]) -> None:
    """Print, per planner, its chain length over the reference in every metric."""
    print(f"\n== {family}: {len(rows)} paths ==")
    if not rows:
        return
    header = " ".join(f"{short:>12}" for short in METRICS.values())
    print(f"{'':>13} {'ok':>6} {header}   motion   plan")
    for name in planners:
        ran = [r for r in rows if name in r]
        ok = [r for r in ran if r[name].get("success", True)]
        if not ran:
            continue
        cells = []
        for short in METRICS.values():
            ratio = [r[name]["lengths"][short] / r["reference"][short] for r in ok]
            cells.append(f"{sum(ratio) / len(ratio):.3f}/{max(ratio):.2f}" if ratio else "-")
        line = f"{name:>13} {len(ok):>2}/{len(ran):<3} " + " ".join(f"{c:>12}" for c in cells)
        if not name.startswith("greedy") and ok:
            motion = sum(r[name]["motion_time"] for r in ok) / len(ok)
            plan = sum(r[name]["plan_time"] for r in ran) / len(ran)
            line += f"  {motion:5.2f}s {plan:6.2f}s"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--robot", default="ur10e.yml", choices=sorted(ROBOTS))
    parser.add_argument("--num_problems", type=int, default=8, help="paths per family")
    parser.add_argument("--multi_problems", type=int, default=8, help="paths per family, multi")
    parser.add_argument("--ik_rounds", type=int, default=25, help="reference IK calls per waypoint")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--families", default="wrap_tool,wrap_pan,double_wrap,random_reach,pillar")
    parser.add_argument("--out", default=None, help="write the per-path results to this JSON")
    args = parser.parse_args()

    spec = ROBOTS[args.robot]
    common = dict(
        robot=args.robot,
        scene_model="collision_table.yml",
        collision_cache={"cuboid": 4},
        use_cuda_graph=True,
    )
    via = ViaPointMotionPlanner(ViaPointMotionPlannerCfg.create(**common))
    via.warmup()
    multi = None
    if args.multi_problems > 0:
        multi = MultiWaypointMotionPlanner(MultiWaypointMotionPlannerCfg.create(**common))
        multi.motion_planner.warmup(enable_graph=False)

    device = via._motion_planner.device_cfg.device
    lower, upper = (limit.cpu() for limit in reference_limits(via))
    vmax = via._motion_planner.kinematics.get_joint_limits().velocity[1]
    q0 = via.default_joint_state.position.view(-1).cpu()
    generator = torch.Generator().manual_seed(args.seed)
    print(
        f"robot {args.robot}: limits {[round(float(v) / DEG) for v in upper]} deg, full-turn "
        f"joints {revolute_joint_mask(via._motion_planner.kinematics.kinematics_config).tolist()}"
    )

    wrap = spec["wrap_joints"]
    families = {
        "wrap_tool": lambda: wrap_problems(q0, lower, upper, wrap[:1], args.num_problems, generator),
        "wrap_pan": lambda: wrap_problems(q0, lower, upper, wrap[1:], args.num_problems, generator),
        "double_wrap": lambda: wrap_problems(q0, lower, upper, wrap, args.num_problems, generator),
        "random_reach": lambda: random_reach_problems(
            q0, lower, upper, spec, args.num_problems, generator
        ),
        "pillar": lambda: pillar_problems(via, q0.to(device), spec, args.num_problems, generator),
    }
    planners = (
        [f"greedy_{short}" for short in METRICS.values()]
        + [f"via_{short}" for short in METRICS.values()]
        + [f"multi_{short}" for short in METRICS.values()]
    )
    report = {}
    for family in args.families.split(","):
        rows = []
        for index, problem in enumerate(families[family]()):
            pillar = problem["pillar"]
            start = problem["start"].to(device)
            state = JointState.from_position(start.view(1, -1), joint_names=via.joint_names)
            waypoints = make_waypoints(via, problem["configs"].to(device))
            if pillar is not None:
                # The tallest pillar that leaves the start and every waypoint collision-free.
                valid = False
                for rise in (0.25, 0.1, 0.0, -0.1, -0.2, -0.3):
                    pillar = problem["pillar"][:2] + [problem["pillar"][2] + rise]
                    via.update_world(table_scene(pillar))
                    valid = reachable(via, waypoints, state)
                    if valid:
                        break
                if not valid:
                    via.update_world(table_scene())
                    print(f"{family} {index}: no pillar leaves the path collision-free, skipped")
                    continue
                if multi is not None:
                    multi.update_world(table_scene(pillar))
            row = {"index": index, "start": start, "pillar": pillar}
            extra = [[] for _ in range(waypoints.horizon)]
            for metric, short in METRICS.items():
                row[f"via_{short}"] = run_via(via, waypoints, state, metric)
            if multi is not None and index < args.multi_problems:
                start_pose = via.compute_kinematics(state).tool_poses
                for metric, short in METRICS.items():
                    row[f"multi_{short}"] = run_multi(multi, start_pose, waypoints, state, metric)
            for name in planners:
                if name in row and row[name]["success"]:
                    chain = row[name]["chain"][-waypoints.horizon :]
                    for j in range(waypoints.horizon):
                        extra[j].append(chain[j])
            layers = dense_ik_layers(via, waypoints, state, args.ik_rounds, extra)
            if pillar is not None:
                for planner in [via] + ([multi] if multi is not None else []):
                    planner.update_world(table_scene())
            if layers is None:
                print(f"{family} {index}: a waypoint has no IK solution, skipped")
                continue
            row["layer_sizes"] = [int(layer.shape[0]) for layer in layers]
            row["reference"], row["reference_chains"] = {}, {}
            for metric, short in METRICS.items():
                reference = shortest_chain(start, layers, metric, vmax)
                row["reference"][short] = chain_lengths(start, reference, vmax)[short]
                row["reference_chains"][short] = reference
                greedy = greedy_chain(start, layers, metric, vmax)
                row[f"greedy_{short}"] = {"lengths": chain_lengths(start, greedy, vmax)}
            for name in planners:
                if name in row and row[name].get("success"):
                    row[name]["lengths"] = chain_lengths(start, row[name]["chain"], vmax)
            rows.append(row)
        summarize(family, rows, planners)
        report[family] = rows

    if args.out is not None:

        def plain(value):
            if isinstance(value, torch.Tensor):
                return [[round(v, 4) for v in r] for r in value.view(-1, value.shape[-1]).tolist()]
            if isinstance(value, dict):
                return {k: plain(v) for k, v in value.items()}
            if isinstance(value, list):
                return [plain(v) for v in value]
            return value

        with open(args.out, "w") as file:
            json.dump({"robot": args.robot, "families": plain(report)}, file)
    via.destroy()
    if multi is not None:
        multi.destroy()


if __name__ == "__main__":
    main()
