# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Measure how close waypoint plans are to the shortest joint-space path.

A trajectory through Cartesian waypoints passes one IK solution of every waypoint, so it is at
least as long as the shortest *IK chain*: the choice of one IK solution per waypoint with the
smallest sum of joint-space distances. This script builds a reference for that chain from a
dense set of IK solutions per waypoint (many IK rounds, plus the full-turn variants ``q ± 2π`` of
joints whose limits span more than one turn), and compares the chains and trajectories of
:class:`ViaPointMotionPlanner` and :class:`MultiWaypointMotionPlanner` with it.

The paths are built to punish a planner that only looks one waypoint ahead:

- ``wrap``: a joint with more than one turn of range keeps turning the same way until the
  next waypoint lies beyond its limit. A myopic planner unwinds a full turn there; the shortest
  path takes the long way at the step where that costs least.
- ``double_wrap``: two such joints with different step patterns.
- ``random_reach``: waypoints at the tool poses of widely spread configurations, from a start
  near a joint limit.
- ``pillar``: sweeps from one side of a pillar to the other; the shortest IK chain ignores it.

Reported per planner: success, IK chain length over the reference (1.0 is optimal), trajectory
arc length over the reference, how often the chain is more than 5 % longer than the reference,
motion time, and planning time. ``greedy`` is the chain of a planner that always moves to the
nearest IK solution of the next waypoint, from the same dense IK sets.

Example::

    python benchmark/waypoint_optimality_benchmark.py --robot ur10e.yml --num_problems 8
"""

import argparse
import json
import math
import time
from typing import Dict, List, Optional

import torch

from curobo._src.motion.ik_chain import select_ik_chains
from curobo._src.motion.ik_revolution import (
    dedup_joint_revolutions,
    expand_joint_revolutions,
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
        "pillar_sweep": (0.6, 0.9),
    },
}

#: ViaPointMotionPlanner settings compared: (expand_revolutions, seed_ik_from_previous,
#: chain_length_slack, seed_bulge, ik_return_seeds) and the number of IK seeds of the planner.
#: ``via_base`` is the planner before this study.
VIA_VARIANTS = {
    "via_base": ((False, False, None, 0.0, 16), 32),
    "via_noprev": ((True, False, 0.1, 0.4, 32), 32),
    "via_new": ((True, True, 0.1, 0.4, 32), 32),
    "via_ik64": ((True, True, 0.1, 0.4, 64), 64),
}

#: MultiWaypointMotionPlanner settings compared:
#: (expand_revolutions, max_layer_nodes, start_state_cost, ik_return_seeds).
#: ``multi_base`` is the planner before this study.
MULTI_VARIANTS = {
    "multi_base": (False, None, False, 20),
    "multi_new": (True, 8, True, 32),
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


def pair_cost(a: torch.Tensor, b: torch.Tensor, vmax: Optional[torch.Tensor]) -> torch.Tensor:
    """Cost ``[A, B]`` between configurations: L2 distance, or the time bound with ``vmax``."""
    if vmax is None:
        return torch.cdist(a, b)
    return ((a.unsqueeze(1) - b.unsqueeze(0)).abs() / vmax).amax(dim=-1)


def shortest_chain(
    start: torch.Tensor, layers: List[torch.Tensor], vmax: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Globally shortest IK chain ``[N, dof]`` through the layers, by dynamic programming."""
    cost = pair_cost(start.view(1, -1), layers[0], vmax)[0]
    parents = []
    for previous, layer in zip(layers[:-1], layers[1:]):
        cost, parent = (cost.unsqueeze(1) + pair_cost(previous, layer, vmax)).min(dim=0)
        parents.append(parent)
    node = int(cost.argmin())
    nodes = [node]
    for parent in reversed(parents):
        node = int(parent[node])
        nodes.append(node)
    nodes.reverse()
    return torch.stack([layer[i] for layer, i in zip(layers, nodes)])


def greedy_chain(start: torch.Tensor, layers: List[torch.Tensor]) -> torch.Tensor:
    """IK chain ``[N, dof]`` that always moves to the nearest solution of the next waypoint."""
    chain, current = [], start
    for layer in layers:
        current = layer[(layer - current).norm(dim=-1).argmin()]
        chain.append(current)
    return torch.stack(chain)


def chain_metrics(start: torch.Tensor, chain: torch.Tensor, vmax: torch.Tensor) -> Dict:
    """Joint-space length (L2 and total rotation) and time bound of a chain from ``start``."""
    steps = torch.cat((start.view(1, -1), chain))
    delta = steps[1:] - steps[:-1]
    return {
        "l2": float(delta.norm(dim=-1).sum()),
        "l1": float(delta.abs().sum()),
        "time_bound": float((delta.abs() / vmax).amax(dim=-1).sum()),
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


def reachable(planner: ViaPointMotionPlanner, waypoints: GoalToolPose, state: JointState) -> bool:
    """Whether every waypoint has a collision-free IK solution in the planner's scene."""
    ik_solver = planner._motion_planner.ik_solver
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
    lower, upper = planner._position_limits
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
        solutions = dedup_joint_revolutions(solutions, 0.02, lower, upper, revolute)
        layers.append(expand_joint_revolutions(solutions, lower, upper, revolute))
    return layers


# -- Planner runs --


def run_via(planner: ViaPointMotionPlanner, waypoints, state, variant: str) -> Dict:
    config = planner.config
    (
        config.expand_revolutions,
        config.seed_ik_from_previous,
        config.chain_length_slack,
        config.seed_bulge,
        config.ik_return_seeds,
    ) = VIA_VARIANTS[variant][0]
    # The IK solver rebuilds its result graphs when the number of returned seeds changes;
    # keep that out of the planning time.
    planner.plan_waypoints(waypoints, state, max_attempts=1)

    # Record, per attempt, the shortest chain through the planner's own IK solutions and how
    # many distinct chains it seeds.
    attempts = []
    plan_through = planner._plan_through

    def record(*args):
        layers, slack = args[3], args[-1]
        seeds = planner._motion_planner.trajopt_solver.config.num_seeds
        _, cost, distinct = select_ik_chains(state.position[0], layers, seeds, slack)
        attempts.append({"shortest": float(cost[0]), "distinct": distinct})
        return plan_through(*args)

    planner._plan_through = record
    torch.cuda.synchronize()
    started = time.perf_counter()
    try:
        result = planner.plan_waypoints(waypoints, state)
    finally:
        planner._plan_through = plan_through
    torch.cuda.synchronize()
    row = {"success": bool(result.success.item()), "plan_time": time.perf_counter() - started}
    row["status"] = result.status
    row["attempts"] = attempts
    if row["success"]:
        row["chain"] = result.waypoint_configs
        trajectory = result.get_interpolated_plan()
        row["arc"] = arc_length(trajectory.position.view(-1, trajectory.position.shape[-1]))
        row["motion_time"] = float(result.trajopt_result.motion_time().view(-1)[0])
    return row


def run_multi(
    planner: MultiWaypointMotionPlanner, start_pose, waypoints, state, variant: str
) -> Dict:
    config = planner.config
    (
        config.expand_revolutions,
        config.max_layer_nodes,
        config.start_state_cost,
        config.ik_return_seeds,
    ) = MULTI_VARIANTS[variant]
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
        nodes = torch.stack(
            [layer[i] for layer, i in zip(captured["layers"], result.chosen_ik_indices)]
        )
        # The planner starts at an IK solution of the first target, not at the current state.
        row["start_gap"] = float((nodes[0] - state.position.view(-1)).norm())
        row["chain"] = nodes
        position = result.composed_trajectory.position
        row["arc"] = arc_length(position.view(-1, position.shape[-1]))
        row["motion_time"] = float(
            sum(float(r.motion_time().view(-1)[0]) for r in result.segment_results)
        )
    return row


def summarize(family: str, rows: List[Dict], planners: List[str]) -> None:
    print(f"\n== {family}: {len(rows)} paths ==")
    if not rows:
        return
    greedy = [r["greedy"]["l2"] / r["reference"]["l2"] for r in rows]
    print(
        f"{'greedy':>9}: chain/ref mean {sum(greedy) / len(greedy):.3f} max {max(greedy):.3f}, "
        f"longer than 1.05x in {sum(g > 1.05 for g in greedy)}/{len(greedy)}"
    )
    for name in planners:
        ran = [r for r in rows if name in r]
        ok = [r for r in ran if r[name]["success"]]
        if not ran:
            continue
        line = f"{name:>9}: success {len(ok)}/{len(ran)}"
        if ok:
            chain = [r[name]["chain_l2"] / r["reference"]["l2"] for r in ok]
            arc = [r[name]["arc"] / r["reference"]["l2"] for r in ok]
            motion = [r[name]["motion_time"] / r["reference"]["time_bound"] for r in ok]
            plan = [r[name]["plan_time"] for r in ran]
            line += (
                f", chain/ref mean {sum(chain) / len(chain):.3f} max {max(chain):.3f} "
                f"(>1.05x in {sum(c > 1.05 for c in chain)}/{len(chain)}), "
                f"arc/ref {sum(arc) / len(arc):.3f}, "
                f"motion/time bound {sum(motion) / len(motion):.2f}, "
                f"motion {sum(r[name]['motion_time'] for r in ok) / len(ok):.2f} s, "
                f"plan {1000 * sum(plan) / len(plan):.0f} ms"
            )
            if name in VIA_VARIANTS:
                first = [r[name]["attempts"][0] for r in ran if r[name]["attempts"]]
                cover = [
                    r[name]["attempts"][0]["shortest"] / r["reference"]["l2"]
                    for r in ran
                    if r[name]["attempts"]
                ]
                line += (
                    f", own IK shortest/ref {sum(cover) / len(cover):.3f} max {max(cover):.3f}, "
                    f"chains seeded {sum(a['distinct'] for a in first) / len(first):.1f}, "
                    f"attempts {sum(len(r[name]['attempts']) for r in ran) / len(ran):.2f}"
                )
            if name in MULTI_VARIANTS:
                gap = [r[name]["start_gap"] for r in ok]
                line += f", start gap mean {sum(gap) / len(gap):.2f} max {max(gap):.2f} rad"
        print(line)


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--robot", default="ur10e.yml", choices=sorted(ROBOTS))
    parser.add_argument("--num_problems", type=int, default=8, help="paths per family")
    parser.add_argument("--multi_problems", type=int, default=8, help="paths per family, multi")
    parser.add_argument("--ik_rounds", type=int, default=40, help="reference IK calls per waypoint")
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
    via_planners = {}
    for seeds in sorted({seeds for _, seeds in VIA_VARIANTS.values()}):
        via_planners[seeds] = ViaPointMotionPlanner(
            ViaPointMotionPlannerCfg.create(num_ik_seeds=seeds, **common)
        )
        via_planners[seeds].warmup()
    via = via_planners[32]
    multi = None
    if args.multi_problems > 0:
        multi = MultiWaypointMotionPlanner(MultiWaypointMotionPlannerCfg.create(**common))
        multi.motion_planner.warmup(enable_graph=False)

    device = via._motion_planner.device_cfg.device
    lower, upper = (limit.cpu() for limit in via._position_limits)
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
    planners = list(VIA_VARIANTS) + list(MULTI_VARIANTS)
    report = {}
    for family in args.families.split(","):
        rows = []
        for index, problem in enumerate(families[family]()):
            pillar = problem["pillar"]
            start = problem["start"].to(device)
            state = JointState.from_position(start.view(1, -1), joint_names=via.joint_names)
            waypoints = make_waypoints(via, problem["configs"].to(device))
            if pillar is not None:
                # The tallest pillar that leaves every waypoint reachable.
                for rise in (0.25, 0.1, 0.0, -0.1, -0.2):
                    pillar = problem["pillar"][:2] + [problem["pillar"][2] + rise]
                    via.update_world(table_scene(pillar))
                    if reachable(via, waypoints, state):
                        break
                scene_planners = [p for p in via_planners.values() if p is not via]
                for planner in scene_planners + ([multi] if multi is not None else []):
                    planner.update_world(table_scene(pillar))
            row = {"index": index, "start": start, "pillar": pillar}
            extra = [[] for _ in range(waypoints.horizon)]
            for name, (_, seeds) in VIA_VARIANTS.items():
                row[name] = run_via(via_planners[seeds], waypoints, state, name)
            if multi is not None and index < args.multi_problems:
                start_pose = via.compute_kinematics(state).tool_poses
                for name in MULTI_VARIANTS:
                    row[name] = run_multi(multi, start_pose, waypoints, state, name)
            for name in planners:
                if name in row and row[name]["success"]:
                    chain = row[name]["chain"][-waypoints.horizon :]
                    for j in range(waypoints.horizon):
                        extra[j].append(chain[j])
            layers = dense_ik_layers(via, waypoints, state, args.ik_rounds, extra)
            if pillar is not None:
                for planner in list(via_planners.values()) + ([multi] if multi is not None else []):
                    planner.update_world(table_scene())
            if layers is None:
                print(f"{family} {index}: a waypoint has no IK solution, skipped")
                continue
            row["layer_sizes"] = [int(layer.shape[0]) for layer in layers]
            reference = shortest_chain(start, layers)
            row["reference"] = chain_metrics(start, reference, vmax)
            row["reference"]["time_bound"] = chain_metrics(
                start, shortest_chain(start, layers, vmax), vmax
            )["time_bound"]
            row["reference"]["chain"] = reference
            row["greedy"] = chain_metrics(start, greedy_chain(start, layers), vmax)
            for name in planners:
                if name in row and row[name]["success"]:
                    # A multi-waypoint chain starts at its own IK solution of the start pose;
                    # measured from the current state it includes the way to that solution.
                    row[name]["chain_l2"] = chain_metrics(start, row[name]["chain"], vmax)["l2"]
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
    for planner in via_planners.values():
        planner.destroy()
    if multi is not None:
        multi.destroy()


if __name__ == "__main__":
    main()
