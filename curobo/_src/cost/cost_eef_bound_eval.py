# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Evaluation of :class:`~curobo._src.cost.tool_pose_bound.ToolPoseBound` limits.

Converts bounds to the per-component intervals the ``eef_bound`` Warp kernel reads, and provides
a PyTorch reference implementation used by the planner's start and goal checks, the
multi-waypoint planner's IK filtering, and the tests.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from curobo._src.cost.tool_pose_bound import (
    COMPONENT_NAMES,
    ToolPoseBound,
    ToolPoseBoundAnchor,
    normalize_tool_pose_bounds,
)
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise

#: Magnitude used for unbounded interval ends inside kernels.
UNBOUNDED = 1.0e9

#: Slack (m or rad) added to every finite side in feasibility checks to absorb float32 noise of
#: forward kinematics, e.g. a start pose lying exactly on a bound.
METRIC_EPSILON = 1.0e-5

_WRAPPED = (False, False, False, False, True, True)


def eef_bound_limits(
    bound: ToolPoseBound, metric: bool
) -> Tuple[List[float], List[float], List[float]]:
    """Return per-component ``(center, lower, upper)`` of ``bound`` in kernel order.

    ``yaw`` and ``roll`` are compared as the wrapped difference to ``center``; the other
    components use a center of 0. Missing ends are ``±UNBOUNDED``.

    Args:
        bound: Bound to convert.
        metric: ``True`` for the feasibility interval, ``False`` for the optimization cost
            interval.

    Returns:
        Three lists of 6 values each.
    """
    anchor_values = _anchor_component_values(bound)
    centers, lowers, uppers = [], [], []
    for i, name in enumerate(COMPONENT_NAMES):
        tolerance = bound.position_tolerance if i < 3 else bound.rotation_tolerance
        c, lo, hi = _interval(
            getattr(bound, name),
            tolerance,
            metric,
            _WRAPPED[i],
            name == "tilt",
            anchor_values[i],
        )
        centers.append(c)
        lowers.append(lo)
        uppers.append(hi)
    return centers, lowers, uppers


def _anchor_component_values(bound: ToolPoseBound) -> List[Optional[float]]:
    """Component values of the anchor pose itself, where they are fixed.

    The cost does not shrink a side past these values, so a goal or start pose lying on the
    edge of an interval does not fight the pose cost.
    """
    if bound.anchor == ToolPoseBoundAnchor.BASE:
        return [None] * 6
    values: List[Optional[float]] = [0.0, 0.0, 0.0, None, None, None]
    if bound.reference_axes == "anchor":
        v = torch.tensor(bound.reference_rotation)
        u = torch.tensor(bound.tool_rotation)
        angles, _ = _zyz(_quat_mul(_quat_conj(v), u))
        values[3:] = [float(a) for a in angles]
    return values


def eef_bound_center_pull(bound: ToolPoseBound, metric: bool) -> Tuple[List[float], List[float]]:
    """Return per-component ``(target, weight)`` of the pull toward the middle of each limit.

    The pull is ``0.5 * weight * (d - target)²``, where ``d`` is the value compared against the
    interval returned by :func:`eef_bound_limits` (the wrapped difference to the center for
    ``yaw`` and ``roll``). Limits with one end, equality limits, and feasibility checks get no
    pull (weight 0).

    Args:
        bound: Bound to convert.
        metric: ``True`` for the feasibility check, which never pulls.

    Returns:
        Two lists of 6 values each.
    """
    targets, weights = [0.0] * 6, [0.0] * 6
    if metric or bound.center_weight == 0.0:
        return targets, weights
    for i, name in enumerate(COMPONENT_NAMES):
        limit = getattr(bound, name)
        if limit is None:
            continue
        lo, hi = limit
        if name == "tilt":
            lo = None if lo is not None and lo <= 0.0 else lo
            hi = None if hi is not None and hi >= math.pi else hi
        tolerance = bound.position_tolerance if i < 3 else bound.rotation_tolerance
        if lo is None or hi is None or hi - lo <= tolerance:
            continue
        # Wrapped components are already measured from the middle.
        targets[i] = 0.0 if _WRAPPED[i] else 0.5 * (lo + hi)
        weights[i] = bound.center_weight
    return targets, weights


def _interval(
    limit: Optional[Tuple[Optional[float], Optional[float]]],
    tolerance: float,
    metric: bool,
    wrapped: bool,
    is_tilt: bool,
    anchor_value: Optional[float],
) -> Tuple[float, float, float]:
    """Return ``(center, lower, upper)`` of one component for the kernel."""
    if limit is None:
        return 0.0, -UNBOUNDED, UNBOUNDED
    lo, hi = limit
    if is_tilt:
        # tilt is always within [0, pi]: limits at the ends of that range constrain nothing.
        lo = None if lo is not None and lo <= 0.0 else lo
        hi = None if hi is not None and hi >= math.pi else hi
    center = 0.5 * (lo + hi) if wrapped else 0.0
    lo = None if lo is None else lo - center
    hi = None if hi is None else hi - center
    equality = lo is not None and hi is not None and hi - lo <= tolerance
    if metric:
        slack = (tolerance if equality else 0.0) + METRIC_EPSILON
        return (
            center,
            -UNBOUNDED if lo is None else lo - slack,
            UNBOUNDED if hi is None else hi + slack,
        )
    if equality:
        lo = hi = 0.5 * (lo + hi)
    else:
        lo = None if lo is None else lo + 0.5 * tolerance
        hi = None if hi is None else hi - 0.5 * tolerance
    cost_lo = -UNBOUNDED if lo is None else lo
    cost_hi = UNBOUNDED if hi is None else hi
    if anchor_value is not None:
        a = anchor_value - center
        if wrapped:
            a = a - 2.0 * math.pi * math.floor((a + math.pi) / (2.0 * math.pi))
        user_lo = -UNBOUNDED if limit[0] is None else limit[0] - center
        user_hi = UNBOUNDED if limit[1] is None else limit[1] - center
        if user_lo <= a <= user_hi:
            cost_lo, cost_hi = min(cost_lo, a), max(cost_hi, a)
    return center, cost_lo, cost_hi


def _quat_mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_conj(q: torch.Tensor) -> torch.Tensor:
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q, v = torch.broadcast_tensors(q, torch.cat([torch.zeros_like(v[..., :1]), v], dim=-1))
    u = q[..., 1:]
    v = v[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + q[..., :1] * t + torch.cross(u, t, dim=-1)


def _zyz(quaternion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """``(tilt, yaw, roll)`` of ``R = Rz(yaw) Ry(tilt) Rz(roll)`` and the parallel-axes mask.

    Args:
        quaternion: ``(..., 4)`` wxyz.

    Returns:
        ``(..., 3)`` angles in radians and ``(...)`` mask, True where the tool axis is parallel
        to the reference axis and ``yaw`` and ``roll`` are undefined (reported as 0).
    """
    w, x, y, z = quaternion.unbind(-1)
    m02 = 2.0 * (x * z + w * y)
    m12 = 2.0 * (y * z - w * x)
    m22 = 1.0 - 2.0 * (x * x + y * y)
    m20 = 2.0 * (x * z - w * y)
    m21 = 2.0 * (y * z + w * x)
    s = torch.sqrt(m02 * m02 + m12 * m12)
    tilt = torch.atan2(s, m22)
    parallel = s < 1e-6
    yaw = torch.where(parallel, torch.zeros_like(s), torch.atan2(m12, m02))
    roll = torch.where(parallel, torch.zeros_like(s), torch.atan2(m21, -m20))
    return torch.stack([tilt, yaw, roll], dim=-1), parallel


def tool_pose_bound_reference(
    bound: ToolPoseBound,
    anchor_position: torch.Tensor,
    anchor_quaternion: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the reference frame origin ``(..., 3)`` and orientation ``(..., 4)``."""
    rotation = anchor_quaternion.new_tensor(bound.reference_rotation)
    if bound.reference_axes == "anchor":
        quaternion = _quat_mul(anchor_quaternion, rotation.expand_as(anchor_quaternion))
    else:
        quaternion = rotation.expand_as(anchor_quaternion)
    return anchor_position, quaternion


def tool_pose_bound_components(
    bound: ToolPoseBound,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    anchor_position: torch.Tensor,
    anchor_quaternion: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Express tool poses in the reference frame of ``bound``.

    Args:
        bound: Bound defining the reference frame and tool axis frame.
        position: ``(..., 3)`` tool positions in the base frame.
        quaternion: ``(..., 4)`` tool orientations (wxyz) in the base frame.
        anchor_position: ``(..., 3)`` anchor positions (broadcastable).
        anchor_quaternion: ``(..., 4)`` anchor orientations (broadcastable).

    Returns:
        ``(..., 6)`` components ``(x, y, z, tilt, yaw, roll)`` and ``(...)`` mask where ``yaw``
        and ``roll`` are undefined.
    """
    ref_p, ref_q = tool_pose_bound_reference(bound, anchor_position, anchor_quaternion)
    ref_inv = _quat_conj(ref_q)
    local_position = _quat_rotate(ref_inv, position - ref_p)
    tool_q = quaternion.new_tensor(bound.tool_rotation)
    relative = _quat_mul(_quat_mul(ref_inv, quaternion), tool_q.expand_as(quaternion))
    angles, parallel = _zyz(relative)
    local_position, angles = torch.broadcast_tensors(local_position, angles)
    return torch.cat([local_position, angles], dim=-1), parallel.expand(angles.shape[:-1])


def tool_pose_bound_violation(
    bound: ToolPoseBound,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    anchor_position: torch.Tensor,
    anchor_quaternion: torch.Tensor,
    metric: bool = True,
) -> torch.Tensor:
    """Per-component violation of ``bound`` (0 where satisfied).

    Args:
        bound: Bound to evaluate.
        position: ``(..., 3)`` tool positions.
        quaternion: ``(..., 4)`` tool orientations (wxyz).
        anchor_position: ``(..., 3)`` anchor positions (broadcastable).
        anchor_quaternion: ``(..., 4)`` anchor orientations (broadcastable).
        metric: Use the feasibility interval (``True``) or the optimization cost interval
            (``False``).

    Returns:
        ``(..., 6)`` non-negative violations of ``(x, y, z, tilt, yaw, roll)`` in m / rad.
    """
    components, parallel = tool_pose_bound_components(
        bound, position, quaternion, anchor_position, anchor_quaternion
    )
    center, lower, upper = (components.new_tensor(v) for v in eef_bound_limits(bound, metric))
    d = components - center
    wrapped = components.new_tensor(_WRAPPED, dtype=torch.bool)
    d = torch.where(wrapped, d - 2.0 * math.pi * torch.floor((d + math.pi) / (2.0 * math.pi)), d)
    violation = torch.relu(lower - d) + torch.relu(d - upper)
    undefined = parallel.unsqueeze(-1) & wrapped
    return torch.where(undefined, torch.zeros_like(violation), violation)


def tool_pose_bound_cost(
    bound: ToolPoseBound,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    anchor_position: torch.Tensor,
    anchor_quaternion: torch.Tensor,
    metric: bool = False,
    center_pull: bool = False,
) -> torch.Tensor:
    """Cost of ``bound`` matching the Warp kernel (unweighted).

    Args:
        bound: Bound to evaluate.
        position: ``(..., 3)`` tool positions.
        quaternion: ``(..., 4)`` tool orientations (wxyz).
        anchor_position: ``(..., 3)`` anchor positions (broadcastable).
        anchor_quaternion: ``(..., 4)`` anchor orientations (broadcastable).
        metric: Use the feasibility interval (``True``) or the optimization cost interval.
        center_pull: Add the pull toward the middle of each limit (see
            :attr:`ToolPoseBound.center_weight`), as the kernel does on non-terminal steps.

    Returns:
        ``(...)`` cost: ``0.5 * sum(violation²)`` plus the pull when requested.
    """
    violation = tool_pose_bound_violation(
        bound, position, quaternion, anchor_position, anchor_quaternion, metric
    )
    cost = 0.5 * (violation * violation).sum(dim=-1)
    if not center_pull:
        return cost
    targets, weights = (violation.new_tensor(v) for v in eef_bound_center_pull(bound, metric))
    if not bool((weights > 0).any()):
        return cost
    components, parallel = tool_pose_bound_components(
        bound, position, quaternion, anchor_position, anchor_quaternion
    )
    center, _, _ = (components.new_tensor(v) for v in eef_bound_limits(bound, metric))
    wrapped = components.new_tensor(_WRAPPED, dtype=torch.bool)
    d = components - center
    d = torch.where(wrapped, d - 2.0 * math.pi * torch.floor((d + math.pi) / (2.0 * math.pi)), d)
    pull = 0.5 * weights * (d - targets) ** 2
    pull = torch.where(parallel.unsqueeze(-1) & wrapped, torch.zeros_like(pull), pull)
    return cost + pull.sum(dim=-1)


def tool_pose_bounds_violation(
    bounds: Sequence[ToolPoseBound],
    position: torch.Tensor,
    quaternion: torch.Tensor,
    start_position: Optional[torch.Tensor],
    start_quaternion: Optional[torch.Tensor],
    goal_position: Optional[torch.Tensor],
    goal_quaternion: Optional[torch.Tensor],
) -> torch.Tensor:
    """Feasibility violation of poses against several bounds of one tool frame.

    Goal-anchored bounds are evaluated against every goal in the set, and the goal with the
    smallest violation counts, as in the cost.

    Args:
        bounds: Bounds of the tool frame.
        position: ``(N, 3)`` tool positions.
        quaternion: ``(N, 4)`` tool orientations (wxyz).
        start_position: ``(N, 3)`` start positions, required by start-anchored bounds.
        start_quaternion: ``(N, 4)`` start orientations.
        goal_position: ``(N, G, 3)`` goal positions, required by goal-anchored bounds.
        goal_quaternion: ``(N, G, 4)`` goal orientations.

    Returns:
        ``(N, len(bounds), 6)`` violations, using the best goal for goal-anchored bounds.
    """
    identity_q = quaternion.new_tensor([1.0, 0.0, 0.0, 0.0]).expand_as(quaternion)
    per_bound = []
    goal_bounds = []
    for i, bound in enumerate(bounds):
        if bound.anchor == ToolPoseBoundAnchor.GOAL:
            if goal_position is None:
                log_and_raise("a goal-anchored bound needs goal poses")
            goal_bounds.append(i)
            per_bound.append(None)
            continue
        if bound.anchor == ToolPoseBoundAnchor.START:
            if start_position is None:
                log_and_raise("a start-anchored bound needs start poses")
            anchor_p, anchor_q = start_position, start_quaternion
        else:
            anchor_p, anchor_q = torch.zeros_like(position), identity_q
        per_bound.append(
            tool_pose_bound_violation(bound, position, quaternion, anchor_p, anchor_q)
        )
    if goal_bounds:
        # (N, G, len(goal_bounds), 6): every goal of the set as the anchor.
        goal_violation = torch.stack(
            [
                tool_pose_bound_violation(
                    bounds[i],
                    position.unsqueeze(1),
                    quaternion.unsqueeze(1),
                    goal_position,
                    goal_quaternion,
                )
                for i in goal_bounds
            ],
            dim=2,
        )
        best = goal_violation.amax(dim=(2, 3)).argmin(dim=1)
        chosen = goal_violation[torch.arange(position.shape[0]), best]
        for j, i in enumerate(goal_bounds):
            per_bound[i] = chosen[:, j]
    return torch.stack(per_bound, dim=1)


def find_tool_pose_bound_violations(
    bounds: Dict[str, Sequence[ToolPoseBound]],
    start_tool_poses: ToolPose,
    goal_tool_poses: GoalToolPose,
) -> List[str]:
    """Check that the start and the goals of each problem satisfy the bounds.

    The start must satisfy every bound (goal-anchored bounds for at least one goal), and at least
    one goal of the set must satisfy every bound.

    Args:
        bounds: Bounds per tool frame.
        start_tool_poses: Tool poses at the start state, ``(n, 1, L, 3/4)``.
        goal_tool_poses: Goals ``(n, 1, L, G, 3/4)`` with the same tool frame order.

    Returns:
        One message per violated (problem, tool frame, pose) check; empty when all pass.
    """
    if start_tool_poses.tool_frames != goal_tool_poses.tool_frames:
        log_and_raise("start and goal tool poses must have the same tool frame order")
    messages = []
    for name, frame_bounds in bounds.items():
        frame_bounds = normalize_tool_pose_bounds(frame_bounds)
        if not frame_bounds:
            continue
        li = start_tool_poses.tool_frames.index(name)
        start_p = start_tool_poses.position[:, 0, li]  # (n, 3)
        start_q = start_tool_poses.quaternion[:, 0, li]
        goal_p = goal_tool_poses.position[:, 0, li]  # (n, G, 3)
        goal_q = goal_tool_poses.quaternion[:, 0, li]
        n, g = goal_p.shape[0], goal_p.shape[1]
        start_violation = tool_pose_bounds_violation(
            frame_bounds, start_p, start_q, start_p, start_q, goal_p, goal_q
        )
        # Every goal as the pose, with the full goalset as goal anchors: (n*G, bounds, 6).
        goal_violation = tool_pose_bounds_violation(
            frame_bounds,
            goal_p.reshape(-1, 3),
            goal_q.reshape(-1, 4),
            start_p.repeat_interleave(g, dim=0),
            start_q.repeat_interleave(g, dim=0),
            goal_p.repeat_interleave(g, dim=0),
            goal_q.repeat_interleave(g, dim=0),
        ).reshape(n, g, len(frame_bounds), 6)
        best_goal = goal_violation.amax(dim=(2, 3)).argmin(dim=1)
        goal_violation = goal_violation[torch.arange(n), best_goal]
        for label, violation in (("start", start_violation), ("goal", goal_violation)):
            for problem in torch.nonzero(violation.amax(dim=(1, 2)) > 0.0).view(-1).tolist():
                details = "; ".join(
                    f"bound {b}: "
                    + ", ".join(
                        f"{COMPONENT_NAMES[k]} by {violation[problem, b, k].item():.4g}"
                        for k in range(6)
                        if violation[problem, b, k] > 0.0
                    )
                    for b in range(len(frame_bounds))
                    if violation[problem, b].amax() > 0.0
                )
                messages.append(
                    f"problem {problem}, tool frame {name}: {label} violates {details}"
                )
    return messages
