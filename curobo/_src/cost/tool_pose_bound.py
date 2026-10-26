# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-link bounds on a tool frame, expressed in a user-defined reference frame.

A :class:`ToolPoseBound` constrains a tool frame relative to a reference frame
``T_ref = T_anchor * T_offset``. The anchor is the goal pose, the start pose (forward kinematics
of the start state), or the robot base frame. The tool pose is expressed in the reference frame:

* position components ``(x, y, z) = R_refᵀ (p - t_ref)``;
* rotation components ``(swing_x, swing_y, twist)`` from the swing-twist decomposition of
  ``r = q_ref⁻¹ ⊗ q`` about the reference z axis, with ``r = swing ⊗ twist``. The swing
  rotation vector (radians) says where the tool z axis points; the twist is the rotation about
  the tool z axis.

Each component has an optional ``[lower, upper]`` interval. The optimization cost uses the
interval as is; feasibility checks widen it by ``position_tolerance`` / ``rotation_tolerance``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise

#: Magnitude used for unbounded interval ends inside kernels.
UNBOUNDED = 1.0e9

_AXES = ("x", "y", "z")


class ToolPoseBoundAnchor(Enum):
    """Pose that the reference frame of a :class:`ToolPoseBound` is attached to."""

    #: Goal pose of the tool frame. With a goalset, a pose is feasible if it satisfies the
    #: bound of any goal in the set.
    GOAL = 0
    #: Tool frame pose at the start state of the problem. In MPC the start state is the
    #: current robot state, so the reference moves with the robot at every step; use GOAL or
    #: BASE for a fixed reference.
    START = 1
    #: Robot base frame.
    BASE = 2


def _optional_vec3(value: Optional[Sequence[Optional[float]]], name: str) -> List[Optional[float]]:
    if value is None:
        return [None, None, None]
    if len(value) != 3:
        log_and_raise(f"{name} must have 3 entries, got {len(value)}")
    return [None if v is None else float(v) for v in value]


@dataclass
class ToolPoseBound:
    """Bounds on one tool frame relative to a reference frame.

    Raises at construction if an interval does not contain the reference itself (component
    value 0), so the anchor always satisfies the components the user filled in.
    """

    #: Pose the reference frame is attached to.
    anchor: Union[ToolPoseBoundAnchor, str] = ToolPoseBoundAnchor.GOAL

    #: Reference position relative to the anchor, in the anchor frame (m). Entries left as
    #: ``None`` are 0, i.e. the reference coincides with the anchor along that anchor axis.
    offset_position: Optional[Sequence[Optional[float]]] = None

    #: Reference orientation relative to the anchor, wxyz. ``None`` is the identity.
    offset_quaternion: Optional[Sequence[float]] = None

    #: Lower bounds on ``(x, y, z)`` in the reference frame (m). ``None`` entries are unbounded.
    position_lower: Optional[Sequence[Optional[float]]] = None

    #: Upper bounds on ``(x, y, z)`` in the reference frame (m). ``None`` entries are unbounded.
    position_upper: Optional[Sequence[Optional[float]]] = None

    #: Lower bounds on ``(swing_x, swing_y, twist)`` (rad). ``None`` entries are unbounded.
    rotation_lower: Optional[Sequence[Optional[float]]] = None

    #: Upper bounds on ``(swing_x, swing_y, twist)`` (rad). ``None`` entries are unbounded.
    rotation_upper: Optional[Sequence[Optional[float]]] = None

    #: Upper bound on the swing angle ``|swing|`` (rad), i.e. a cone around the reference z
    #: axis. ``None`` is unbounded.
    max_swing: Optional[float] = None

    #: Slack (m) added to every position interval when checking feasibility.
    position_tolerance: float = 0.0

    #: Slack (rad) added to every rotation interval and to ``max_swing`` when checking
    #: feasibility.
    rotation_tolerance: float = 0.0

    def __post_init__(self):
        """Normalize inputs and validate intervals against the reference pose."""
        if isinstance(self.anchor, str):
            try:
                self.anchor = ToolPoseBoundAnchor[self.anchor.upper()]
            except KeyError:
                log_and_raise(
                    f"anchor must be one of {[a.name.lower() for a in ToolPoseBoundAnchor]}, "
                    + f"got {self.anchor}"
                )
        self.offset_position = _optional_vec3(self.offset_position, "offset_position")
        if self.offset_quaternion is not None:
            if len(self.offset_quaternion) != 4:
                log_and_raise("offset_quaternion must be wxyz with 4 entries")
            norm = math.sqrt(sum(float(v) * float(v) for v in self.offset_quaternion))
            if norm < 1e-6:
                log_and_raise("offset_quaternion must be non-zero")
            self.offset_quaternion = [float(v) / norm for v in self.offset_quaternion]
        self.position_lower = _optional_vec3(self.position_lower, "position_lower")
        self.position_upper = _optional_vec3(self.position_upper, "position_upper")
        self.rotation_lower = _optional_vec3(self.rotation_lower, "rotation_lower")
        self.rotation_upper = _optional_vec3(self.rotation_upper, "rotation_upper")
        if self.position_tolerance < 0.0 or self.rotation_tolerance < 0.0:
            log_and_raise("position_tolerance and rotation_tolerance must be >= 0")
        if self.max_swing is not None and not 0.0 <= self.max_swing <= math.pi:
            log_and_raise(f"max_swing must be in [0, pi], got {self.max_swing}")

        names = [f"position {a}" for a in _AXES] + ["swing_x", "swing_y", "twist"]
        lowers = self.position_lower + self.rotation_lower
        uppers = self.position_upper + self.rotation_upper
        for k, (name, lo, hi) in enumerate(zip(names, lowers, uppers)):
            if k >= 3:
                for v in (lo, hi):
                    if v is not None and not -math.pi <= v <= math.pi:
                        log_and_raise(f"{name} bound must be in [-pi, pi], got {v}")
            if lo is not None and hi is not None and lo > hi:
                log_and_raise(f"{name}: lower bound {lo} > upper bound {hi}")
            # The reference itself (component value 0) must satisfy the bound.
            if (lo is not None and lo > 0.0) or (hi is not None and hi < 0.0):
                log_and_raise(
                    f"{name}: interval [{lo}, {hi}] does not contain the reference pose (0); "
                    + "move the reference with offset_position / offset_quaternion instead"
                )

    @property
    def uses_rotation(self) -> bool:
        """True when any rotation component or the swing cone is bounded."""
        return self.max_swing is not None or any(
            v is not None for v in self.rotation_lower + self.rotation_upper
        )

    def stacked_values(self, use_tolerance: bool) -> Tuple[List[float], ...]:
        """Return kernel-ready values with ``±UNBOUNDED`` for missing bounds.

        Args:
            use_tolerance: Widen intervals by the tolerances (feasibility checks).

        Returns:
            ``(offset_position[3], offset_quaternion[4], position_lower[3], position_upper[3],
            rotation_lower[3], rotation_upper[3], [max_swing])``.
        """
        pos_tol = self.position_tolerance if use_tolerance else 0.0
        rot_tol = self.rotation_tolerance if use_tolerance else 0.0

        def lower(values, tol):
            return [-UNBOUNDED if v is None else v - tol for v in values]

        def upper(values, tol):
            return [UNBOUNDED if v is None else v + tol for v in values]

        offset_q = self.offset_quaternion or [1.0, 0.0, 0.0, 0.0]
        max_swing = UNBOUNDED if self.max_swing is None else self.max_swing + rot_tol
        return (
            [0.0 if v is None else v for v in self.offset_position],
            list(offset_q),
            lower(self.position_lower, pos_tol),
            upper(self.position_upper, pos_tol),
            lower(self.rotation_lower, rot_tol),
            upper(self.rotation_upper, rot_tol),
            [max_swing],
        )

    @staticmethod
    def plane(
        normal_axis: str = "z",
        anchor: Union[ToolPoseBoundAnchor, str] = ToolPoseBoundAnchor.GOAL,
        tolerance: float = 0.0,
        offset_position: Optional[Sequence[Optional[float]]] = None,
        offset_quaternion: Optional[Sequence[float]] = None,
    ) -> ToolPoseBound:
        """Keep the tool frame on the reference plane normal to ``normal_axis``."""
        k = _axis_index(normal_axis)
        bound = [None, None, None]
        bound[k] = 0.0
        return ToolPoseBound(
            anchor=anchor,
            offset_position=offset_position,
            offset_quaternion=offset_quaternion,
            position_lower=bound,
            position_upper=list(bound),
            position_tolerance=tolerance,
        )

    @staticmethod
    def line(
        axis: str = "z",
        lower: Optional[float] = None,
        upper: Optional[float] = None,
        anchor: Union[ToolPoseBoundAnchor, str] = ToolPoseBoundAnchor.GOAL,
        tolerance: float = 0.0,
    ) -> ToolPoseBound:
        """Keep the tool frame on the reference ``axis``, within ``[lower, upper]`` along it."""
        k = _axis_index(axis)
        lo = [0.0, 0.0, 0.0]
        hi = [0.0, 0.0, 0.0]
        lo[k] = lower
        hi[k] = upper
        return ToolPoseBound(
            anchor=anchor,
            position_lower=lo,
            position_upper=hi,
            position_tolerance=tolerance,
        )


def _axis_index(axis: str) -> int:
    if axis not in _AXES:
        log_and_raise(f"axis must be one of {_AXES}, got {axis}")
    return _AXES.index(axis)


# ---------------------------------------------------------------------------
# PyTorch evaluation (used for start/goal checks; mirrors the Warp kernel)
# ---------------------------------------------------------------------------
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
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def _quat_rotate(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    w = q[..., :1]
    u = q[..., 1:]
    t = 2.0 * torch.cross(u, v, dim=-1)
    return v + w * t + torch.cross(u, t, dim=-1)


def swing_twist_components(quaternion: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Swing-twist decomposition about z of wxyz quaternions.

    Args:
        quaternion: ``(..., 4)`` unit quaternions ``r = swing ⊗ twist``.

    Returns:
        ``(..., 3)`` components ``(swing_x, swing_y, twist)`` in radians, and ``(...)`` swing
        angle. The twist is 0 when the swing is 180 degrees, where it is undefined.
    """
    r = torch.where(quaternion[..., :1] < 0.0, -quaternion, quaternion)
    w, x, y, z = r.unbind(-1)
    n = torch.sqrt(w * w + z * z)
    m = torch.sqrt(x * x + y * y)
    degenerate = n < 1e-6
    n_safe = torch.where(degenerate, torch.ones_like(n), n)
    twist = torch.where(degenerate, torch.zeros_like(n), 2.0 * torch.atan2(z, w))
    xp = torch.where(degenerate, x, (x * w - y * z) / n_safe)
    yp = torch.where(degenerate, y, (x * z + y * w) / n_safe)
    angle = 2.0 * torch.atan2(m, n)
    scale = torch.where(m > 1e-6, angle / torch.clamp(m, min=1e-6), 2.0 / n_safe)
    return torch.stack([scale * xp, scale * yp, twist], dim=-1), angle


def tool_pose_bound_components(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    reference_position: torch.Tensor,
    reference_quaternion: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Express tool poses in a reference frame.

    Args:
        position: ``(..., 3)`` tool positions.
        quaternion: ``(..., 4)`` tool orientations (wxyz).
        reference_position: ``(..., 3)`` reference positions.
        reference_quaternion: ``(..., 4)`` reference orientations (wxyz).

    Returns:
        ``(..., 6)`` components ``(x, y, z, swing_x, swing_y, twist)`` and ``(...)`` swing angle.
    """
    ref_inv = _quat_conj(reference_quaternion)
    local_position = _quat_rotate(ref_inv, position - reference_position)
    rotation, swing_angle = swing_twist_components(_quat_mul(ref_inv, quaternion))
    return torch.cat([local_position, rotation], dim=-1), swing_angle


def tool_pose_bound_reference(
    bound: ToolPoseBound,
    anchor_position: torch.Tensor,
    anchor_quaternion: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the reference pose ``T_anchor * T_offset`` for ``(..., 3/4)`` anchors."""
    offset_p, offset_q, *_ = bound.stacked_values(use_tolerance=False)
    offset_p = anchor_position.new_tensor(offset_p)
    offset_q = anchor_quaternion.new_tensor(offset_q)
    position = anchor_position + _quat_rotate(
        anchor_quaternion, offset_p.expand_as(anchor_position)
    )
    quaternion = _quat_mul(anchor_quaternion, offset_q.expand_as(anchor_quaternion))
    return position, quaternion


def tool_pose_bound_violation(
    bound: ToolPoseBound,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    anchor_position: torch.Tensor,
    anchor_quaternion: torch.Tensor,
    use_tolerance: bool = True,
) -> torch.Tensor:
    """Per-component violation of ``bound`` (0 where satisfied).

    Args:
        bound: Bound to evaluate.
        position: ``(..., 3)`` tool positions.
        quaternion: ``(..., 4)`` tool orientations (wxyz).
        anchor_position: ``(..., 3)`` anchor positions (broadcastable).
        anchor_quaternion: ``(..., 4)`` anchor orientations (broadcastable).
        use_tolerance: Widen intervals by the bound tolerances.

    Returns:
        ``(..., 7)`` non-negative violations of ``(x, y, z, swing_x, swing_y, twist,
        swing angle)`` in m / rad.
    """
    ref_p, ref_q = tool_pose_bound_reference(bound, anchor_position, anchor_quaternion)
    components, swing_angle = tool_pose_bound_components(position, quaternion, ref_p, ref_q)
    _, _, pos_lo, pos_hi, rot_lo, rot_hi, max_swing = bound.stacked_values(use_tolerance)
    lower = components.new_tensor(pos_lo + rot_lo)
    upper = components.new_tensor(pos_hi + rot_hi)
    violation = torch.relu(lower - components) + torch.relu(components - upper)
    swing_violation = torch.relu(swing_angle - max_swing[0]).unsqueeze(-1)
    return torch.cat([violation, swing_violation], dim=-1)


def tool_pose_bound_cost(
    bound: ToolPoseBound,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    anchor_position: torch.Tensor,
    anchor_quaternion: torch.Tensor,
    use_tolerance: bool = False,
) -> torch.Tensor:
    """Squared-hinge cost ``0.5 * sum(violation²)`` matching the Warp kernel (unweighted)."""
    violation = tool_pose_bound_violation(
        bound, position, quaternion, anchor_position, anchor_quaternion, use_tolerance
    )
    return 0.5 * (violation * violation).sum(dim=-1)


_COMPONENT_NAMES = ("x", "y", "z", "swing_x", "swing_y", "twist", "swing angle")


def find_tool_pose_bound_violations(
    bounds: Dict[str, ToolPoseBound],
    start_tool_poses: ToolPose,
    goal_tool_poses: GoalToolPose,
) -> List[str]:
    """Check that the start and the goals of each problem satisfy the bounds (with tolerance).

    With a goalset, the start must satisfy the bound of at least one goal (goal anchor), and at
    least one goal must be feasible.

    Args:
        bounds: Bound per tool frame.
        start_tool_poses: Tool poses at the start state, ``(n, 1, L, 3/4)``.
        goal_tool_poses: Goals ``(n, 1, L, G, 3/4)`` with the same tool frame order.

    Returns:
        One message per violated (problem, tool frame, pose) check; empty when all pass.
    """
    if start_tool_poses.tool_frames != goal_tool_poses.tool_frames:
        log_and_raise("start and goal tool poses must have the same tool frame order")
    messages = []
    for name, bound in bounds.items():
        if bound is None:
            continue
        li = start_tool_poses.tool_frames.index(name)
        start_p = start_tool_poses.position[:, 0, li]  # (n, 3)
        start_q = start_tool_poses.quaternion[:, 0, li]
        goal_p = goal_tool_poses.position[:, 0, li]  # (n, G, 3)
        goal_q = goal_tool_poses.quaternion[:, 0, li]
        if bound.anchor == ToolPoseBoundAnchor.GOAL:
            anchor_p, anchor_q = goal_p, goal_q
        elif bound.anchor == ToolPoseBoundAnchor.START:
            anchor_p, anchor_q = start_p.unsqueeze(1), start_q.unsqueeze(1)
        else:
            anchor_p = torch.zeros_like(start_p).unsqueeze(1)
            anchor_q = torch.zeros_like(start_q).unsqueeze(1)
            anchor_q[..., 0] = 1.0
        # Start against every anchor: (n, A, 7); feasible if any anchor is satisfied.
        start_violation = tool_pose_bound_violation(
            bound, start_p.unsqueeze(1), start_q.unsqueeze(1), anchor_p, anchor_q
        )
        # Every goal against every anchor: (n, G, A, 7).
        goal_violation = tool_pose_bound_violation(
            bound,
            goal_p.unsqueeze(2),
            goal_q.unsqueeze(2),
            anchor_p.unsqueeze(1),
            anchor_q.unsqueeze(1),
        )
        for label, violation in (("start", start_violation), ("goal", goal_violation)):
            # Reduce anchors (and goals) to the best candidate per problem.
            flat = violation.reshape(violation.shape[0], -1, violation.shape[-1])
            best = flat.amax(dim=-1).argmin(dim=-1)
            best_violation = flat[torch.arange(flat.shape[0]), best]
            for problem in torch.nonzero(best_violation.amax(dim=-1) > 0.0).view(-1).tolist():
                details = ", ".join(
                    f"{_COMPONENT_NAMES[k]} by {best_violation[problem, k].item():.4g}"
                    for k in range(len(_COMPONENT_NAMES))
                    if best_violation[problem, k] > 0.0
                )
                messages.append(
                    f"problem {problem}, tool frame {name}: {label} violates {details}"
                )
    return messages
