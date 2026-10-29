# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Limits on a tool frame, expressed in a reference frame.

A :class:`ToolPoseBound` limits the tool frame ``E`` relative to a reference frame ``F``:

* ``F`` has its origin at the anchor (goal pose, start pose, or robot base frame ``B``). Its axes
  are ``reference_rotation`` applied to the anchor's axes (``reference_axes="anchor"``) or to the
  base axes (``reference_axes="base"``).
* Position ``(x, y, z) = R_BFᵀ (p_E - t_anchor)``: the tool frame origin relative to the anchor,
  along the axes of ``F``.
* Orientation: the tool axis frame ``U`` is ``R_BU = R_BE · tool_rotation`` and
  ``R_FU = Rz(yaw) · Ry(tilt) · Rz(roll)``:

  - ``tilt`` in ``[0, π]``: angle between the tool axis (z of ``U``) and the reference axis
    (z of ``F``);
  - ``yaw``: direction the tool axis tilts toward, measured about the reference axis from the x
    axis of ``F``;
  - ``roll``: rotation of the tool about the tool axis; 0 when the x axis of ``U`` lies in the
    plane of the two axes, on the side the tool axis tilts toward.

``yaw`` and ``roll`` are undefined when the two axes are parallel (``tilt`` 0 or π), so they can
only be limited together with a ``tilt`` interval away from 0 and π.

Each limit is used in two ways. The feasibility check (metrics, planner start and goal checks)
uses the given interval. The optimization cost moves each side inward by half the tolerance, so
optimized paths stay inside the interval instead of settling on its edge. An interval no wider
than the tolerance is an equality: the cost pulls to its center and the feasibility check widens
it by the tolerance.
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

#: Slack (m or rad) added to every finite side in feasibility checks to absorb float32 noise of
#: forward kinematics, e.g. a start pose lying exactly on a bound.
METRIC_EPSILON = 1.0e-5

#: Smallest distance (rad) of a ``tilt`` interval from 0 and π when ``yaw`` or ``roll`` is limited.
SINGULARITY_MARGIN = math.radians(10.0)

#: Component names in kernel order.
COMPONENT_NAMES = ("x", "y", "z", "tilt", "yaw", "roll")

_WRAPPED = (False, False, False, False, True, True)

#: A limit: ``None`` (unlimited), a number (equality) or a ``(lower, upper)`` pair whose ends
#: may be ``None``.
Limit = Optional[Union[float, Sequence[Optional[float]]]]


class ToolPoseBoundAnchor(Enum):
    """Pose that the reference frame of a :class:`ToolPoseBound` is attached to."""

    #: Goal pose of the tool frame. With a goalset, a pose is feasible if it satisfies the
    #: bounds of any goal in the set.
    GOAL = 0
    #: Tool frame pose at the start state of the problem. In MPC the start state is the
    #: current robot state, so the reference moves with the robot at every step; use GOAL or
    #: BASE for a fixed reference.
    START = 1
    #: Robot base frame.
    BASE = 2


@dataclass
class ToolPoseBound:
    """Limits on one tool frame relative to a reference frame.

    Construct directly for full control, or use :meth:`plane`, :meth:`line`, :meth:`axis_tilt`
    and :meth:`orientation_box`, which take plain vectors and angles. A tool frame can have
    several bounds; a pose must satisfy all of them.
    """

    #: Pose the reference frame origin is attached to: ``"goal"``, ``"start"`` or ``"base"``.
    anchor: Union[ToolPoseBoundAnchor, str] = ToolPoseBoundAnchor.GOAL

    #: Axes ``reference_rotation`` is applied to: ``"anchor"`` (the reference axes turn with the
    #: anchor) or ``"base"`` (the reference axes are fixed in the robot base frame).
    reference_axes: str = "anchor"

    #: Rotation (wxyz) from the axes chosen by :attr:`reference_axes` to the reference frame.
    #: Its z axis is the reference axis of ``tilt`` and ``yaw``; its x axis is where ``yaw`` is 0.
    #: ``None`` is the identity.
    reference_rotation: Optional[Sequence[float]] = None

    #: Rotation (wxyz) from the tool frame to the tool axis frame. Its z axis is the tool axis of
    #: ``tilt`` and ``roll``; its x axis is where ``roll`` is 0. It does not move the point whose
    #: position is limited, which stays the tool frame origin. ``None`` is the identity.
    tool_rotation: Optional[Sequence[float]] = None

    #: Position of the tool frame origin along the reference x axis, relative to the anchor (m).
    x: Limit = None

    #: Position along the reference y axis, relative to the anchor (m).
    y: Limit = None

    #: Position along the reference z axis, relative to the anchor (m).
    z: Limit = None

    #: Angle between the tool axis and the reference axis (rad), within ``[0, π]``.
    tilt: Limit = None

    #: Direction the tool axis tilts toward, about the reference axis from the reference x axis
    #: (rad). Both ends are required; the interval may cross ±π and spans at most 2π.
    yaw: Limit = None

    #: Rotation of the tool about the tool axis (rad). Both ends are required; the interval may
    #: cross ±π and spans at most 2π.
    roll: Limit = None

    #: Position tolerance (m): the cost keeps paths half of it inside inequality limits, and the
    #: feasibility check accepts this much error on equality limits.
    position_tolerance: float = 0.002

    #: Rotation tolerance (rad), used like :attr:`position_tolerance` for ``tilt``, ``yaw`` and
    #: ``roll``.
    rotation_tolerance: float = 0.02

    def __post_init__(self):
        """Normalize inputs and validate the limits."""
        if isinstance(self.anchor, str):
            try:
                self.anchor = ToolPoseBoundAnchor[self.anchor.upper()]
            except KeyError:
                log_and_raise(
                    f"anchor must be one of {[a.name.lower() for a in ToolPoseBoundAnchor]}, "
                    + f"got {self.anchor}"
                )
        if self.reference_axes not in ("anchor", "base"):
            log_and_raise(f"reference_axes must be 'anchor' or 'base', got {self.reference_axes}")
        self.reference_rotation = _unit_quaternion(self.reference_rotation, "reference_rotation")
        self.tool_rotation = _unit_quaternion(self.tool_rotation, "tool_rotation")
        for name in COMPONENT_NAMES:
            setattr(self, name, _parse_limit(getattr(self, name), name))
        for name in ("position_tolerance", "rotation_tolerance"):
            if getattr(self, name) < 0.0:
                log_and_raise(f"{name} must be >= 0, got {getattr(self, name)}")
        if all(getattr(self, name) is None for name in COMPONENT_NAMES):
            log_and_raise(
                "ToolPoseBound has no limits; set at least one of " + str(COMPONENT_NAMES)
            )

        if self.tilt is not None:
            for v in self.tilt:
                if v is not None and not 0.0 <= v <= math.pi:
                    log_and_raise(f"tilt must be within [0, pi], got {self.tilt}")
        for name in ("yaw", "roll"):
            limit = getattr(self, name)
            if limit is None:
                continue
            if limit[0] is None or limit[1] is None:
                log_and_raise(f"{name} needs both a lower and an upper end, got {limit}")
            if limit[1] - limit[0] > 2.0 * math.pi:
                log_and_raise(f"{name} interval must span at most 2*pi, got {limit}")
        if self.yaw is not None or self.roll is not None:
            lo, hi = self.tilt if self.tilt is not None else (None, None)
            if (
                lo is None
                or hi is None
                or lo < SINGULARITY_MARGIN
                or hi > math.pi - SINGULARITY_MARGIN
            ):
                log_and_raise(
                    "yaw and roll are undefined when the tool axis is parallel to the reference "
                    + "axis; limiting them requires a tilt interval within "
                    + f"[{SINGULARITY_MARGIN:.4f}, {math.pi - SINGULARITY_MARGIN:.4f}] rad, "
                    + f"got tilt={self.tilt}. Choose tool_rotation so that the tool axis is "
                    + "perpendicular to the reference axis, or use orientation_box"
                )

    @property
    def uses_rotation(self) -> bool:
        """True when ``tilt``, ``yaw`` or ``roll`` is limited."""
        return any(getattr(self, name) is not None for name in ("tilt", "yaw", "roll"))

    def kernel_values(self, metric: bool) -> Tuple[List[float], List[float], List[float]]:
        """Return per-component ``(center, lower, upper)`` in kernel order.

        ``yaw`` and ``roll`` are compared as the wrapped difference to ``center``; the other
        components use a center of 0. Missing ends are ``±UNBOUNDED``.

        Args:
            metric: ``True`` for the feasibility interval, ``False`` for the optimization cost
                interval.

        Returns:
            Three lists of 6 values each.
        """
        anchor_values = self._anchor_values()
        centers, lowers, uppers = [], [], []
        for i, name in enumerate(COMPONENT_NAMES):
            tolerance = self.position_tolerance if i < 3 else self.rotation_tolerance
            c, lo, hi = _interval(
                getattr(self, name),
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

    def _anchor_values(self) -> List[Optional[float]]:
        """Component values of the anchor pose itself, where they are fixed.

        The cost does not shrink a side past these values, so a goal or start pose lying on the
        edge of an interval does not fight the pose cost.
        """
        if self.anchor == ToolPoseBoundAnchor.BASE:
            return [None] * 6
        values: List[Optional[float]] = [0.0, 0.0, 0.0, None, None, None]
        if self.reference_axes == "anchor":
            v = torch.tensor(self.reference_rotation)
            u = torch.tensor(self.tool_rotation)
            angles, _ = _zyz(_quat_mul(_quat_conj(v), u))
            values[3:] = [float(a) for a in angles]
        return values

    @staticmethod
    def plane(
        normal: Sequence[float] = (0.0, 0.0, 1.0),
        point: Optional[Sequence[float]] = None,
        min_distance: Optional[float] = 0.0,
        max_distance: Optional[float] = 0.0,
        anchor: Union[ToolPoseBoundAnchor, str] = ToolPoseBoundAnchor.GOAL,
        reference_axes: str = "anchor",
        tolerance: Optional[float] = None,
    ) -> ToolPoseBound:
        """Limit the signed distance of the tool frame origin to a plane.

        The defaults keep the tool on the plane; ``max_distance=None`` keeps it on the side the
        normal points to, and a ``[min_distance, max_distance]`` pair keeps it in a slab.

        Args:
            normal: Plane normal, in the axes chosen by ``reference_axes``.
            point: A point on the plane relative to the anchor, in the same axes. ``None`` is the
                anchor origin.
            min_distance: Lower limit of the signed distance (m); ``None`` is unlimited.
            max_distance: Upper limit of the signed distance (m); ``None`` is unlimited.
            anchor: See :attr:`anchor`.
            reference_axes: See :attr:`reference_axes`.
            tolerance: Position tolerance (m); ``None`` keeps the default.
        """
        n = _unit_vector(normal, "normal")
        offset = sum(a * b for a, b in zip(n, point)) if point is not None else 0.0
        return ToolPoseBound(
            anchor=anchor,
            reference_axes=reference_axes,
            reference_rotation=_rotation_between((0.0, 0.0, 1.0), n),
            z=(_shift(min_distance, offset), _shift(max_distance, offset)),
            **_tolerance_kwargs("position_tolerance", tolerance),
        )

    @staticmethod
    def line(
        direction: Sequence[float] = (0.0, 0.0, 1.0),
        point: Optional[Sequence[float]] = None,
        min_along: Optional[float] = None,
        max_along: Optional[float] = None,
        anchor: Union[ToolPoseBoundAnchor, str] = ToolPoseBoundAnchor.GOAL,
        reference_axes: str = "anchor",
        tolerance: Optional[float] = None,
    ) -> ToolPoseBound:
        """Keep the tool frame origin on a line, optionally within a segment of it.

        Args:
            direction: Line direction, in the axes chosen by ``reference_axes``.
            point: A point on the line relative to the anchor, in the same axes. ``None`` is the
                anchor origin.
            min_along: Lower limit of the position along ``direction``, measured from ``point``
                (m); ``None`` is unlimited.
            max_along: Upper limit of the position along ``direction`` (m); ``None`` is
                unlimited.
            anchor: See :attr:`anchor`.
            reference_axes: See :attr:`reference_axes`.
            tolerance: Position tolerance (m); ``None`` keeps the default.
        """
        d = _unit_vector(direction, "direction")
        rotation = _rotation_between((0.0, 0.0, 1.0), d)
        p = [0.0, 0.0, 0.0] if point is None else [float(v) for v in point]
        if len(p) != 3:
            log_and_raise(f"point must have 3 entries, got {point}")
        local = _quat_rotate(_quat_conj(torch.tensor(rotation)), torch.tensor(p)).tolist()
        return ToolPoseBound(
            anchor=anchor,
            reference_axes=reference_axes,
            reference_rotation=rotation,
            x=local[0],
            y=local[1],
            z=(_shift(min_along, local[2]), _shift(max_along, local[2])),
            **_tolerance_kwargs("position_tolerance", tolerance),
        )

    @staticmethod
    def axis_tilt(
        tool_axis: Sequence[float] = (0.0, 0.0, 1.0),
        reference_axis: Sequence[float] = (0.0, 0.0, 1.0),
        min_angle: Optional[float] = None,
        max_angle: Optional[float] = None,
        anchor: Union[ToolPoseBoundAnchor, str] = ToolPoseBoundAnchor.GOAL,
        reference_axes: str = "anchor",
        tolerance: Optional[float] = None,
    ) -> ToolPoseBound:
        """Limit the angle between an axis of the tool and a reference axis.

        With ``max_angle`` only, the tool axis stays in a cone around the reference axis (for
        example, a cup kept upright). With ``min_angle == max_angle``, the tool axis keeps a fixed
        angle to the reference axis while rotating freely about both axes.

        Args:
            tool_axis: Axis of the tool, in the tool frame.
            reference_axis: Reference axis, in the axes chosen by ``reference_axes``.
            min_angle: Lower limit of the angle (rad); ``None`` is unlimited.
            max_angle: Upper limit of the angle (rad); ``None`` is unlimited.
            anchor: See :attr:`anchor`.
            reference_axes: See :attr:`reference_axes`.
            tolerance: Rotation tolerance (rad); ``None`` keeps the default.
        """
        return ToolPoseBound(
            anchor=anchor,
            reference_axes=reference_axes,
            reference_rotation=_rotation_between(
                (0.0, 0.0, 1.0), _unit_vector(reference_axis, "reference_axis")
            ),
            tool_rotation=_rotation_between((0.0, 0.0, 1.0), _unit_vector(tool_axis, "tool_axis")),
            tilt=(min_angle, max_angle),
            **_tolerance_kwargs("rotation_tolerance", tolerance),
        )

    @staticmethod
    def orientation_box(
        max_roll: Optional[float] = None,
        max_pitch: Optional[float] = None,
        max_yaw: Optional[float] = None,
        anchor: Union[ToolPoseBoundAnchor, str] = ToolPoseBoundAnchor.GOAL,
        reference_axes: str = "anchor",
        tolerance: Optional[float] = None,
    ) -> ToolPoseBound:
        """Limit the tool orientation to roll, pitch and yaw ranges around the reference axes.

        The tool frame orientation relative to the reference frame is written as
        ``Rz(yaw) · Ry(pitch) · Rx(roll)``, and each angle is limited to ``±max``. With the
        default anchor this keeps the tool orientation close to the goal orientation.

        Args:
            max_roll: Largest rotation about the x axis (rad); ``None`` is unlimited.
            max_pitch: Largest rotation about the y axis (rad), at most 80 degrees. ``None`` is
                unlimited, or 80 degrees when ``max_roll`` or ``max_yaw`` is set.
            max_yaw: Largest rotation about the z axis (rad); ``None`` is unlimited.
            anchor: See :attr:`anchor`.
            reference_axes: See :attr:`reference_axes`.
            tolerance: Rotation tolerance (rad); ``None`` keeps the default.
        """
        largest_pitch = math.pi / 2.0 - SINGULARITY_MARGIN
        if max_pitch is not None and not 0.0 <= max_pitch <= largest_pitch + 1e-9:
            log_and_raise(
                f"max_pitch must be within [0, {largest_pitch:.4f}] rad, got {max_pitch}"
            )
        if max_pitch is None and (max_roll is not None or max_yaw is not None):
            max_pitch = largest_pitch
        # The tool x axis is the tool axis, perpendicular to the reference z axis at the reference
        # orientation, so yaw, tilt - 90 degrees and roll are yaw, pitch and roll.
        half = math.pi / 4.0
        return ToolPoseBound(
            anchor=anchor,
            reference_axes=reference_axes,
            tool_rotation=[math.cos(half), 0.0, math.sin(half), 0.0],
            tilt=None
            if max_pitch is None
            else (math.pi / 2.0 - max_pitch, math.pi / 2.0 + max_pitch),
            yaw=None if max_yaw is None else (-max_yaw, max_yaw),
            roll=None if max_roll is None else (-max_roll, max_roll),
            **_tolerance_kwargs("rotation_tolerance", tolerance),
        )


#: Bounds of one tool frame as accepted by ``update_tool_pose_bounds``: a bound, a list of bounds,
#: or ``None`` (no bounds).
ToolPoseBounds = Optional[Union["ToolPoseBound", Sequence["ToolPoseBound"]]]


def normalize_tool_pose_bounds(
    bounds: Optional[Union[ToolPoseBound, Sequence[ToolPoseBound]]],
) -> List[ToolPoseBound]:
    """Return ``bounds`` as a list: ``None`` is empty and a single bound becomes one entry."""
    if bounds is None:
        return []
    if isinstance(bounds, ToolPoseBound):
        return [bounds]
    out = list(bounds)
    for b in out:
        if not isinstance(b, ToolPoseBound):
            log_and_raise(f"expected ToolPoseBound entries, got {type(b).__name__}")
    return out


def _parse_limit(limit: Limit, name: str) -> Optional[Tuple[Optional[float], Optional[float]]]:
    if limit is None:
        return None
    if isinstance(limit, (int, float)):
        return (float(limit), float(limit))
    if len(limit) != 2:
        log_and_raise(f"{name} must be a number or a (lower, upper) pair, got {limit}")
    lo, hi = (None if v is None else float(v) for v in limit)
    if lo is None and hi is None:
        return None
    if lo is not None and hi is not None and lo > hi:
        log_and_raise(f"{name}: lower limit {lo} > upper limit {hi}")
    return (lo, hi)


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


def _shift(value: Optional[float], offset: float) -> Optional[float]:
    return None if value is None else float(value) + offset


def _tolerance_kwargs(name: str, tolerance: Optional[float]) -> Dict[str, float]:
    return {} if tolerance is None else {name: float(tolerance)}


def _unit_quaternion(q: Optional[Sequence[float]], name: str) -> List[float]:
    if q is None:
        return [1.0, 0.0, 0.0, 0.0]
    q = [float(v) for v in q]
    if len(q) != 4:
        log_and_raise(f"{name} must be a wxyz quaternion with 4 entries, got {q}")
    norm = math.sqrt(sum(v * v for v in q))
    if norm < 1e-6:
        log_and_raise(f"{name} must be non-zero")
    return [v / norm for v in q]


def _unit_vector(v: Sequence[float], name: str) -> List[float]:
    v = [float(a) for a in v]
    if len(v) != 3:
        log_and_raise(f"{name} must have 3 entries, got {v}")
    norm = math.sqrt(sum(a * a for a in v))
    if norm < 1e-9:
        log_and_raise(f"{name} must be non-zero")
    return [a / norm for a in v]


def _rotation_between(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Shortest-arc rotation (wxyz) taking unit vector ``a`` to unit vector ``b``."""
    dot = sum(x * y for x, y in zip(a, b))
    if dot < -1.0 + 1e-9:
        # Opposite vectors: rotate by pi about any axis perpendicular to a.
        axis = [0.0, -a[2], a[1]] if abs(a[0]) < 0.9 else [-a[1], a[0], 0.0]
        n = math.sqrt(sum(v * v for v in axis))
        return [0.0] + [v / n for v in axis]
    cross = [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]]
    q = [1.0 + dot] + cross
    n = math.sqrt(sum(v * v for v in q))
    return [v / n for v in q]


# -- PyTorch reference implementation -------------------------------------------------------


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
    center, lower, upper = (components.new_tensor(v) for v in bound.kernel_values(metric))
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
) -> torch.Tensor:
    """Squared-hinge cost ``0.5 * sum(violation²)`` matching the Warp kernel (unweighted)."""
    violation = tool_pose_bound_violation(
        bound, position, quaternion, anchor_position, anchor_quaternion, metric
    )
    return 0.5 * (violation * violation).sum(dim=-1)


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
