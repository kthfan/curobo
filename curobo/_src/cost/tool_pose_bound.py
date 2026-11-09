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

Inside an interval the cost is zero, so paths are free to run along an edge. ``center_weight``
adds a pull toward the middle of every interval with two ends, on every trajectory step except
the last, so the end of the path can still reach a goal that is not at the middle.

Limits are evaluated by :mod:`curobo._src.cost.cost_eef_bound_eval`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Dict, List, Optional, Sequence, Tuple, Union

from curobo._src.util.logging import log_and_raise

#: Smallest distance (rad) of a ``tilt`` interval from 0 and π when ``yaw`` or ``roll`` is limited.
SINGULARITY_MARGIN = math.radians(10.0)

#: Component names in kernel order.
COMPONENT_NAMES = ("x", "y", "z", "tilt", "yaw", "roll")


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

    #: Weight, relative to the cost weight, of a pull toward the middle of each limit with two
    #: ends that is wider than the tolerance: ``0.5 * center_weight * (value - middle)²`` per
    #: component (m or rad). It acts on every trajectory step except the last and never affects
    #: feasibility. One-sided limits have no middle and are not pulled. 0 disables it.
    center_weight: float = 0.0

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
        for name in ("position_tolerance", "rotation_tolerance", "center_weight"):
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

    def anchored_at_base(
        self, anchor_position: Sequence[float], anchor_quaternion: Sequence[float]
    ) -> ToolPoseBound:
        """Return the same limits with the anchor replaced by a fixed pose in the base frame.

        A goal- or start-anchored bound whose anchor pose is known becomes a base-anchored
        bound that limits the tool frame exactly the same way. Base-anchored bounds are
        returned unchanged.

        Args:
            anchor_position: Anchor position in the base frame (m).
            anchor_quaternion: Anchor orientation (wxyz) in the base frame.

        Returns:
            Base-anchored bound with base reference axes.
        """
        if self.anchor == ToolPoseBoundAnchor.BASE:
            return self
        anchor_quaternion = _unit_quaternion(anchor_quaternion, "anchor_quaternion")
        reference = self.reference_rotation
        if self.reference_axes == "anchor":
            reference = _quat_multiply(anchor_quaternion, reference)
        # (x, y, z) = R_refᵀ (p - t_anchor) = R_refᵀ p - R_refᵀ t_anchor.
        offset = _rotate_vector(_conjugate(reference), [float(v) for v in anchor_position])
        shifted = {}
        for axis, name in enumerate(("x", "y", "z")):
            limit = getattr(self, name)
            if limit is not None:
                shifted[name] = (_shift(limit[0], offset[axis]), _shift(limit[1], offset[axis]))
        return replace(
            self,
            anchor=ToolPoseBoundAnchor.BASE,
            reference_axes="base",
            reference_rotation=reference,
            **shifted,
        )

    @property
    def uses_rotation(self) -> bool:
        """True when ``tilt``, ``yaw`` or ``roll`` is limited."""
        return any(getattr(self, name) is not None for name in ("tilt", "yaw", "roll"))

    @staticmethod
    def plane(
        normal: Sequence[float] = (0.0, 0.0, 1.0),
        point: Optional[Sequence[float]] = None,
        min_distance: Optional[float] = 0.0,
        max_distance: Optional[float] = 0.0,
        anchor: Union[ToolPoseBoundAnchor, str] = ToolPoseBoundAnchor.GOAL,
        reference_axes: str = "anchor",
        tolerance: Optional[float] = None,
        center_weight: float = 0.0,
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
            center_weight: See :attr:`center_weight`.
        """
        n = _unit_vector(normal, "normal")
        offset = sum(a * b for a, b in zip(n, point)) if point is not None else 0.0
        return ToolPoseBound(
            anchor=anchor,
            reference_axes=reference_axes,
            reference_rotation=_rotation_between((0.0, 0.0, 1.0), n),
            z=(_shift(min_distance, offset), _shift(max_distance, offset)),
            center_weight=center_weight,
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
        center_weight: float = 0.0,
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
            center_weight: See :attr:`center_weight`.
        """
        d = _unit_vector(direction, "direction")
        rotation = _rotation_between((0.0, 0.0, 1.0), d)
        p = [0.0, 0.0, 0.0] if point is None else [float(v) for v in point]
        if len(p) != 3:
            log_and_raise(f"point must have 3 entries, got {point}")
        local = _rotate_vector(_conjugate(rotation), p)
        return ToolPoseBound(
            anchor=anchor,
            reference_axes=reference_axes,
            reference_rotation=rotation,
            x=local[0],
            y=local[1],
            z=(_shift(min_along, local[2]), _shift(max_along, local[2])),
            center_weight=center_weight,
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
        center_weight: float = 0.0,
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
            center_weight: See :attr:`center_weight`.
        """
        return ToolPoseBound(
            anchor=anchor,
            reference_axes=reference_axes,
            reference_rotation=_rotation_between(
                (0.0, 0.0, 1.0), _unit_vector(reference_axis, "reference_axis")
            ),
            tool_rotation=_rotation_between((0.0, 0.0, 1.0), _unit_vector(tool_axis, "tool_axis")),
            tilt=(min_angle, max_angle),
            center_weight=center_weight,
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
        center_weight: float = 0.0,
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
            center_weight: See :attr:`center_weight`.
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
            center_weight=center_weight,
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


def _conjugate(q: Sequence[float]) -> List[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def _quat_multiply(a: Sequence[float], b: Sequence[float]) -> List[float]:
    """Hamilton product ``a ⊗ b`` of wxyz quaternions."""
    return [
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    ]


def _rotate_vector(q: Sequence[float], v: Sequence[float]) -> List[float]:
    """Rotate ``v`` by the unit quaternion ``q`` (wxyz)."""
    w, u = q[0], q[1:]
    t = [
        2.0 * (u[1] * v[2] - u[2] * v[1]),
        2.0 * (u[2] * v[0] - u[0] * v[2]),
        2.0 * (u[0] * v[1] - u[1] * v[0]),
    ]
    return [
        v[0] + w * t[0] + u[1] * t[2] - u[2] * t[1],
        v[1] + w * t[1] + u[2] * t[0] - u[0] * t[2],
        v[2] + w * t[2] + u[0] * t[1] - u[1] * t[0],
    ]


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
