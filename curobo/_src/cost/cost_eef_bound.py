# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""End-effector corridor / bound cost.

Penalises a trajectory for leaving the axis-aligned bounding corridor
defined between the start pose and the goal pose, all expressed in the
goal's local coordinate frame.  Each axis can be independently enabled
or disabled via the ``margins`` tensor.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.types.math import Pose

if TYPE_CHECKING:
    from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg


class EEFBoundCost(BaseCost):
    """Axis-aligned EEF position + orientation bound cost.

    Call :meth:`update_margins` to activate the cost and set the per-axis
    corridor widths before running the optimizer.  When ``margins`` is
    ``None`` the cost tensor is all-zeros (disabled).

    Margin tensor layout (shape ``(4, 3)``)::

        margins[0]  – position start-side  (m),  negative → axis unconstrained
        margins[1]  – position goal-side   (m),  negative → axis unconstrained
        margins[2]  – rotation start-side  (rad), negative → axis unconstrained
        margins[3]  – rotation goal-side   (rad), negative → axis unconstrained
    """

    def __init__(self, config: EEFBoundCostCfg):
        super().__init__(config)
        self.config: EEFBoundCostCfg = config
        device = self.device_cfg.device
        dtype = self.device_cfg.dtype
        self._margins = torch.full((4, 3), -1.0, device=device, dtype=dtype)
        self._enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def margins_enabled(self) -> bool:
        """True when at least one axis is actively constrained."""
        return self._enabled

    def update_margins(self, margins: Optional[torch.Tensor] = None) -> None:
        """Set corridor margins and (de-)activate the cost.

        Args:
            margins: Float tensor of shape ``(4, 3)``.  Pass ``None`` to
                disable all constraints.
        """
        if margins is None:
            self._margins.fill_(-1.0)
            self._enabled = False
        else:
            self._margins.copy_(margins.to(self._margins.device, dtype=self._margins.dtype))
            self._enabled = True

    # ------------------------------------------------------------------
    # BaseCost interface
    # ------------------------------------------------------------------

    def forward(
        self,
        ee_pos_batch: torch.Tensor,
        ee_rot_batch: torch.Tensor,
        goal_pos: torch.Tensor,
        goal_rot: torch.Tensor,
    ) -> torch.Tensor:
        """Compute the EEF bound cost.

        Args:
            ee_pos_batch: ``(B, T, 3)`` EEF positions.
            ee_rot_batch: ``(B, T, 4)`` EEF quaternions ``(w, x, y, z)``.
            goal_pos:     ``(B, 3)`` goal position.
            goal_rot:     ``(B, 4)`` goal quaternion ``(w, x, y, z)``.

        Returns:
            Cost tensor of shape ``(B, T)``.
        """
        if not self._enabled:
            return torch.zeros(
                *ee_pos_batch.shape[:-1],
                device=ee_pos_batch.device,
                dtype=ee_pos_batch.dtype,
            )

        margins = self._margins
        if self.config.disable_margin:
            margins = torch.zeros_like(margins)
        else:
            pos_off = self.config.pos_margin_offset
            rot_off = self.config.rot_margin_offset
            offset = torch.tensor(
                [pos_off, pos_off, rot_off, rot_off],
                dtype=margins.dtype,
                device=margins.device,
            )
            # Widen positive margins; keep negative (disabled) flags intact.
            margins = torch.nn.functional.relu(margins + offset[:, None]) + margins.clip(
                -torch.inf, 0
            )

        pos_cost = _eef_position_bound_cost(
            ee_pos_batch,
            ee_rot_batch,
            goal_pos,
            goal_rot,
            margins[:2],
        )
        rot_cost = _eef_rotation_bound_cost(
            ee_pos_batch,
            ee_rot_batch,
            goal_pos,
            goal_rot,
            margins[2:],
        )
        return self._weight[0] * (pos_cost + rot_cost)


# ---------------------------------------------------------------------------
# Functional helpers (module-private)
# ---------------------------------------------------------------------------


def _eef_position_bound_cost(
    ee_pos_batch: torch.Tensor,
    ee_rot_batch: torch.Tensor,
    ee_pos_goal: torch.Tensor,
    ee_rot_goal: torch.Tensor,
    margins: torch.Tensor,
) -> torch.Tensor:
    """Axis-aligned position corridor cost in the goal frame.

    Args:
        margins: ``(2, 3)`` – ``[start-side, goal-side]`` per axis.
            Negative values disable that side on the corresponding axis.

    Returns:
        ``(B, T)`` cost tensor.
    """
    dtype = ee_pos_batch.dtype

    constrain_begin = (margins[0] >= 0).to(dtype)
    constrain_end = (margins[1] >= 0).to(dtype)
    margin_begin = margins[0].clamp(min=0)
    margin_end = margins[1].clamp(min=0)

    start_pose = Pose(position=ee_pos_batch[:, 0], quaternion=ee_rot_batch[:, 0])
    current_poses = Pose(position=ee_pos_batch, quaternion=ee_rot_batch)
    goal_pose = Pose(position=ee_pos_goal, quaternion=ee_rot_goal)

    local_current = goal_pose.compute_local_pose(current_poses)
    local_start = goal_pose.compute_local_pose(start_pose)

    # (B, 1, 3)
    local_start_pos = local_start.position.unsqueeze(1)
    local_end_pos = torch.zeros_like(local_start_pos)

    is_end_larger = (local_end_pos > local_start_pos).to(dtype)

    violation = sum(
        [
            is_end_larger
            * constrain_begin
            * torch.relu((local_start_pos - margin_begin) - local_current.position),
            is_end_larger
            * constrain_end
            * torch.relu(local_current.position - (local_end_pos + margin_end)),
            (1 - is_end_larger)
            * constrain_begin
            * torch.relu(local_current.position - (local_start_pos + margin_begin)),
            (1 - is_end_larger)
            * constrain_end
            * torch.relu((local_end_pos - margin_end) - local_current.position),
        ]
    )
    return torch.sum(violation, dim=-1)


def _quat_to_rotvec(q: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """Quaternion ``(w, x, y, z)`` → rotation vector ``(..., 3)``."""
    q = q / q.norm(dim=-1, keepdim=True).clamp_min(eps)
    qw = q[..., 0:1]
    q_vec = q[..., 1:4]

    # Shortest-arc: ensure w >= 0
    flip = (qw < 0).to(q.dtype)
    q_vec = torch.where(flip > 0, -q_vec, q_vec)
    qw = torch.where(flip > 0, -qw, qw)

    sin_half = q_vec.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, qw.clamp_min(eps))

    scale = angle / sin_half.clamp_min(eps)
    scale.masked_fill_(sin_half < 1e-6, 2.0)
    return q_vec * scale


def _eef_rotation_bound_cost(
    ee_pos_batch: torch.Tensor,
    ee_rot_batch: torch.Tensor,
    ee_pos_goal: torch.Tensor,
    ee_rot_goal: torch.Tensor,
    margins: torch.Tensor,
) -> torch.Tensor:
    """Per-axis rotation-vector corridor cost in the goal frame.

    Args:
        margins: ``(2, 3)`` – ``[start-side, goal-side]`` in radians.
            Negative values disable that side on the corresponding axis.

    Returns:
        ``(B, T)`` cost tensor.
    """
    dtype = ee_rot_batch.dtype

    constrain_begin = (margins[0] >= 0).to(dtype)
    constrain_end = (margins[1] >= 0).to(dtype)
    margin_begin = margins[0].clamp(min=0)
    margin_end = margins[1].clamp(min=0)

    start_pose = Pose(position=ee_pos_batch[:, 0], quaternion=ee_rot_batch[:, 0])
    current_poses = Pose(position=ee_pos_batch, quaternion=ee_rot_batch)
    goal_pose = Pose(position=ee_pos_goal, quaternion=ee_rot_goal)

    local_current = goal_pose.compute_local_pose(current_poses)
    local_start = goal_pose.compute_local_pose(start_pose)

    # Map to rotation vectors
    local_current_rv = _quat_to_rotvec(local_current.quaternion)       # (B, T, 3)
    local_start_rv = _quat_to_rotvec(local_start.quaternion).unsqueeze(1)  # (B, 1, 3)
    local_end_rv = torch.zeros_like(local_start_rv)                     # (B, 1, 3)

    is_end_larger = (local_end_rv > local_start_rv).to(dtype)

    violation = sum(
        [
            is_end_larger
            * constrain_begin
            * torch.relu((local_start_rv - margin_begin) - local_current_rv),
            is_end_larger
            * constrain_end
            * torch.relu(local_current_rv - (local_end_rv + margin_end)),
            (1 - is_end_larger)
            * constrain_begin
            * torch.relu(local_current_rv - (local_start_rv + margin_begin)),
            (1 - is_end_larger)
            * constrain_end
            * torch.relu((local_end_rv - margin_end) - local_current_rv),
        ]
    )
    return torch.sum(violation, dim=-1)