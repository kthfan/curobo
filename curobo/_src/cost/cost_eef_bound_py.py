# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Legacy pure-PyTorch implementation of the EEF bound / corridor cost.

This implementation is the reference for the Warp implementation in
``cost_eef_bound``.  It is intentionally simple (no Warp, no manual gradient
kernels) and serves as the "ground truth" for unit tests that compare the
forward cost and backward gradient against the optimised Warp version.

Public surface:

* :class:`EEFBoundCostPyTorch` – PyTorch equivalent of ``EEFBoundCost``.
* :func:`quat_to_rotvec`       – wxyz quaternion → axis-angle rotation vector.
* :func:`corridor_cost_3d`     – scalar axis-aligned 3D corridor cost.
* :func:`quat_conj_wxyz`, :func:`quat_mul_wxyz` – quaternion utilities.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.cost.cost_base import BaseCost

if TYPE_CHECKING:
    from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg


# ---------------------------------------------------------------------------
# Quaternion utilities (wxyz convention)
# ---------------------------------------------------------------------------
def quat_conj_wxyz(q: torch.Tensor) -> torch.Tensor:
    """Conjugate of a (..., 4) wxyz quaternion."""
    return torch.cat([q[..., :1], -q[..., 1:]], dim=-1)


def quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two (..., 4) wxyz quaternions."""
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


def _rotate_vector_by_quat_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vector ``v`` by quaternion ``q`` (wxyz):  ``v' = q (0,v) q^{-1}``.

    Uses the cross-product formulation which is cheaper than the full
    quaternion multiply when only the vector part is needed.
    """
    qw = q[..., 0:1]
    qxyz = q[..., 1:]
    t = 2.0 * torch.cross(qxyz, v, dim=-1)
    return v + qw * t + torch.cross(qxyz, t, dim=-1)


# ---------------------------------------------------------------------------
# Rotation-vector conversion (PyTorch autograd-friendly)
# ---------------------------------------------------------------------------
def quat_to_rotvec(q: torch.Tensor) -> torch.Tensor:
    """Convert ``(..., 4)`` wxyz quaternion to rotation vector ``(..., 3)``.

    The rotation vector is ``axis * angle`` where ``axis`` is the (unit)
    rotation axis and ``angle`` is the rotation magnitude.  The
    implementation mirrors :func:`wp_torch_eef_bound.quat_to_rotvec_warp` so
    that Warp and PyTorch agree on the conventions exactly.
    """
    eps = 1e-8
    norm = torch.sqrt((q * q).sum(dim=-1, keepdim=True)).clamp(min=eps)
    qn = q / norm
    # Canonical hemisphere.
    qn = torch.where(qn[..., :1] < 0, -qn, qn)
    w = qn[..., 0]
    xyz = qn[..., 1:]
    sin_half = torch.sqrt((xyz * xyz).sum(dim=-1, keepdim=True))
    angle = 2.0 * torch.atan2(sin_half, w)
    # scale = angle / sin_half for sin_half > eps, else 2.
    scale = torch.where(
        sin_half < 1e-6,
        torch.full_like(sin_half, 2.0),
        angle / (sin_half + eps),
    )
    return xyz * scale


# ---------------------------------------------------------------------------
# 3D corridor cost (PyTorch)
# ---------------------------------------------------------------------------
def corridor_cost_3d(
    current: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    margins: torch.Tensor,
) -> torch.Tensor:
    """3D axis-aligned corridor cost.

    Args:
        current: ``(N, H, 3)`` current position or rotation-vector.
        start:   ``(N, 1, 3)`` start point (in same frame as ``end``).
        end:     ``(N, 1, 3)`` end point (in same frame as ``start``).
        margins: ``(2, 3)`` margin tensor, rows ``[margin_begin, margin_end]``
            and columns ``[x, y, z]``.  Negative value disables the axis.

    Returns:
        ``(N, H)`` cost – sum of per-axis violations.
    """
    # is_end_larger[i, 0, k] = 1.0 iff end > start on axis k.
    is_end_larger = (end > start).to(current.dtype)
    one_minus = 1.0 - is_end_larger

    # margins: (2, 3) -> broadcastable (1, 1, 3) for begin/end rows.
    mb = margins[0:1, :].unsqueeze(0)  # margin on the "begin" (start) side
    me = margins[1:2, :].unsqueeze(0)  # margin on the "end"   side
    cb = (mb >= 0).to(current.dtype)
    ce = (me >= 0).to(current.dtype)
    mb_safe = torch.relu(mb)
    me_safe = torch.relu(me)

    start_e = start.expand_as(current)
    end_e = end.expand_as(current)

    begin_viol = (
        is_end_larger * torch.relu((start_e - mb_safe) - current)
        + one_minus * torch.relu(current - (start_e + mb_safe))
    )
    end_viol = (
        is_end_larger * torch.relu(current - (end_e + me_safe))
        + one_minus * torch.relu((end_e - me_safe) - current)
    )
    return (cb * begin_viol + ce * end_viol).sum(dim=-1)


# ---------------------------------------------------------------------------
# Legacy PyTorch cost class
# ---------------------------------------------------------------------------
class EEFBoundCostPyTorch(BaseCost):
    """PyTorch reference implementation of :class:`cost_eef_bound.EEFBoundCost`.

    Mirrors the public API of the Warp version (``update_margins``,
    ``setup_batch_tensors``, ``forward``) so the same tests can target both.
    """

    def __init__(self, config: "EEFBoundCostCfg"):
        super().__init__(config)
        self.config: "EEFBoundCostCfg" = config
        self.tool_frames = config.tool_frames
        self.num_links = len(self.tool_frames)

        device = self.device_cfg.device
        dtype = self.device_cfg.dtype

        # Per-link margins, shape (L, 4, 3).
        self._margins = torch.full((self.num_links, 4, 3), -1.0, device=device, dtype=dtype)
        self._enabled = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    @property
    def margins_enabled(self) -> bool:
        return self._enabled

    def update_margins(self, margins: Optional[torch.Tensor] = None) -> None:
        """Set per-link margins.  Accepts ``(4, 3)`` (broadcast to all links)
        or ``(num_links, 4, 3)``.
        """
        if margins is None:
            self._margins.fill_(-1.0)
            self._enabled = False
            return

        margins = margins.to(self._margins.device, dtype=self._margins.dtype)
        if margins.dim() == 2 and margins.shape == (4, 3):
            margins = margins.unsqueeze(0).expand(self.num_links, -1, -1).contiguous()
        elif tuple(margins.shape) != (self.num_links, 4, 3):
            raise ValueError(
                f"margins must have shape (4, 3) or ({self.num_links}, 4, 3); "
                f"got {tuple(margins.shape)}"
            )

        self._margins.copy_(margins)
        self._enabled = True

    # ------------------------------------------------------------------
    # BaseCost interface
    # ------------------------------------------------------------------
    def setup_batch_tensors(self, batch_size: int, horizon: int, **kwargs):
        if batch_size != self._batch_size or horizon != self._horizon:
            num_links = self.num_links
            self._out_distance = torch.zeros(
                (batch_size, horizon, 2 * num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_position_distance = torch.zeros(
                (batch_size, horizon, num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_rotation_distance = torch.zeros(
                (batch_size, horizon, num_links),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_goalset_idx = torch.zeros(
                (batch_size, horizon, num_links),
                dtype=torch.int32,
                device=self.device_cfg.device,
            )
            self._out_position_gradient = torch.zeros(
                (batch_size, horizon, num_links, 3),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
            self._out_rotation_gradient = torch.zeros(
                (batch_size, horizon, num_links, 4),
                dtype=torch.float32,
                device=self.device_cfg.device,
            )
        super().setup_batch_tensors(batch_size, horizon)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(
        self,
        current_tool_poses: ToolPose,
        goal_tool_poses: GoalToolPose,
        **kwargs,
    ) -> torch.Tensor:
        B, H, L, _ = current_tool_poses.position.shape
        G = goal_tool_poses.num_goalset

        if not self._enabled or not self._cost_enabled:
            return torch.zeros(
                (B, H, 1),
                device=current_tool_poses.position.device,
                dtype=current_tool_poses.position.dtype,
            )

        # Static goals only (H_goal == 1) — pick t=0.
        goal_pos0 = goal_tool_poses.position[:, 0]   # (B, L, G, 3)
        goal_quat0 = goal_tool_poses.quaternion[:, 0]  # (B, L, G, 4)

        # Apply margin offsets (only on enabled axes).
        margins = self._margins
        if self.config.disable_margin:
            margins = torch.zeros_like(margins)
        else:
            offset = torch.tensor(
                [self.config.pos_margin_offset, self.config.pos_margin_offset,
                 self.config.rot_margin_offset, self.config.rot_margin_offset],
                dtype=margins.dtype, device=margins.device,
            )  # (4,)
            margins = torch.where(
                margins >= 0,
                torch.relu(margins + offset.view(1, 4, 1)),
                margins,
            )

        # Expand everything to (B, H, L, G, ...) for vectorised cost evaluation.
        cur_pos_E = current_tool_poses.position.unsqueeze(3).expand(B, H, L, G, 3)
        cur_quat_E = current_tool_poses.quaternion.unsqueeze(3).expand(B, H, L, G, 4)
        goal_pos_E = goal_pos0.unsqueeze(1).expand(B, H, L, G, 3)
        goal_quat_E = goal_quat0.unsqueeze(1).expand(B, H, L, G, 4)

        # Transform to goal frame.
        g_inv = quat_conj_wxyz(goal_quat_E)  # (B, H, L, G, 4)
        local_cur_pos = _rotate_vector_by_quat_wxyz(
            g_inv, cur_pos_E - goal_pos_E
        )
        local_cur_quat = quat_mul_wxyz(g_inv, cur_quat_E)
        local_cur_rv = quat_to_rotvec(local_cur_quat)

        # Start pose = current pose at t=0 (broadcast over H).
        start_pos0 = current_tool_poses.position[:, 0]   # (B, L, 3)
        start_quat0 = current_tool_poses.quaternion[:, 0]  # (B, L, 4)
        start_pos_E = start_pos0.unsqueeze(1).unsqueeze(3).expand(B, H, L, G, 3)
        start_quat_E = start_quat0.unsqueeze(1).unsqueeze(3).expand(B, H, L, G, 4)
        local_start_pos = _rotate_vector_by_quat_wxyz(
            g_inv, start_pos_E - goal_pos_E
        )
        local_start_quat = quat_mul_wxyz(g_inv, start_quat_E)
        local_start_rv = quat_to_rotvec(local_start_quat)

        # Goal in goal frame = (0, 0, 0) and identity.
        end_pos = torch.zeros_like(local_cur_pos)
        end_rv = torch.zeros_like(local_cur_rv)

        # Per-(b, h, l, g) cost: position corridor + rotation corridor.
        cost_per_axis_pos = _corridor_per_axis(
            local_cur_pos, local_start_pos, end_pos,
            margins[:, 0:2, :],  # (L, 2, 3)
        )  # (B, H, L, G)
        cost_per_axis_rot = _corridor_per_axis(
            local_cur_rv, local_start_rv, end_rv,
            margins[:, 2:4, :],
        )
        cost_BHLG = cost_per_axis_pos + cost_per_axis_rot

        # Best goal per (B, H, L) → min.
        cost_BHL = cost_BHLG.min(dim=-1).values  # (B, H, L)

        # Per-link weight.
        w = self._weight
        if w.numel() == L:
            cost_BHL_weighted = cost_BHL * w  # broadcast over (B, H)
        elif w.numel() == 1:
            cost_BHL_weighted = cost_BHL  # scalar weight applied after sum
        else:
            raise ValueError(f"weight must be scalar or shape ({L},); got {tuple(w.shape)}")

        cost_BH = cost_BHL_weighted.sum(dim=-1)  # (B, H)

        if w.numel() == 1:
            cost = (cost_BH * w).unsqueeze(-1)
        else:
            cost = cost_BH.unsqueeze(-1)
        return cost


def _corridor_per_axis(
    current: torch.Tensor,
    start: torch.Tensor,
    end: torch.Tensor,
    margins: torch.Tensor,
) -> torch.Tensor:
    """Vectorised 3D corridor cost that respects per-link margins.

    Args:
        current:  ``(B, H, L, G, 3)``.
        start:    ``(B, H, L, G, 3)``.
        end:      ``(B, H, L, G, 3)``.
        margins:  ``(L, 2, 3)`` — ``[margin_begin, margin_end]`` per link.

    Returns:
        ``(B, H, L, G)`` per-axis cost summed over the 3 axes.
    """
    L = margins.shape[0]

    # Reshape margins to broadcast against (B, H, L, G, ...): (1, 1, L, 1, 3).
    mb = margins[:, 0, :].view(1, 1, L, 1, 3)
    me = margins[:, 1, :].view(1, 1, L, 1, 3)
    cb = (mb >= 0).to(current.dtype)
    ce = (me >= 0).to(current.dtype)
    mb_safe = torch.relu(mb)
    me_safe = torch.relu(me)

    is_end_larger = (end > start).to(current.dtype)
    one_minus = 1.0 - is_end_larger

    begin_viol = (
        is_end_larger * torch.relu((start - mb_safe) - current)
        + one_minus * torch.relu(current - (start + mb_safe))
    )
    end_viol = (
        is_end_larger * torch.relu(current - (end + me_safe))
        + one_minus * torch.relu((end - me_safe) - current)
    )
    per_axis_cost = cb * begin_viol + ce * end_viol
    return per_axis_cost.sum(dim=-1)
