# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-effector corridor / bound cost (Warp-based).

This is the default cuRoboV2 implementation.  It uses the Warp kernel in
``wp_torch_eef_bound`` for the forward pass; the Warp kernel itself computes
the per-(B, H, L) cost **and** the per-link position / rotation gradient in
the same launch.  A ``torch.autograd.Function`` caches those gradients and
reuses them in ``backward`` so no ``wp.Tape`` or PyTorch re-evaluation is
required.

A legacy pure-PyTorch version is available in ``cost_eef_bound_py`` for
reference and as the gradient ground-truth in unit tests.

Per-link margins
----------------
``update_margins`` accepts a ``(4, 3)`` tensor (broadcast to all tool frames)
or a ``(num_links, 4, 3)`` tensor so each link can have its own corridor
widths.  Margins are ``(l, s, ax)`` shaped:

    margins[l, 0, :]  – position start-side  (m),  negative → axis disabled
    margins[l, 1, :]  – position goal-side   (m),  negative → axis disabled
    margins[l, 2, :]  – rotation start-side  (rad), negative → axis disabled
    margins[l, 3, :]  – rotation goal-side   (rad), negative → axis disabled

Per-link / scalar weight
------------------------
``BaseCostCfg.weight`` may be:

* a scalar tensor of shape ``()`` or ``(1,)`` — applies the same scalar to
  every per-link cost before the sum over links;
* a tensor of shape ``(num_links,)`` — applies per-link scaling so the
  contribution of each tool frame can be tuned independently.

The weight is applied inside :class:`EEFBoundCostFunction` so the cached
position / rotation gradients can be re-used in ``backward`` without a
second pass through the cost computation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import torch

from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.cost.cost_base import BaseCost
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import init_warp

from curobo._src.cost.wp_torch_eef_bound import EEFBoundCostFunction

if TYPE_CHECKING:
    from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg


class EEFBoundCost(BaseCost):
    """Axis-aligned EEF position + orientation bound cost (per-link margins)."""

    def __init__(self, config: "EEFBoundCostCfg"):
        super().__init__(config)
        self.config: "EEFBoundCostCfg" = config

        if not config.tool_frames:
            log_and_raise(
                "EEFBoundCostCfg.tool_frames must be a non-empty list of link names; "
                "call cfg.set_tool_frames([...]) before constructing the cost"
            )
        self.tool_frames = list(config.tool_frames)
        self.num_links = len(self.tool_frames)

        init_warp()

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
        """True when at least one axis on at least one link is constrained."""
        return self._enabled

    def update_margins(self, margins: Optional[torch.Tensor] = None) -> None:
        """Set per-link corridor margins.

        Args:
            margins: Float tensor of shape ``(num_links, 4, 3)`` for per-link
                margins or ``(4, 3)`` to broadcast the same margins to every
                link.  Pass ``None`` to disable all constraints.

        Raises:
            ValueError: If ``margins`` shape is incompatible with ``num_links``.
        """
        if margins is None:
            self._margins.fill_(-1.0)
            self._enabled = False
            return

        margins_t = margins.to(self._margins.device, dtype=self._margins.dtype)
        if margins_t.dim() == 2 and tuple(margins_t.shape) == (4, 3):
            margins_t = margins_t.unsqueeze(0).expand(self.num_links, -1, -1).contiguous()
        elif tuple(margins_t.shape) != (self.num_links, 4, 3):
            raise ValueError(
                f"margins must have shape (4, 3) or ({self.num_links}, 4, 3); "
                f"got {tuple(margins_t.shape)}"
            )

        self._margins.copy_(margins_t)
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

    def forward(
        self,
        current_tool_poses: ToolPose,
        goal_tool_poses: GoalToolPose,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the EEF bound cost.

        Args:
            current_tool_poses: Current EEF poses, shape ``(B, H, L, 3/4)``.
            goal_tool_poses: Goal EEF poses, shape ``(B, H_goal, L, G, 3/4)``.

        Returns:
            Cost tensor of shape ``(B, H, 1)``.
        """
        B, H, L, _ = current_tool_poses.position.shape

        if not self._enabled or not self._cost_enabled:
            return torch.zeros(
                (B, H, 1),
                device=current_tool_poses.position.device,
                dtype=current_tool_poses.position.dtype,
            )

        if L != self.num_links:
            raise ValueError(
                f"current_tool_poses has L={L} links but cfg declares "
                f"{self.num_links} tool_frames: {self.tool_frames}"
            )

        self.setup_batch_tensors(B, H)

        # ---------- pick a single goal pose across horizon (static goals) ----
        goal_pos = goal_tool_poses.position
        goal_quat = goal_tool_poses.quaternion
        if goal_pos.shape[1] == 1:
            goal_pos0 = goal_pos[:, 0]   # (B, L, G, 3)
            goal_quat0 = goal_quat[:, 0]  # (B, L, G, 4)
        elif goal_pos.shape[1] > 1:
            raise ValueError(
                f"goal_tool_poses.horizon > 1 not supported, got {goal_pos.shape[1]}"
            )
        else:
            raise ValueError(
                f"goal_tool_poses.horizon must be >= 1, got {goal_pos.shape[1]}"
            )

        # ---------- start pose = current pose at t=0 ----------------------
        start_pos = current_tool_poses.position[:, 0:1]    # (B, 1, L, 3)
        start_quat = current_tool_poses.quaternion[:, 0:1]  # (B, 1, L, 4)

        # ---------- per-axis margin offsets (applied to enabled axes) -------
        margins = self._margins
        if self.config.disable_margin:
            margins = torch.zeros_like(margins)
        else:
            offset = torch.tensor(
                [
                    self.config.pos_margin_offset,
                    self.config.pos_margin_offset,
                    self.config.rot_margin_offset,
                    self.config.rot_margin_offset,
                ],
                dtype=margins.dtype,
                device=margins.device,
            )  # shape (4,)
            margins = torch.where(
                margins >= 0,
                torch.relu(margins + offset.view(1, 4, 1)),
                margins,
            )

        # ---------- validate weight shape ----------------------------------
        w = self._weight
        if w.numel() not in (1, L):
            raise ValueError(
                f"weight must be a scalar tensor or have shape ({L},); "
                f"got {tuple(w.shape)}"
            )

        # ---------- Warp kernel + autograd Function -------------------------
        # The autograd function applies the weight (scalar or per-link) to
        # the per-link cost internally, so backward can re-use the cached
        # gradients from the kernel without any extra pass.
        cost_BHL_w = EEFBoundCostFunction.apply(
            current_tool_poses.position,
            current_tool_poses.quaternion,
            start_pos,
            start_quat,
            goal_pos0,
            goal_quat0,
            margins,
            w,
            self._out_distance,
            self._out_position_distance,
            self._out_rotation_distance,
            self._out_goalset_idx,
            self._out_position_gradient,
            self._out_rotation_gradient,
        )  # shape (B, H, L) — per-link weighted cost

        # ---------- sum across links ---------------------------------------
        cost_BH = cost_BHL_w.sum(dim=-1)  # (B, H)
        cost = cost_BH.unsqueeze(-1)       # (B, H, 1)
        return cost
