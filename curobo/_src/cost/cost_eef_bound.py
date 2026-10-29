# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tool-frame bound cost: keeps tool frames inside their :class:`ToolPoseBound` limits.

Bounds are set per tool frame through
:attr:`ToolPoseCriteria.bounds <curobo._src.cost.tool_pose_criteria.ToolPoseCriteria.bounds>`
or ``update_tool_pose_bounds``, and stored in fixed-size tensors with
:attr:`EEFBoundCostCfg.max_bounds_per_tool_frame` slots per tool frame, so updating them never
changes the launched kernel (CUDA graph safe). Tool frames without bounds contribute zero cost.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Dict, List, Optional, Sequence, Union

import torch

from curobo._src.cost.cost_base import BaseCost
from curobo._src.cost.tool_pose_bound import (
    UNBOUNDED,
    ToolPoseBound,
    ToolPoseBoundAnchor,
    normalize_tool_pose_bounds,
)
from curobo._src.cost.wp_eef_bound import EEFBoundCostFunction
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import init_warp

if TYPE_CHECKING:
    from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg
    from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria


class EEFBoundCost(BaseCost):
    """Squared-hinge cost on tool-frame components expressed in reference frames."""

    def __init__(self, config: EEFBoundCostCfg):
        """Allocate bound slot tensors; all tool frames start unbounded."""
        super().__init__(config)
        if not config.tool_frames:
            log_and_raise("EEFBoundCostCfg.tool_frames must be set before creating the cost")
        if config.max_bounds_per_tool_frame < 1:
            log_and_raise(
                "EEFBoundCostCfg.max_bounds_per_tool_frame must be >= 1, "
                + f"got {config.max_bounds_per_tool_frame}"
            )
        self.tool_frames = list(config.tool_frames)
        num_links = len(self.tool_frames)
        if self._weight.numel() not in (1, num_links):
            log_and_raise(
                f"EEFBoundCost weight must have 1 or {num_links} values, "
                + f"got {self._weight.numel()}"
            )
        init_warp()

        device = self.device_cfg.device
        dtype = self.device_cfg.dtype
        slots = (num_links, config.max_bounds_per_tool_frame)
        self._bounds: Dict[str, List[ToolPoseBound]] = {name: [] for name in self.tool_frames}
        self._anchor = torch.full(slots, -1, device=device, dtype=torch.int32)
        self._axes = torch.zeros(slots, device=device, dtype=torch.int32)
        self._reference_rotation = torch.zeros((*slots, 4), device=device, dtype=dtype)
        self._reference_rotation[..., 0] = 1.0
        self._tool_rotation = self._reference_rotation.clone()
        self._center = torch.zeros((*slots, 6), device=device, dtype=dtype)
        self._lower = torch.full((*slots, 6), -UNBOUNDED, device=device, dtype=dtype)
        self._upper = torch.full((*slots, 6), UNBOUNDED, device=device, dtype=dtype)
        self._use_rotation = torch.zeros(slots, device=device, dtype=torch.int32)
        # Placeholders for anchors that are not in use.
        self._unused_position = torch.zeros((1, 3), device=device, dtype=dtype)
        self._unused_quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=dtype)

    @property
    def tool_pose_bounds(self) -> Dict[str, List[ToolPoseBound]]:
        """Current bounds per tool frame (empty when unbounded)."""
        return {name: list(bounds) for name, bounds in self._bounds.items()}

    def uses_anchor(self, anchor: ToolPoseBoundAnchor) -> bool:
        """True if any bound is attached to ``anchor``."""
        return any(b.anchor == anchor for bounds in self._bounds.values() for b in bounds)

    def update_tool_pose_bounds(
        self,
        bounds: Dict[str, Optional[Union[ToolPoseBound, Sequence[ToolPoseBound]]]],
    ) -> None:
        """Replace the bounds of the given tool frames.

        Args:
            bounds: Map from tool frame name to a bound, a list of bounds, or ``None`` to clear.
        """
        num_slots = self._anchor.shape[1]
        for name, value in bounds.items():
            if name not in self._bounds:
                log_and_raise(f"EEFBoundCost: unknown tool frame {name}, have {self.tool_frames}")
            frame_bounds = normalize_tool_pose_bounds(value)
            if len(frame_bounds) > num_slots:
                log_and_raise(
                    f"EEFBoundCost: tool frame {name} has {len(frame_bounds)} bounds, at most "
                    + f"{num_slots} are supported (EEFBoundCostCfg.max_bounds_per_tool_frame)"
                )
            i = self.tool_frames.index(name)
            self._bounds[name] = frame_bounds
            self._anchor[i] = -1
            for k, bound in enumerate(frame_bounds):
                center, lower, upper = bound.kernel_values(self.config.metric)
                self._axes[i, k] = 0 if bound.reference_axes == "anchor" else 1
                self._reference_rotation[i, k] = self._reference_rotation.new_tensor(
                    bound.reference_rotation
                )
                self._tool_rotation[i, k] = self._tool_rotation.new_tensor(bound.tool_rotation)
                self._center[i, k] = self._center.new_tensor(center)
                self._lower[i, k] = self._lower.new_tensor(lower)
                self._upper[i, k] = self._upper.new_tensor(upper)
                self._use_rotation[i, k] = int(bound.uses_rotation)
                self._anchor[i, k] = bound.anchor.value

    def update_tool_pose_criteria(self, tool_pose_criteria: Dict[str, ToolPoseCriteria]) -> None:
        """Take the bounds from ``ToolPoseCriteria.bounds`` of the given tool frames."""
        self.update_tool_pose_bounds({k: v.bounds for k, v in tool_pose_criteria.items()})

    def setup_batch_tensors(self, batch_size: int, horizon: int):
        """Allocate output buffers for a ``(batch_size, horizon)`` trajectory batch."""
        if batch_size != self._batch_size or horizon != self._horizon:
            num_links = len(self.tool_frames)
            kwargs = {"device": self.device_cfg.device, "dtype": self.device_cfg.dtype}
            self._out_cost = torch.zeros((batch_size, horizon, num_links), **kwargs)
            self._out_position_gradient = torch.zeros(
                (batch_size, horizon, num_links, 3), **kwargs
            )
            self._out_quaternion_gradient = torch.zeros(
                (batch_size, horizon, num_links, 4), **kwargs
            )
            self._unused_idxs = torch.zeros(
                (batch_size,), device=self.device_cfg.device, dtype=torch.int32
            )
        super().setup_batch_tensors(batch_size, horizon)

    def forward(
        self,
        current_tool_poses: ToolPose,
        goal_tool_poses: Optional[GoalToolPose] = None,
        idxs_goal: Optional[torch.Tensor] = None,
        start_tool_poses: Optional[ToolPose] = None,
        idxs_start: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Compute the bound cost.

        Args:
            current_tool_poses: Tool poses ``(B, H, L, 3/4)``.
            goal_tool_poses: Goals ``(B_goal, 1, L, G, 3/4)``; required by goal-anchored bounds.
            idxs_goal: ``(B,)`` or ``(B, 1)`` index into the goal batch.
            start_tool_poses: Start poses ``(B_start, 1, L, 3/4)``; required by start-anchored
                bounds.
            idxs_start: ``(B,)`` or ``(B, 1)`` index into the start batch.
            **kwargs: Ignored; accepted for the common cost interface.

        Returns:
            Cost ``(B, H, 1)``.
        """
        if current_tool_poses.tool_frames != self.tool_frames:
            log_and_raise(
                f"EEFBoundCost: tool frames {current_tool_poses.tool_frames} do not match "
                + f"{self.tool_frames}"
            )
        b, h, _, _ = current_tool_poses.position.shape
        self.setup_batch_tensors(b, h)

        # Real goal / start arrays are passed whenever they are usable, so a CUDA graph
        # captured before a bound switches anchor still reads valid memory.
        goal_position, goal_quaternion = self._unused_position, self._unused_quaternion
        idxs_goal_flat = self._unused_idxs
        num_goalset = 1
        goal_usable = (
            goal_tool_poses is not None
            and idxs_goal is not None
            and goal_tool_poses.tool_frames == self.tool_frames
            and goal_tool_poses.horizon == 1
        )
        if self.uses_anchor(ToolPoseBoundAnchor.GOAL) and not goal_usable:
            log_and_raise(
                "EEFBoundCost: a goal-anchored bound needs goal poses with horizon 1 for tool "
                + f"frames {self.tool_frames} and idxs_goal"
            )
        if goal_usable:
            num_goalset = goal_tool_poses.num_goalset
            goal_position = goal_tool_poses.position.reshape(-1, 3)
            goal_quaternion = goal_tool_poses.quaternion.reshape(-1, 4)
            idxs_goal_flat = self._flat_idxs(idxs_goal, b, "idxs_goal")

        start_position, start_quaternion = self._unused_position, self._unused_quaternion
        idxs_start_flat = self._unused_idxs
        start_usable = (
            start_tool_poses is not None
            and idxs_start is not None
            and start_tool_poses.tool_frames == self.tool_frames
        )
        if self.uses_anchor(ToolPoseBoundAnchor.START) and not start_usable:
            log_and_raise(
                "EEFBoundCost: a start-anchored bound needs the start state of the problem "
                + f"(start poses for tool frames {self.tool_frames} and idxs_start)"
            )
        if start_usable:
            start_position = start_tool_poses.position.reshape(-1, 3)
            start_quaternion = start_tool_poses.quaternion.reshape(-1, 4)
            idxs_start_flat = self._flat_idxs(idxs_start, b, "idxs_start")

        cost = EEFBoundCostFunction.apply(
            current_tool_poses.position,
            current_tool_poses.quaternion,
            goal_position.detach().contiguous(),
            goal_quaternion.detach().contiguous(),
            idxs_goal_flat,
            start_position.detach().contiguous(),
            start_quaternion.detach().contiguous(),
            idxs_start_flat,
            self._anchor,
            self._axes,
            self._reference_rotation,
            self._tool_rotation,
            self._center,
            self._lower,
            self._upper,
            self._use_rotation,
            self._weight,
            num_goalset,
            self.config.use_grad_input,
            self._out_cost,
            self._out_position_gradient,
            self._out_quaternion_gradient,
        )
        cost = cost.sum(dim=-1, keepdim=True)
        if self.config.convert_to_binary:
            cost = torch.clamp(cost, max=1.0)
            cost = torch.where(cost > 0, cost + 1.0, cost)
        return cost

    @staticmethod
    def _flat_idxs(idxs: torch.Tensor, batch_size: int, name: str) -> torch.Tensor:
        flat = idxs.reshape(-1)
        if flat.shape[0] != batch_size:
            log_and_raise(f"EEFBoundCost: {name} must have {batch_size} entries, got {flat.shape}")
        if flat.dtype != torch.int32:
            log_and_raise(f"EEFBoundCost: {name} must be int32, got {flat.dtype}")
        return flat.contiguous()
