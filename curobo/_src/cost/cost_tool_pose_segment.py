# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Per-segment tool-pose criteria for a trajectory that passes through ordered waypoints.

A trajectory with ``N`` waypoints has ``N`` segments: segment ``k`` ends at waypoint ``k`` and
starts at waypoint ``k - 1`` (the start state for ``k = 0``). Every trajectory sample reads the
criteria of one *slot*:

- path slot ``k`` (index ``k``): samples inside segment ``k``, compared with the reference
  interpolated between the segment's ends, using the path fields of its criteria;
- reach slot ``j`` (index ``capacity + j``): the sample at waypoint ``j``, using the terminal
  fields of segment ``j``'s criteria and an enforced reach tolerance. The samples after the last
  waypoint, where the trajectory rests at the goal, use the last reach slot.

:class:`ToolPoseSegmentTable` holds the fixed-capacity criteria tables shared by every tool-pose
cost of one trajectory optimizer. :class:`ToolPoseSegmentState` maps each cost's samples to slots,
for native samples (fixed per plan) or dense validation samples (from their reference progress).
All buffers have fixed addresses, so updating a plan never requires a new CUDA graph capture.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch

from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise


def native_sample_slots(
    waypoint_indices: Sequence[int], horizon: int, capacity: int
) -> List[int]:
    """Return the criteria slot of every native sample.

    Args:
        waypoint_indices: Strictly increasing native sample index of each waypoint.
        horizon: Number of native samples.
        capacity: Maximum number of segments; reach slots start at this index.

    Returns:
        ``horizon`` slot indices.
    """
    indices = list(waypoint_indices)
    if not indices or any(b <= a for a, b in zip(indices, indices[1:])):
        log_and_raise("waypoint indices must be a non-empty, strictly increasing sequence")
    if indices[-1] >= horizon or len(indices) > capacity:
        log_and_raise("waypoint indices exceed the native horizon or the segment capacity")
    reach = {index: j for j, index in enumerate(indices)}
    slots = []
    segment = 0
    for h in range(horizon):
        if h in reach:
            slots.append(capacity + reach[h])
            segment = reach[h] + 1
        elif h > indices[-1]:
            slots.append(capacity + len(indices) - 1)
        else:
            slots.append(segment)
    return slots


class ToolPoseSegmentTable:
    """Fixed-capacity criteria of every segment and waypoint of one planned trajectory."""

    def __init__(
        self,
        tool_frames: Sequence[str],
        capacity: int,
        horizon: int,
        padding: int,
        device_cfg: DeviceCfg,
    ):
        """Allocate the tables for up to ``capacity`` segments.

        Args:
            tool_frames: Ordered tool frames of the tool-pose costs.
            capacity: Maximum number of segments (waypoints) of one trajectory.
            horizon: Number of native samples of the trajectory optimizer.
            padding: Stationary native intervals at each end of the reference.
            device_cfg: Device and dtype of the costs.
        """
        if capacity < 1:
            log_and_raise("segment capacity must be positive")
        self.tool_frames = list(tool_frames)
        self.capacity = capacity
        self.horizon = horizon
        self.padding = padding
        self.device_cfg = device_cfg
        links = len(self.tool_frames)
        entries = 2 * capacity * links
        tensor_args = device_cfg.as_torch_dict()
        device = device_cfg.device
        #: Quadratic position metric per ``slot * links + link``.
        self.position_metric = torch.zeros((entries, 3, 3), **tensor_args)
        #: Quadratic rotation metric per ``slot * links + link``.
        self.rotation_metric = torch.zeros((entries, 3, 3), **tensor_args)
        #: Cost deadband; zero so the optimizers pull every tracked sample to the reference.
        self.deadband = torch.zeros((entries, 2), **tensor_args)
        #: :class:`PoseErrorFrame` per entry.
        self.error_frame = torch.zeros((entries,), device=device, dtype=torch.int32)
        #: Acceptance tolerance ``[2 * capacity, links, 2]`` (m, 2·sin(angle/2)).
        self.tolerance = torch.zeros((2 * capacity, links, 2), **tensor_args)
        #: Whether a slot's tolerance decides feasibility ``[2 * capacity, links]``.
        self.enforce = torch.zeros((2 * capacity, links), **tensor_args)
        #: Slot of every native sample ``[horizon]``.
        self.native_slots = torch.zeros((horizon,), device=device, dtype=torch.int32)
        #: Native index of every waypoint, ``inf`` beyond the planned count ``[capacity]``.
        self.waypoint_indices = torch.full((capacity,), torch.inf, **tensor_args)
        #: Index of the last segment of the plan (0-d, int64).
        self.last_segment = torch.zeros((), device=device, dtype=torch.int64)
        #: Reach slot of the last waypoint (0-d, int32).
        self.final_slot = torch.full((), capacity, device=device, dtype=torch.int32)
        #: Dense validation progress and sample counts, bound by the dense tool-pose cost.
        self.dense_progress: Optional[torch.Tensor] = None
        self.dense_counts: Optional[torch.Tensor] = None
        self.num_segments = 0

    def set_plan(
        self,
        path_criteria: Sequence[Dict[str, ToolPoseCriteria]],
        reach_criteria: Sequence[Dict[str, ToolPoseCriteria]],
        waypoint_indices: Sequence[int],
    ) -> None:
        """Write the criteria of one plan in place.

        Args:
            path_criteria: Criteria per segment; their path fields apply inside the segment.
                Every dict holds every tool frame.
            reach_criteria: Criteria per waypoint; their terminal fields apply at the waypoint,
                where ``terminal_tolerance`` (m, 2·sin(angle/2)) is always enforced.
            waypoint_indices: Native sample index of every waypoint, strictly increasing.
        """
        n = len(waypoint_indices)
        if not (len(path_criteria) == len(reach_criteria) == n):
            log_and_raise("every segment needs path criteria and reach criteria")
        slots = native_sample_slots(waypoint_indices, self.horizon, self.capacity)
        cap, links = self.capacity, len(self.tool_frames)
        rows, fields = [], {key: [] for key in ("pp", "ps", "rp", "rs", "frame", "tol", "enf")}
        for k in range(n):
            for slot, criteria, prefix in (
                (k, path_criteria[k], "path"),
                (cap + k, reach_criteria[k], "terminal"),
            ):
                for li, frame in enumerate(self.tool_frames):
                    if frame not in criteria:
                        log_and_raise(f"criteria of segment {k} miss tool frame {frame}")
                    c = criteria[frame]
                    rows.append(slot * links + li)
                    fields["pp"].append(getattr(c, f"{prefix}_position_projector"))
                    fields["ps"].append(getattr(c, f"{prefix}_position_axis_scale"))
                    fields["rp"].append(getattr(c, f"{prefix}_rotation_projector"))
                    fields["rs"].append(getattr(c, f"{prefix}_rotation_axis_scale"))
                    fields["frame"].append(c.error_frame)
                    if prefix == "path":
                        fields["tol"].append(c.path_tolerance)
                        fields["enf"].append(c.enforce_path_tolerance)
                    else:
                        fields["tol"].append(c.terminal_tolerance)
                        fields["enf"].append(self.enforce.new_ones(()))
        stacked = {key: torch.stack(values) for key, values in fields.items()}
        index = torch.tensor(rows, device=self.device_cfg.device, dtype=torch.long)
        self.position_metric.zero_()
        self.rotation_metric.zero_()
        self.error_frame.zero_()
        self.tolerance.zero_()
        self.enforce.zero_()
        self.position_metric.index_copy_(0, index, _metric(stacked["pp"], stacked["ps"]))
        self.rotation_metric.index_copy_(0, index, _metric(stacked["rp"], stacked["rs"]))
        self.error_frame.index_copy_(0, index, stacked["frame"].to(torch.int32))
        self.tolerance.view(-1, 2).index_copy_(0, index, stacked["tol"])
        self.enforce.view(-1).index_copy_(0, index, stacked["enf"].to(self.enforce.dtype))
        self.native_slots.copy_(self.native_slots.new_tensor(slots))
        indices = torch.full((cap,), torch.inf, dtype=self.waypoint_indices.dtype)
        indices[:n] = torch.tensor(list(waypoint_indices), dtype=indices.dtype)
        self.waypoint_indices.copy_(indices)
        self.last_segment.fill_(n - 1)
        self.final_slot.fill_(cap + n - 1)
        self.num_segments = n

    def write_sample_slots(
        self,
        out: torch.Tensor,
        counts: Optional[torch.Tensor] = None,
        progress: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Write the slot of every sample into ``out`` ``[batch, horizon]`` and return it.

        Args:
            out: Fixed slot buffer ``[batch, horizon]`` (int32).
            counts: Valid samples per problem ``[batch]``; required with ``progress``.
            progress: Dense reference progress in ``[0, 1]`` ``[batch, horizon]``. ``None``
                means the samples are the native samples.
        """
        if progress is None:
            if out.shape[1] != self.horizon:
                # Evaluations off the native grid (such as goal-only checks) reach the goal.
                out.copy_(self.final_slot.expand_as(out))
            else:
                out.copy_(self.native_slots.unsqueeze(0).expand_as(out))
            return out
        active = self.horizon - 2 * self.padding
        native = torch.nan_to_num(progress, nan=0.0).clamp(0.0, 1.0) * float(active - 1)
        native = native + float(self.padding)
        segment = torch.searchsorted(self.waypoint_indices, native.contiguous(), right=False)
        segment = torch.minimum(segment, self.last_segment)
        sample = torch.arange(out.shape[1], device=out.device).unsqueeze(0)
        last = sample == (counts.view(-1, 1) - 1)
        out.copy_(torch.where(last, self.final_slot, segment.to(torch.int32)))
        return out

    def segment_of_slot(self, slots: torch.Tensor) -> torch.Tensor:
        """Return the segment index of every slot: a reach slot belongs to the segment it ends."""
        return torch.where(slots >= self.capacity, slots - self.capacity, slots)


class ToolPoseSegmentState:
    """Sample-to-slot mapping of one cost evaluated on native or dense samples."""

    def __init__(self, table: ToolPoseSegmentTable, dense: bool):
        """Bind a cost to a shared table.

        Args:
            table: Shared criteria of the planned trajectory.
            dense: The cost evaluates dense validation samples instead of native samples.
        """
        self.table = table
        self.dense = dense
        self.slots: Optional[torch.Tensor] = None

    def sample_slots(
        self,
        batch_size: int,
        horizon: int,
        counts: Optional[torch.Tensor] = None,
        progress: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Return the fixed-address slot buffer ``[batch, horizon]`` of the current samples."""
        if self.slots is None or self.slots.shape != (batch_size, horizon):
            self.slots = torch.zeros(
                (batch_size, horizon), device=self.table.device_cfg.device, dtype=torch.int32
            )
        if self.dense:
            if progress is None or counts is None:
                # Dense samples without a reference rest at the goal reached by the last slot.
                self.slots.copy_(self.table.final_slot.expand_as(self.slots))
                return self.slots
            return self.table.write_sample_slots(
                self.slots, counts, progress.reshape(batch_size, horizon)
            )
        return self.table.write_sample_slots(self.slots)

    def acceptance(self, slots: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return enforced flags ``[batch, horizon, links]`` and tolerances ``[..., 2]``."""
        return self.table.enforce[slots], self.table.tolerance[slots]


def _metric(projector: torch.Tensor, axis_scale: torch.Tensor) -> torch.Tensor:
    """Return ``P.T @ diag(axis_scale**2) @ P``."""
    scaled = axis_scale.unsqueeze(-1) * projector
    return scaled.transpose(-1, -2) @ scaled
