# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Stitch per-segment joint-space trajectories into one trajectory.

The multi-waypoint planner solves each segment (edge between two waypoints)
independently and then concatenates the surviving segment trajectories rather
than solving one giant trajectory.  Consecutive segments share a waypoint
configuration (the end of segment ``s`` equals the start of segment ``s+1``), so
the duplicated boundary sample is dropped to avoid a one-timestep pause artifact.
"""

from __future__ import annotations

# Standard Library
from typing import List

# Third Party
import torch

# CuRobo
from curobo._src.state.state_joint import JointState
from curobo._src.util.logging import log_and_raise


def stitch_jointstate_trajectories(
    segments: List[JointState],
    drop_duplicate_boundary: bool = True,
) -> JointState:
    """Concatenate single-trajectory joint states along the time axis.

    Args:
        segments: List of joint states, each with position of shape
            ``[horizon, dof]`` (a single trajectory, not batched).
        drop_duplicate_boundary: When ``True``, drop the first sample of every
            segment after the first, since it coincides with the last sample of
            the previous segment.

    Returns:
        A single :class:`JointState` whose position is the concatenation of all
        segment positions along the time axis.  Velocity/acceleration/jerk are
        concatenated only when present in every segment, otherwise set to
        ``None``.  ``dt`` is taken from the first segment.
    """
    if len(segments) == 0:
        log_and_raise("stitch_jointstate_trajectories requires at least one segment")

    has_velocity = all(s.velocity is not None for s in segments)
    has_acceleration = all(s.acceleration is not None for s in segments)
    has_jerk = all(s.jerk is not None for s in segments)

    position_parts: List[torch.Tensor] = []
    velocity_parts: List[torch.Tensor] = []
    acceleration_parts: List[torch.Tensor] = []
    jerk_parts: List[torch.Tensor] = []

    for i, segment in enumerate(segments):
        if segment.position.ndim != 2:
            log_and_raise(
                "stitch_jointstate_trajectories expects [horizon, dof] segments, "
                f"got {segment.position.shape}"
            )
        start = 1 if (drop_duplicate_boundary and i > 0) else 0
        position_parts.append(segment.position[start:])
        if has_velocity:
            velocity_parts.append(segment.velocity[start:])
        if has_acceleration:
            acceleration_parts.append(segment.acceleration[start:])
        if has_jerk:
            jerk_parts.append(segment.jerk[start:])

    position = torch.cat(position_parts, dim=0)
    velocity = torch.cat(velocity_parts, dim=0) if has_velocity else None
    acceleration = torch.cat(acceleration_parts, dim=0) if has_acceleration else None
    jerk = torch.cat(jerk_parts, dim=0) if has_jerk else None

    return JointState(
        position=position,
        velocity=velocity,
        acceleration=acceleration,
        jerk=jerk,
        dt=segments[0].dt,
        joint_names=segments[0].joint_names,
    )
