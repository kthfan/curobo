# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Full-turn variants of IK solutions.

A revolute joint whose limits span more than one turn reaches the same robot pose at ``q`` and
``q ± 2π``. An IK solver returns only one of them per solution, so a planner that chooses among
IK solutions by joint distance would not see that a waypoint can be reached "the other way
round". :func:`dedup_joint_revolutions` reduces IK solutions to one per pose and
:func:`expand_joint_revolutions` adds every variant within the limits.
"""

from __future__ import annotations

import math

import torch

from curobo._src.robot.types.joint_types import JointType
from curobo._src.robot.types.kinematics_params import KinematicsParams

_REVOLUTE_TYPES = (
    JointType.X_ROT.value,
    JointType.Y_ROT.value,
    JointType.Z_ROT.value,
    JointType.X_ROT_NEG.value,
    JointType.Y_ROT_NEG.value,
    JointType.Z_ROT_NEG.value,
)


def revolute_joint_mask(kinematics_config: KinematicsParams) -> torch.Tensor:
    """Tell which active joints return to the same robot pose after a full turn.

    That holds for a revolute joint whose mimic joints, if any, are revolute and follow it with
    a multiplier of ±1.

    Args:
        kinematics_config: Kinematics parameters of the robot.

    Returns:
        Boolean mask ``[dof]`` over the active joints.
    """
    joint_map = kinematics_config.joint_map.view(-1).long()
    joint_type = kinematics_config.joint_map_type.view(-1).long()
    multiplier = kinematics_config.joint_offset_map.view(-1, 2)[:, 0]
    dof = kinematics_config.joint_limits.position.shape[-1]
    revolute = torch.zeros_like(joint_type, dtype=torch.bool)
    for value in _REVOLUTE_TYPES:
        revolute |= joint_type == value
    same_pose = revolute & ((multiplier.abs() - 1.0).abs() < 1e-6)
    mask = torch.zeros(dof, dtype=torch.bool, device=joint_map.device)
    for j in range(dof):
        links = joint_map == j
        mask[j] = bool(links.any()) and bool(same_pose[links].all())
    return mask


def dedup_joint_revolutions(
    solutions: torch.Tensor,
    tol: float,
    lower: torch.Tensor,
    upper: torch.Tensor,
    revolute: torch.Tensor,
) -> torch.Tensor:
    """De-duplicate joint configurations that differ by full turns or by less than ``tol``.

    Revolute joints whose limits span more than one turn are first moved into the turn that
    starts at their lower limit; distances on those joints are measured around the circle.
    Like :func:`~curobo._src.motion.ik_dedup.dedup_joint_solutions`, the first of every group
    of duplicates is kept.

    Args:
        solutions: Joint configurations ``[N, dof]`` within the limits.
        tol: L-infinity distance (rad or m) under which two configurations count as one.
        lower: Lower position limits ``[dof]``.
        upper: Upper position limits ``[dof]``.
        revolute: Joints that reach the same pose after a full turn, boolean ``[dof]``.

    Returns:
        Unique configurations ``[M, dof]``, ``M <= N``, in their original order.
    """
    turn = 2.0 * math.pi
    wide = revolute & ((upper - lower) > turn)
    canonical = torch.where(wide, lower + torch.remainder(solutions - lower, turn), solutions)
    if canonical.shape[0] <= 1:
        return canonical
    difference = (canonical.unsqueeze(1) - canonical.unsqueeze(0)).abs()
    difference = torch.where(wide, torch.minimum(difference, turn - difference), difference)
    close = (difference.amax(dim=-1) < tol).cpu()
    alive = torch.ones(canonical.shape[0], dtype=torch.bool)
    keep = []
    for i in range(canonical.shape[0]):
        if alive[i]:
            keep.append(i)
            alive &= ~close[i]
    return canonical[torch.tensor(keep, device=canonical.device)]


def expand_joint_revolutions(
    solutions: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    revolute: torch.Tensor,
    max_turns: int = 1,
    max_solutions: int = 4096,
) -> torch.Tensor:
    """Add the copies of joint configurations that differ by full turns of revolute joints.

    Args:
        solutions: Joint configurations ``[N, dof]`` within the limits.
        lower: Lower position limits ``[dof]``.
        upper: Upper position limits ``[dof]``.
        revolute: Joints that reach the same pose after a full turn, boolean ``[dof]``.
        max_turns: Most full turns added to or subtracted from one joint.
        max_solutions: Most configurations returned. When the variants exceed it, the ones with
            the fewest shifted joints are kept.

    Returns:
        Configurations ``[M, dof]``, ``M >= N``: ``solutions`` first, then their variants within
        the limits.
    """
    n, dof = solutions.shape
    turn = 2.0 * math.pi
    wide = revolute & ((upper - lower) > turn)
    if n == 0 or max_turns < 1 or not bool(wide.any()):
        return solutions
    turns = torch.arange(-max_turns, max_turns + 1, device=solutions.device)
    zero = torch.zeros(1, dtype=turns.dtype, device=solutions.device)
    options = [turns if bool(wide[j]) else zero for j in range(dof)]
    shifts = torch.cartesian_prod(*options).view(-1, dof)
    shifts = shifts[shifts.abs().sum(dim=-1) > 0]
    # Fewest shifted joints first, so a size cap keeps the variants closest to the solutions.
    shifts = shifts[torch.argsort(shifts.abs().sum(dim=-1), stable=True)]
    variants = solutions.unsqueeze(0) + shifts.to(solutions.dtype).unsqueeze(1) * turn
    valid = ((variants >= lower) & (variants <= upper)).all(dim=-1)
    variants = variants[valid][: max(0, max_solutions - n)]
    return torch.cat((solutions, variants), dim=0)
