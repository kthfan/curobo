# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only adapters for verifying :class:`EEFBoundCost` without a GPU.

NOT for main: see ``cpu_verify/README.md``. Nothing in ``curobo/`` imports this module and it
never patches library code. It only provides:

* builders for the cost and its pose inputs on ``cpu``,
* a Python mirror of the CUDA ``quaternion_gradient_to_angular_velocity`` used by the
  kinematics backward pass, so the rotation gradient can be checked in the frame the robot
  actually consumes it,
* central finite differences under world-frame perturbations (the ground truth),
* a goal gather that mimics ``RobotCostManager`` goal indexing for ``num_seeds > 1``.
"""

from __future__ import annotations

import math
import os
from typing import Callable, List, Optional, Sequence, Tuple, Type

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from curobo._src.cost.cost_eef_bound import EEFBoundCost
from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg
from curobo._src.cost.cost_eef_bound_py import EEFBoundCostPyTorch, quat_mul_wxyz
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.warp import init_warp

CPU = torch.device("cpu")
DTYPE = torch.float32


def device_cfg() -> DeviceCfg:
    """Return a CPU float32 device config."""
    return DeviceCfg(device=CPU, dtype=DTYPE)


def init() -> None:
    """Initialize Warp; kernels compile for the ``cpu`` (LLVM) device."""
    init_warp()


def quat_axis_angle(axis: Sequence[float], angle: float) -> torch.Tensor:
    """Return a wxyz quaternion of shape ``(4,)`` for a rotation of ``angle`` about ``axis``."""
    a = torch.tensor(axis, dtype=DTYPE)
    a = a / a.norm()
    return torch.cat([torch.tensor([math.cos(angle / 2.0)], dtype=DTYPE), a * math.sin(angle / 2.0)])


def random_quat(n: int, generator: torch.Generator) -> torch.Tensor:
    """Return ``(n, 4)`` uniformly random unit wxyz quaternions with ``w >= 0``."""
    q = torch.randn(n, 4, generator=generator, dtype=DTYPE)
    q = q / q.norm(dim=-1, keepdim=True)
    return torch.where(q[:, :1] < 0, -q, q)


def make_cost(
    cls: Type,
    margins: torch.Tensor,
    tool_frames: Optional[List[str]] = None,
    weight: float = 1.0,
    **cfg_kwargs,
):
    """Build ``EEFBoundCost`` or ``EEFBoundCostPyTorch`` on CPU with ``margins`` applied.

    Args:
        cls: Cost class to build.
        margins: ``(4, 3)`` or ``(L, 4, 3)`` margins; negative disables an axis.
        tool_frames: Link names. Defaults to ``["ee"]``.
        weight: Scalar cost weight.
        **cfg_kwargs: Extra :class:`EEFBoundCostCfg` fields.
    """
    cfg = EEFBoundCostCfg(
        weight=weight,
        device_cfg=device_cfg(),
        tool_frames=tool_frames or ["ee"],
        **cfg_kwargs,
    )
    cost = cls(cfg)
    cost.update_margins(margins)
    return cost


def make_poses(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    goal_position: torch.Tensor,
    goal_quaternion: torch.Tensor,
    tool_frames: Optional[List[str]] = None,
) -> Tuple[ToolPose, GoalToolPose]:
    """Wrap tensors into cuRobo pose containers.

    Args:
        position: ``(B, H, L, 3)``; ``[:, 0]`` is used by the cost as the corridor start.
        quaternion: ``(B, H, L, 4)`` wxyz.
        goal_position: ``(B_goal, 1, L, G, 3)``.
        goal_quaternion: ``(B_goal, 1, L, G, 4)`` wxyz.
        tool_frames: Link names. Defaults to ``["ee"]`` or ``link_i`` for ``L > 1``.
    """
    num_links = position.shape[2]
    if tool_frames is None:
        tool_frames = ["ee"] if num_links == 1 else [f"link_{i}" for i in range(num_links)]
    return (
        ToolPose(tool_frames=tool_frames, position=position, quaternion=quaternion),
        GoalToolPose(tool_frames=tool_frames, position=goal_position, quaternion=goal_quaternion),
    )


def single_link_problem(
    start_position: torch.Tensor,
    start_quaternion: torch.Tensor,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    goal_position: torch.Tensor,
    goal_quaternion: torch.Tensor,
) -> Tuple[ToolPose, GoalToolPose]:
    """Build a ``B=1, H=2, L=1, G=1`` problem: ``t=0`` is the start, ``t=1`` the evaluated pose."""
    pos = torch.stack([start_position, position]).view(1, 2, 1, 3).clone()
    quat = torch.stack([start_quaternion, quaternion]).view(1, 2, 1, 4).clone()
    return make_poses(pos, quat, goal_position.view(1, 1, 1, 1, 3), goal_quaternion.view(1, 1, 1, 1, 4))


def quat_grad_to_world_omega(quaternion: torch.Tensor, quat_grad: torch.Tensor) -> torch.Tensor:
    """Map a cuRobo quaternion gradient to the world-frame angular gradient the robot uses.

    Mirrors ``quaternion_gradient_to_angular_velocity`` in
    ``curobolib/kernels/common/quaternion_util.cuh`` (``omega = 0.5 * vec(conj(q) * dq)``) and
    undoes its ``0.5`` so that ``dq = q * (v, 0)`` maps back to ``v``. The kinematics backward
    pass treats ``omega`` as a world-frame torque-like vector (``xyz_rot_backward``).

    Args:
        quaternion: ``(..., 4)`` wxyz link orientation.
        quat_grad: ``(..., 4)`` wxyz gradient written by a pose cost.

    Returns:
        ``(..., 3)`` world-frame angular gradient.
    """
    w, x, y, z = quaternion.unbind(-1)
    dw, dx, dy, dz = quat_grad.unbind(-1)
    return torch.stack(
        [
            -x * dw + w * dx + z * dy - y * dz,
            -y * dw - z * dx + w * dy + x * dz,
            -z * dw + y * dx - x * dy + w * dz,
        ],
        dim=-1,
    )


def finite_difference_world(
    cost_fn: Callable[[torch.Tensor, torch.Tensor], float],
    position: torch.Tensor,
    quaternion: torch.Tensor,
    eps: float = 1.0e-3,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Central differences of ``cost_fn`` w.r.t. world-frame translation and rotation.

    Rotation is perturbed as ``q' = exp(eps * e_k) * q`` (left, i.e. world frame), matching
    how a revolute joint upstream of the link moves it.

    Args:
        cost_fn: Maps ``(position (3,), quaternion (4,))`` to a scalar cost.
        position: ``(3,)`` evaluation position.
        quaternion: ``(4,)`` wxyz evaluation orientation.
        eps: Step size; float32 inputs need ``>= 1e-3``.

    Returns:
        Position gradient ``(3,)`` and world-frame angular gradient ``(3,)``.
    """
    grad_pos = torch.zeros(3, dtype=DTYPE)
    grad_rot = torch.zeros(3, dtype=DTYPE)
    for k in range(3):
        step = torch.zeros(3, dtype=DTYPE)
        step[k] = eps
        grad_pos[k] = (cost_fn(position + step, quaternion) - cost_fn(position - step, quaternion)) / (
            2.0 * eps
        )
        axis = [0.0, 0.0, 0.0]
        axis[k] = 1.0
        q_plus = quat_mul_wxyz(quat_axis_angle(axis, eps), quaternion)
        q_minus = quat_mul_wxyz(quat_axis_angle(axis, -eps), quaternion)
        grad_rot[k] = (cost_fn(position, q_plus) - cost_fn(position, q_minus)) / (2.0 * eps)
    return grad_pos, grad_rot


def gather_goal_for_seeds(goal: GoalToolPose, idxs_goal: torch.Tensor) -> GoalToolPose:
    """Expand goals to the rollout batch the way ``RobotCostManager`` indexes them.

    ``ToolPoseCost`` receives ``goal.idxs_link_pose`` (shape ``(B, 1)``) and looks goals up
    per rollout row; ``EEFBoundCost.forward`` takes no index, so this gather is what the
    cost manager hook would have to do (or the kernel would have to accept ``idxs_goal``).

    Args:
        goal: Goal with batch ``B_goal``.
        idxs_goal: ``(B,)`` or ``(B, 1)`` int index into the goal batch.
    """
    idx = idxs_goal.view(-1).long()
    return GoalToolPose(
        tool_frames=goal.tool_frames,
        position=goal.position[idx].contiguous(),
        quaternion=goal.quaternion[idx].contiguous(),
    )


__all__ = [
    "EEFBoundCost",
    "EEFBoundCostPyTorch",
    "finite_difference_world",
    "gather_goal_for_seeds",
    "init",
    "make_cost",
    "make_poses",
    "quat_axis_angle",
    "quat_grad_to_world_omega",
    "random_quat",
    "single_link_problem",
]
