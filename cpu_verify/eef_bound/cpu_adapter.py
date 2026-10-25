# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only adapters for verifying the tool pose bound cost without a GPU.

NOT for main: see ``cpu_verify/README.md``. Nothing in ``curobo/`` imports this module and it
never patches library code. It provides:

* builders for :class:`EEFBoundCost` and single-link problems on ``cpu``,
* a Python mirror of the CUDA ``quaternion_gradient_to_angular_velocity`` used by the
  kinematics backward pass, so the orientation gradient is checked in the frame the robot
  actually consumes it,
* central finite differences under base-frame perturbations (the ground truth).
"""

from __future__ import annotations

import math
import os
from typing import Callable, Dict, List, Optional, Sequence, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

import torch

from curobo._src.cost.cost_eef_bound import EEFBoundCost
from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg
from curobo._src.cost.tool_pose_bound import ToolPoseBound, _quat_mul
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.warp import init_warp

CPU = torch.device("cpu")
DTYPE = torch.float32
IDENTITY = torch.tensor([1.0, 0.0, 0.0, 0.0])


def device_cfg() -> DeviceCfg:
    """Return a CPU float32 device config."""
    return DeviceCfg(device=CPU, dtype=DTYPE)


def init() -> None:
    """Initialize Warp; kernels compile for the ``cpu`` (LLVM) device."""
    init_warp()


def quat_axis_angle(axis: Sequence[float], angle: float) -> torch.Tensor:
    """Return a wxyz quaternion ``(4,)`` for a rotation of ``angle`` about ``axis``."""
    a = torch.tensor(axis, dtype=DTYPE)
    a = a / a.norm()
    head = torch.tensor([math.cos(angle / 2.0)], dtype=DTYPE)
    return torch.cat([head, a * math.sin(angle / 2.0)])


def random_quat(n: int, generator: torch.Generator) -> torch.Tensor:
    """Return ``(n, 4)`` uniformly random unit wxyz quaternions."""
    q = torch.randn(n, 4, generator=generator, dtype=DTYPE)
    return q / q.norm(dim=-1, keepdim=True)


def make_cost(
    bounds: Dict[str, Optional[ToolPoseBound]],
    tool_frames: Optional[List[str]] = None,
    weight: float = 1.0,
    **cfg_kwargs,
) -> EEFBoundCost:
    """Build an :class:`EEFBoundCost` on CPU with ``bounds`` applied."""
    cfg = EEFBoundCostCfg(
        weight=weight, device_cfg=device_cfg(), tool_frames=tool_frames or ["ee"], **cfg_kwargs
    )
    cost = EEFBoundCost(cfg)
    cost.update_tool_pose_bounds(bounds)
    return cost


def single_link_problem(
    position: torch.Tensor,
    quaternion: torch.Tensor,
    goal_position: torch.Tensor,
    goal_quaternion: torch.Tensor,
    start_position: Optional[torch.Tensor] = None,
    start_quaternion: Optional[torch.Tensor] = None,
) -> Tuple[ToolPose, GoalToolPose, ToolPose, torch.Tensor]:
    """Build a ``B=1, H=1, L=1`` problem. Returns tool, goal, start poses and the index."""
    start_position = goal_position if start_position is None else start_position
    start_quaternion = goal_quaternion if start_quaternion is None else start_quaternion
    tool = ToolPose(["ee"], position.view(1, 1, 1, 3), quaternion.view(1, 1, 1, 4))
    goal = GoalToolPose(
        ["ee"], goal_position.view(1, 1, 1, 1, 3), goal_quaternion.view(1, 1, 1, 1, 4)
    )
    start = ToolPose(["ee"], start_position.view(1, 1, 1, 3), start_quaternion.view(1, 1, 1, 4))
    return tool, goal, start, torch.zeros(1, dtype=torch.int32)


def evaluate(cost: EEFBoundCost, *problem_args) -> float:
    """Scalar cost of a :func:`single_link_problem`."""
    tool, goal, start, idx = single_link_problem(*problem_args)
    return cost.forward(tool, goal, idx, start, idx)[0, 0, 0].item()


def analytic_gradient(cost: EEFBoundCost, *problem_args) -> Tuple[torch.Tensor, torch.Tensor]:
    """Position gradient and base-frame angular gradient as the kinematics backward reads it."""
    tool, goal, start, idx = single_link_problem(*problem_args)
    tool.requires_grad_(True)
    cost.forward(tool, goal, idx, start, idx).sum().backward()
    omega = quat_grad_to_angular_gradient(
        tool.quaternion.detach()[0, 0, 0], tool.quaternion.grad[0, 0, 0]
    )
    return tool.position.grad[0, 0, 0].clone(), omega


def quat_grad_to_angular_gradient(
    quaternion: torch.Tensor, quat_grad: torch.Tensor
) -> torch.Tensor:
    """Python mirror of CUDA ``quaternion_gradient_to_angular_velocity``.

    ``omega = 0.5 * vec(conj(q) ⊗ dq)``; ``xyz_rot_backward`` dots it with base-frame joint
    axes, so it must equal the base-frame angular gradient of the cost.
    """
    w, x, y, z = quaternion.unbind(-1)
    dw, dx, dy, dz = quat_grad.unbind(-1)
    return 0.5 * torch.stack(
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
    """Central differences w.r.t. base-frame translation and rotation ``q' = exp(eps e_k) q``."""
    grad_pos = torch.zeros(3, dtype=DTYPE)
    grad_rot = torch.zeros(3, dtype=DTYPE)
    for k in range(3):
        step = torch.zeros(3, dtype=DTYPE)
        step[k] = eps
        grad_pos[k] = (
            cost_fn(position + step, quaternion) - cost_fn(position - step, quaternion)
        ) / (2.0 * eps)
        axis = [0.0, 0.0, 0.0]
        axis[k] = 1.0
        q_plus = _quat_mul(quat_axis_angle(axis, eps), quaternion)
        q_minus = _quat_mul(quat_axis_angle(axis, -eps), quaternion)
        grad_rot[k] = (cost_fn(position, q_plus) - cost_fn(position, q_minus)) / (2.0 * eps)
    return grad_pos, grad_rot
