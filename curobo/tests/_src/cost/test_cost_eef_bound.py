# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EEFBoundCost (Warp kernel, runs on CPU and CUDA)."""

import math

import pytest
import torch

from curobo._src.cost.cost_eef_bound import EEFBoundCost
from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg
from curobo._src.cost.tool_pose_bound import ToolPoseBound, _quat_mul, tool_pose_bound_cost
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.warp import init_warp


@pytest.fixture(scope="module", autouse=True)
def setup_warp():
    init_warp(quiet=True)


@pytest.fixture(params=["cpu", "cuda:0"])
def device_cfg(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(request.param), dtype=torch.float32)


def axis_angle(axis, angle, device):
    a = torch.tensor(axis, dtype=torch.float32, device=device)
    a = a / a.norm()
    head = torch.tensor([math.cos(angle / 2)], device=device)
    return torch.cat([head, a * math.sin(angle / 2)])


def random_quat(n, generator, device):
    q = torch.randn(n, 4, generator=generator).to(device)
    return q / q.norm(dim=-1, keepdim=True)


def make_cost(device_cfg, bounds, weight=1.0, tool_frames=("ee",), **cfg_kwargs):
    cfg = EEFBoundCostCfg(
        weight=weight, device_cfg=device_cfg, tool_frames=list(tool_frames), **cfg_kwargs
    )
    cost = EEFBoundCost(cfg)
    cost.update_tool_pose_bounds(bounds)
    return cost


BOUNDS = {
    "goal_position_offset": dict(
        anchor="goal",
        offset_position=[0.05, None, 0.1],
        offset_quaternion=[0.8, 0.2, 0.4, 0.4],
        position_lower=[-0.02, None, -0.3],
        position_upper=[0.03, 0.01, None],
    ),
    "goal_swing_twist": dict(
        anchor="goal", rotation_lower=[-0.1, -0.05, -0.3], rotation_upper=[0.05, 0.1, 0.2]
    ),
    "base_cone": dict(
        anchor="base",
        offset_quaternion=[0.98, 0.0, 0.2, 0.0],
        max_swing=0.2,
        rotation_lower=[None, None, -0.5],
        rotation_upper=[None, None, 0.4],
        position_lower=[None, None, -0.1],
    ),
    "start_all": dict(
        anchor="start",
        position_lower=[-0.05] * 3,
        position_upper=[0.05] * 3,
        rotation_lower=[-0.1] * 3,
        rotation_upper=[0.1] * 3,
        max_swing=0.12,
    ),
}


class Problem:
    """Random batch with 2 goal problems, a goalset of 2, and rollout rows mapped by index."""

    def __init__(self, device, seed=0, batch=3, horizon=4):
        g = torch.Generator().manual_seed(seed)
        self.position = (torch.randn(batch, horizon, 1, 3, generator=g) * 0.3).to(device)
        self.quaternion = random_quat(batch * horizon, g, device).view(batch, horizon, 1, 4)
        self.goal = GoalToolPose(
            tool_frames=["ee"],
            position=(torch.randn(2, 1, 1, 2, 3, generator=g) * 0.3).to(device),
            quaternion=random_quat(4, g, device).view(2, 1, 1, 2, 4),
        )
        self.start = ToolPose(
            tool_frames=["ee"],
            position=(torch.randn(2, 1, 1, 3, generator=g) * 0.3).to(device),
            quaternion=random_quat(2, g, device).view(2, 1, 1, 4),
        )
        self.idxs = torch.tensor([0, 1, 1], dtype=torch.int32, device=device)[:batch]

    def evaluate(self, cost, position=None, quaternion=None):
        position = self.position if position is None else position
        quaternion = self.quaternion if quaternion is None else quaternion
        tool_poses = ToolPose(tool_frames=["ee"], position=position, quaternion=quaternion)
        return cost.forward(tool_poses, self.goal, self.idxs, self.start, self.idxs)

    def reference(self, bound):
        return self.reference_per_anchor(bound).min(dim=-1).values

    def reference_per_anchor(self, bound):
        """PyTorch cost against every anchor candidate, ``(B, H, A)``."""
        idx = self.idxs.long()
        if bound.anchor.name == "GOAL":
            anchor_p = self.goal.position[idx, 0, 0]
            anchor_q = self.goal.quaternion[idx, 0, 0]
        elif bound.anchor.name == "START":
            anchor_p = self.start.position[idx, 0]
            anchor_q = self.start.quaternion[idx, 0]
        else:
            anchor_p = torch.zeros_like(self.position[:, 0])
            anchor_q = torch.zeros_like(self.quaternion[:, 0])
            anchor_q[..., 0] = 1.0
        cost = tool_pose_bound_cost(
            bound,
            self.position[:, :, 0].unsqueeze(2),
            self.quaternion[:, :, 0].unsqueeze(2),
            anchor_p.unsqueeze(1),
            anchor_q.unsqueeze(1),
        )
        return cost


def world_omega(quaternion, quaternion_grad):
    """Angular gradient as the kinematics backward reads it: 0.5 * vec(q⁻¹ ⊗ dq)."""
    w, x, y, z = quaternion.unbind(-1)
    dw, dx, dy, dz = quaternion_grad.unbind(-1)
    return 0.5 * torch.stack(
        [
            -x * dw + w * dx + z * dy - y * dz,
            -y * dw - z * dx + w * dy + x * dz,
            -z * dw + y * dx - x * dy + w * dz,
        ],
        dim=-1,
    )


@pytest.mark.parametrize("name", list(BOUNDS))
def test_forward_matches_reference(device_cfg, name):
    bound = ToolPoseBound(**BOUNDS[name])
    problem = Problem(device_cfg.device)
    cost = make_cost(device_cfg, {"ee": bound})
    out = problem.evaluate(cost)[..., 0]
    assert (out > 0).any()
    assert torch.allclose(out, problem.reference(bound), atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("name", list(BOUNDS))
def test_gradient_matches_finite_difference(device_cfg, name):
    """Gradients equal world-frame finite differences, in the frame kinematics consumes."""
    bound = ToolPoseBound(**BOUNDS[name])
    problem = Problem(device_cfg.device, seed=1)
    cost = make_cost(device_cfg, {"ee": bound}, use_grad_input=True)
    position = problem.position.clone().requires_grad_(True)
    quaternion = problem.quaternion.clone().requires_grad_(True)
    problem.evaluate(cost, position, quaternion).sum().backward()
    omega = world_omega(problem.quaternion[..., 0, :], quaternion.grad[..., 0, :])
    position_grad = position.grad[..., 0, :].clone()

    eps = 1e-3
    fd_position = torch.zeros_like(position_grad)
    fd_rotation = torch.zeros_like(omega)
    for k in range(3):
        step = torch.zeros(3, device=device_cfg.device)
        step[k] = eps
        plus = problem.evaluate(cost, problem.position + step).clone()
        minus = problem.evaluate(cost, problem.position - step).clone()
        fd_position[..., k] = (plus - minus)[..., 0] / (2 * eps)
        axis = [0.0, 0.0, 0.0]
        axis[k] = 1.0
        q_plus = _quat_mul(axis_angle(axis, eps, device_cfg.device), problem.quaternion)
        q_minus = _quat_mul(axis_angle(axis, -eps, device_cfg.device), problem.quaternion)
        plus = problem.evaluate(cost, quaternion=q_plus).clone()
        minus = problem.evaluate(cost, quaternion=q_minus).clone()
        fd_rotation[..., k] = (plus - minus)[..., 0] / (2 * eps)

    # Skip samples where two goalset candidates have nearly equal cost: the minimum switches
    # goals there, so the cost is not differentiable and finite differences straddle the switch.
    per_anchor = problem.reference_per_anchor(bound)
    if per_anchor.shape[-1] > 1:
        best_two = per_anchor.topk(2, dim=-1, largest=False).values
        clear = (best_two[..., 1] - best_two[..., 0]) > 1e-2
    else:
        clear = torch.ones(per_anchor.shape[:2], dtype=torch.bool, device=per_anchor.device)
    assert clear.sum() >= clear.numel() // 2
    assert torch.allclose(position_grad[clear], fd_position[clear], atol=2e-3, rtol=1e-2)
    assert torch.allclose(omega[clear], fd_rotation[clear], atol=5e-3, rtol=1e-2)


def test_no_bound_is_zero(device_cfg):
    problem = Problem(device_cfg.device)
    cost = make_cost(device_cfg, {})
    assert problem.evaluate(cost).abs().sum().item() == 0.0


def test_criteria_update_sets_and_clears(device_cfg):
    problem = Problem(device_cfg.device)
    cost = make_cost(device_cfg, {})
    bound = ToolPoseBound(**BOUNDS["start_all"])
    cost.update_tool_pose_criteria({"ee": ToolPoseCriteria(bound=bound, device_cfg=device_cfg)})
    assert problem.evaluate(cost).sum().item() > 0.0
    cost.update_tool_pose_criteria({"ee": ToolPoseCriteria(device_cfg=device_cfg)})
    assert problem.evaluate(cost).sum().item() == 0.0


def test_unknown_tool_frame_raises(device_cfg):
    cost = make_cost(device_cfg, {})
    with pytest.raises(ValueError):
        cost.update_tool_pose_bounds({"other": ToolPoseBound.plane()})


def test_start_anchor_without_start_raises(device_cfg):
    problem = Problem(device_cfg.device)
    cost = make_cost(device_cfg, {"ee": ToolPoseBound(**BOUNDS["start_all"])})
    tool_poses = ToolPose(
        tool_frames=["ee"], position=problem.position, quaternion=problem.quaternion
    )
    with pytest.raises(ValueError, match="start"):
        cost.forward(tool_poses, problem.goal, problem.idxs)


def test_tolerance_and_binary(device_cfg):
    device = device_cfg.device
    bound = ToolPoseBound.plane("z", anchor="base", tolerance=0.01)
    position = torch.zeros(1, 3, 1, 3, device=device)
    position[0, :, 0, 2] = torch.tensor([0.0, 0.005, 0.02], device=device)
    quaternion = torch.zeros(1, 3, 1, 4, device=device)
    quaternion[..., 0] = 1.0
    tool_poses = ToolPose(tool_frames=["ee"], position=position, quaternion=quaternion)

    optim = make_cost(device_cfg, {"ee": bound})
    expected = 0.5 * torch.tensor([0.0, 0.005, 0.02], device=device) ** 2
    assert torch.allclose(optim.forward(tool_poses)[0, :, 0], expected, atol=1e-8)

    metrics = make_cost(
        device_cfg, {"ee": bound}, weight=5000.0, metric=True, convert_to_binary=True
    )
    feasible = metrics.forward(tool_poses)[0, :, 0]
    assert feasible[0].item() == 0.0 and feasible[1].item() == 0.0
    assert feasible[2].item() > 1.0


def test_per_link_weight(device_cfg):
    device = device_cfg.device
    bound = ToolPoseBound.plane("z", anchor="base")
    cost = make_cost(
        device_cfg, {"a": bound, "b": bound}, weight=[1.0, 3.0], tool_frames=("a", "b")
    )
    position = torch.zeros(1, 1, 2, 3, device=device)
    position[..., 2] = 0.1
    quaternion = torch.zeros(1, 1, 2, 4, device=device)
    quaternion[..., 0] = 1.0
    position.requires_grad_(True)
    tool_poses = ToolPose(tool_frames=["a", "b"], position=position, quaternion=quaternion)
    out = cost.forward(tool_poses)
    assert out[0, 0, 0].item() == pytest.approx(4.0 * 0.5 * 0.01, rel=1e-5)
    out.sum().backward()
    assert position.grad[0, 0, 1, 2].item() == pytest.approx(
        3.0 * position.grad[0, 0, 0, 2].item()
    )


def test_out_of_range_anchor_index_is_ignored(device_cfg):
    """The kernel guards anchor indices instead of reading past the anchor arrays."""
    problem = Problem(device_cfg.device)
    cost = make_cost(device_cfg, {"ee": ToolPoseBound(**BOUNDS["start_all"])})
    tool_poses = ToolPose(
        tool_frames=["ee"], position=problem.position, quaternion=problem.quaternion
    )
    idxs = torch.full_like(problem.idxs, 7)
    out = cost.forward(tool_poses, problem.goal, problem.idxs, problem.start, idxs)
    assert out.abs().sum().item() == 0.0


def test_unused_goal_with_horizon_is_ignored(device_cfg):
    problem = Problem(device_cfg.device)
    cost = make_cost(device_cfg, {"ee": ToolPoseBound(**BOUNDS["base_cone"])})
    goal = GoalToolPose(
        tool_frames=["ee"],
        position=problem.goal.position.expand(2, 3, 1, 2, 3).contiguous(),
        quaternion=problem.goal.quaternion.expand(2, 3, 1, 2, 4).contiguous(),
    )
    tool_poses = ToolPose(
        tool_frames=["ee"], position=problem.position, quaternion=problem.quaternion
    )
    reference = problem.evaluate(cost).clone()
    assert torch.allclose(cost.forward(tool_poses, goal, problem.idxs), reference)
