# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU-only verification of :class:`EEFBoundCost` (Warp kernel on the ``cpu`` device).

NOT for main: see ``cpu_verify/README.md``.

Tests marked ``xfail(strict=True)`` encode defects found in review (see ``REVIEW.md``). They
must fail today; once a defect is fixed the test XPASSes, strict mode turns that into a
failure, and the marker must be removed so the check becomes a regression test.

Run::

    python -m pytest cpu_verify
"""

from __future__ import annotations

from typing import Tuple

import pytest
import torch

from cpu_adapter import (
    EEFBoundCost,
    EEFBoundCostPyTorch,
    finite_difference_world,
    gather_goal_for_seeds,
    init,
    make_cost,
    make_poses,
    quat_axis_angle,
    quat_grad_to_world_omega,
    random_quat,
    single_link_problem,
)
from curobo._src.cost.cost_eef_bound_py import quat_mul_wxyz

IDENTITY = quat_axis_angle([1.0, 0.0, 0.0], 0.0)
ORIGIN = torch.zeros(3)


@pytest.fixture(scope="module", autouse=True)
def _warp_cpu():
    init()


def _rotation_margins(value: float = 0.02) -> torch.Tensor:
    margins = torch.full((4, 3), -1.0)
    margins[2:] = value
    return margins


def _position_margins(value: float = 0.05) -> torch.Tensor:
    margins = torch.full((4, 3), -1.0)
    margins[:2] = value
    return margins


def _cost_at(
    cost, start_p: torch.Tensor, start_q: torch.Tensor, goal_p: torch.Tensor, goal_q: torch.Tensor
):
    """Return ``f(position, quaternion) -> cost at t=1`` for a fixed start and goal."""

    def fn(position: torch.Tensor, quaternion: torch.Tensor) -> float:
        tp, gp = single_link_problem(start_p, start_q, position, quaternion, goal_p, goal_q)
        return cost.forward(tp, gp)[0, 1, 0].item()

    return fn


def _analytic_world_grad(
    cost,
    position: torch.Tensor,
    quaternion: torch.Tensor,
    start_p: torch.Tensor,
    start_q: torch.Tensor,
    goal_p: torch.Tensor,
    goal_q: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return the cost's position gradient and world-frame angular gradient at ``t=1``."""
    tp, gp = single_link_problem(start_p, start_q, position, quaternion, goal_p, goal_q)
    tp.position.requires_grad_(True)
    tp.quaternion.requires_grad_(True)
    cost.forward(tp, gp).sum().backward()
    grad_pos = tp.position.grad[0, 1, 0]
    omega = quat_grad_to_world_omega(tp.quaternion.detach()[0, 1, 0], tp.quaternion.grad[0, 1, 0])
    return grad_pos, omega


# ---------------------------------------------------------------------------
# Expected to pass
# ---------------------------------------------------------------------------
def test_forward_warp_matches_torch_random():
    """Warp and PyTorch costs agree on random multi-link, multi-goal, rotated-goal inputs."""
    gen = torch.Generator().manual_seed(0)
    B, H, L, G = 6, 7, 2, 3
    margins = torch.rand(L, 4, 3, generator=gen) * 0.2
    margins[torch.rand(L, 4, 3, generator=gen) < 0.3] = -1.0
    frames = [f"link_{i}" for i in range(L)]
    wp_cost = make_cost(EEFBoundCost, margins, tool_frames=frames)
    pt_cost = make_cost(EEFBoundCostPyTorch, margins, tool_frames=frames)
    weight = torch.tensor([1.0, 2.5])
    wp_cost._weight = weight.clone()
    pt_cost._weight = weight.clone()

    pos = torch.randn(B, H, L, 3, generator=gen) * 0.3
    quat = random_quat(B * H * L, gen).view(B, H, L, 4)
    goal_pos = torch.randn(B, 1, L, G, 3, generator=gen) * 0.3
    goal_quat = random_quat(B * L * G, gen).view(B, 1, L, G, 4)
    tp, gp = make_poses(pos, quat, goal_pos, goal_quat)

    assert torch.allclose(wp_cost.forward(tp, gp), pt_cost.forward(tp, gp), atol=1e-4, rtol=1e-4)


def test_position_gradient_matches_finite_difference():
    """Position gradient equals world-frame finite differences for rotated goals."""
    gen = torch.Generator().manual_seed(1)
    cost = make_cost(EEFBoundCost, _position_margins())
    checked = 0
    for _ in range(40):
        goal_p = torch.randn(3, generator=gen) * 0.2
        goal_q = random_quat(1, gen)[0]
        start_p = goal_p + torch.randn(3, generator=gen) * 0.4
        position = goal_p + torch.randn(3, generator=gen) * 0.6
        fn = _cost_at(cost, start_p, goal_q, goal_p, goal_q)
        fd, _ = finite_difference_world(fn, position, goal_q)
        fd_half, _ = finite_difference_world(fn, position, goal_q, eps=5e-4)
        if not torch.allclose(fd, fd_half, atol=1e-2):
            continue  # within eps of a corridor kink; the derivative is not defined there
        grad_pos, _ = _analytic_world_grad(cost, position, goal_q, start_p, goal_q, goal_p, goal_q)
        assert torch.allclose(grad_pos, fd, atol=2e-2), (grad_pos, fd)
        checked += 1
    assert checked >= 20


def test_rotation_gradient_world_frame_identity_goal():
    """With an identity goal, the rotation gradient points along the world-frame descent."""
    cost = make_cost(EEFBoundCost, _rotation_margins())
    quaternion = quat_axis_angle([1.0, 0.0, 0.0], 0.3)
    fn = _cost_at(cost, ORIGIN, IDENTITY, ORIGIN, IDENTITY)
    _, fd = finite_difference_world(fn, ORIGIN, quaternion)
    _, omega = _analytic_world_grad(cost, ORIGIN, quaternion, ORIGIN, IDENTITY, ORIGIN, IDENTITY)
    assert torch.allclose(omega, fd, atol=2e-2), (omega, fd)


def test_gathered_goal_matches_per_problem_evaluation():
    """Gathering goals by ``idxs_link_pose`` before the call reproduces per-problem costs."""
    gen = torch.Generator().manual_seed(2)
    cost = make_cost(EEFBoundCost, _position_margins())
    B, H = 4, 5
    pos = torch.randn(B, H, 1, 3, generator=gen) * 0.3
    quat = random_quat(B * H, gen).view(B, H, 1, 4)
    goal_pos = torch.randn(2, 1, 1, 1, 3, generator=gen) * 0.3
    goal_quat = random_quat(2, gen).view(2, 1, 1, 1, 4)
    idxs = torch.tensor([0, 0, 1, 1], dtype=torch.int32)
    tp, gp = make_poses(pos, quat, goal_pos, goal_quat)
    batched = cost.forward(tp, gather_goal_for_seeds(gp, idxs)).clone()
    for b in range(B):
        g = int(idxs[b])
        tp_b, gp_b = make_poses(pos[b : b + 1], quat[b : b + 1], goal_pos[g : g + 1], goal_quat[g : g + 1])
        assert torch.allclose(batched[b : b + 1], cost.forward(tp_b, gp_b), atol=1e-5)


# ---------------------------------------------------------------------------
# Known defects (strict xfail). Remove the marker together with the fix.
# ---------------------------------------------------------------------------
@pytest.mark.xfail(
    strict=True,
    reason="D1: rotation gradient is c*(v_goal,0) instead of c*(R_goal v_goal,0); "
    "the goal-frame gradient is never rotated to world frame",
)
@pytest.mark.parametrize("goal_yaw", [0.5, 1.2, 2.5])
def test_rotation_gradient_world_frame_rotated_goal(goal_yaw: float):
    cost = make_cost(EEFBoundCost, _rotation_margins())
    goal_q = quat_axis_angle([0.0, 0.0, 1.0], goal_yaw)
    # 0.3 rad about the goal's own x axis: a single-axis violation, so the log-map
    # Jacobian is exact and any mismatch is the frame error alone.
    quaternion = quat_mul_wxyz(goal_q, quat_axis_angle([1.0, 0.0, 0.0], 0.3))
    fn = _cost_at(cost, ORIGIN, goal_q, ORIGIN, goal_q)
    _, fd = finite_difference_world(fn, ORIGIN, quaternion)
    _, omega = _analytic_world_grad(cost, ORIGIN, quaternion, ORIGIN, goal_q, ORIGIN, goal_q)
    assert torch.allclose(omega, fd, atol=2e-2), (omega, fd)


@pytest.mark.xfail(
    strict=True,
    reason="D2: disable_margin=True zeroes every margin, so disabled (negative) axes "
    "become hard zero-width constraints",
)
def test_disable_margin_keeps_disabled_axes_off():
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.1
    margins[1, 0] = 0.1  # only x position is constrained
    cost = make_cost(EEFBoundCost, margins, disable_margin=True)
    start = torch.tensor([0.0, 0.0, 0.5])
    tp, gp = single_link_problem(start, IDENTITY, torch.tensor([0.0, 0.3, 0.5]), IDENTITY, ORIGIN, IDENTITY)
    assert cost.forward(tp, gp)[0, 1, 0].item() == pytest.approx(0.0, abs=1e-6)


@pytest.mark.xfail(
    strict=True,
    reason="D3: PyTorch reference does not detach the t=0 start pose, so gradient leaks "
    "into position[:, 0]; Warp treats the start as a constant",
)
def test_reference_matches_warp_gradient_at_start():
    margins = _position_margins()
    wp_cost = make_cost(EEFBoundCost, margins)
    pt_cost = make_cost(EEFBoundCostPyTorch, margins)
    # Start at x=0.5, goal at 0, evaluated pose beyond the start: begin side active.
    grads = []
    for cost in (wp_cost, pt_cost):
        tp, gp = single_link_problem(
            torch.tensor([0.5, 0.0, 0.0]), IDENTITY, torch.tensor([0.8, 0.0, 0.0]), IDENTITY, ORIGIN, IDENTITY
        )
        tp.position.requires_grad_(True)
        cost.forward(tp, gp).sum().backward()
        grads.append(tp.position.grad.clone())
    assert torch.allclose(grads[0], grads[1], atol=1e-5), grads


@pytest.mark.xfail(
    strict=True,
    raises=RuntimeError,
    reason="D4: goals are indexed by rollout batch row; with num_seeds > 1 the goal batch is "
    "smaller and the reshape fails (ToolPoseCost uses goal.idxs_link_pose instead)",
)
def test_goal_batch_smaller_than_rollout_batch():
    cost = make_cost(EEFBoundCost, _position_margins())
    B, H = 4, 3
    pos = torch.zeros(B, H, 1, 3)
    pos[:, 1:, 0, 0] = -0.5
    quat = IDENTITY.expand(B, H, 1, 4).contiguous()
    goal_pos = torch.zeros(2, 1, 1, 1, 3)
    goal_quat = IDENTITY.expand(2, 1, 1, 1, 4).contiguous()
    tp, gp = make_poses(pos, quat, goal_pos, goal_quat)
    expected = cost.forward(tp, gather_goal_for_seeds(gp, torch.tensor([0, 0, 1, 1]))).clone()
    assert torch.allclose(cost.forward(tp, gp), expected)
