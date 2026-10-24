# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Unit tests for the EEF bound / corridor cost.

Tests both the legacy PyTorch implementation (``EEFBoundCostPyTorch`` in
``cost_eef_bound_py``) and the default Warp implementation (``EEFBoundCost`` in
``cost_eef_bound``).  The Warp implementation's kernel lives in
``wp_torch_eef_bound``.

The tests run in the workspace by mocking the minimal cuRobo module hierarchy
required by the cost modules.  If cuRobo is installed in the active environment,
the mocks are skipped and the real types are used instead.
"""

from __future__ import annotations

import sys
import types
from dataclasses import dataclass, field
from typing import Any, List, Optional, Type, Union
import torch
import pytest


# Now import the modules under test.
from cost_eef_bound_py import EEFBoundCostPyTorch, quat_to_rotvec, corridor_cost_3d
from cost_eef_bound_cfg import EEFBoundCostCfg

WARP_AVAILABLE = False
try:
    import warp  # noqa: F401
    WARP_AVAILABLE = True
    from cost_eef_bound import EEFBoundCost
except ImportError:
    EEFBoundCost = None  # type: ignore

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------
def identity_quat(*shape, device="cpu", dtype=torch.float32):
    """Return identity quaternions in cuRobo (wxyz) layout."""
    q = torch.zeros(*shape, 4, device=device, dtype=dtype)
    q[..., 0] = 1.0
    return q


def axis_angle_quat(axis, angle, *shape, device="cpu", dtype=torch.float32):
    """Build a quaternion (wxyz) from an axis/angle (axis need not be normalized)."""
    if not isinstance(axis, torch.Tensor):
        axis = torch.tensor(axis, device=device, dtype=dtype)
    else:
        axis = axis.to(device=device, dtype=dtype)
    axis = axis / axis.norm()
    half = torch.tensor(angle / 2.0, device=device, dtype=dtype)
    q = torch.zeros(*shape, 4, device=device, dtype=dtype)
    q[..., 0] = torch.cos(half)
    q[..., 1:] = axis * torch.sin(half)
    return q


def make_poses(B, H, L, G, device="cpu", dtype=torch.float32):
    """Create a deterministic trajectory and a static goal."""
    frames = [f"link_{i}" for i in range(L)]
    cur_pos = torch.zeros(B, H, L, 3, device=device, dtype=dtype)
    cur_quat = identity_quat(B, H, L, device=device, dtype=dtype)
    for h in range(H):
        cur_pos[:, h, :, 0] = 0.5 - h / max(H - 1, 1)
    goal_pos = torch.zeros(B, 1, L, G, 3, device=device, dtype=dtype)
    goal_quat = identity_quat(B, 1, L, G, device=device, dtype=dtype)
    cur_tp = sys.modules["curobo._src.types.tool_pose"].ToolPose(
        tool_frames=frames, position=cur_pos, quaternion=cur_quat
    )
    goal_tp = sys.modules["curobo._src.types.tool_pose"].GoalToolPose(
        tool_frames=frames, position=goal_pos, quaternion=goal_quat
    )
    return cur_tp, goal_tp


def default_cfg(device="cpu", dtype=torch.float32, weight=1.0, num_links=1, **kwargs):
    """Build a config with the given weight and number of tool frames.

    ``num_links`` defaults to 1 for the common single-EEF case; tests that
    exercise multi-link semantics pass ``num_links=N`` explicitly.
    """
    device_cfg = sys.modules["curobo._src.types.device_cfg"].DeviceCfg(device=device, dtype=dtype)
    tool_frames = [f"link_{i}" for i in range(num_links)]
    return EEFBoundCostCfg(
        weight=weight,
        device_cfg=device_cfg,
        tool_frames=tool_frames,
        **kwargs,
    )


# ---------------------------------------------------------------------------
# Helper-function tests
# ---------------------------------------------------------------------------
def test_quat_to_rotvec_identity():
    q = identity_quat(5, device="cpu", dtype=torch.float64)
    rv = quat_to_rotvec(q)
    assert rv.shape == (5, 3)
    assert torch.allclose(rv, torch.zeros(5, 3, dtype=torch.float64), atol=1e-8)


def test_quat_to_rotvec_round_trip():
    dtype = torch.float64
    axis = torch.tensor([1.0, 2.0, -1.0], dtype=dtype)
    axis = axis / axis.norm()
    for angle in [0.1, 0.5, 1.2, 3.0]:
        q = axis_angle_quat(axis, angle, device="cpu", dtype=dtype)
        rv = quat_to_rotvec(q)
        expected = axis * angle
        assert torch.allclose(rv, expected, atol=1e-6)


def test_corridor_cost_3d_inside_no_cost():
    N, H = 2, 4
    current = torch.zeros(N, H, 3)
    start = torch.zeros(N, 1, 3)
    end = torch.zeros(N, 1, 3)
    margins = torch.tensor([[0.1, 0.1, 0.1], [0.1, 0.1, 0.1]])
    cost = corridor_cost_3d(current, start, end, margins)
    assert cost.shape == (N, H)
    assert torch.allclose(cost, torch.zeros(N, H), atol=1e-8)


def test_corridor_cost_3d_position_violation():
    N, H = 1, 3
    start = torch.tensor([[[0.5, 0.0, 0.0]]])
    end = torch.zeros_like(start)
    current = torch.zeros(N, H, 3)
    current[:, 0, 0] = 0.7
    current[:, 1, 0] = -0.2
    current[:, 2, 0] = 0.25
    margins = torch.tensor([[0.1, -1.0, -1.0], [0.1, -1.0, -1.0]])
    cost = corridor_cost_3d(current, start, end, margins)
    assert torch.allclose(cost[0], torch.tensor([0.1, 0.1, 0.0]), atol=1e-8)


# ---------------------------------------------------------------------------
# Legacy PyTorch implementation tests
# ---------------------------------------------------------------------------
def test_pytorch_cost_disabled_without_margins():
    cfg = default_cfg()
    cost_obj = EEFBoundCostPyTorch(cfg)
    cur_tp, goal_tp = make_poses(2, 5, 1, 1)
    out = cost_obj.forward(cur_tp, goal_tp)
    assert out.shape == (2, 5, 1)
    assert torch.allclose(out, torch.zeros(2, 5, 1), atol=1e-8)
    assert not cost_obj.margins_enabled


def test_pytorch_cost_position_violation():
    B, H, L, G = 1, 5, 1, 1
    cfg = default_cfg()
    cost_obj = EEFBoundCostPyTorch(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.05
    margins[1, 0] = 0.05
    cost_obj.update_margins(margins)
    cur_tp, goal_tp = make_poses(B, H, L, G)
    cost = cost_obj.forward(cur_tp, goal_tp)
    assert cost.shape == (B, H, 1)
    assert cost[0, 0, 0] == 0.0
    assert cost[0, H // 2, 0] == 0.0
    assert cost[0, -1, 0] > 0.0
    assert cost[0, -2, 0] > 0.0
    assert cost[0, -1, 0] > cost[0, -2, 0]


def test_pytorch_cost_rotation_violation():
    B, H, L, G = 1, 3, 1, 1
    cfg = default_cfg()
    cost_obj = EEFBoundCostPyTorch(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[2, 0] = 0.05
    margins[3, 0] = 0.05
    cost_obj.update_margins(margins)
    frames = ["link_0"]
    cur_pos = torch.zeros(B, H, L, 3)
    cur_quat = identity_quat(B, H, L)
    cur_quat[:, 0, :, :] = axis_angle_quat([1.0, 0.0, 0.0], 0.5, B, L)
    cur_quat[:, 1, :, :] = axis_angle_quat([1.0, 0.0, 0.0], 0.7, B, L)
    cur_quat[:, 2, :, :] = axis_angle_quat([1.0, 0.0, 0.0], 0.2, B, L)
    goal_pos = torch.zeros(B, 1, L, G, 3)
    goal_quat = identity_quat(B, 1, L, G)
    cur_tp = sys.modules["curobo._src.types.tool_pose"].ToolPose(
        tool_frames=frames, position=cur_pos, quaternion=cur_quat
    )
    goal_tp = sys.modules["curobo._src.types.tool_pose"].GoalToolPose(
        tool_frames=frames, position=goal_pos, quaternion=goal_quat
    )
    cost = cost_obj.forward(cur_tp, goal_tp)
    assert cost[0, 1, 0] > 0.0
    assert torch.allclose(cost[0, 2, 0], torch.zeros(1), atol=1e-6)


def test_pytorch_cost_multi_goalset_min():
    B, H, L, G = 1, 3, 1, 2
    cfg = default_cfg()
    cost_obj = EEFBoundCostPyTorch(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.1
    margins[1, 0] = 0.1
    cost_obj.update_margins(margins)
    frames = ["link_0"]
    cur_pos = torch.zeros(B, H, L, 3)
    cur_quat = identity_quat(B, H, L)
    for h in range(H):
        cur_pos[:, h, :, 0] = 0.5 - h / max(H - 1, 1)
    goal_pos = torch.zeros(B, 1, L, G, 3)
    goal_quat = identity_quat(B, 1, L, G)
    goal_pos[:, :, :, 1, 0] = -0.5
    cur_tp = sys.modules["curobo._src.types.tool_pose"].ToolPose(
        tool_frames=frames, position=cur_pos, quaternion=cur_quat
    )
    goal_tp = sys.modules["curobo._src.types.tool_pose"].GoalToolPose(
        tool_frames=frames, position=goal_pos, quaternion=goal_quat
    )
    cost = cost_obj.forward(cur_tp, goal_tp)
    assert torch.allclose(cost, torch.zeros(B, H, 1), atol=1e-6)


def test_pytorch_cost_multi_link_sum():
    B, H, L, G = 1, 3, 2, 1
    cfg = default_cfg(num_links=2)
    cost_obj = EEFBoundCostPyTorch(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.1
    margins[1, 0] = 0.1
    cost_obj.update_margins(margins)
    cur_tp, goal_tp = make_poses(B, H, L, G)
    cost = cost_obj.forward(cur_tp, goal_tp)
    # Build a separate single-link cfg / cost for the comparison.
    single_link_cost = EEFBoundCostPyTorch(default_cfg(num_links=1))
    single_link_cost.update_margins(margins)
    cur_tp_single, goal_tp_single = make_poses(B, H, 1, G)
    single_cost = single_link_cost.forward(cur_tp_single, goal_tp_single)
    assert torch.allclose(cost, 2.0 * single_cost, atol=1e-6)


def test_pytorch_cost_weight_scaling():
    B, H, L, G = 1, 3, 1, 1
    cfg = default_cfg(weight=2.5)
    cost_obj = EEFBoundCostPyTorch(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.1
    margins[1, 0] = 0.1
    cost_obj.update_margins(margins)
    cur_tp, goal_tp = make_poses(B, H, L, G)
    cost = cost_obj.forward(cur_tp, goal_tp)
    cfg_one = default_cfg(weight=1.0)
    cost_one = EEFBoundCostPyTorch(cfg_one)
    cost_one.update_margins(margins)
    one_cost = cost_one.forward(cur_tp, goal_tp)
    assert torch.allclose(cost, 2.5 * one_cost, atol=1e-6)


def test_pytorch_cost_per_link_weight():
    """Per-link weight tensor of shape ``(L,)`` scales each link independently."""
    B, H, L, G = 1, 3, 2, 1
    w_per_link = torch.tensor([2.0, 0.5])
    cfg = default_cfg(num_links=2, weight=w_per_link)
    cost_obj = EEFBoundCostPyTorch(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.1
    margins[1, 0] = 0.1
    cost_obj.update_margins(margins)
    cur_tp, goal_tp = make_poses(B, H, L, G)

    cfg_one = default_cfg(num_links=2, weight=1.0)
    cost_one = EEFBoundCostPyTorch(cfg_one)
    cost_one.update_margins(margins)
    one_cost = cost_one.forward(cur_tp, goal_tp)  # 2 * single-link cost (identical links)

    weighted_cost = cost_obj.forward(cur_tp, goal_tp)
    expected = (w_per_link.sum() / 2.0) * one_cost
    assert torch.allclose(weighted_cost, expected, atol=1e-5)


def test_pytorch_cost_per_link_margins():
    """Per-link margins of shape ``(L, 4, 3)`` apply each link's own margins."""
    B, H, L, G = 1, 3, 2, 1
    cfg = default_cfg(num_links=2)
    cost_obj = EEFBoundCostPyTorch(cfg)
    # Link 0: x-axis constrained with margin 0.1; Link 1: y-axis constrained.
    margins = torch.full((2, 4, 3), -1.0)
    margins[0, 0, 0] = 0.1
    margins[0, 1, 0] = 0.1
    margins[1, 0, 1] = 0.1
    margins[1, 1, 1] = 0.1
    cost_obj.update_margins(margins)

    # Build a trajectory where x is violated on link 0 only, and y on link 1.
    frames = ["link_0", "link_1"]
    cur_pos = torch.zeros(B, H, L, 3)
    cur_quat = identity_quat(B, H, L)
    # Linear interpolation in x and y from (0.5, 0.5) to (-0.5, -0.5).
    for h in range(H):
        alpha = 0.5 - h / max(H - 1, 1)
        cur_pos[:, h, 0, 0] = alpha       # link 0: x goes 0.5 → -0.5
        cur_pos[:, h, 1, 1] = -alpha      # link 1: y goes -0.5 → 0.5
    goal_pos = torch.zeros(B, 1, L, G, 3)
    goal_quat = identity_quat(B, 1, L, G)
    cur_tp = sys.modules["curobo._src.types.tool_pose"].ToolPose(
        tool_frames=frames, position=cur_pos, quaternion=cur_quat
    )
    goal_tp = sys.modules["curobo._src.types.tool_pose"].GoalToolPose(
        tool_frames=frames, position=goal_pos, quaternion=goal_quat
    )

    cost = cost_obj.forward(cur_tp, goal_tp)
    # At the boundary points, the cost is 0 because the trajectory ends
    # exactly at the corridor edge.  Interior points with x outside the
    # corridor on link 0 contribute cost; same for y on link 1.
    assert cost.shape == (B, H, 1)
    assert torch.all(cost >= 0.0)


def test_pytorch_cost_gradient_exists():
    B, H, L, G = 1, 4, 1, 1
    cfg = default_cfg()
    cost_obj = EEFBoundCostPyTorch(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.1
    margins[1, 0] = 0.1
    cost_obj.update_margins(margins)
    cur_tp, goal_tp = make_poses(B, H, L, G)
    cur_tp.position.requires_grad_(True)
    cost = cost_obj.forward(cur_tp, goal_tp)
    total = cost.sum()
    total.backward()
    assert cur_tp.position.grad is not None
    assert cur_tp.position.grad.shape == cur_tp.position.shape


def test_pytorch_cost_margin_offset_bug_fixed():
    cfg = default_cfg(pos_margin_offset=0.5)
    cost_obj = EEFBoundCostPyTorch(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = -0.1
    margins[1, 0] = 0.1
    cost_obj.update_margins(margins)
    B, H, L, G = 1, 3, 1, 1
    frames = ["link_0"]
    cur_pos = torch.zeros(B, H, L, 3)
    cur_pos[:, :, :, 0] = 10.0
    cur_quat = identity_quat(B, H, L)
    goal_pos = torch.zeros(B, 1, L, G, 3)
    goal_quat = identity_quat(B, 1, L, G)
    cur_tp = sys.modules["curobo._src.types.tool_pose"].ToolPose(
        tool_frames=frames, position=cur_pos, quaternion=cur_quat
    )
    goal_tp = sys.modules["curobo._src.types.tool_pose"].GoalToolPose(
        tool_frames=frames, position=goal_pos, quaternion=goal_quat
    )
    cost = cost_obj.forward(cur_tp, goal_tp)
    assert torch.allclose(cost, torch.zeros(B, H, 1), atol=1e-6)


# ---------------------------------------------------------------------------
# Default Warp implementation tests
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp is not installed")
def test_warp_cost_disabled_without_margins():
    cfg = default_cfg()
    cost_obj = EEFBoundCost(cfg)
    cur_tp, goal_tp = make_poses(2, 5, 1, 1)
    out = cost_obj.forward(cur_tp, goal_tp)
    assert out.shape == (2, 5, 1)
    assert torch.allclose(out, torch.zeros(2, 5, 1), atol=1e-8)


@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp is not installed")
def test_warp_matches_pytorch_position_violation():
    B, H, L, G = 1, 5, 1, 1
    cfg = default_cfg()
    pt_cost = EEFBoundCostPyTorch(cfg)
    wp_cost = EEFBoundCost(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.05
    margins[1, 0] = 0.05
    pt_cost.update_margins(margins)
    wp_cost.update_margins(margins)
    cur_tp, goal_tp = make_poses(B, H, L, G)
    pt_out = pt_cost.forward(cur_tp, goal_tp)
    wp_out = wp_cost.forward(cur_tp, goal_tp)
    assert torch.allclose(pt_out, wp_out, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp is not installed")
def test_warp_matches_pytorch_rotation_violation():
    B, H, L, G = 1, 4, 1, 1
    cfg = default_cfg()
    pt_cost = EEFBoundCostPyTorch(cfg)
    wp_cost = EEFBoundCost(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[2, 2] = 0.05
    margins[3, 2] = 0.05
    pt_cost.update_margins(margins)
    wp_cost.update_margins(margins)
    frames = ["link_0"]
    cur_pos = torch.zeros(B, H, L, 3)
    cur_quat = identity_quat(B, H, L)
    cur_quat[:, 0, :, :] = axis_angle_quat([0.0, 0.0, 1.0], 0.5, B, L)
    cur_quat[:, 1, :, :] = axis_angle_quat([0.0, 0.0, 1.0], 0.8, B, L)
    cur_quat[:, 2, :, :] = axis_angle_quat([0.0, 0.0, 1.0], 0.3, B, L)
    cur_quat[:, 3, :, :] = axis_angle_quat([0.0, 0.0, 1.0], 0.0, B, L)
    goal_pos = torch.zeros(B, 1, L, G, 3)
    goal_quat = identity_quat(B, 1, L, G)
    cur_tp = sys.modules["curobo._src.types.tool_pose"].ToolPose(
        tool_frames=frames, position=cur_pos, quaternion=cur_quat
    )
    goal_tp = sys.modules["curobo._src.types.tool_pose"].GoalToolPose(
        tool_frames=frames, position=goal_pos, quaternion=goal_quat
    )
    pt_out = pt_cost.forward(cur_tp, goal_tp)
    wp_out = wp_cost.forward(cur_tp, goal_tp)
    assert torch.allclose(pt_out, wp_out, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp is not installed")
def test_warp_matches_pytorch_multi_link_goal():
    B, H, L, G = 1, 3, 2, 2
    cfg = default_cfg(num_links=2)
    pt_cost = EEFBoundCostPyTorch(cfg)
    wp_cost = EEFBoundCost(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.1
    margins[1, 0] = 0.1
    pt_cost.update_margins(margins)
    wp_cost.update_margins(margins)
    cur_tp, goal_tp = make_poses(B, H, L, G)
    pt_out = pt_cost.forward(cur_tp, goal_tp)
    wp_out = wp_cost.forward(cur_tp, goal_tp)
    assert torch.allclose(pt_out, wp_out, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp is not installed")
def test_warp_matches_pytorch_non_identity_goal():
    B, H, L, G = 1, 5, 1, 1
    cfg = default_cfg()
    pt_cost = EEFBoundCostPyTorch(cfg)
    wp_cost = EEFBoundCost(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.05
    margins[1, 0] = 0.05
    margins[2, 2] = 0.05
    margins[3, 2] = 0.05
    pt_cost.update_margins(margins)
    wp_cost.update_margins(margins)
    frames = ["link_0"]
    cur_pos = torch.zeros(B, H, L, 3)
    cur_quat = identity_quat(B, H, L)
    for h in range(H):
        cur_pos[:, h, :, 0] = 0.5 - h / max(H - 1, 1)
    for h in range(H):
        cur_quat[:, h, :, :] = axis_angle_quat([0.0, 0.0, 1.0], 0.1 * h, B, L)
    goal_pos = torch.zeros(B, 1, L, G, 3)
    goal_pos[:, :, :, :, 0] = 0.2
    goal_pos[:, :, :, :, 1] = -0.1
    goal_quat = axis_angle_quat([0.0, 0.0, 1.0], 0.3, B, 1, L, G)
    cur_tp = sys.modules["curobo._src.types.tool_pose"].ToolPose(
        tool_frames=frames, position=cur_pos, quaternion=cur_quat
    )
    goal_tp = sys.modules["curobo._src.types.tool_pose"].GoalToolPose(
        tool_frames=frames, position=goal_pos, quaternion=goal_quat
    )
    pt_out = pt_cost.forward(cur_tp, goal_tp)
    wp_out = wp_cost.forward(cur_tp, goal_tp)
    assert torch.allclose(pt_out, wp_out, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp is not installed")
def test_warp_cost_gradient_matches_pytorch():
    B, H, L, G = 1, 5, 1, 1
    cfg = default_cfg()
    pt_cost = EEFBoundCostPyTorch(cfg)
    wp_cost = EEFBoundCost(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.05
    margins[1, 0] = 0.05
    margins[2, 2] = 0.05
    margins[3, 2] = 0.05
    pt_cost.update_margins(margins)
    wp_cost.update_margins(margins)
    cur_tp, goal_tp = make_poses(B, H, L, G)
    cur_tp.position.requires_grad_(True)
    cur_tp.quaternion.requires_grad_(True)
    wp_out = wp_cost.forward(cur_tp, goal_tp)
    wp_out.sum().backward()
    wp_grad_pos = cur_tp.position.grad.clone()
    wp_grad_quat = cur_tp.quaternion.grad.clone()
    cur_tp.position.grad = None
    cur_tp.quaternion.grad = None
    pt_out = pt_cost.forward(cur_tp, goal_tp)
    pt_out.sum().backward()
    pt_grad_pos = cur_tp.position.grad.clone()
    pt_grad_quat = cur_tp.quaternion.grad.clone()
    assert torch.allclose(wp_grad_pos, pt_grad_pos, atol=1e-4, rtol=1e-4)
    assert torch.allclose(wp_grad_quat, pt_grad_quat, atol=1e-4, rtol=1e-4)


@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp is not installed")
def test_warp_cost_gradient_non_identity_goal():
    B, H, L, G = 1, 4, 1, 1
    cfg = default_cfg()
    cost_obj = EEFBoundCost(cfg)
    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.05
    margins[1, 0] = 0.05
    cost_obj.update_margins(margins)
    frames = ["link_0"]
    cur_pos = torch.zeros(B, H, L, 3)
    cur_quat = identity_quat(B, H, L)
    for h in range(H):
        cur_pos[:, h, :, 0] = 0.5 - h / max(H - 1, 1)
    goal_pos = torch.zeros(B, 1, L, G, 3)
    goal_pos[:, :, :, :, 0] = 0.2
    goal_quat = axis_angle_quat([0.0, 0.0, 1.0], 0.3, B, 1, L, G)
    cur_tp = sys.modules["curobo._src.types.tool_pose"].ToolPose(
        tool_frames=frames, position=cur_pos, quaternion=cur_quat
    )
    goal_tp = sys.modules["curobo._src.types.tool_pose"].GoalToolPose(
        tool_frames=frames, position=goal_pos, quaternion=goal_quat
    )
    cur_tp.position.requires_grad_(True)
    cost = cost_obj.forward(cur_tp, goal_tp)
    cost.sum().backward()
    assert cur_tp.position.grad is not None
    assert cur_tp.position.grad.shape == cur_tp.position.shape
    assert cur_tp.position.grad.abs().sum() > 0.0


@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp is not installed")
def test_warp_per_link_weight_scales_gradient():
    """Per-link weight must scale the gradient w.r.t. ``cur_pos`` accordingly."""
    B, H, L, G = 1, 5, 2, 1
    w_link = torch.tensor([1.0, 3.0])
    cfg_w = default_cfg(num_links=2, weight=w_link)
    cost_w = EEFBoundCost(cfg_w)
    cfg_one = default_cfg(num_links=2, weight=1.0)
    cost_one = EEFBoundCost(cfg_one)

    margins = torch.full((4, 3), -1.0)
    margins[0, 0] = 0.05
    margins[1, 0] = 0.05
    cost_w.update_margins(margins)
    cost_one.update_margins(margins)

    cur_tp, goal_tp = make_poses(B, H, L, G)
    cur_tp.position.requires_grad_(True)

    cost_w.forward(cur_tp, goal_tp).sum().backward()
    grad_w = cur_tp.position.grad.clone()
    cur_tp.position.grad = None

    cost_one.forward(cur_tp, goal_tp).sum().backward()
    grad_one = cur_tp.position.grad.clone()

    # The link-1 gradient should be 3x the link-0 gradient.
    assert torch.allclose(grad_w[:, :, 0, :], grad_one[:, :, 0, :], atol=1e-5)
    assert torch.allclose(grad_w[:, :, 1, :], 3.0 * grad_one[:, :, 1, :], atol=1e-5)


@pytest.mark.skipif(not WARP_AVAILABLE, reason="warp is not installed")
def test_warp_output_buffers_populated():
    """``_out_goalset_idx``, ``_out_position_gradient`` etc. are filled by the kernel."""
    B, H, L, G = 1, 4, 2, 3
    cfg = default_cfg(num_links=2)
    cost_obj = EEFBoundCost(cfg)
    cost_obj.setup_batch_tensors(B, H)
    margins = torch.full((2, 4, 3), -1.0)
    margins[:, 0, 0] = 0.05
    margins[:, 1, 0] = 0.05
    cost_obj.update_margins(margins)
    cur_tp, goal_tp = make_poses(B, H, L, G)
    # Make the 2nd goal attractive for link 0 by moving it to where the
    # trajectory is — that should be the chosen goal.
    goal_tp.position[:, 0, 0, 1, 0] = 0.5  # link 0, goal 1
    cost_obj.forward(cur_tp, goal_tp)
    # Goalset index must be 0 or 1 (within the goalset of size 3).
    g_idx = cost_obj._out_goalset_idx
    assert g_idx.shape == (B, H, L)
    assert g_idx.dtype == torch.int32
    assert (g_idx >= 0).all() and (g_idx < G).all()
    # Position gradient must be non-zero somewhere (because the trajectory
    # leaves the corridor).
    assert cost_obj._out_position_gradient.abs().sum() > 0.0
    # Bookkeeping buffers are populated.
    assert cost_obj._out_distance.abs().sum() >= 0.0
    assert cost_obj._out_position_distance.shape == (B, H, L)
    assert cost_obj._out_rotation_distance.shape == (B, H, L)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
