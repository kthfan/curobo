# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unit tests for EEFBoundCost and EEFBoundCostCfg."""

import math

import pytest
import torch

from curobo._src.cost.cost_eef_bound import (
    EEFBoundCost,
    _eef_position_bound_cost,
    _eef_rotation_bound_cost,
    _quat_to_rotvec,
)
from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg
from curobo._src.types.device_cfg import DeviceCfg


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _identity_quat(shape, device, dtype):
    """Return identity quaternion (w,x,y,z) = (1,0,0,0) with given shape."""
    q = torch.zeros(*shape, 4, device=device, dtype=dtype)
    q[..., 0] = 1.0
    return q


def _rot_z_quat(angle_rad: float, device, dtype) -> torch.Tensor:
    """Return a (1, 4) wxyz quaternion for a rotation of *angle_rad* about Z."""
    half = angle_rad * 0.5
    return torch.tensor(
        [[math.cos(half), 0.0, 0.0, math.sin(half)]],
        device=device,
        dtype=dtype,
    )


def _all_disabled_margins(device, dtype) -> torch.Tensor:
    """Return (4, 3) margins tensor with all axes disabled (all -1)."""
    return torch.full((4, 3), -1.0, device=device, dtype=dtype)


def _all_zero_margins(device, dtype) -> torch.Tensor:
    """Return (4, 3) margins tensor with all axes constrained, zero dead-zone."""
    return torch.zeros((4, 3), device=device, dtype=dtype)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(params=["cpu", "cuda:0"])
def device_cfg(request):
    device = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(device))


# ---------------------------------------------------------------------------
# EEFBoundCostCfg tests
# ---------------------------------------------------------------------------

class TestEEFBoundCostCfg:

    def test_init_float_weight(self, device_cfg):
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        assert cfg.weight.shape == (1,)
        assert torch.allclose(cfg.weight, torch.tensor([1.0], device=device_cfg.device, dtype=device_cfg.dtype))

    def test_init_list_weight(self, device_cfg):
        cfg = EEFBoundCostCfg(weight=[2.0], device_cfg=device_cfg)
        assert torch.allclose(cfg.weight, torch.tensor([2.0], device=device_cfg.device, dtype=device_cfg.dtype))

    def test_class_type_is_eef_bound_cost(self, device_cfg):
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        assert cfg.class_type is EEFBoundCost

    def test_default_flags(self, device_cfg):
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        assert cfg.disable_margin is False
        assert cfg.pos_margin_offset == 0.0
        assert cfg.rot_margin_offset == 0.0

    def test_custom_flags(self, device_cfg):
        cfg = EEFBoundCostCfg(
            weight=1.0,
            disable_margin=True,
            pos_margin_offset=0.05,
            rot_margin_offset=0.1,
            device_cfg=device_cfg,
        )
        assert cfg.disable_margin is True
        assert cfg.pos_margin_offset == pytest.approx(0.05)
        assert cfg.rot_margin_offset == pytest.approx(0.1)

    def test_int_weight_raises(self, device_cfg):
        with pytest.raises(Exception):
            EEFBoundCostCfg(weight=1, device_cfg=device_cfg)


# ---------------------------------------------------------------------------
# EEFBoundCost initialisation and update_margins
# ---------------------------------------------------------------------------

class TestEEFBoundCostInit:

    def test_init_disabled_by_default(self, device_cfg):
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = EEFBoundCost(cfg)
        assert cost.margins_enabled is False
        assert cost.enabled is True  # BaseCost: weight > 0, so enabled

    def test_margins_all_negative_on_construction(self, device_cfg):
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = EEFBoundCost(cfg)
        assert torch.all(cost._margins < 0)

    def test_update_margins_enables(self, device_cfg):
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = EEFBoundCost(cfg)
        margins = _all_zero_margins(device_cfg.device, device_cfg.dtype)
        cost.update_margins(margins)
        assert cost.margins_enabled is True

    def test_update_margins_none_disables(self, device_cfg):
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = EEFBoundCost(cfg)
        cost.update_margins(_all_zero_margins(device_cfg.device, device_cfg.dtype))
        cost.update_margins(None)
        assert cost.margins_enabled is False
        assert torch.all(cost._margins < 0)

    def test_update_margins_copies_values(self, device_cfg):
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = EEFBoundCost(cfg)
        margins = torch.tensor(
            [[0.01, 0.01, -1.0], [0.01, 0.01, -1.0],
             [-1.0, -1.0, -1.0], [-1.0, -1.0, -1.0]],
            device=device_cfg.device, dtype=device_cfg.dtype,
        )
        cost.update_margins(margins)
        assert torch.allclose(cost._margins, margins)


# ---------------------------------------------------------------------------
# EEFBoundCost.forward – disabled path
# ---------------------------------------------------------------------------

class TestEEFBoundCostForwardDisabled:

    def test_forward_returns_zeros_when_not_enabled(self, device_cfg):
        """When margins have never been set, forward must return all-zero cost."""
        B, T = 3, 8
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = EEFBoundCost(cfg)

        ee_pos = torch.randn(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        assert out.shape == (B, T)
        assert torch.all(out == 0.0)

    def test_forward_returns_zeros_after_disable(self, device_cfg):
        """After calling update_margins(None), forward must return zeros."""
        B, T = 2, 5
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg)
        cost = EEFBoundCost(cfg)
        cost.update_margins(_all_zero_margins(device_cfg.device, device_cfg.dtype))
        cost.update_margins(None)

        ee_pos = torch.randn(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        assert torch.all(out == 0.0)


# ---------------------------------------------------------------------------
# EEFBoundCost.forward – position cost
# ---------------------------------------------------------------------------

class TestEEFBoundCostForwardPosition:
    """Tests using identity goal orientation so goal frame == world frame."""

    def _make_cost(self, device_cfg, weight=1.0, **cfg_kwargs):
        cfg = EEFBoundCostCfg(weight=weight, device_cfg=device_cfg, **cfg_kwargs)
        return EEFBoundCost(cfg)

    def _pos_only_margins(self, device, dtype):
        """Constrain position (zero dead-zone), leave rotation disabled."""
        m = torch.zeros(4, 3, device=device, dtype=dtype)
        m[2:] = -1.0   # disable rotation rows
        return m

    def test_trajectory_at_start_has_zero_cost(self, device_cfg):
        """
        All timesteps equal the start pose.
        In goal frame: start_pos ≠ 0, but every step equals start_pos.
        The bound is [end=0, start], so a step that equals start is exactly
        on the start boundary (margin = 0 → no dead-zone) → zero violation.
        """
        B, T = 2, 6
        cost = self._make_cost(device_cfg)
        margins = self._pos_only_margins(device_cfg.device, device_cfg.dtype)
        cost.update_margins(margins)

        start = torch.tensor([0.3, 0.0, 0.0], device=device_cfg.device, dtype=device_cfg.dtype)
        ee_pos = start.view(1, 1, 3).expand(B, T, 3).clone()
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        assert out.shape == (B, T)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_trajectory_at_goal_has_zero_cost(self, device_cfg):
        """All timesteps at origin (goal). Bound includes goal → zero cost."""
        B, T = 2, 6
        cost = self._make_cost(device_cfg)
        margins = self._pos_only_margins(device_cfg.device, device_cfg.dtype)
        cost.update_margins(margins)

        start = torch.tensor([0.3, 0.0, 0.0], device=device_cfg.device, dtype=device_cfg.dtype)
        # t=0 must be start; rest at goal (origin)
        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_pos[:, 0] = start
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        # non-start steps are at goal (origin) → within [0, 0.3] → zero
        assert torch.allclose(out[:, 1:], torch.zeros_like(out[:, 1:]), atol=1e-5)

    def test_trajectory_outside_start_bound_has_positive_cost(self, device_cfg):
        """
        Start at x=0.3 (goal at origin), so bound is x ∈ [0, 0.3].
        Moving to x=0.5 violates the start bound → positive cost.
        """
        B, T = 2, 4
        cost = self._make_cost(device_cfg)
        margins = self._pos_only_margins(device_cfg.device, device_cfg.dtype)
        cost.update_margins(margins)

        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_pos[:, 0, 0] = 0.3   # start at x=0.3
        ee_pos[:, 1, 0] = 0.5   # violates upper bound (x > 0.3)
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        # step t=1 should have positive cost
        assert torch.all(out[:, 1] > 0.0)

    def test_trajectory_outside_goal_bound_has_positive_cost(self, device_cfg):
        """
        Start at x=0.3, goal at origin, bound x ∈ [0, 0.3].
        Moving to x=-0.1 violates the goal-side lower bound → positive cost.
        """
        B, T = 2, 4
        cost = self._make_cost(device_cfg)
        margins = self._pos_only_margins(device_cfg.device, device_cfg.dtype)
        cost.update_margins(margins)

        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_pos[:, 0, 0] = 0.3   # start
        ee_pos[:, 1, 0] = -0.1  # violates lower bound (x < 0)
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        assert torch.all(out[:, 1] > 0.0)

    def test_disabled_axes_not_penalised(self, device_cfg):
        """When an axis margin is negative, arbitrarily large movement is not penalised."""
        B, T = 2, 4
        cost = self._make_cost(device_cfg)

        # Only constrain X; Y and Z disabled
        margins = torch.full((4, 3), -1.0, device=device_cfg.device, dtype=device_cfg.dtype)
        margins[0, 0] = 0.0   # position start-side X
        margins[1, 0] = 0.0   # position goal-side X
        cost.update_margins(margins)

        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_pos[:, 0, 0] = 0.3   # start x=0.3
        # Stay within X bound, but go far in Y and Z
        ee_pos[:, 1] = torch.tensor([0.15, 100.0, 100.0], device=device_cfg.device, dtype=device_cfg.dtype)
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        # t=1: X is within [0, 0.3], Y/Z unconstrained → zero cost
        assert torch.allclose(out[:, 1], torch.zeros(B, device=device_cfg.device, dtype=device_cfg.dtype), atol=1e-5)

    def test_weight_scales_cost(self, device_cfg):
        """Cost with weight=2 should be exactly 2× cost with weight=1."""
        B, T = 2, 4
        margins = self._pos_only_margins(device_cfg.device, device_cfg.dtype)

        cost1 = self._make_cost(device_cfg, weight=1.0)
        cost1.update_margins(margins)
        cost2 = self._make_cost(device_cfg, weight=2.0)
        cost2.update_margins(margins)

        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_pos[:, 0, 0] = 0.3
        ee_pos[:, 1, 0] = 0.5   # out of bounds
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out1 = cost1.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        out2 = cost2.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        assert torch.allclose(out2, 2.0 * out1, atol=1e-5)

    def test_disable_margin_ignores_offset(self, device_cfg):
        """With disable_margin=True, pos_margin_offset is ignored."""
        B, T = 2, 4
        margins = self._pos_only_margins(device_cfg.device, device_cfg.dtype)

        # disable_margin=True: margin widening is skipped
        cost_no_margin = self._make_cost(device_cfg, disable_margin=True, pos_margin_offset=0.5)
        cost_no_margin.update_margins(margins)

        # Plain cost (no offset, no widening)
        cost_plain = self._make_cost(device_cfg)
        cost_plain.update_margins(margins)

        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_pos[:, 0, 0] = 0.3
        ee_pos[:, 1, 0] = 0.5   # outside [0, 0.3]
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out_no_margin = cost_no_margin.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        out_plain = cost_plain.forward(ee_pos, ee_rot, goal_pos, goal_rot)

        # Both treat margin as 0 → results must be identical
        assert torch.allclose(out_no_margin, out_plain, atol=1e-5)

    def test_pos_margin_offset_creates_dead_zone(self, device_cfg):
        """A pos_margin_offset should allow slight violations without penalty."""
        B, T = 2, 4
        offset = 0.05
        margins = self._pos_only_margins(device_cfg.device, device_cfg.dtype)  # base = 0

        cost = self._make_cost(device_cfg, pos_margin_offset=offset)
        cost.update_margins(margins)

        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_pos[:, 0, 0] = 0.3        # start
        # t=1 slightly beyond start: 0.3 + 0.03 = 0.33, within 0.05 dead-zone
        ee_pos[:, 1, 0] = 0.33
        # t=2 far beyond dead-zone: 0.3 + 0.1 = 0.4 > 0.35
        ee_pos[:, 2, 0] = 0.4
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        # Within dead-zone → zero
        assert torch.allclose(out[:, 1], torch.zeros(B, device=device_cfg.device, dtype=device_cfg.dtype), atol=1e-5)
        # Beyond dead-zone → positive
        assert torch.all(out[:, 2] > 0.0)


# ---------------------------------------------------------------------------
# EEFBoundCost.forward – rotation cost
# ---------------------------------------------------------------------------

class TestEEFBoundCostForwardRotation:

    def _rot_only_margins(self, device, dtype):
        """Constrain rotation (zero dead-zone), leave position disabled."""
        m = torch.zeros(4, 3, device=device, dtype=dtype)
        m[:2] = -1.0   # disable position rows
        return m

    def _make_cost(self, device_cfg, weight=1.0, **cfg_kwargs):
        cfg = EEFBoundCostCfg(weight=weight, device_cfg=device_cfg, **cfg_kwargs)
        return EEFBoundCost(cfg)

    def test_identity_trajectory_has_zero_rotation_cost(self, device_cfg):
        """Trajectory at identity rotation, goal at identity → zero cost."""
        B, T = 2, 6
        cost = self._make_cost(device_cfg)
        cost.update_margins(self._rot_only_margins(device_cfg.device, device_cfg.dtype))

        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_rotation_outside_bound_has_positive_cost(self, device_cfg):
        """
        Start with a 90° rotation around Z; goal = identity.
        Bound is rotvec_z ∈ [0, π/2].  Moving to 100° violates start bound.
        """
        B, T = 2, 4
        cost = self._make_cost(device_cfg)
        cost.update_margins(self._rot_only_margins(device_cfg.device, device_cfg.dtype))

        start_q = _rot_z_quat(math.pi / 2, device_cfg.device, device_cfg.dtype)  # 90°
        exceed_q = _rot_z_quat(math.radians(100), device_cfg.device, device_cfg.dtype)  # 100°

        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        ee_rot[:, 0] = start_q   # t=0: start
        ee_rot[:, 1] = exceed_q  # t=1: beyond start bound

        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        assert torch.all(out[:, 1] > 0.0)

    def test_rotation_within_bound_has_zero_cost(self, device_cfg):
        """
        Start 90°, goal identity.  Intermediate 45° is inside [0°, 90°] → zero cost.
        """
        B, T = 2, 4
        cost = self._make_cost(device_cfg)
        cost.update_margins(self._rot_only_margins(device_cfg.device, device_cfg.dtype))

        start_q = _rot_z_quat(math.pi / 2, device_cfg.device, device_cfg.dtype)
        mid_q = _rot_z_quat(math.pi / 4, device_cfg.device, device_cfg.dtype)

        ee_pos = torch.zeros(B, T, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        ee_rot = _identity_quat((B, T), device_cfg.device, device_cfg.dtype)
        ee_rot[:, 0] = start_q
        ee_rot[:, 1] = mid_q   # within [0, π/2]

        goal_pos = torch.zeros(B, 3, device=device_cfg.device, dtype=device_cfg.dtype)
        goal_rot = _identity_quat((B,), device_cfg.device, device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        assert torch.allclose(out[:, 1], torch.zeros(B, device=device_cfg.device, dtype=device_cfg.dtype), atol=1e-4)


# ---------------------------------------------------------------------------
# _quat_to_rotvec helper
# ---------------------------------------------------------------------------

class TestQuatToRotvec:
    """Tests for the module-private _quat_to_rotvec helper."""

    @pytest.fixture(params=["cpu", "cuda:0"])
    def device(self, request):
        if request.param.startswith("cuda") and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device(request.param)

    def test_identity_maps_to_zero(self, device):
        q = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device, dtype=torch.float32)
        rv = _quat_to_rotvec(q)
        assert torch.allclose(rv, torch.zeros(1, 3, device=device, dtype=torch.float32), atol=1e-5)

    def test_90_deg_z_rotation(self, device):
        """90° around Z: rotvec should be (0, 0, π/2)."""
        angle = math.pi / 2
        q = torch.tensor(
            [[math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]],
            device=device, dtype=torch.float32,
        )
        rv = _quat_to_rotvec(q)
        expected = torch.tensor([[0.0, 0.0, angle]], device=device, dtype=torch.float32)
        assert torch.allclose(rv, expected, atol=1e-4)

    def test_180_deg_x_rotation(self, device):
        """180° around X: rotvec should be (π, 0, 0)."""
        angle = math.pi
        q = torch.tensor(
            [[math.cos(angle / 2), math.sin(angle / 2), 0.0, 0.0]],
            device=device, dtype=torch.float32,
        )
        rv = _quat_to_rotvec(q)
        expected = torch.tensor([[angle, 0.0, 0.0]], device=device, dtype=torch.float32)
        assert torch.allclose(rv, expected, atol=1e-4)

    def test_negative_w_shortest_arc(self, device):
        """Quaternion with w < 0 must be flipped to shortest arc before conversion."""
        angle = math.pi / 3  # 60°
        half = angle / 2
        q_pos = torch.tensor(
            [[math.cos(half), 0.0, math.sin(half), 0.0]],
            device=device, dtype=torch.float32,
        )
        # Negated quaternion represents same rotation
        q_neg = -q_pos
        rv_pos = _quat_to_rotvec(q_pos)
        rv_neg = _quat_to_rotvec(q_neg)
        assert torch.allclose(rv_pos, rv_neg, atol=1e-5)

    def test_batch_shape_preserved(self, device):
        """Output shape must be (..., 3)."""
        q = torch.rand(4, 6, 4, device=device, dtype=torch.float32)
        q = q / q.norm(dim=-1, keepdim=True)
        q[..., 0] = q[..., 0].abs()   # keep w positive
        rv = _quat_to_rotvec(q)
        assert rv.shape == (4, 6, 3)


# ---------------------------------------------------------------------------
# _eef_position_bound_cost functional
# ---------------------------------------------------------------------------

class TestEEFPositionBoundCostFunctional:

    @pytest.fixture(params=["cpu", "cuda:0"])
    def device(self, request):
        if request.param.startswith("cuda") and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device(request.param)

    def test_output_shape(self, device):
        B, T = 3, 7
        dtype = torch.float32
        ee_pos = torch.zeros(B, T, 3, device=device, dtype=dtype)
        ee_rot = _identity_quat((B, T), device, dtype)
        goal_pos = torch.zeros(B, 3, device=device, dtype=dtype)
        goal_rot = _identity_quat((B,), device, dtype)
        margins = torch.zeros(2, 3, device=device, dtype=dtype)
        out = _eef_position_bound_cost(ee_pos, ee_rot, goal_pos, goal_rot, margins)
        assert out.shape == (B, T)

    def test_all_disabled_margins_zero_cost(self, device):
        """All-negative margins → no constraint → zero cost even for wild trajectories."""
        B, T = 2, 5
        dtype = torch.float32
        ee_pos = torch.randn(B, T, 3, device=device, dtype=dtype) * 10.0
        ee_rot = _identity_quat((B, T), device, dtype)
        goal_pos = torch.zeros(B, 3, device=device, dtype=dtype)
        goal_rot = _identity_quat((B,), device, dtype)
        margins = torch.full((2, 3), -1.0, device=device, dtype=dtype)
        out = _eef_position_bound_cost(ee_pos, ee_rot, goal_pos, goal_rot, margins)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_output_non_negative(self, device):
        """Cost must always be ≥ 0."""
        B, T = 4, 10
        dtype = torch.float32
        torch.manual_seed(0)
        ee_pos = torch.randn(B, T, 3, device=device, dtype=dtype)
        ee_rot = _identity_quat((B, T), device, dtype)
        goal_pos = torch.zeros(B, 3, device=device, dtype=dtype)
        goal_rot = _identity_quat((B,), device, dtype)
        margins = torch.zeros(2, 3, device=device, dtype=dtype)
        out = _eef_position_bound_cost(ee_pos, ee_rot, goal_pos, goal_rot, margins)
        assert torch.all(out >= 0.0)


# ---------------------------------------------------------------------------
# _eef_rotation_bound_cost functional
# ---------------------------------------------------------------------------

class TestEEFRotationBoundCostFunctional:

    @pytest.fixture(params=["cpu", "cuda:0"])
    def device(self, request):
        if request.param.startswith("cuda") and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return torch.device(request.param)

    def test_output_shape(self, device):
        B, T = 3, 7
        dtype = torch.float32
        ee_pos = torch.zeros(B, T, 3, device=device, dtype=dtype)
        ee_rot = _identity_quat((B, T), device, dtype)
        goal_pos = torch.zeros(B, 3, device=device, dtype=dtype)
        goal_rot = _identity_quat((B,), device, dtype)
        margins = torch.zeros(2, 3, device=device, dtype=dtype)
        out = _eef_rotation_bound_cost(ee_pos, ee_rot, goal_pos, goal_rot, margins)
        assert out.shape == (B, T)

    def test_all_disabled_margins_zero_cost(self, device):
        B, T = 2, 5
        dtype = torch.float32
        ee_pos = torch.zeros(B, T, 3, device=device, dtype=dtype)
        ee_rot = torch.rand(B, T, 4, device=device, dtype=dtype)
        ee_rot = ee_rot / ee_rot.norm(dim=-1, keepdim=True)
        goal_pos = torch.zeros(B, 3, device=device, dtype=dtype)
        goal_rot = _identity_quat((B,), device, dtype)
        margins = torch.full((2, 3), -1.0, device=device, dtype=dtype)
        out = _eef_rotation_bound_cost(ee_pos, ee_rot, goal_pos, goal_rot, margins)
        assert torch.allclose(out, torch.zeros_like(out), atol=1e-5)

    def test_output_non_negative(self, device):
        B, T = 4, 8
        dtype = torch.float32
        torch.manual_seed(1)
        ee_pos = torch.zeros(B, T, 3, device=device, dtype=dtype)
        ee_rot = torch.rand(B, T, 4, device=device, dtype=dtype)
        ee_rot = ee_rot / ee_rot.norm(dim=-1, keepdim=True)
        goal_pos = torch.zeros(B, 3, device=device, dtype=dtype)
        goal_rot = _identity_quat((B,), device, dtype)
        margins = torch.zeros(2, 3, device=device, dtype=dtype)
        out = _eef_rotation_bound_cost(ee_pos, ee_rot, goal_pos, goal_rot, margins)
        assert torch.all(out >= 0.0)


# ---------------------------------------------------------------------------
# Gradient tests
# ---------------------------------------------------------------------------

class TestEEFBoundCostGradients:

    @pytest.fixture
    def cuda_device_cfg(self):
        if not torch.cuda.is_available():
            pytest.skip("CUDA required for gradient tests")
        return DeviceCfg(device=torch.device("cuda:0"))

    def test_position_gradient_flows(self, cuda_device_cfg):
        """Gradients must flow back through ee_pos_batch."""
        B, T = 2, 4
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        cost = EEFBoundCost(cfg)
        margins = torch.zeros(4, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        cost.update_margins(margins)

        ee_pos = torch.zeros(
            B, T, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype,
            requires_grad=True,
        )
        with torch.no_grad():
            ee_pos.data[:, 0, 0] = 0.3   # start
            ee_pos.data[:, 1, 0] = 0.5   # out of bounds

        ee_rot = _identity_quat((B, T), cuda_device_cfg.device, cuda_device_cfg.dtype)
        goal_pos = torch.zeros(B, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_rot = _identity_quat((B,), cuda_device_cfg.device, cuda_device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        out.sum().backward()

        assert ee_pos.grad is not None
        assert ee_pos.grad.shape == ee_pos.shape
        # Gradient at the violating step must be non-zero
        assert torch.any(ee_pos.grad[:, 1] != 0.0)

    def test_rotation_gradient_flows(self, cuda_device_cfg):
        """Gradients must flow back through ee_rot_batch."""
        B, T = 2, 4
        cfg = EEFBoundCostCfg(weight=1.0, device_cfg=cuda_device_cfg)
        cost = EEFBoundCost(cfg)
        # Enable only rotation rows
        margins = torch.full((4, 3), -1.0, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        margins[2:] = 0.0
        cost.update_margins(margins)

        ee_pos = torch.zeros(B, T, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        start_q = _rot_z_quat(math.pi / 2, cuda_device_cfg.device, cuda_device_cfg.dtype)
        exceed_q = _rot_z_quat(math.radians(100), cuda_device_cfg.device, cuda_device_cfg.dtype)

        ee_rot = _identity_quat((B, T), cuda_device_cfg.device, cuda_device_cfg.dtype).requires_grad_(False)
        ee_rot = ee_rot.detach().clone().requires_grad_(True)
        with torch.no_grad():
            ee_rot.data[:, 0] = start_q
            ee_rot.data[:, 1] = exceed_q

        goal_pos = torch.zeros(B, 3, device=cuda_device_cfg.device, dtype=cuda_device_cfg.dtype)
        goal_rot = _identity_quat((B,), cuda_device_cfg.device, cuda_device_cfg.dtype)

        out = cost.forward(ee_pos, ee_rot, goal_pos, goal_rot)
        out.sum().backward()

        assert ee_rot.grad is not None
        assert ee_rot.grad.shape == ee_rot.shape