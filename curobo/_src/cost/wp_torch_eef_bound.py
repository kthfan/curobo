# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Warp kernel and PyTorch autograd wrapper for the EEF bound cost.

Layout
------
* ``eef_bound_cost_warp`` (the Warp kernel) computes the per-(B, H, L) cost
  **and** the position / rotation gradients in the same launch.  One thread
  per ``(b, h, l)`` triple, looping over the ``G`` goalset inside; for each
  (b, h, l) we keep the lowest-cost goal and its associated gradient.
* :class:`EEFBoundCostFunction` is a ``torch.autograd.Function`` whose
  ``forward`` calls the kernel and applies the per-link (or scalar) weight,
  and whose ``backward`` propagates the **pre-computed** gradients through
  the rest of the cost graph.  No ``wp.Tape`` and no PyTorch autograd
  re-evaluation is used.

Conventions
-----------
* cuRobo uses **wxyz** quaternions; Warp uses **xyzw**.  The kernel takes
  xyzw, computes everything internally, and writes the rotation gradient
  back to ``out_rotation_gradient`` already converted to **wxyz**.
* Per-link margins are flattened as ``(L * 4 * 3)`` row-major: link ``l``
  reads ``margins[l*12 + s*3 + ax]`` for ``s in 0..3`` (start-pos, end-pos,
  start-rot, end-rot) and ``ax in 0..2``.
* Negative margins are pre-converted to a large positive sentinel
  (``1e6``) on the Python side so the kernel does not need separate
  constrain masks — the cost and gradient are both zero on disabled axes.
"""

from __future__ import annotations

from typing import Optional

import torch
import warp as wp
from curobo._src.util.warp import init_warp, get_warp_device_stream

wp.init()


# ---------------------------------------------------------------------------
# Warp @wp.func helpers
# ---------------------------------------------------------------------------
@wp.func
def _corridor_axis(
    current: float,
    start: float,
    end: float,
    margin_begin: float,
    margin_end: float,
) -> wp.vec2:
    """Single-axis corridor cost (begin + end) and its gradient w.r.t. ``current``.

    Returns ``wp.vec2(violation, d_violation_d_current)``.  The gradient
    mirrors PyTorch autograd semantics of ``max(0, ...)`` — the slope
    (±1) gated by the ReLU activation indicator.

    Disabled axes (negative margin) are encoded as a large positive
    sentinel so both the violation and the gradient are zero.
    """
    is_end_larger = float(end > start)
    one_minus = 1.0 - is_end_larger

    # Begin-side violation.
    begin_arg_begin = (start - margin_begin) - current
    begin_arg_end = current - (start + margin_begin)
    begin_viol_begin = wp.max(0.0, begin_arg_begin)
    begin_viol_end = wp.max(0.0, begin_arg_end)
    begin_viol = is_end_larger * begin_viol_begin + one_minus * begin_viol_end
    begin_active_begin = float(begin_arg_begin > 0.0)
    begin_active_end = float(begin_arg_end > 0.0)
    begin_active = is_end_larger * begin_active_begin + one_minus * begin_active_end
    begin_grad = -is_end_larger + one_minus
    grad_begin = begin_active * begin_grad

    # End-side violation.
    end_arg_begin = current - (end + margin_end)
    end_arg_end = (end - margin_end) - current
    end_viol_begin = wp.max(0.0, end_arg_begin)
    end_viol_end = wp.max(0.0, end_arg_end)
    end_viol = is_end_larger * end_viol_begin + one_minus * end_viol_end
    end_active_begin = float(end_arg_begin > 0.0)
    end_active_end = float(end_arg_end > 0.0)
    end_active = is_end_larger * end_active_begin + one_minus * end_active_end
    end_grad = is_end_larger - one_minus
    grad_end = end_active * end_grad

    return wp.vec2(begin_viol + end_viol, grad_begin + grad_end)


@wp.func
def quat_to_rotvec_warp(q: wp.quat, eps: float) -> wp.vec3:
    """Quaternion ``(x, y, z, w)`` (Warp convention) → rotation vector.

    Near-identity uses ``scale = 2`` so the limit matches ``ω ≈ 2 * q_xyz``.
    """
    norm = wp.sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w)
    if norm < eps:
        return wp.vec3(0.0)
    qn = q / norm
    if qn.w < 0.0:
        qn = -qn
    sin_half = wp.sqrt(qn.x * qn.x + qn.y * qn.y + qn.z * qn.z)
    angle = 2.0 * wp.atan2(sin_half, qn.w)
    scale = angle / (sin_half + eps)
    if sin_half < 1e-6:
        scale = 2.0
    return wp.vec3(qn.x * scale, qn.y * scale, qn.z * scale)


# ---------------------------------------------------------------------------
# Main kernel
# ---------------------------------------------------------------------------
@wp.kernel
def eef_bound_cost_warp(
    # Current and start poses (xyzw quaternions).
    current_pos: wp.array(dtype=wp.vec3),       # (B*H*L, 3)
    current_quat: wp.array(dtype=wp.quat),      # (B*H*L, 4) xyzw
    start_pos: wp.array(dtype=wp.vec3),         # (B*L, 3)
    start_quat: wp.array(dtype=wp.quat),        # (B*L, 4) xyzw
    # Goal poses (xyzw quaternions), one per (b, l, g).
    goal_pos: wp.array(dtype=wp.vec3),          # (B*L*G, 3)
    goal_quat: wp.array(dtype=wp.quat),         # (B*L*G, 4) xyzw
    # Per-link margins, flattened row-major: l*12 + s*3 + ax.
    # Negative margins are replaced with 1e6 on the Python side.
    margins: wp.array(dtype=wp.float32),        # (L*4*3,)
    # Output buffers (per-(B,H,L) tuple, addressed by tid = b*H*L + h*L + l).
    out_cost: wp.array(dtype=wp.float32),                   # (B*H*L,)
    out_position_distance: wp.array(dtype=wp.float32),      # (B*H*L,)
    out_rotation_distance: wp.array(dtype=wp.float32),      # (B*H*L,)
    out_distance: wp.array(dtype=wp.float32),               # (2*B*H*L,) interleaved [pos, rot]
    out_goalset_idx: wp.array(dtype=wp.int32),              # (B*H*L,)
    out_position_gradient: wp.array(dtype=wp.vec3),         # (B*H*L, 3) world frame
    out_rotation_gradient: wp.array(dtype=wp.vec4),         # (B*H*L, 4) world frame, wxyz
    # Sizes.
    H: int,
    L: int,
    G: int,
):
    """One thread per ``(b, h, l)`` triple.  Inside, iterate ``g`` and keep
    the lowest-cost goal (and its gradient).
    """
    tid = wp.tid()
    b = tid / (H * L)
    rem = tid - b * H * L
    h = rem / L
    l = rem - h * L

    margin_base = l * 4 * 3

    bl_idx = b * L + l
    s_pos = start_pos[bl_idx]
    s_quat_xyzw = start_quat[bl_idx]
    c_pos = current_pos[tid]
    c_quat_xyzw = current_quat[tid]

    best_cost = wp.float32(-1.0)
    best_pos_dist = wp.float32(0.0)
    best_rot_dist = wp.float32(0.0)
    best_pos_grad = wp.vec3(0.0, 0.0, 0.0)
    best_rot_grad_wxyz = wp.vec4(0.0, 0.0, 0.0, 0.0)
    best_g = wp.int32(0)

    goal_base = b * L * G + l * G
    for g in range(G):
        g_idx = goal_base + g
        g_pos = goal_pos[g_idx]
        g_quat_xyzw = goal_quat[g_idx]
        g_inv_xyzw = wp.quat_inverse(g_quat_xyzw)

        # Transform current and start into goal frame.
        local_c_pos = wp.quat_rotate(g_inv_xyzw, c_pos - g_pos)
        local_c_quat_xyzw = wp.mul(g_inv_xyzw, c_quat_xyzw)
        local_c_rv = quat_to_rotvec_warp(local_c_quat_xyzw, 1e-8)

        local_s_pos = wp.quat_rotate(g_inv_xyzw, s_pos - g_pos)
        local_s_quat_xyzw = wp.mul(g_inv_xyzw, s_quat_xyzw)
        local_s_rv = quat_to_rotvec_warp(local_s_quat_xyzw, 1e-8)

        # Goal-in-own-frame origin / identity for the corridor end-point.
        end_pos = wp.vec3(0.0, 0.0, 0.0)
        end_rv = wp.vec3(0.0, 0.0, 0.0)

        cost = wp.float32(0.0)
        pos_dist_acc = wp.float32(0.0)
        rot_dist_acc = wp.float32(0.0)
        pos_grad_local = wp.vec3(0.0, 0.0, 0.0)   # d cost / d local_c_pos
        rot_grad_local = wp.vec3(0.0, 0.0, 0.0)   # d cost / d local_c_rv

        for ax in range(3):
            # Position corridor (rows 0 and 1 of per-link margins).
            mb_pos = margins[margin_base + 0 * 3 + ax]
            me_pos = margins[margin_base + 1 * 3 + ax]
            pos_res = _corridor_axis(
                local_c_pos[ax], local_s_pos[ax], end_pos[ax],
                mb_pos, me_pos,
            )
            cost += pos_res[0]
            pos_dist_acc += pos_res[0]
            pos_grad_local[ax] = pos_res[1]

            # Rotation corridor (rows 2 and 3).
            mb_rot = margins[margin_base + 2 * 3 + ax]
            me_rot = margins[margin_base + 3 * 3 + ax]
            rot_res = _corridor_axis(
                local_c_rv[ax], local_s_rv[ax], end_rv[ax],
                mb_rot, me_rot,
            )
            cost += rot_res[0]
            rot_dist_acc += rot_res[0]
            rot_grad_local[ax] = rot_res[1]

        if best_cost < 0.0 or cost < best_cost:
            best_cost = cost
            best_pos_dist = pos_dist_acc
            best_rot_dist = rot_dist_acc
            # Position gradient in goal frame → world frame:
            #   local_c_pos = R(g_inv) (c_pos - g_pos)
            #   d(cost)/d(c_pos) = R(g) d(cost)/d(local_c_pos)
            best_pos_grad = wp.quat_rotate(g_quat_xyzw, pos_grad_local)

            # Rotation gradient via the cuRobo convention:
            #   d(cost)/d(local_c_quat_xyzw) = local_c_quat * (rot_grad, 0)
            #   d(cost)/d(c_quat_xyzw)       = g * d(cost)/d(local_c_quat_xyzw)
            rot_grad_omega = wp.quat(
                rot_grad_local[0], rot_grad_local[1], rot_grad_local[2], 0.0
            )
            grad_local_q_xyzw = wp.mul(local_c_quat_xyzw, rot_grad_omega)
            grad_q_world_xyzw = wp.mul(g_quat_xyzw, grad_local_q_xyzw)
            # xyzw → wxyz for the output buffer (cuRobo convention).
            best_rot_grad_wxyz = wp.vec4(
                grad_q_world_xyzw[3],  # w
                grad_q_world_xyzw[0],  # x
                grad_q_world_xyzw[1],  # y
                grad_q_world_xyzw[2],  # z
            )
            best_g = g

    # Write outputs.
    out_cost[tid] = best_cost
    out_position_distance[tid] = best_pos_dist
    out_rotation_distance[tid] = best_rot_dist
    out_distance[2 * tid + 0] = best_pos_dist
    out_distance[2 * tid + 1] = best_rot_dist
    out_goalset_idx[tid] = best_g
    out_position_gradient[tid] = best_pos_grad
    out_rotation_gradient[tid] = best_rot_grad_wxyz


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Convert a ``(..., 4)`` wxyz quaternion tensor to xyzw."""
    return torch.stack([q[..., 1], q[..., 2], q[..., 3], q[..., 0]], dim=-1)


_DISABLED_MARGIN_SENTINEL = 1.0e6


def _disable_negative_margins(margins: torch.Tensor) -> torch.Tensor:
    """Replace negative margins with a large sentinel so they neither cost
    nor gradient.  ``1e6`` is well below fp32 overflow and dominates any
    plausible trajectory value.
    """
    return torch.where(margins < 0, torch.full_like(margins, _DISABLED_MARGIN_SENTINEL), margins)


# ---------------------------------------------------------------------------
# Autograd Function
# ---------------------------------------------------------------------------
class EEFBoundCostFunction(torch.autograd.Function):
    """Warp-accelerated forward + cached-gradient backward for the EEF bound cost.

    ``forward``
        Calls the Warp kernel, which fills per-(B, H, L) cost and gradients,
        then applies the per-link or scalar ``weight`` to the per-link cost.
        Returns the weighted per-link cost tensor ``(B, H, L)``.

    ``backward``
        Receives ``grad_output`` = gradient w.r.t. the (weighted) per-link
        cost.  Multiplies by ``weight`` (chain rule) and then by the cached
        position / rotation gradients and returns the corresponding gradients
        for ``cur_pos`` and ``cur_quat``.  Everything else is ``None``.
    """

    @staticmethod
    def forward(
        ctx,
        cur_pos: torch.Tensor,
        cur_quat: torch.Tensor,
        start_pos: torch.Tensor,
        start_quat: torch.Tensor,
        goal_pos: torch.Tensor,
        goal_quat: torch.Tensor,
        margins: torch.Tensor,
        weight: torch.Tensor,
        out_distance: torch.Tensor,
        out_position_distance: torch.Tensor,
        out_rotation_distance: torch.Tensor,
        out_goalset_idx: torch.Tensor,
        out_position_gradient: torch.Tensor,
        out_rotation_gradient: torch.Tensor,
    ):
        B, H, L, _ = cur_pos.shape
        G = goal_pos.shape[2]
        N_threads = B * H * L

        # Negative margins become a large positive sentinel so the kernel does
        # not need separate constrain masks.
        margins_k = _disable_negative_margins(margins).contiguous()

        # ---------------- reshape to Warp-friendly layout -----------------
        # Kernel ordering: tid = b*H*L + h*L + l.  Reshape cur_pos from
        # (B, H, L, 3) to (B, L, H, 3) and then to (N_threads, 3).
        cur_pos_N = cur_pos.permute(0, 2, 1, 3).reshape(N_threads, 3).contiguous()
        cur_quat_N = cur_quat.permute(0, 2, 1, 3).reshape(N_threads, 4).contiguous()
        start_pos_N = start_pos.squeeze(1).reshape(B * L, 3).contiguous()
        start_quat_N = start_quat.squeeze(1).reshape(B * L, 4).contiguous()
        goal_pos_N = goal_pos.reshape(B * L * G, 3).contiguous()
        goal_quat_N = goal_quat.reshape(B * L * G, 4).contiguous()
        margins_N = margins_k.reshape(-1).contiguous()

        # cuRobo (wxyz) → Warp (xyzw).
        cur_quat_xyzw = _wxyz_to_xyzw(cur_quat_N)
        start_quat_xyzw = _wxyz_to_xyzw(start_quat_N)
        goal_quat_xyzw = _wxyz_to_xyzw(goal_quat_N)

        # ---------------- allocate / wrap Warp arrays ---------------------
        wp_device, wp_stream = get_warp_device_stream(cur_pos)
        cur_pos_w = wp.from_torch(cur_pos_N, dtype=wp.vec3)
        cur_quat_w = wp.from_torch(cur_quat_xyzw, dtype=wp.quat)
        start_pos_w = wp.from_torch(start_pos_N, dtype=wp.vec3)
        start_quat_w = wp.from_torch(start_quat_xyzw, dtype=wp.quat)
        goal_pos_w = wp.from_torch(goal_pos_N, dtype=wp.vec3)
        goal_quat_w = wp.from_torch(goal_quat_xyzw, dtype=wp.quat)
        margins_w = wp.from_torch(margins_N, dtype=wp.float32)

        # Out buffers: 1D reshapes that match the kernel ordering
        # ``tid = b*H*L + h*L + l``.  The output buffers in
        # ``cost_eef_bound.setup_batch_tensors`` already have shape
        # ``(B, H, L, ...)`` with row-major strides, so a plain reshape
        # without permute is what we need.
        out_cost_w = wp.zeros(N_threads, dtype=wp.float32, device=wp_device)
        out_pos_dist_1d = out_position_distance.reshape(-1).contiguous()
        out_rot_dist_1d = out_rotation_distance.reshape(-1).contiguous()
        out_dist_1d = out_distance.reshape(-1).contiguous()  # (2*B*H*L,)
        out_g_idx_1d = out_goalset_idx.reshape(-1).contiguous().to(torch.int32)
        out_pos_grad_1d = out_position_gradient.reshape(-1, 3).contiguous()
        out_rot_grad_1d = out_rotation_gradient.reshape(-1, 4).contiguous()

        out_pos_dist_w = wp.from_torch(out_pos_dist_1d, dtype=wp.float32)
        out_rot_dist_w = wp.from_torch(out_rot_dist_1d, dtype=wp.float32)
        out_dist_w = wp.from_torch(out_dist_1d, dtype=wp.float32)
        out_g_idx_w = wp.from_torch(out_g_idx_1d, dtype=wp.int32)
        out_pos_grad_w = wp.from_torch(out_pos_grad_1d, dtype=wp.vec3)
        out_rot_grad_w = wp.from_torch(out_rot_grad_1d, dtype=wp.vec4)

        # ---------------- launch -----------------------------------------
        wp.launch(
            eef_bound_cost_warp,
            dim=N_threads,
            inputs=[
                cur_pos_w, cur_quat_w,
                start_pos_w, start_quat_w,
                goal_pos_w, goal_quat_w,
                margins_w,
                out_cost_w,
                out_pos_dist_w, out_rot_dist_w,
                out_dist_w,
                out_g_idx_w,
                out_pos_grad_w, out_rot_grad_w,
                H, L, G,
            ],
            device=wp_device,
            stream=wp_stream,
        )

        # ---------------- read back cost ---------------------------------
        # The kernel writes at ``tid = b*H*L + h*L + l``, so a plain
        # ``reshape`` recovers the natural (B, H, L) ordering without any
        # permute.
        cost_N = wp.to_torch(out_cost_w).reshape(N_threads)
        cost_BHL = cost_N.reshape(B, H, L).contiguous()

        # ---------------- apply weight (scalar or per-link) ---------------
        if weight.numel() == L:
            cost_BHL_w = cost_BHL * weight.view(1, 1, L)
        elif weight.numel() == 1:
            cost_BHL_w = cost_BHL * weight
        else:
            raise ValueError(f"weight must have numel 1 or {L}; got {tuple(weight.shape)}")

        # ---------------- save state for backward -------------------------
        # Clone the gradients so subsequent Warp launches cannot overwrite
        # them between forward and backward.
        ctx.save_for_backward(
            out_position_gradient.detach().clone(),
            out_rotation_gradient.detach().clone(),
            weight.detach().clone(),
        )
        return cost_BHL_w

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        # grad_output: (B, H, L) — gradient w.r.t. weighted per-link cost.
        pos_grad, rot_grad, weight = ctx.saved_tensors  # (B,H,L,3), (B,H,L,4), scalar or (L,)
        # Chain rule:  d(loss)/d(cur_pos) = d(loss)/d(cost_w) * d(cost_w)/d(cur_pos)
        #   d(cost_w)/d(cur_pos) = weight * d(cost_kernel)/d(cur_pos) = weight * pos_grad
        grad_per_link = grad_output * weight  # broadcasts: scalar → (B,H,L); (L,) → (B,H,L)
        grad_pos = pos_grad * grad_per_link.unsqueeze(-1)   # (B, H, L, 3)
        grad_quat = rot_grad * grad_per_link.unsqueeze(-1)  # (B, H, L, 4)

        # Return in the same order as forward inputs; None for non-tensor args
        # and for the output buffers (which we never re-derive).
        return (
            grad_pos, grad_quat,
            None, None, None, None,
            None,
            None,
            None, None, None, None, None, None,
        )
