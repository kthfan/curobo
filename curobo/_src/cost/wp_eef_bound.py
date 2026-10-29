# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernel and autograd function for :class:`~curobo._src.cost.cost_eef_bound.EEFBoundCost`.

One thread per ``(batch, horizon, link)``. Each link has a fixed number of bound slots. For every
used slot the thread builds the reference frame, expresses the tool pose in it as
``(x, y, z, tilt, yaw, roll)`` (see :mod:`curobo._src.cost.tool_pose_bound`), and evaluates a
squared hinge on each limited component. Goal-anchored slots are evaluated against every goal of
the set and the goal with the lowest total cost is kept.

Gradients are exact and written in cuRobo's convention: position gradient in the base frame,
and orientation gradient ``q ⊗ (0, 2ω)`` (wxyz) where ``ω`` is the base-frame angular gradient,
so that the kinematics backward (``0.5 * vec(q⁻¹ ⊗ dq)``) recovers ``ω``. All quaternions here
are wxyz.
"""

from __future__ import annotations

import torch
import warp as wp

from curobo._src.util.warp import get_warp_device_stream

_PI = 3.14159265358979
# Smallest sin(tilt)² used for the yaw and roll gradients, which grow as 1 / sin(tilt) near the
# parallel-axes singularity (sin(5 deg)²).
_MIN_SIN2 = 0.0076


@wp.func
def _quat_mul(a: wp.vec4, b: wp.vec4) -> wp.vec4:
    return wp.vec4(
        a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
        a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
        a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
        a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
    )


@wp.func
def _quat_conj(q: wp.vec4) -> wp.vec4:
    return wp.vec4(q[0], -q[1], -q[2], -q[3])


@wp.func
def _quat_rotate(q: wp.vec4, v: wp.vec3) -> wp.vec3:
    u = wp.vec3(q[1], q[2], q[3])
    t = 2.0 * wp.cross(u, v)
    return v + q[0] * t + wp.cross(u, t)


@wp.func
def _squared_hinge(value: float, lower: float, upper: float) -> wp.vec2:
    """Return ``(0.5 * violation², d/d value)`` for the interval ``[lower, upper]``."""
    below = wp.max(0.0, lower - value)
    above = wp.max(0.0, value - upper)
    return wp.vec2(0.5 * (below * below + above * above), above - below)


@wp.func
def _wrap(angle: float) -> float:
    """Wrap to ``[-π, π)``."""
    return angle - 2.0 * _PI * wp.floor((angle + _PI) / (2.0 * _PI))


@wp.func
def _slot_cost(
    c_pos: wp.vec3,
    c_quat: wp.vec4,
    a_pos: wp.vec3,
    a_quat: wp.vec4,
    s: int,
    axes: wp.array(dtype=wp.int32),
    reference_rotation: wp.array(dtype=wp.vec4),
    tool_rotation: wp.array(dtype=wp.vec4),
    center: wp.array(dtype=wp.float32),
    lower: wp.array(dtype=wp.float32),
    upper: wp.array(dtype=wp.float32),
    use_rotation: wp.array(dtype=wp.int32),
):
    """Return cost, base-frame position gradient and base-frame angular gradient of slot ``s``."""
    ref_quat = reference_rotation[s]
    if axes[s] == 0:
        ref_quat = _quat_mul(a_quat, reference_rotation[s])
    ref_inv = _quat_conj(ref_quat)
    local_pos = _quat_rotate(ref_inv, c_pos - a_pos)
    cost = float(0.0)
    grad_local = wp.vec3(0.0)
    for k in range(3):
        h = _squared_hinge(local_pos[k], lower[s * 6 + k], upper[s * 6 + k])
        cost += h[0]
        grad_local[k] = h[1]
    grad_pos = _quat_rotate(ref_quat, grad_local)
    omega = wp.vec3(0.0)

    if use_rotation[s] != 0:
        q = _quat_mul(_quat_mul(ref_inv, c_quat), tool_rotation[s])
        w = q[0]
        x = q[1]
        y = q[2]
        z = q[3]
        # Third column u (tool axis in F) and third row r (reference axis in U) of R_FU.
        u = wp.vec3(2.0 * (x * z + w * y), 2.0 * (y * z - w * x), 1.0 - 2.0 * (x * x + y * y))
        r = wp.vec3(2.0 * (x * z - w * y), 2.0 * (y * z + w * x), u[2])
        sin2 = u[0] * u[0] + u[1] * u[1]
        sin_tilt = wp.sqrt(sin2)
        tilt = wp.atan2(sin_tilt, u[2])
        omega_ref = wp.vec3(0.0)

        h = _squared_hinge(tilt, lower[s * 6 + 3], upper[s * 6 + 3])
        cost += h[0]
        if h[1] != 0.0:
            # d tilt / d omega = (z × u) / |z × u|; any horizontal axis where it is undefined.
            d_tilt = wp.vec3(1.0, 0.0, 0.0)
            if u[2] < 0.0:
                d_tilt = wp.vec3(-1.0, 0.0, 0.0)
            if sin_tilt > 1.0e-6:
                d_tilt = wp.vec3(-u[1] / sin_tilt, u[0] / sin_tilt, 0.0)
            omega_ref += h[1] * d_tilt

        if sin_tilt > 1.0e-6:
            g_sin2 = wp.max(sin2, _MIN_SIN2)
            yaw = wp.atan2(u[1], u[0])
            h = _squared_hinge(_wrap(yaw - center[s * 6 + 4]), lower[s * 6 + 4], upper[s * 6 + 4])
            cost += h[0]
            if h[1] != 0.0:
                d_yaw = wp.vec3(-u[0] * u[2] / g_sin2, -u[1] * u[2] / g_sin2, 1.0)
                omega_ref += h[1] * d_yaw

            roll = wp.atan2(r[1], -r[0])
            h = _squared_hinge(_wrap(roll - center[s * 6 + 5]), lower[s * 6 + 5], upper[s * 6 + 5])
            cost += h[0]
            if h[1] != 0.0:
                # Gradient in the tool axis frame, rotated into the reference frame.
                d_roll = wp.vec3(-r[0] * r[2] / g_sin2, -r[1] * r[2] / g_sin2, 1.0)
                omega_ref += h[1] * _quat_rotate(q, d_roll)

        omega = _quat_rotate(ref_quat, omega_ref)

    return cost, grad_pos, omega


@wp.kernel
def eef_bound_cost_kernel(
    current_position: wp.array(dtype=wp.vec3),  # (B*H*L)
    current_quaternion: wp.array(dtype=wp.vec4),  # (B*H*L) wxyz
    goal_position: wp.array(dtype=wp.vec3),  # (B_goal*L*G)
    goal_quaternion: wp.array(dtype=wp.vec4),  # (B_goal*L*G) wxyz
    idxs_goal: wp.array(dtype=wp.int32),  # (B)
    start_position: wp.array(dtype=wp.vec3),  # (B_start*L)
    start_quaternion: wp.array(dtype=wp.vec4),  # (B_start*L) wxyz
    idxs_start: wp.array(dtype=wp.int32),  # (B)
    anchor: wp.array(dtype=wp.int32),  # (L*K) -1 unused, 0 goal, 1 start, 2 base
    axes: wp.array(dtype=wp.int32),  # (L*K) 0 anchor axes, 1 base axes
    reference_rotation: wp.array(dtype=wp.vec4),  # (L*K) wxyz
    tool_rotation: wp.array(dtype=wp.vec4),  # (L*K) wxyz
    center: wp.array(dtype=wp.float32),  # (L*K*6)
    lower: wp.array(dtype=wp.float32),  # (L*K*6)
    upper: wp.array(dtype=wp.float32),  # (L*K*6)
    use_rotation: wp.array(dtype=wp.int32),  # (L*K)
    weight: wp.array(dtype=wp.float32),  # (1) or (L)
    weight_per_link: int,
    horizon: int,
    num_links: int,
    num_slots: int,
    num_goalset: int,
    out_cost: wp.array(dtype=wp.float32),  # (B*H*L)
    out_position_gradient: wp.array(dtype=wp.vec3),  # (B*H*L)
    out_quaternion_gradient: wp.array(dtype=wp.vec4),  # (B*H*L) wxyz
):
    """Write the weighted cost and gradients of one ``(batch, horizon, link)`` element."""
    tid = wp.tid()
    b = tid / (horizon * num_links)
    li = tid - (tid / num_links) * num_links
    c_pos = current_position[tid]
    c_quat = current_quaternion[tid]

    total = float(0.0)
    grad_pos = wp.vec3(0.0)
    omega = wp.vec3(0.0)
    has_goal_slot = int(0)
    for k in range(num_slots):
        s = li * num_slots + k
        a = anchor[s]
        if a < 0:
            continue
        if a == 0:
            has_goal_slot = 1
            continue
        a_pos = wp.vec3(0.0)
        a_quat = wp.vec4(1.0, 0.0, 0.0, 0.0)
        if a == 1:
            si = idxs_start[b] * num_links + li
            if si >= start_position.shape[0]:
                continue  # anchor data missing; the Python wrapper reports this case
            a_pos = start_position[si]
            a_quat = start_quaternion[si]
        cost, g_pos, g_omega = _slot_cost(
            c_pos,
            c_quat,
            a_pos,
            a_quat,
            s,
            axes,
            reference_rotation,
            tool_rotation,
            center,
            lower,
            upper,
            use_rotation,
        )
        total += cost
        grad_pos += g_pos
        omega += g_omega

    if has_goal_slot != 0:
        best_cost = float(-1.0)
        best_grad_pos = wp.vec3(0.0)
        best_omega = wp.vec3(0.0)
        for g in range(num_goalset):
            gi = idxs_goal[b] * num_links * num_goalset + li * num_goalset + g
            if gi >= goal_position.shape[0]:
                continue
            a_pos = goal_position[gi]
            a_quat = goal_quaternion[gi]
            goal_cost = float(0.0)
            goal_grad_pos = wp.vec3(0.0)
            goal_omega = wp.vec3(0.0)
            for k in range(num_slots):
                s = li * num_slots + k
                if anchor[s] != 0:
                    continue
                cost, g_pos, g_omega = _slot_cost(
                    c_pos,
                    c_quat,
                    a_pos,
                    a_quat,
                    s,
                    axes,
                    reference_rotation,
                    tool_rotation,
                    center,
                    lower,
                    upper,
                    use_rotation,
                )
                goal_cost += cost
                goal_grad_pos += g_pos
                goal_omega += g_omega
            if best_cost < 0.0 or goal_cost < best_cost:
                best_cost = goal_cost
                best_grad_pos = goal_grad_pos
                best_omega = goal_omega
        if best_cost > 0.0:
            total += best_cost
            grad_pos += best_grad_pos
            omega += best_omega

    w = weight[0]
    if weight_per_link != 0:
        w = weight[li]
    out_cost[tid] = w * total
    out_position_gradient[tid] = w * grad_pos
    out_quaternion_gradient[tid] = w * _quat_mul(
        c_quat, wp.vec4(0.0, 2.0 * omega[0], 2.0 * omega[1], 2.0 * omega[2])
    )


class EEFBoundCostFunction(torch.autograd.Function):
    """Launch :func:`eef_bound_cost_kernel`; backward reuses the gradients it wrote.

    Returns the weighted per-link cost ``(B, H, L)``. With ``use_grad_input=False`` the incoming
    gradient is assumed to be all ones (the cost is only summed), as in the other pose costs.
    """

    @staticmethod
    def forward(
        ctx,
        current_position: torch.Tensor,
        current_quaternion: torch.Tensor,
        goal_position: torch.Tensor,
        goal_quaternion: torch.Tensor,
        idxs_goal: torch.Tensor,
        start_position: torch.Tensor,
        start_quaternion: torch.Tensor,
        idxs_start: torch.Tensor,
        anchor: torch.Tensor,
        axes: torch.Tensor,
        reference_rotation: torch.Tensor,
        tool_rotation: torch.Tensor,
        center: torch.Tensor,
        lower: torch.Tensor,
        upper: torch.Tensor,
        use_rotation: torch.Tensor,
        weight: torch.Tensor,
        num_goalset: int,
        use_grad_input: bool,
        out_cost: torch.Tensor,
        out_position_gradient: torch.Tensor,
        out_quaternion_gradient: torch.Tensor,
    ) -> torch.Tensor:
        """Launch the kernel and return the weighted per-link cost.

        Shapes: current ``(B, H, L, 3/4)``; goal ``(B_goal*L*G, 3/4)``; start
        ``(B_start*L, 3/4)``; idxs ``(B,)`` int32; slot tensors ``(L, K[, 4 or 6])``; outputs
        ``(B, H, L[, 3/4])``.
        """
        b, h, num_links, _ = current_position.shape
        num_slots = anchor.shape[-1]
        device, stream = get_warp_device_stream(current_position)
        wp.launch(
            eef_bound_cost_kernel,
            dim=b * h * num_links,
            inputs=[
                wp.from_torch(current_position.detach().reshape(-1, 3), dtype=wp.vec3),
                wp.from_torch(current_quaternion.detach().reshape(-1, 4), dtype=wp.vec4),
                wp.from_torch(goal_position, dtype=wp.vec3),
                wp.from_torch(goal_quaternion, dtype=wp.vec4),
                wp.from_torch(idxs_goal, dtype=wp.int32),
                wp.from_torch(start_position, dtype=wp.vec3),
                wp.from_torch(start_quaternion, dtype=wp.vec4),
                wp.from_torch(idxs_start, dtype=wp.int32),
                wp.from_torch(anchor.view(-1), dtype=wp.int32),
                wp.from_torch(axes.view(-1), dtype=wp.int32),
                wp.from_torch(reference_rotation.view(-1, 4), dtype=wp.vec4),
                wp.from_torch(tool_rotation.view(-1, 4), dtype=wp.vec4),
                wp.from_torch(center.view(-1), dtype=wp.float32),
                wp.from_torch(lower.view(-1), dtype=wp.float32),
                wp.from_torch(upper.view(-1), dtype=wp.float32),
                wp.from_torch(use_rotation.view(-1), dtype=wp.int32),
                wp.from_torch(weight.view(-1), dtype=wp.float32),
                int(weight.numel() == num_links and num_links > 1),
                h,
                num_links,
                num_slots,
                num_goalset,
                wp.from_torch(out_cost.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_gradient.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(out_quaternion_gradient.view(-1, 4), dtype=wp.vec4),
            ],
            device=device,
            stream=stream,
        )
        ctx.use_grad_input = use_grad_input
        ctx.save_for_backward(out_position_gradient, out_quaternion_gradient)
        return out_cost

    @staticmethod
    @torch.autograd.function.once_differentiable
    def backward(ctx, grad_cost: torch.Tensor):
        """Return the cached pose gradients, scaled by ``grad_cost`` when requested."""
        position_gradient, quaternion_gradient = ctx.saved_tensors
        grads = [None] * 22
        if ctx.use_grad_input:
            grad_cost = grad_cost.unsqueeze(-1)
            position_gradient = position_gradient * grad_cost
            quaternion_gradient = quaternion_gradient * grad_cost
        if ctx.needs_input_grad[0]:
            grads[0] = position_gradient
        if ctx.needs_input_grad[1]:
            grads[1] = quaternion_gradient
        return tuple(grads)
