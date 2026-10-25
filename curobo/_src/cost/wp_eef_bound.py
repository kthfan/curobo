# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernel and autograd function for :class:`~curobo._src.cost.cost_eef_bound.EEFBoundCost`.

One thread per ``(batch, horizon, link)``. Each thread builds the reference frame
``T_anchor * T_offset`` (looping over the goalset when anchored to the goal and keeping the
lowest cost), expresses the tool pose in it, and evaluates a squared hinge on every bounded
component of ``(x, y, z, swing_x, swing_y, twist)`` and on the swing angle.

Gradients are exact and written in cuRobo's convention: position gradient in the base frame,
and orientation gradient ``q ⊗ (0, 2ω)`` (wxyz) where ``ω`` is the base-frame angular gradient,
so that the kinematics backward (``0.5 * vec(q⁻¹ ⊗ dq)``) recovers ``ω``. All quaternions here
are wxyz.
"""

from __future__ import annotations

import torch
import warp as wp

from curobo._src.util.warp import get_warp_device_stream


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
def _bound_cost(
    c_pos: wp.vec3,
    c_quat: wp.vec4,
    ref_pos: wp.vec3,
    ref_quat: wp.vec4,
    pos_lower: wp.vec3,
    pos_upper: wp.vec3,
    rot_lower: wp.vec3,
    rot_upper: wp.vec3,
    max_swing: float,
    use_rotation: int,
):
    """Return cost, base-frame position gradient and base-frame angular gradient."""
    ref_inv = _quat_conj(ref_quat)
    local_pos = _quat_rotate(ref_inv, c_pos - ref_pos)
    cost = float(0.0)
    grad_local = wp.vec3(0.0)
    for k in range(3):
        h = _squared_hinge(local_pos[k], pos_lower[k], pos_upper[k])
        cost += h[0]
        grad_local[k] = h[1]
    grad_pos = _quat_rotate(ref_quat, grad_local)
    omega = wp.vec3(0.0)

    if use_rotation != 0:
        r = _quat_mul(ref_inv, c_quat)
        if r[0] < 0.0:
            r = -r
        w = r[0]
        x = r[1]
        y = r[2]
        z = r[3]
        n = wp.sqrt(w * w + z * z)
        m = wp.sqrt(x * x + y * y)
        s2 = m * m + n * n
        angle = 2.0 * wp.atan2(m, n)
        if n < 1.0e-6:
            # Swing of 180 degrees: twist is undefined. Report the swing, no gradient.
            sx = angle * x / wp.max(m, 1.0e-6)
            sy = angle * y / wp.max(m, 1.0e-6)
            cost += _squared_hinge(sx, rot_lower[0], rot_upper[0])[0]
            cost += _squared_hinge(sy, rot_lower[1], rot_upper[1])[0]
            cost += _squared_hinge(0.0, rot_lower[2], rot_upper[2])[0]
            swing_excess = wp.max(0.0, angle - max_swing)
            cost += 0.5 * swing_excess * swing_excess
        else:
            twist = 2.0 * wp.atan2(z, w)
            n3 = n * n * n
            xp = (x * w - y * z) / n
            yp = (x * z + y * w) / n
            scale = 2.0 / n
            dscale_dm = float(0.0)
            dscale_dn = -2.0 / (n * n)
            dangle_dm = 2.0 * n / s2
            dangle_dn = -2.0 * m / s2
            if m > 1.0e-6:
                scale = angle / m
                dscale_dm = (dangle_dm * m - angle) / (m * m)
                dscale_dn = dangle_dn / m
            sx = scale * xp
            sy = scale * yp

            hx = _squared_hinge(sx, rot_lower[0], rot_upper[0])
            hy = _squared_hinge(sy, rot_lower[1], rot_upper[1])
            ht = _squared_hinge(twist, rot_lower[2], rot_upper[2])
            swing_excess = wp.max(0.0, angle - max_swing)
            cost += hx[0] + hy[0] + ht[0] + 0.5 * swing_excess * swing_excess

            # Euclidean gradient w.r.t. r = (w, x, y, z).
            gw = float(0.0)
            gx = float(0.0)
            gy = float(0.0)
            gz = float(0.0)
            # Twist: tau = 2 atan2(z, w).
            gw += ht[1] * (-2.0 * z / (n * n))
            gz += ht[1] * (2.0 * w / (n * n))
            # Swing vector: (sx, sy) = scale(m, n) * (xp, yp).
            gsx = hx[1]
            gsy = hy[1]
            gp = gsx * xp + gsy * yp
            gw += scale * (
                gsx * (x / n - (x * w - y * z) * w / n3) + gsy * (y / n - (x * z + y * w) * w / n3)
            )
            gx += scale * (gsx * (w / n) + gsy * (z / n))
            gy += scale * (gsx * (-z / n) + gsy * (w / n))
            gz += scale * (
                gsx * (-y / n - (x * w - y * z) * z / n3)
                + gsy * (x / n - (x * z + y * w) * z / n3)
            )
            gw += gp * dscale_dn * w / n
            gz += gp * dscale_dn * z / n
            # Swing angle: angle = 2 atan2(m, n).
            if m > 1.0e-6:
                gx += gp * dscale_dm * x / m + swing_excess * dangle_dm * x / m
                gy += gp * dscale_dm * y / m + swing_excess * dangle_dm * y / m
            gw += swing_excess * dangle_dn * w / n
            gz += swing_excess * dangle_dn * z / n

            # Left perturbation r <- exp(eps) r gives d/d eps = 0.5 (w g_v - g_w v + v x g_v).
            v = wp.vec3(x, y, z)
            g_v = wp.vec3(gx, gy, gz)
            omega_ref = 0.5 * (w * g_v - gw * v + wp.cross(v, g_v))
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
    anchor: wp.array(dtype=wp.int32),  # (L) -1 disabled, 0 goal, 1 start, 2 base
    offset_position: wp.array(dtype=wp.vec3),  # (L)
    offset_quaternion: wp.array(dtype=wp.vec4),  # (L) wxyz
    position_lower: wp.array(dtype=wp.vec3),  # (L)
    position_upper: wp.array(dtype=wp.vec3),  # (L)
    rotation_lower: wp.array(dtype=wp.vec3),  # (L)
    rotation_upper: wp.array(dtype=wp.vec3),  # (L)
    max_swing: wp.array(dtype=wp.float32),  # (L)
    use_rotation: wp.array(dtype=wp.int32),  # (L)
    weight: wp.array(dtype=wp.float32),  # (1) or (L)
    weight_per_link: int,
    horizon: int,
    num_links: int,
    num_goalset: int,
    out_cost: wp.array(dtype=wp.float32),  # (B*H*L)
    out_position_gradient: wp.array(dtype=wp.vec3),  # (B*H*L)
    out_quaternion_gradient: wp.array(dtype=wp.vec4),  # (B*H*L) wxyz
):
    """Write the weighted cost and gradients of one ``(batch, horizon, link)`` element."""
    tid = wp.tid()
    b = tid / (horizon * num_links)
    li = tid - (tid / num_links) * num_links
    a = anchor[li]
    if a < 0:
        out_cost[tid] = 0.0
        out_position_gradient[tid] = wp.vec3(0.0)
        out_quaternion_gradient[tid] = wp.vec4(0.0)
        return

    c_pos = current_position[tid]
    c_quat = current_quaternion[tid]
    num_anchors = 1
    if a == 0:
        num_anchors = num_goalset

    best_cost = float(-1.0)
    best_grad_pos = wp.vec3(0.0)
    best_omega = wp.vec3(0.0)
    for g in range(num_anchors):
        a_pos = wp.vec3(0.0)
        a_quat = wp.vec4(1.0, 0.0, 0.0, 0.0)
        if a == 0:
            gi = idxs_goal[b] * num_links * num_goalset + li * num_goalset + g
            if gi >= goal_position.shape[0]:
                continue  # anchor data missing; the Python wrapper reports this case
            a_pos = goal_position[gi]
            a_quat = goal_quaternion[gi]
        elif a == 1:
            si = idxs_start[b] * num_links + li
            if si >= start_position.shape[0]:
                continue
            a_pos = start_position[si]
            a_quat = start_quaternion[si]
        ref_pos = a_pos + _quat_rotate(a_quat, offset_position[li])
        ref_quat = _quat_mul(a_quat, offset_quaternion[li])
        cost, grad_pos, omega = _bound_cost(
            c_pos,
            c_quat,
            ref_pos,
            ref_quat,
            position_lower[li],
            position_upper[li],
            rotation_lower[li],
            rotation_upper[li],
            max_swing[li],
            use_rotation[li],
        )
        if best_cost < 0.0 or cost < best_cost:
            best_cost = cost
            best_grad_pos = grad_pos
            best_omega = omega

    w = weight[0]
    if weight_per_link != 0:
        w = weight[li]
    if best_cost < 0.0:
        best_cost = 0.0
    out_cost[tid] = w * best_cost
    out_position_gradient[tid] = w * best_grad_pos
    out_quaternion_gradient[tid] = w * _quat_mul(
        c_quat, wp.vec4(0.0, 2.0 * best_omega[0], 2.0 * best_omega[1], 2.0 * best_omega[2])
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
        offset_position: torch.Tensor,
        offset_quaternion: torch.Tensor,
        position_lower: torch.Tensor,
        position_upper: torch.Tensor,
        rotation_lower: torch.Tensor,
        rotation_upper: torch.Tensor,
        max_swing: torch.Tensor,
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
        ``(B_start*L, 3/4)``; idxs ``(B,)`` int32; per-link tensors ``(L, ...)``; outputs
        ``(B, H, L[, 3/4])``.
        """
        b, h, num_links, _ = current_position.shape
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
                wp.from_torch(anchor, dtype=wp.int32),
                wp.from_torch(offset_position, dtype=wp.vec3),
                wp.from_torch(offset_quaternion, dtype=wp.vec4),
                wp.from_torch(position_lower, dtype=wp.vec3),
                wp.from_torch(position_upper, dtype=wp.vec3),
                wp.from_torch(rotation_lower, dtype=wp.vec3),
                wp.from_torch(rotation_upper, dtype=wp.vec3),
                wp.from_torch(max_swing, dtype=wp.float32),
                wp.from_torch(use_rotation, dtype=wp.int32),
                wp.from_torch(weight.view(-1), dtype=wp.float32),
                int(weight.numel() == num_links and num_links > 1),
                h,
                num_links,
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
        grads = [None] * 23
        if ctx.use_grad_input:
            grad_cost = grad_cost.unsqueeze(-1)
            position_gradient = position_gradient * grad_cost
            quaternion_gradient = quaternion_gradient * grad_cost
        if ctx.needs_input_grad[0]:
            grads[0] = position_gradient
        if ctx.needs_input_grad[1]:
            grads[1] = quaternion_gradient
        return tuple(grads)
