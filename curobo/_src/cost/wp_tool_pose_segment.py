# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Warp kernel for the tool-pose cost with one criteria slot per trajectory sample.

The upstream kernel in :mod:`~curobo._src.cost.wp_tool_pose` selects one of two criteria per
tool frame: terminal on the last valid sample and path on every other sample. This kernel reads
a slot index per ``(batch, horizon)`` sample instead, so a trajectory can use different criteria
in different segments. Criteria tables hold ``num_slots * num_links`` entries, indexed by
``slot * num_links + link``. The analytic gradients and outputs match the upstream kernel.
"""

from __future__ import annotations

from typing import Optional

import torch
import warp as wp

from curobo._src.cost.tool_pose_criteria import PoseErrorFrame
from curobo._src.cost.wp_pose_metric import (
    angular_cotangent_to_wxyz_quaternion_cotangent,
    compute_quadratic_pose_metric,
)
from curobo._src.cost.wp_tool_pose import ToolPoseDistance
from curobo._src.util.logging import log_and_raise
from curobo._src.util.warp import get_warp_device_stream, warp_kernel

_GOAL_ERROR_FRAME = wp.constant(wp.int32(PoseErrorFrame.GOAL))


def create_segment_pose_distance_kernel(num_goalset: int):
    """Create a slot-indexed goal-set kernel specialized by its compile-time loop bound."""

    def _segment_pose_distance_template(
        current_position: wp.array(dtype=wp.vec3),
        current_quat: wp.array(dtype=wp.vec4),
        goal_position: wp.array(dtype=wp.vec3),
        goal_quat: wp.array(dtype=wp.vec4),
        idxs_goal: wp.array(dtype=wp.int32),
        valid_steps: wp.array(dtype=wp.int32),
        sample_slot: wp.array(dtype=wp.int32),
        position_orientation_weight: wp.array(dtype=wp.float32),
        position_metric: wp.array(dtype=wp.mat33),
        rotation_metric: wp.array(dtype=wp.mat33),
        pose_tolerance: wp.array(dtype=wp.vec2),
        error_frame: wp.array(dtype=wp.int32),
        out_distance: wp.array(dtype=wp.float32),
        out_position_distance: wp.array(dtype=wp.float32),
        out_rotation_distance: wp.array(dtype=wp.float32),
        out_position_gradient: wp.array(dtype=wp.vec3),
        out_rotation_gradient: wp.array(dtype=wp.vec4),
        out_goalset_idx: wp.array(dtype=wp.int32),
        batch_size: wp.int32,
        horizon: wp.int32,
        goal_horizon: wp.int32,
        num_links: wp.int32,
    ):
        thread_index = wp.tid()
        if thread_index >= batch_size * horizon * num_links:
            return

        batch_index = thread_index / (horizon * num_links)
        horizon_index = (thread_index - batch_index * horizon * num_links) / num_links
        link_index = thread_index - batch_index * horizon * num_links - horizon_index * num_links

        slot = sample_slot[batch_index * horizon + horizon_index]
        criteria_index = slot * num_links + link_index
        tolerance = pose_tolerance[criteria_index]
        local_position_metric = position_metric[criteria_index]
        local_rotation_metric = rotation_metric[criteria_index]
        metric_trace = (
            local_position_metric[0, 0]
            + local_position_metric[1, 1]
            + local_position_metric[2, 2]
            + local_rotation_metric[0, 0]
            + local_rotation_metric[1, 1]
            + local_rotation_metric[2, 2]
        )
        if metric_trace <= 0.0 or horizon_index >= valid_steps[batch_index]:
            out_distance[2 * thread_index] = 0.0
            out_distance[2 * thread_index + 1] = 0.0
            out_position_distance[thread_index] = 0.0
            out_rotation_distance[thread_index] = 0.0
            out_position_gradient[thread_index] = wp.vec3(0.0, 0.0, 0.0)
            out_rotation_gradient[thread_index] = wp.vec4(0.0, 0.0, 0.0, 0.0)
            out_goalset_idx[thread_index] = 0
            return
        position_weight = position_orientation_weight[0]
        rotation_weight = position_orientation_weight[1]
        selected_goal_batch = idxs_goal[batch_index]
        use_goal_frame = error_frame[criteria_index] == _GOAL_ERROR_FRAME

        current_position_world = current_position[thread_index]
        current_quat_wxyz = current_quat[thread_index]
        current_quat_world = wp.quat(
            current_quat_wxyz[1],
            current_quat_wxyz[2],
            current_quat_wxyz[3],
            current_quat_wxyz[0],
        )
        best_cost = wp.float32(-1.0)
        best_goal_index = wp.int32(0)
        best_position_cost = wp.float32(0.0)
        best_rotation_cost = wp.float32(0.0)
        best_position_distance = wp.float32(0.0)
        best_rotation_distance = wp.float32(0.0)
        best_position_gradient = wp.vec3(0.0, 0.0, 0.0)
        best_rotation_gradient = wp.vec3(0.0, 0.0, 0.0)
        best_goal_transform = wp.transform(wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0))

        for goalset_index in range(num_goalset):
            target_step = wp.int32(0)
            if goal_horizon > 1:
                target_step = horizon_index
            goal_flat_index = (
                (selected_goal_batch * goal_horizon + target_step) * num_links * num_goalset
                + link_index * num_goalset
                + goalset_index
            )
            goal_position_world = goal_position[goal_flat_index]
            goal_quat_wxyz = goal_quat[goal_flat_index]
            goal_quat_world = wp.quat(
                goal_quat_wxyz[1],
                goal_quat_wxyz[2],
                goal_quat_wxyz[3],
                goal_quat_wxyz[0],
            )
            goal_transform_world = wp.transform(
                wp.vec3(0.0, 0.0, 0.0), wp.quat(0.0, 0.0, 0.0, 1.0)
            )

            metric_current_position = current_position_world
            metric_current_quat = current_quat_world
            metric_goal_position = goal_position_world
            metric_goal_quat = goal_quat_world
            if use_goal_frame:
                goal_transform_world = wp.transform(goal_position_world, goal_quat_world)
                current_transform_world = wp.transform(current_position_world, current_quat_world)
                current_transform_goal = wp.transform_multiply(
                    wp.transform_inverse(goal_transform_world),
                    current_transform_world,
                )
                metric_current_position = wp.transform_get_translation(current_transform_goal)
                metric_current_quat = wp.transform_get_rotation(current_transform_goal)
                metric_goal_position = wp.vec3(0.0, 0.0, 0.0)
                metric_goal_quat = wp.quat(0.0, 0.0, 0.0, 1.0)

            (
                position_cost,
                rotation_cost,
                position_distance,
                rotation_distance,
                position_gradient,
                rotation_gradient,
            ) = compute_quadratic_pose_metric(
                metric_current_position,
                metric_goal_position,
                metric_current_quat,
                metric_goal_quat,
                local_position_metric,
                local_rotation_metric,
                tolerance[0],
                tolerance[1],
                position_weight,
                rotation_weight,
            )

            total_cost = position_cost + rotation_cost
            if best_cost < 0.0 or total_cost < best_cost:
                best_cost = total_cost
                best_goal_index = goalset_index
                best_position_cost = position_cost
                best_rotation_cost = rotation_cost
                best_position_distance = position_distance
                best_rotation_distance = rotation_distance
                best_position_gradient = position_gradient
                best_rotation_gradient = rotation_gradient
                best_goal_transform = goal_transform_world

        if use_goal_frame:
            best_position_gradient = wp.transform_vector(
                best_goal_transform, best_position_gradient
            )
            best_rotation_gradient = wp.transform_vector(
                best_goal_transform, best_rotation_gradient
            )

        quaternion_gradient_wxyz = angular_cotangent_to_wxyz_quaternion_cotangent(
            best_rotation_gradient, current_quat_world
        )

        out_distance[2 * thread_index] = best_position_cost
        out_distance[2 * thread_index + 1] = best_rotation_cost
        out_position_distance[thread_index] = best_position_distance
        out_rotation_distance[thread_index] = best_rotation_distance
        out_position_gradient[thread_index] = best_position_gradient
        out_rotation_gradient[thread_index] = quaternion_gradient_wxyz
        out_goalset_idx[thread_index] = best_goal_index

    kernel_name = f"segment_pose_distance_{num_goalset}_v1"
    return warp_kernel(kernel_name)(_segment_pose_distance_template)


class SegmentToolPoseDistance(ToolPoseDistance):
    """Evaluate the slot-indexed pose metric; the backward pass is the upstream one."""

    @staticmethod
    def forward(
        ctx,
        current_position: torch.Tensor,
        current_quat: torch.Tensor,
        goal_position: torch.Tensor,
        goal_quat: torch.Tensor,
        idxs_goal: torch.Tensor,
        position_orientation_weight: torch.Tensor,
        position_metric: torch.Tensor,
        rotation_metric: torch.Tensor,
        pose_tolerance: torch.Tensor,
        error_frame: torch.Tensor,
        sample_slot: torch.Tensor,
        out_distance: torch.Tensor,
        out_position_distance: torch.Tensor,
        out_rotation_distance: torch.Tensor,
        out_position_gradient: torch.Tensor,
        out_rotation_gradient: torch.Tensor,
        out_goalset_idx: torch.Tensor,
        use_grad_input: bool,
        warp_kernel,
        valid_steps: Optional[torch.Tensor] = None,
    ):
        """Compute tool-pose costs with the criteria slot of every sample.

        Args:
            current_position: Tool positions ``[batch, horizon, links, 3]``.
            current_quat: Tool quaternions (wxyz) ``[batch, horizon, links, 4]``.
            goal_position: Goals ``[goals, goal_horizon, links, goalset, 3]``, with
                ``goal_horizon`` one or ``horizon``.
            goal_quat: Goal quaternions (wxyz) with the same leading shape.
            idxs_goal: Goal batch index per problem ``[batch, 1]``.
            position_orientation_weight: Position and rotation weights ``[2]``.
            position_metric: Position metrics ``[num_slots * links, 3, 3]``.
            rotation_metric: Rotation metrics ``[num_slots * links, 3, 3]``.
            pose_tolerance: Cost deadbands ``[num_slots * links, 2]``.
            error_frame: Error frame per entry ``[num_slots * links]`` (int32).
            sample_slot: Criteria slot per sample ``[batch, horizon]`` (int32).
            out_distance: Output costs ``[batch, horizon, 2 * links]``.
            out_position_distance: Output position distances ``[batch, horizon, links]``.
            out_rotation_distance: Output rotation distances ``[batch, horizon, links]``.
            out_position_gradient: Output position gradients ``[batch, horizon, links, 3]``.
            out_rotation_gradient: Output quaternion gradients ``[batch, horizon, links, 4]``.
            out_goalset_idx: Output selected goal ``[batch, horizon, links]`` (int32).
            use_grad_input: Scale gradients by the incoming cotangent.
            warp_kernel: Kernel from :func:`create_segment_pose_distance_kernel`.
            valid_steps: Valid samples per problem ``[batch]`` (int32); later samples are zero.
        """
        ctx.set_materialize_grads(False)
        if goal_position.ndim == 4:
            goal_position = goal_position.unsqueeze(1)
            goal_quat = goal_quat.unsqueeze(1)
        batch_size, horizon, num_links, _ = current_position.shape
        if current_quat.shape != (batch_size, horizon, num_links, 4):
            log_and_raise("current_quat must have shape (b, h, num_links, 4)")
        if goal_position.ndim != 5 or goal_position.shape[:-1] != goal_quat.shape[:-1]:
            log_and_raise("goal poses must have shape (goals, horizon, links, goalset, 3/4)")
        goal_horizon = goal_position.shape[1]
        if goal_horizon not in (1, horizon) or goal_position.shape[2] != num_links:
            log_and_raise("goal poses must match the evaluation horizon and links")
        if valid_steps is None:
            valid_steps = torch.full(
                (batch_size,), horizon, device=current_position.device, dtype=torch.int32
            )
        if valid_steps.shape != (batch_size,) or valid_steps.dtype != torch.int32:
            log_and_raise("valid_steps must be int32 with shape (batch_size,)")
        if sample_slot.shape != (batch_size, horizon) or sample_slot.dtype != torch.int32:
            log_and_raise("sample_slot must be int32 with shape (batch_size, horizon)")
        entries = error_frame.shape[0]
        if entries % num_links != 0:
            log_and_raise("criteria tables must hold num_slots * num_links entries")
        for name, tensor, shape in (
            ("position_metric", position_metric, (entries, 3, 3)),
            ("rotation_metric", rotation_metric, (entries, 3, 3)),
            ("pose_tolerance", pose_tolerance, (entries, 2)),
        ):
            if tensor.shape != shape:
                log_and_raise(f"{name} must have shape {shape}, got {tuple(tensor.shape)}")

        ctx.use_grad_input = use_grad_input
        ctx.input_count = len(ctx.needs_input_grad)
        wp_device, wp_stream = get_warp_device_stream(current_position)
        wp.launch(
            kernel=warp_kernel,
            dim=batch_size * horizon * num_links,
            inputs=[
                wp.from_torch(current_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(current_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(goal_position.detach().view(-1, 3), dtype=wp.vec3),
                wp.from_torch(goal_quat.detach().view(-1, 4), dtype=wp.vec4),
                wp.from_torch(idxs_goal.detach().view(-1), dtype=wp.int32),
                wp.from_torch(valid_steps.view(-1), dtype=wp.int32),
                wp.from_torch(sample_slot.view(-1), dtype=wp.int32),
                wp.from_torch(position_orientation_weight.view(-1), dtype=wp.float32),
                wp.from_torch(position_metric.view(-1, 3, 3), dtype=wp.mat33),
                wp.from_torch(rotation_metric.view(-1, 3, 3), dtype=wp.mat33),
                wp.from_torch(pose_tolerance.view(-1, 2), dtype=wp.vec2),
                wp.from_torch(error_frame.view(-1), dtype=wp.int32),
                wp.from_torch(out_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_rotation_distance.view(-1), dtype=wp.float32),
                wp.from_torch(out_position_gradient.view(-1, 3), dtype=wp.vec3),
                wp.from_torch(out_rotation_gradient.view(-1, 4), dtype=wp.vec4),
                wp.from_torch(out_goalset_idx.view(-1), dtype=wp.int32),
                batch_size,
                horizon,
                goal_horizon,
                num_links,
            ],
            device=wp_device,
            stream=wp_stream,
            adjoint=False,
        )
        ctx.mark_non_differentiable(out_position_distance, out_rotation_distance, out_goalset_idx)
        ctx.save_for_backward(out_position_gradient, out_rotation_gradient)
        return out_distance, out_position_distance, out_rotation_distance, out_goalset_idx
