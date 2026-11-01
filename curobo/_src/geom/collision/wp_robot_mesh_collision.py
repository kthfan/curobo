# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Launch wrapper for the exact robot-mesh collision constraint.

Unlike the sphere collision path there is no autograd function here: the mesh constraint is
a boolean gate, not a differentiable cost, so this is a plain function that launches one
kernel per obstacle type present in the scene.
"""

from __future__ import annotations

# Standard Library
from typing import TYPE_CHECKING, Optional, Tuple

# Third Party
import torch
import warp as wp

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_kernel import (
    robot_cuboid_collision_kernel,
    robot_link_near_kernel,
    robot_mesh_collision_kernel,
    robot_voxel_collision_kernel,
)
from curobo._src.geom.collision.wp_robot_mesh_self_collision_kernel import (
    robot_pair_near_kernel,
    robot_self_collision_kernel,
)
from curobo._src.util.warp import get_warp_device_stream

if TYPE_CHECKING:
    from curobo._src.geom.data.data_robot_mesh import RobotMeshData
    from curobo._src.geom.data.data_scene import SceneData


def check_robot_mesh_collision(
    robot_mesh: "RobotMeshData",
    link_transforms: torch.Tensor,
    scene: "SceneData",
    env_query_idx: torch.Tensor,
    use_multi_env: bool,
    margin: float = 0.0,
    check_containment: bool = True,
    in_collision: Optional[torch.Tensor] = None,
    contact_link: Optional[torch.Tensor] = None,
    link_near: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Test the robot's exact triangle geometry against every obstacle in a scene.

    Args:
        robot_mesh: Packed robot collision triangles in link-local frame.
        link_transforms: Per-link transforms ``[batch, horizon, num_links, 3, 4]``.
        scene: Scene holding cuboid, mesh and voxel obstacle data.
        env_query_idx: Environment index per batch element.
        use_multi_env: Whether to use batch-specific environments.
        margin: Clearance distance the robot must keep from obstacles.
        check_containment: If True, also flag links fully contained in a mesh obstacle.
        in_collision: Optional pre-allocated output ``[batch * horizon]`` int32 buffer.
        contact_link: Optional pre-allocated output ``[batch * horizon]`` int32 buffer.
        link_near: Optional pre-allocated ``[3, batch * horizon * num_links]`` int32 scratch
            buffer for the link-level culling, one row per obstacle type.

    Returns:
        Tuple of ``(in_collision, contact_link)``, each of shape ``[batch * horizon]``.
        ``in_collision`` is 1 where the configuration collides, 0 otherwise. ``contact_link``
        holds a colliding link's body index, or -1 where there is no collision.
    """
    batch, horizon = link_transforms.shape[0], link_transforms.shape[1]
    batch_horizon = batch * horizon
    device = link_transforms.device

    if in_collision is None:
        in_collision = torch.zeros(batch_horizon, dtype=torch.int32, device=device)
    else:
        in_collision.zero_()

    if contact_link is None:
        contact_link = torch.empty(batch_horizon, dtype=torch.int32, device=device)
    contact_link.fill_(-1)

    near_size = batch_horizon * robot_mesh.num_links
    if link_near is None:
        link_near = torch.zeros(3, near_size, dtype=torch.int32, device=device)
    else:
        link_near.zero_()

    wp_device, stream = get_warp_device_stream(link_transforms)

    robot_wp = robot_mesh.to_warp(link_transforms.detach())
    env_idx_wp = wp.from_torch(env_query_idx.view(-1), dtype=wp.int32)
    in_collision_wp = wp.from_torch(in_collision.view(-1), dtype=wp.int32)
    contact_link_wp = wp.from_torch(contact_link.view(-1), dtype=wp.int32)

    dim = batch_horizon * robot_mesh.total_tris
    use_multi_env_wp = wp.uint8(use_multi_env)
    margin_wp = wp.float32(margin)

    def flag_near_links(obs_wp: object, row: int) -> wp.array:
        """Flag the links near obstacles of one type; the per-triangle kernel skips the rest."""
        near_wp = wp.from_torch(link_near[row], dtype=wp.int32)
        wp.launch(
            kernel=robot_link_near_kernel,
            dim=near_size,
            inputs=[
                robot_wp,
                obs_wp,
                env_idx_wp,
                near_wp,
                batch_horizon,
                horizon,
                margin_wp,
                use_multi_env_wp,
            ],
            stream=stream,
            device=wp_device,
        )
        return near_wp
    if scene.cuboids is not None:
        cuboids_wp = scene.cuboids.to_warp()
        near_wp = flag_near_links(cuboids_wp, 0)
        wp.launch(
            kernel=robot_cuboid_collision_kernel,
            dim=dim,
            inputs=[
                robot_wp,
                cuboids_wp,
                env_idx_wp,
                in_collision_wp,
                contact_link_wp,
                near_wp,
                batch_horizon,
                horizon,
                margin_wp,
                use_multi_env_wp,
            ],
            stream=stream,
            device=wp_device,
        )

    if scene.meshes is not None:
        meshes_wp = scene.meshes.to_warp()
        near_wp = flag_near_links(meshes_wp, 1)
        wp.launch(
            kernel=robot_mesh_collision_kernel,
            dim=dim,
            inputs=[
                robot_wp,
                meshes_wp,
                env_idx_wp,
                in_collision_wp,
                contact_link_wp,
                near_wp,
                batch_horizon,
                horizon,
                margin_wp,
                use_multi_env_wp,
                wp.uint8(check_containment),
            ],
            stream=stream,
            device=wp_device,
        )

    if scene.voxels is not None:
        voxels_wp = scene.voxels.to_warp()
        near_wp = flag_near_links(voxels_wp, 2)
        wp.launch(
            kernel=robot_voxel_collision_kernel,
            dim=dim,
            inputs=[
                robot_wp,
                voxels_wp,
                env_idx_wp,
                in_collision_wp,
                contact_link_wp,
                near_wp,
                batch_horizon,
                horizon,
                margin_wp,
                use_multi_env_wp,
            ],
            stream=stream,
            device=wp_device,
        )

    return in_collision, contact_link


def check_robot_mesh_self_collision(
    robot_mesh: "RobotMeshData",
    link_transforms: torch.Tensor,
    margin: float = 0.0,
    in_collision: Optional[torch.Tensor] = None,
    contact_pair: Optional[torch.Tensor] = None,
    pair_near: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Test the robot's link meshes against each other, for the pairs it was prepared with.

    Args:
        robot_mesh: Packed robot collision triangles, prepared with
            :meth:`~curobo._src.geom.data.data_robot_mesh.RobotMeshData.set_self_collision_pairs`.
        link_transforms: Per-link transforms ``[batch, horizon, num_links, 3, 4]``.
        margin: Clearance distance the links must keep from each other.
        in_collision: Optional pre-allocated output ``[batch * horizon]`` int32 buffer.
        contact_pair: Optional pre-allocated output ``[batch * horizon]`` int32 buffer.
        pair_near: Optional pre-allocated ``[batch * horizon * num_pairs]`` int32 scratch
            buffer for the pair-level culling.

    Returns:
        Tuple of ``(in_collision, contact_pair)``, each of shape ``[batch * horizon]``.
        ``in_collision`` is 1 where two links collide, 0 otherwise. ``contact_pair`` holds the
        index of a colliding pair in ``robot_mesh.self_collision_pairs``, or -1.
    """
    batch, horizon = link_transforms.shape[0], link_transforms.shape[1]
    batch_horizon = batch * horizon
    device = link_transforms.device

    if in_collision is None:
        in_collision = torch.zeros(batch_horizon, dtype=torch.int32, device=device)
    else:
        in_collision.zero_()
    if contact_pair is None:
        contact_pair = torch.empty(batch_horizon, dtype=torch.int32, device=device)
    contact_pair.fill_(-1)

    pairs = robot_mesh.self_collision_pairs
    if pairs is None:
        return in_collision, contact_pair
    num_pairs = pairs.shape[0]
    num_work = robot_mesh.self_work_tri.shape[0]
    if pair_near is None:
        pair_near = torch.empty(batch_horizon * num_pairs, dtype=torch.int32, device=device)

    wp_device, stream = get_warp_device_stream(link_transforms)
    robot_wp = robot_mesh.to_warp(link_transforms.detach())
    pairs_wp = wp.from_torch(pairs, dtype=wp.int32)
    pair_near_wp = wp.from_torch(pair_near, dtype=wp.int32)
    margin_wp = wp.float32(margin)

    wp.launch(
        kernel=robot_pair_near_kernel,
        dim=batch_horizon * num_pairs,
        inputs=[robot_wp, pairs_wp, pair_near_wp, num_pairs, batch_horizon, margin_wp],
        stream=stream,
        device=wp_device,
    )
    wp.launch(
        kernel=robot_self_collision_kernel,
        dim=batch_horizon * num_work,
        inputs=[
            robot_wp,
            pairs_wp,
            pair_near_wp,
            wp.from_torch(robot_mesh.link_mesh_ids, dtype=wp.uint64),
            wp.from_torch(robot_mesh.link_tri_start, dtype=wp.int32),
            wp.from_torch(robot_mesh.self_work_tri, dtype=wp.int32),
            wp.from_torch(robot_mesh.self_work_pair, dtype=wp.int32),
            wp.from_torch(in_collision.view(-1), dtype=wp.int32),
            wp.from_torch(contact_pair.view(-1), dtype=wp.int32),
            num_pairs,
            num_work,
            batch_horizon,
            margin_wp,
        ],
        stream=stream,
        device=wp_device,
    )
    return in_collision, contact_pair
