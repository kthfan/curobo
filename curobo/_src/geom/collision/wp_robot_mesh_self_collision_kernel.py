# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Warp kernels for exact robot self-collision between link meshes.

Each tested link pair ``(a, b)`` is checked like a mesh obstacle: every triangle of link ``a``
is moved into link ``b``'s frame and queried against a BVH over ``b``'s triangles, built once
in ``b``'s link frame. The narrowphase is the exact triangle-triangle intersection, plus the
exact triangle-triangle distance against ``margin`` when it is positive. The kernels are
forward-only: there is no gradient.

A pre-pass skips pairs whose link bounding spheres are farther apart than the margin, so the
per-triangle kernel only runs for pairs that are close in a configuration.
"""

# Third Party
import warp as wp

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_kernel import (
    LINK_CULL_EPS,
    tri_aabb_hi,
    tri_aabb_lo,
    triangles_within,
)
from curobo._src.geom.data.data_robot_mesh import RobotMeshWarp, transform_point_link


@wp.func
def inverse_transform_point_link(
    link_tf: wp.array2d(dtype=wp.float32),
    row: wp.int32,
    p: wp.vec3,
) -> wp.vec3:
    """Map a world point into a link's frame with the inverse of its ``[R | t]`` transform.

    Args:
        link_tf: Flattened transform array, 12 floats per link.
        row: Flat index ``cfg_idx * num_links + link_idx``.
        p: Point in world frame.

    Returns:
        The point in the link's frame, ``R^T (p - t)``.
    """
    d = wp.vec3(p[0] - link_tf[row, 3], p[1] - link_tf[row, 7], p[2] - link_tf[row, 11])
    return wp.vec3(
        link_tf[row, 0] * d[0] + link_tf[row, 4] * d[1] + link_tf[row, 8] * d[2],
        link_tf[row, 1] * d[0] + link_tf[row, 5] * d[1] + link_tf[row, 9] * d[2],
        link_tf[row, 2] * d[0] + link_tf[row, 6] * d[1] + link_tf[row, 10] * d[2],
    )


@wp.kernel(enable_backward=False)
def robot_pair_near_kernel(
    robot: RobotMeshWarp,
    pair_links: wp.array2d(dtype=wp.int32),
    pair_near: wp.array(dtype=wp.int32),
    num_pairs: wp.int32,
    batch_horizon: wp.int32,
    margin: wp.float32,
):
    """Flag the link pairs whose bounding spheres are within ``margin`` of each other.

    Args:
        robot: Robot collision triangles, link transforms and link bounding spheres.
        pair_links: Body indices ``[num_pairs, 2]`` of the tested pairs.
        pair_near: Output per ``cfg_idx * num_pairs + pair``: 1 if the pair may touch, else 0.
        num_pairs: Number of tested pairs.
        batch_horizon: Number of configurations (batch * horizon).
        margin: Clearance distance of the check.
    """
    tid = wp.tid()
    cfg_idx = tid / num_pairs
    pair = tid - cfg_idx * num_pairs
    if cfg_idx >= batch_horizon:
        return
    a = pair_links[pair, 0]
    b = pair_links[pair, 1]
    bound_a = robot.link_bounds[a]
    bound_b = robot.link_bounds[b]
    if bound_a[3] < 0.0 or bound_b[3] < 0.0:
        pair_near[tid] = 0
        return
    center_a = transform_point_link(
        robot.link_tf, cfg_idx * robot.num_links + a, wp.vec3(bound_a[0], bound_a[1], bound_a[2])
    )
    center_b = transform_point_link(
        robot.link_tf, cfg_idx * robot.num_links + b, wp.vec3(bound_b[0], bound_b[1], bound_b[2])
    )
    reach = bound_a[3] + bound_b[3] + margin + LINK_CULL_EPS
    if wp.length(center_a - center_b) > reach:
        pair_near[tid] = 0
    else:
        pair_near[tid] = 1


@wp.kernel(enable_backward=False)
def robot_self_collision_kernel(
    robot: RobotMeshWarp,
    pair_links: wp.array2d(dtype=wp.int32),
    pair_near: wp.array(dtype=wp.int32),
    link_mesh_ids: wp.array(dtype=wp.uint64),
    link_tri_start: wp.array(dtype=wp.int32),
    work_tri: wp.array(dtype=wp.int32),
    work_pair: wp.array(dtype=wp.int32),
    in_collision: wp.array(dtype=wp.int32),
    contact_pair: wp.array(dtype=wp.int32),
    num_pairs: wp.int32,
    num_work: wp.int32,
    batch_horizon: wp.int32,
    margin: wp.float32,
):
    """Exact triangle test of one triangle of a pair's first link against its second link.

    One thread per ``(configuration, work item)``; a work item is one triangle slot of the
    first link of a pair.

    Args:
        robot: Robot collision triangles and per-link transforms.
        pair_links: Body indices ``[num_pairs, 2]``; triangles of the first link are tested
            against the BVH of the second.
        pair_near: Per ``cfg_idx * num_pairs + pair`` flag from
            :func:`robot_pair_near_kernel`; pairs flagged 0 are skipped.
        link_mesh_ids: Warp mesh id of each paired link's BVH, by body index. The BVH holds
            the link's triangle slots in its link frame.
        link_tri_start: First triangle slot of each paired link, by body index.
        work_tri: Triangle slot of each work item.
        work_pair: Pair index of each work item.
        in_collision: Output flag per configuration, pre-zeroed.
        contact_pair: Output colliding pair index per configuration, pre-filled with -1.
        num_pairs: Number of tested pairs.
        num_work: Number of work items.
        batch_horizon: Number of configurations (batch * horizon).
        margin: Clearance distance; pairs closer than this collide.
    """
    tid = wp.tid()
    cfg_idx = tid / num_work
    k = tid - cfg_idx * num_work
    if cfg_idx >= batch_horizon:
        return
    if in_collision[cfg_idx] == 1:
        return
    pair = work_pair[k]
    if pair_near[cfg_idx * num_pairs + pair] == 0:
        return
    a = pair_links[pair, 0]
    b = pair_links[pair, 1]
    tri = work_tri[k]
    # Reserved-but-unfilled and disabled slots carry a negative link.
    if robot.tri_link[tri] != a:
        return

    row_a = cfg_idx * robot.num_links + a
    row_b = cfg_idx * robot.num_links + b
    a0 = inverse_transform_point_link(
        robot.link_tf, row_b, transform_point_link(robot.link_tf, row_a, robot.tri_verts[tri * 3])
    )
    a1 = inverse_transform_point_link(
        robot.link_tf,
        row_b,
        transform_point_link(robot.link_tf, row_a, robot.tri_verts[tri * 3 + 1]),
    )
    a2 = inverse_transform_point_link(
        robot.link_tf,
        row_b,
        transform_point_link(robot.link_tf, row_a, robot.tri_verts[tri * 3 + 2]),
    )

    mesh_id = link_mesh_ids[b]
    start_b = link_tri_start[b]
    margin_sq = margin * margin
    query = wp.mesh_query_aabb(
        mesh_id, tri_aabb_lo(a0, a1, a2, margin), tri_aabb_hi(a0, a1, a2, margin)
    )
    face = wp.int32(0)
    while wp.mesh_query_aabb_next(query, face):
        if robot.tri_link[start_b + face] != b:
            continue
        b0 = wp.mesh_get_point(mesh_id, face * 3 + 0)
        b1 = wp.mesh_get_point(mesh_id, face * 3 + 1)
        b2 = wp.mesh_get_point(mesh_id, face * 3 + 2)
        hit = wp.intersect_tri_tri(a0, a1, a2, b0, b1, b2) != 0
        if not hit and margin_sq > 0.0:
            hit = triangles_within(a0, a1, a2, b0, b1, b2, margin_sq)
        if hit:
            wp.atomic_max(in_collision, cfg_idx, 1)
            wp.atomic_max(contact_pair, cfg_idx, pair)
            return
