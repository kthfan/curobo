# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Warp kernels for the signed clearance between the robot's link meshes and the scene.

Where the collision kernels answer *does a configuration collide?*, these kernels measure *how
far is each link from the scene?* The result is a signed distance per (configuration, link):

    - Positive: the exact smallest distance between the link's triangles and the obstacles
      (cuboids and meshes; resolution-limited for voxels).
    - Zero: the link's surface crosses an obstacle's surface, but no link vertex lies inside it.
    - Negative: the deepest link vertex inside an obstacle, read from the obstacle's SDF. It
      bounds the penetration depth from below.

Distances are capped at ``max_distance``: links farther away report ``max_distance``, which
keeps the link-level culling of the collision kernels effective. Links without geometry, or
skipped by the scene check, report ``max_distance`` as well.

Parallelization matches the collision kernels: one thread per ``(configuration, robot
triangle)``, each folding its distance into its link's entry with an atomic minimum. The
kernels are forward-only: there is no gradient.

Cuboids and meshes take two passes. :func:`robot_vertex_bound_kernel` first writes the
distance of the link's vertices to the scene, an upper bound on the link's distance. The exact
kernels then start from that bound and skip every triangle that cannot come closer, which keeps
large caps and dense meshes affordable: without it, every triangle would test every obstacle
feature within ``max_distance``.
"""

# Standard Library
from typing import Any

# Third Party
import warp as wp

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_kernel import (
    MAX_VOXEL_SAMPLES_PER_EDGE,
    compute_local_sdf,
    is_obs_enabled,
    load_obstacle_transform,
    segment_segment_dist_sq,
    tri_aabb_hi,
    tri_aabb_lo,
    triangle_box_overlap,
    triangles_intersect,
)
from curobo._src.geom.data.data_cuboid import CuboidDataWarp
from curobo._src.geom.data.data_mesh import MeshDataWarp
from curobo._src.geom.data.data_robot_mesh import RobotMeshWarp, transform_point_link
from curobo._src.geom.data.data_voxel import VoxelDataWarp

#: Squared distance returned when no feature pair is closer than the query bound.
FAR_DIST_SQ = wp.constant(1.0e30)

# =============================================================================
# Exact distance helpers
# =============================================================================


@wp.func
def point_face_dist_sq(p: wp.vec3, a: wp.vec3, b: wp.vec3, c: wp.vec3) -> wp.float32:
    """Squared distance from ``p`` to the interior of triangle ``abc``.

    Returns :data:`FAR_DIST_SQ` when ``p`` does not project inside the triangle, or when the
    triangle is degenerate: a closest point on the triangle's boundary is measured by the
    edge-edge pairs of :func:`triangle_triangle_dist_sq`.
    """
    n = wp.cross(b - a, c - a)
    n_sq = wp.dot(n, n)
    if n_sq <= 1.0e-20:
        return FAR_DIST_SQ
    if wp.dot(wp.cross(b - a, p - a), n) < 0.0:
        return FAR_DIST_SQ
    if wp.dot(wp.cross(c - b, p - b), n) < 0.0:
        return FAR_DIST_SQ
    if wp.dot(wp.cross(a - c, p - c), n) < 0.0:
        return FAR_DIST_SQ
    h = wp.dot(p - a, n)
    return h * h / n_sq


@wp.func
def triangle_triangle_dist_sq(
    a0: wp.vec3,
    a1: wp.vec3,
    a2: wp.vec3,
    b0: wp.vec3,
    b1: wp.vec3,
    b2: wp.vec3,
) -> wp.float32:
    """Squared distance between two non-intersecting triangles.

    The distance between two disjoint triangles is attained between a vertex of one and the
    face interior of the other, or between an edge of each, so the 6 vertex-face and 9
    edge-edge pairs give it exactly. Intersecting triangles must be caught beforehand with
    :func:`~curobo._src.geom.collision.wp_robot_mesh_kernel.triangles_intersect`.
    """
    d = point_face_dist_sq(a0, b0, b1, b2)
    d = wp.min(d, point_face_dist_sq(a1, b0, b1, b2))
    d = wp.min(d, point_face_dist_sq(a2, b0, b1, b2))
    d = wp.min(d, point_face_dist_sq(b0, a0, a1, a2))
    d = wp.min(d, point_face_dist_sq(b1, a0, a1, a2))
    d = wp.min(d, point_face_dist_sq(b2, a0, a1, a2))
    for i in range(3):
        pa = a0
        qa = a1
        if i == 1:
            pa = a1
            qa = a2
        elif i == 2:
            pa = a2
            qa = a0
        d = wp.min(d, segment_segment_dist_sq(pa, qa, b0, b1))
        d = wp.min(d, segment_segment_dist_sq(pa, qa, b1, b2))
        d = wp.min(d, segment_segment_dist_sq(pa, qa, b2, b0))
    return d


@wp.func
def box_corner(half: wp.vec3, k: wp.int32) -> wp.vec3:
    """Corner ``k`` in ``[0, 8)`` of box ``[-half, half]``; bit ``i`` sets the sign of axis i."""
    sx = wp.float32(1.0)
    sy = wp.float32(1.0)
    sz = wp.float32(1.0)
    if (k & 1) == 0:
        sx = -1.0
    if (k & 2) == 0:
        sy = -1.0
    if (k & 4) == 0:
        sz = -1.0
    return wp.vec3(sx * half[0], sy * half[1], sz * half[2])


@wp.func
def aabb_dist_sq(lo0: wp.vec3, hi0: wp.vec3, lo1: wp.vec3, hi1: wp.vec3) -> wp.float32:
    """Squared distance between two axis-aligned boxes, 0 where they overlap."""
    gap = wp.vec3(
        wp.max(wp.max(lo0[0] - hi1[0], lo1[0] - hi0[0]), 0.0),
        wp.max(wp.max(lo0[1] - hi1[1], lo1[1] - hi0[1]), 0.0),
        wp.max(wp.max(lo0[2] - hi1[2], lo1[2] - hi0[2]), 0.0),
    )
    return wp.dot(gap, gap)


@wp.func
def triangle_box_dist_sq(
    a0: wp.vec3,
    a1: wp.vec3,
    a2: wp.vec3,
    half: wp.vec3,
    outside0: wp.float32,
    outside1: wp.float32,
    outside2: wp.float32,
    bound_sq: wp.float32,
) -> wp.float32:
    """Squared distance between a triangle and a disjoint box ``[-half, half]``, up to a bound.

    The closest pair of two disjoint convex polytopes is a vertex of one and a face of the
    other, or an edge of each. The triangle vertices against the box are given as
    ``outside0..2`` (the box SDF); the box corners are measured against the triangle face and
    the 12 box edges against the 3 triangle edges. Corners and edges whose bounding boxes are
    farther from the triangle's than the running minimum are skipped.

    Args:
        a0: First triangle vertex in the box frame.
        a1: Second triangle vertex in the box frame.
        a2: Third triangle vertex in the box frame.
        half: Box half extents.
        outside0: Box SDF at ``a0``, positive.
        outside1: Box SDF at ``a1``, positive.
        outside2: Box SDF at ``a2``, positive.
        bound_sq: Squared distance already known; the result is at most this.
    """
    m = wp.min(outside0, wp.min(outside1, outside2))
    d = wp.min(m * m, bound_sq)
    lo = tri_aabb_lo(a0, a1, a2, 0.0)
    hi = tri_aabb_hi(a0, a1, a2, 0.0)
    for k in range(8):
        corner = box_corner(half, k)
        if aabb_dist_sq(lo, hi, corner, corner) < d:
            d = wp.min(d, point_face_dist_sq(corner, a0, a1, a2))
        # Each corner starts the edges along the axes where its coordinate is negative, so
        # every one of the 12 edges is visited once.
        for axis in range(3):
            if (k & (1 << axis)) == 0:
                other = wp.vec3(corner)
                other[axis] = half[axis]
                if aabb_dist_sq(lo, hi, corner, other) < d:
                    d = wp.min(d, segment_segment_dist_sq(a0, a1, corner, other))
                    d = wp.min(d, segment_segment_dist_sq(a1, a2, corner, other))
                    d = wp.min(d, segment_segment_dist_sq(a2, a0, corner, other))
    return d


@wp.func
def deepest_vertex(
    s0: wp.float32,
    s1: wp.float32,
    s2: wp.float32,
) -> wp.float32:
    """Signed distance of a triangle crossing an obstacle: its deepest vertex, at most 0."""
    return wp.min(wp.min(wp.min(s0, s1), s2), 0.0)


@wp.func
def triangle_centroid_radius(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3) -> wp.vec4:
    """Centroid of a triangle and the largest distance from it to a vertex, as ``(c, r)``.

    Every point of the triangle lies within ``r`` of the centroid, so an obstacle whose
    distance to the centroid is ``d`` is at least ``d - r`` away from the triangle.
    """
    c = (a0 + a1 + a2) / 3.0
    r = wp.max(wp.length(a0 - c), wp.max(wp.length(a1 - c), wp.length(a2 - c)))
    return wp.vec4(c[0], c[1], c[2], r)


@wp.func
def vertex_bound(
    obs: CuboidDataWarp,
    env_idx: wp.int32,
    o: wp.int32,
    p: wp.vec3,
    cap: wp.float32,
) -> wp.float32:
    """Signed distance from a link vertex to a cuboid, an upper bound on the link's distance."""
    return compute_local_sdf(obs, env_idx, o, p)


@wp.func
def vertex_bound(  # noqa: F811 -- Warp overload for mesh obstacles.
    obs: MeshDataWarp,
    env_idx: wp.int32,
    o: wp.int32,
    p: wp.vec3,
    cap: wp.float32,
) -> wp.float32:
    """Unsigned distance from a link vertex to a mesh surface, or ``cap`` when it is farther.

    The unsigned distance bounds the link's distance from above with or without containment,
    and skips the ray casts that decide the sign.
    """
    mesh_id = obs.mesh_ids[env_idx * obs.max_n + o]
    query = wp.mesh_query_point_no_sign(mesh_id, p, cap)
    if not query.result:
        return cap
    closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
    return wp.length(closest - p)


# =============================================================================
# Kernels
# =============================================================================


@wp.kernel(enable_backward=False)
def robot_vertex_bound_kernel(
    robot: RobotMeshWarp,
    obs: Any,
    env_query_idx: wp.array(dtype=wp.int32),
    link_distance: wp.array(dtype=wp.float32),
    link_near: wp.array(dtype=wp.int32),
    batch_horizon: wp.int32,
    horizon: wp.int32,
    max_distance: wp.float32,
    use_multi_env: wp.uint8,
):
    """Lower each link's distance to the distance of its vertices to cuboids or meshes.

    The distance from any point of a link to an obstacle bounds the link's distance from
    above, so the exact kernels can start from it and skip the triangles that cannot come
    closer. Run it for every cuboid and mesh set before the exact kernels.

    Args:
        robot: Robot collision triangles and per-link transforms.
        obs: Cuboid or mesh obstacle set.
        env_query_idx: Environment index per batch element.
        link_distance: Output per (configuration, link), pre-filled with ``max_distance``.
        link_near: Per (configuration, link) flag from ``robot_link_near_kernel``.
        batch_horizon: Number of configurations (batch * horizon).
        horizon: Time steps per batch element; maps a configuration to its batch index.
        max_distance: Distance cap.
        use_multi_env: If 1, look up the environment from ``env_query_idx``.
    """
    tid = wp.tid()
    cfg_idx = tid / robot.total_tris
    tri_idx = tid - cfg_idx * robot.total_tris
    if cfg_idx >= batch_horizon:
        return

    link_idx = robot.tri_link[tri_idx]
    if link_idx < 0:
        return
    row = cfg_idx * robot.num_links + link_idx
    if link_near[row] == 0:
        return

    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[cfg_idx / horizon]

    best = max_distance
    n_obs = obs.n_per_env[env_idx]
    for o in range(n_obs):
        if not is_obs_enabled(obs, env_idx, o):
            continue
        inv_tf = load_obstacle_transform(obs, env_idx, o)
        for k in range(3):
            p = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + k])
            best = wp.min(best, vertex_bound(obs, env_idx, o, wp.transform_point(inv_tf, p), best))

    if best < max_distance:
        wp.atomic_min(link_distance, row, best)


@wp.kernel(enable_backward=False)
def robot_cuboid_distance_kernel(
    robot: RobotMeshWarp,
    obs: CuboidDataWarp,
    env_query_idx: wp.array(dtype=wp.int32),
    link_distance: wp.array(dtype=wp.float32),
    link_near: wp.array(dtype=wp.int32),
    batch_horizon: wp.int32,
    horizon: wp.int32,
    max_distance: wp.float32,
    use_multi_env: wp.uint8,
):
    """Exact signed distance between robot links and cuboid obstacles.

    Thread ``tid`` decodes to ``(cfg_idx, tri_idx)`` via ``tid = cfg_idx * total_tris + tri_idx``.

    Args:
        robot: Robot collision triangles and per-link transforms.
        obs: Cuboid obstacle set.
        env_query_idx: Environment index per batch element.
        link_distance: Output per (configuration, link), pre-filled with ``max_distance``.
        link_near: Per (configuration, link) flag from ``robot_link_near_kernel`` run with
            ``max_distance`` as margin; links flagged 0 are farther away and skipped.
        batch_horizon: Number of configurations (batch * horizon).
        horizon: Time steps per batch element; maps a configuration to its batch index.
        max_distance: Distance cap.
        use_multi_env: If 1, look up the environment from ``env_query_idx``.
    """
    tid = wp.tid()
    cfg_idx = tid / robot.total_tris
    tri_idx = tid - cfg_idx * robot.total_tris
    if cfg_idx >= batch_horizon:
        return

    link_idx = robot.tri_link[tri_idx]
    if link_idx < 0:
        return
    row = cfg_idx * robot.num_links + link_idx
    if link_near[row] == 0:
        return

    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[cfg_idx / horizon]

    v0_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 0])
    v1_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 1])
    v2_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 2])

    # The vertex bound, possibly already lowered by other triangles of the link.
    best = link_distance[row]
    n_obs = obs.n_per_env[env_idx]
    for o in range(n_obs):
        if not is_obs_enabled(obs, env_idx, o):
            continue
        flat = env_idx * obs.max_n + o
        inv_tf = load_obstacle_transform(obs, env_idx, o)
        a0 = wp.transform_point(inv_tf, v0_w)
        a1 = wp.transform_point(inv_tf, v1_w)
        a2 = wp.transform_point(inv_tf, v2_w)

        # A triangle whose box is apart from the obstacle is at least that gap away. Otherwise
        # the box SDF, changing no faster than the distance, keeps every point of the
        # triangle, and hence a crossing's deepest vertex, above the centroid's SDF less r.
        half = wp.vec3(obs.dims[flat, 0] * 0.5, obs.dims[flat, 1] * 0.5, obs.dims[flat, 2] * 0.5)
        gap_sq = aabb_dist_sq(
            tri_aabb_lo(a0, a1, a2, 0.0), tri_aabb_hi(a0, a1, a2, 0.0), -half, half
        )
        if gap_sq > 0.0:
            if best <= 0.0 or gap_sq >= best * best:
                continue
        else:
            cr = triangle_centroid_radius(a0, a1, a2)
            if compute_local_sdf(obs, env_idx, o, wp.vec3(cr[0], cr[1], cr[2])) - cr[3] >= best:
                continue

        s0 = compute_local_sdf(obs, env_idx, o, a0)
        s1 = compute_local_sdf(obs, env_idx, o, a1)
        s2 = compute_local_sdf(obs, env_idx, o, a2)
        if gap_sq == 0.0 and triangle_box_overlap(a0, a1, a2, half):
            best = wp.min(best, deepest_vertex(s0, s1, s2))
        elif best > 0.0:
            dist_sq = triangle_box_dist_sq(a0, a1, a2, half, s0, s1, s2, best * best)
            best = wp.min(best, wp.sqrt(dist_sq))

    if best < max_distance:
        wp.atomic_min(link_distance, row, best)


@wp.kernel(enable_backward=False)
def robot_mesh_distance_kernel(
    robot: RobotMeshWarp,
    obs: MeshDataWarp,
    env_query_idx: wp.array(dtype=wp.int32),
    link_distance: wp.array(dtype=wp.float32),
    link_near: wp.array(dtype=wp.int32),
    batch_horizon: wp.int32,
    horizon: wp.int32,
    max_distance: wp.float32,
    use_multi_env: wp.uint8,
    check_containment: wp.uint8,
):
    """Exact signed distance between robot links and mesh obstacles.

    Args:
        robot: Robot collision triangles and per-link transforms.
        obs: Mesh obstacle set.
        env_query_idx: Environment index per batch element.
        link_distance: Output per (configuration, link), pre-filled with ``max_distance``.
        link_near: Per (configuration, link) flag from ``robot_link_near_kernel``.
        batch_horizon: Number of configurations (batch * horizon).
        horizon: Time steps per batch element; maps a configuration to its batch index.
        max_distance: Distance cap; also inflates the BVH query box.
        use_multi_env: If 1, look up the environment from ``env_query_idx``.
        check_containment: If 1, read the obstacle SDF at the link's vertices, so that
            penetration is negative and a link fully inside an obstacle is found. If 0, a
            crossing reports 0 and the obstacle SDF is never read, for open meshes whose
            inside is not defined.
    """
    tid = wp.tid()
    cfg_idx = tid / robot.total_tris
    tri_idx = tid - cfg_idx * robot.total_tris
    if cfg_idx >= batch_horizon:
        return

    link_idx = robot.tri_link[tri_idx]
    if link_idx < 0:
        return
    row = cfg_idx * robot.num_links + link_idx
    if link_near[row] == 0:
        return

    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[cfg_idx / horizon]

    v0_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 0])
    v1_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 1])
    v2_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 2])

    # The vertex bound, possibly already lowered by other triangles of the link.
    best = link_distance[row]
    n_obs = obs.n_per_env[env_idx]
    for o in range(n_obs):
        if not is_obs_enabled(obs, env_idx, o):
            continue
        flat = env_idx * obs.max_n + o
        mesh_id = obs.mesh_ids[flat]
        inv_tf = load_obstacle_transform(obs, env_idx, o)
        a0 = wp.transform_point(inv_tf, v0_w)
        a1 = wp.transform_point(inv_tf, v1_w)
        a2 = wp.transform_point(inv_tf, v2_w)

        # A surface farther than ``r + best`` from the centroid can neither cross the triangle
        # nor come closer than ``best``; only containment is left to check.
        cr = triangle_centroid_radius(a0, a1, a2)
        centroid = wp.vec3(cr[0], cr[1], cr[2])
        near = wp.mesh_query_point_no_sign(mesh_id, centroid, cr[3] + wp.max(best, 0.0))
        best_sq = wp.float32(0.0)
        crossing = wp.bool(False)
        if near.result:
            # The centroid is a point of the triangle: its distance to the surface bounds the
            # triangle's, unless the triangle turns out to be inside.
            closest = wp.mesh_eval_position(mesh_id, near.face, near.u, near.v)
            reach = wp.min(wp.max(best, 0.0), wp.length(closest - centroid))
            best_sq = reach * reach
            lo = tri_aabb_lo(a0, a1, a2, 0.0)
            hi = tri_aabb_hi(a0, a1, a2, 0.0)
            # Only obstacle triangles within ``reach`` of this triangle's box can lower it.
            query = wp.mesh_query_aabb(
                mesh_id, tri_aabb_lo(a0, a1, a2, reach), tri_aabb_hi(a0, a1, a2, reach)
            )
            obs_tri = wp.int32(0)
            while wp.mesh_query_aabb_next(query, obs_tri):
                b0 = wp.mesh_get_point(mesh_id, obs_tri * 3 + 0)
                b1 = wp.mesh_get_point(mesh_id, obs_tri * 3 + 1)
                b2 = wp.mesh_get_point(mesh_id, obs_tri * 3 + 2)
                gap_sq = aabb_dist_sq(
                    lo, hi, tri_aabb_lo(b0, b1, b2, 0.0), tri_aabb_hi(b0, b1, b2, 0.0)
                )
                if gap_sq == 0.0:
                    if triangles_intersect(a0, a1, a2, b0, b1, b2):
                        crossing = True
                        break
                # Boxes farther apart than ``best`` hold no closer pair of points.
                if gap_sq < best_sq:
                    best_sq = wp.min(best_sq, triangle_triangle_dist_sq(a0, a1, a2, b0, b1, b2))

        if crossing:
            depth = wp.float32(0.0)
            if check_containment == wp.uint8(1):
                depth = deepest_vertex(
                    compute_local_sdf(obs, env_idx, o, a0),
                    compute_local_sdf(obs, env_idx, o, a1),
                    compute_local_sdf(obs, env_idx, o, a2),
                )
            best = wp.min(best, depth)
            continue

        # Without a crossing the triangle lies wholly on one side of the surface: one vertex
        # tells whether it is inside, and then its deepest vertex is the distance.
        if check_containment == wp.uint8(1):
            s0 = compute_local_sdf(obs, env_idx, o, a0)
            if s0 < 0.0:
                s1 = compute_local_sdf(obs, env_idx, o, a1)
                s2 = compute_local_sdf(obs, env_idx, o, a2)
                best = wp.min(best, deepest_vertex(s0, s1, s2))
                continue
        if near.result:
            best = wp.min(best, wp.sqrt(best_sq))

    if best < max_distance:
        wp.atomic_min(link_distance, row, best)


@wp.kernel(enable_backward=False)
def robot_voxel_distance_kernel(
    robot: RobotMeshWarp,
    obs: VoxelDataWarp,
    env_query_idx: wp.array(dtype=wp.int32),
    link_distance: wp.array(dtype=wp.float32),
    link_near: wp.array(dtype=wp.int32),
    batch_horizon: wp.int32,
    horizon: wp.int32,
    max_distance: wp.float32,
    use_multi_env: wp.uint8,
):
    """Sampled signed distance between robot links and voxel ESDF obstacles.

    The triangle is sampled as in the voxel collision kernel, at most half a voxel apart, and
    the smallest ESDF value over the samples is kept. This is **resolution-limited**: the ESDF
    itself is interpolated between voxels, and geometry thinner than a voxel can be missed.
    Where the sampling cap makes the samples sparser than half a voxel, the value is lowered
    by the extra reach, so it never exceeds what a half-voxel lattice would report.

    Args:
        robot: Robot collision triangles and per-link transforms.
        obs: Voxel grid obstacle set.
        env_query_idx: Environment index per batch element.
        link_distance: Output per (configuration, link), pre-filled with ``max_distance``.
        link_near: Per (configuration, link) flag from ``robot_link_near_kernel``.
        batch_horizon: Number of configurations (batch * horizon).
        horizon: Time steps per batch element; maps a configuration to its batch index.
        max_distance: Distance cap.
        use_multi_env: If 1, look up the environment from ``env_query_idx``.
    """
    tid = wp.tid()
    cfg_idx = tid / robot.total_tris
    tri_idx = tid - cfg_idx * robot.total_tris
    if cfg_idx >= batch_horizon:
        return

    link_idx = robot.tri_link[tri_idx]
    if link_idx < 0:
        return
    row = cfg_idx * robot.num_links + link_idx
    if link_near[row] == 0:
        return

    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[cfg_idx / horizon]

    v0_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 0])
    v1_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 1])
    v2_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 2])

    best = max_distance
    n_obs = obs.n_per_env[env_idx]
    for o in range(n_obs):
        if not is_obs_enabled(obs, env_idx, o):
            continue
        flat = env_idx * obs.max_n + o
        voxel_size = obs.params[flat, 3]
        inv_tf = load_obstacle_transform(obs, env_idx, o)
        a0 = wp.transform_point(inv_tf, v0_w)
        a1 = wp.transform_point(inv_tf, v1_w)
        a2 = wp.transform_point(inv_tf, v2_w)

        longest = wp.max(wp.length(a1 - a0), wp.max(wp.length(a2 - a1), wp.length(a0 - a2)))
        half_voxel = voxel_size * 0.5
        steps = wp.int32(wp.ceil(longest / wp.max(half_voxel, 1.0e-6)))
        steps = wp.clamp(steps, 1, MAX_VOXEL_SAMPLES_PER_EDGE)
        spacing = longest / wp.float32(steps)
        widening = wp.max(spacing - half_voxel, 0.0) / 1.7320508

        inv_steps = 1.0 / wp.float32(steps)
        for i in range(steps + 1):
            u = wp.float32(i) * inv_steps
            for j in range(steps + 1 - i):
                v = wp.float32(j) * inv_steps
                p = a0 + (a1 - a0) * u + (a2 - a0) * v
                best = wp.min(best, compute_local_sdf(obs, env_idx, o, p) - widening)

    if best < max_distance:
        wp.atomic_min(link_distance, row, best)
