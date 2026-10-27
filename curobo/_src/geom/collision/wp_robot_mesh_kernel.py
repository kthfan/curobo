# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Warp kernels for exact robot-mesh vs obstacle collision detection.

These kernels answer a single boolean question per configuration: *is the robot's exact
triangle geometry in collision with the scene?* They are forward-only
(``enable_backward=False``) -- there is no gradient and no adjoint code is generated. The
differentiable sphere-based collision cost is unaffected and still drives the optimizer.

Parallelization is one thread per ``(configuration, robot triangle)``. Each thread walks the
obstacle's own acceleration structure, so once a configuration is known to collide its
remaining threads exit immediately.

Accuracy per obstacle type:
    - Mesh: exact triangle-triangle intersection, plus an exact triangle-triangle distance
      test against ``margin`` when it is positive, plus an optional containment pass so a
      link fully swallowed by a large obstacle is still detected.
    - Cuboid: exact triangle-box overlap via the separating axis theorem. Containment is
      handled inherently by the test, so no second pass is needed.
    - Voxel: **resolution-limited, not exact.** The robot triangle is point-sampled against
      the ESDF, so features thinner than the voxel size can be missed.

Known limitation shared by all three: an obstacle small enough to sit entirely *inside* a
robot link, touching none of its triangles, is not detected. Detecting that would require a
point-in-robot-mesh test, which has no acceleration structure here because the robot
geometry is re-posed for every configuration.
"""

# Standard Library
from importlib import import_module

# Third Party
import warp as wp

# CuRobo
from curobo._src.geom.data.data_cuboid import CuboidDataWarp
from curobo._src.geom.data.data_mesh import MeshDataWarp
from curobo._src.geom.data.data_robot_mesh import RobotMeshWarp, transform_point_link
from curobo._src.geom.data.data_voxel import VoxelDataWarp
from curobo._src.geom.data.registry import OBSTACLE_SDF_MODULES

# =============================================================================
# SDF Function Overload Registration
# =============================================================================
#
# Mirrors wp_collision_kernel.py: keeping a variable of the same name in scope makes each
# subsequent wp.func call register an overload rather than a new function.

is_obs_enabled = None
load_obstacle_transform = None
compute_local_sdf = None

for _module_path in OBSTACLE_SDF_MODULES:
    _data_module = import_module(_module_path)
    is_obs_enabled = wp.func(getattr(_data_module, "is_obs_enabled"), module=__name__)
    load_obstacle_transform = wp.func(
        getattr(_data_module, "load_obstacle_transform"), module=__name__
    )
    compute_local_sdf = wp.func(getattr(_data_module, "compute_local_sdf"), module=__name__)

del _module_path, _data_module


# =============================================================================
# Shared Warp Helpers
# =============================================================================


@wp.func
def report_collision(
    in_collision: wp.array(dtype=wp.int32),
    contact_link: wp.array(dtype=wp.int32),
    cfg_idx: wp.int32,
    link_idx: wp.int32,
):
    """Flag a configuration as colliding and record a contacting link."""
    wp.atomic_max(in_collision, cfg_idx, 1)
    wp.atomic_max(contact_link, cfg_idx, link_idx)


@wp.func
def tri_aabb_lo(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, margin: wp.float32) -> wp.vec3:
    """Lower corner of a triangle's axis-aligned bounding box, inflated by ``margin``."""
    return wp.vec3(
        wp.min(wp.min(a0[0], a1[0]), a2[0]) - margin,
        wp.min(wp.min(a0[1], a1[1]), a2[1]) - margin,
        wp.min(wp.min(a0[2], a1[2]), a2[2]) - margin,
    )


@wp.func
def tri_aabb_hi(a0: wp.vec3, a1: wp.vec3, a2: wp.vec3, margin: wp.float32) -> wp.vec3:
    """Upper corner of a triangle's axis-aligned bounding box, inflated by ``margin``."""
    return wp.vec3(
        wp.max(wp.max(a0[0], a1[0]), a2[0]) + margin,
        wp.max(wp.max(a0[1], a1[1]), a2[1]) + margin,
        wp.max(wp.max(a0[2], a1[2]), a2[2]) + margin,
    )


# =============================================================================
# Mesh Obstacles: exact triangle-triangle
# =============================================================================


@wp.func
def point_face_within(
    p: wp.vec3,
    a: wp.vec3,
    b: wp.vec3,
    c: wp.vec3,
    dist_sq: wp.float32,
) -> wp.bool:
    """Check whether ``p`` projects inside triangle ``abc`` closer than ``sqrt(dist_sq)``.

    Only the face interior is tested. A closest point on the triangle boundary is covered by
    the edge-edge tests in :func:`triangles_within`, and a degenerate triangle, having no face
    interior, is skipped.
    """
    n = wp.cross(b - a, c - a)
    n_sq = wp.dot(n, n)
    if n_sq <= 1.0e-20:
        return False
    h = wp.dot(p - a, n)
    # Compare h^2 / |n|^2 < dist_sq without dividing.
    if h * h >= dist_sq * n_sq:
        return False
    if wp.dot(wp.cross(b - a, p - a), n) < 0.0:
        return False
    if wp.dot(wp.cross(c - b, p - b), n) < 0.0:
        return False
    if wp.dot(wp.cross(a - c, p - c), n) < 0.0:
        return False
    return True


@wp.func
def segment_segment_dist_sq(
    p1: wp.vec3,
    q1: wp.vec3,
    p2: wp.vec3,
    q2: wp.vec3,
) -> wp.float32:
    """Squared distance between segments ``p1q1`` and ``p2q2`` (Ericson, RTCD 5.1.9)."""
    d1 = q1 - p1
    d2 = q2 - p2
    r = p1 - p2
    a = wp.dot(d1, d1)
    e = wp.dot(d2, d2)
    f = wp.dot(d2, r)
    eps = wp.float32(1.0e-12)

    s = wp.float32(0.0)
    t = wp.float32(0.0)
    if a <= eps and e <= eps:
        s = 0.0
        t = 0.0
    elif a <= eps:
        s = 0.0
        t = wp.clamp(f / e, 0.0, 1.0)
    else:
        c = wp.dot(d1, r)
        if e <= eps:
            t = 0.0
            s = wp.clamp(-c / a, 0.0, 1.0)
        else:
            b = wp.dot(d1, d2)
            denom = a * e - b * b
            # Near-parallel segments: any s works, the clamps below fix up t.
            if denom > 1.0e-6 * a * e:
                s = wp.clamp((b * f - c * e) / denom, 0.0, 1.0)
            else:
                s = 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t = 0.0
                s = wp.clamp(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t = 1.0
                s = wp.clamp((b - c) / a, 0.0, 1.0)

    diff = (p1 + d1 * s) - (p2 + d2 * t)
    return wp.dot(diff, diff)


@wp.func
def triangles_within(
    a0: wp.vec3,
    a1: wp.vec3,
    a2: wp.vec3,
    b0: wp.vec3,
    b1: wp.vec3,
    b2: wp.vec3,
    dist_sq: wp.float32,
) -> wp.bool:
    """Check whether two non-intersecting triangles are closer than ``sqrt(dist_sq)``.

    The distance between two disjoint triangles is attained either between a vertex of one
    and the face interior of the other, or between an edge of each. Testing the 6
    vertex-face and 9 edge-edge pairs is therefore exact. Intersecting triangles must be
    caught beforehand with ``wp.intersect_tri_tri``.
    """
    if point_face_within(a0, b0, b1, b2, dist_sq):
        return True
    if point_face_within(a1, b0, b1, b2, dist_sq):
        return True
    if point_face_within(a2, b0, b1, b2, dist_sq):
        return True
    if point_face_within(b0, a0, a1, a2, dist_sq):
        return True
    if point_face_within(b1, a0, a1, a2, dist_sq):
        return True
    if point_face_within(b2, a0, a1, a2, dist_sq):
        return True

    for i in range(3):
        pa = a0
        qa = a1
        if i == 1:
            pa = a1
            qa = a2
        elif i == 2:
            pa = a2
            qa = a0
        if segment_segment_dist_sq(pa, qa, b0, b1) < dist_sq:
            return True
        if segment_segment_dist_sq(pa, qa, b1, b2) < dist_sq:
            return True
        if segment_segment_dist_sq(pa, qa, b2, b0) < dist_sq:
            return True
    return False


@wp.kernel(enable_backward=False)
def robot_mesh_collision_kernel(
    robot: RobotMeshWarp,
    obs: MeshDataWarp,
    env_query_idx: wp.array(dtype=wp.int32),
    in_collision: wp.array(dtype=wp.int32),
    contact_link: wp.array(dtype=wp.int32),
    batch_horizon: wp.int32,
    horizon: wp.int32,
    margin: wp.float32,
    use_multi_env: wp.uint8,
    check_containment: wp.uint8,
):
    """Exact triangle-triangle collision between robot links and mesh obstacles.

    Thread ``tid`` decodes to ``(cfg_idx, tri_idx)`` via ``tid = cfg_idx * total_tris + tri_idx``.

    Args:
        robot: Robot collision triangles and per-link transforms.
        obs: Mesh obstacle set.
        env_query_idx: Environment index per configuration.
        in_collision: Output flag per configuration, pre-zeroed.
        contact_link: Output contacting link index per configuration, pre-filled with -1.
        batch_horizon: Number of configurations (batch * horizon).
        margin: Clearance distance. A pair of triangles closer than ``margin`` collides.
        use_multi_env: If 1, look up the environment from ``env_query_idx``.
        check_containment: If 1, also flag links that lie entirely inside an obstacle.
    """
    tid = wp.tid()

    cfg_idx = tid / robot.total_tris
    tri_idx = tid - cfg_idx * robot.total_tris

    if cfg_idx >= batch_horizon:
        return

    # Early out: another thread already proved this configuration collides.
    if in_collision[cfg_idx] == 1:
        return

    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[cfg_idx / horizon]

    link_idx = robot.tri_link[tri_idx]
    # Reserved-but-unfilled slots carry a negative link, and hold no geometry.
    if link_idx < 0:
        return
    row = cfg_idx * robot.num_links + link_idx
    v0_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 0])
    v1_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 1])
    v2_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 2])
    margin_sq = margin * margin

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

        # Broadphase: the obstacle BVH returns only triangles whose AABB overlaps.
        query = wp.mesh_query_aabb(
            mesh_id, tri_aabb_lo(a0, a1, a2, margin), tri_aabb_hi(a0, a1, a2, margin)
        )
        obs_tri = wp.int32(0)
        while wp.mesh_query_aabb_next(query, obs_tri):
            b0 = wp.mesh_get_point(mesh_id, obs_tri * 3 + 0)
            b1 = wp.mesh_get_point(mesh_id, obs_tri * 3 + 1)
            b2 = wp.mesh_get_point(mesh_id, obs_tri * 3 + 2)

            # Narrowphase: exact triangle-triangle predicate, then exact distance < margin.
            if wp.intersect_tri_tri(a0, a1, a2, b0, b1, b2) != 0:
                report_collision(in_collision, contact_link, cfg_idx, link_idx)
                return
            if margin_sq > 0.0 and triangles_within(a0, a1, a2, b0, b1, b2, margin_sq):
                report_collision(in_collision, contact_link, cfg_idx, link_idx)
                return

        # Containment: surface crossing alone misses a link sitting entirely inside a large
        # obstacle, because no triangle pair intersects. A point-in-mesh test on one vertex
        # suffices: a triangle that passed the test above keeps at least ``margin`` from the
        # surface everywhere, so it is wholly inside or wholly outside by that much.
        if check_containment == wp.uint8(1):
            if compute_local_sdf(obs, env_idx, o, a0) < margin:
                report_collision(in_collision, contact_link, cfg_idx, link_idx)
                return


# =============================================================================
# Cuboid Obstacles: exact triangle-box (separating axis theorem)
# =============================================================================


@wp.func
def axis_separates(
    p0: wp.float32,
    p1: wp.float32,
    p2: wp.float32,
    radius: wp.float32,
) -> wp.bool:
    """Check whether one candidate axis separates the projected triangle from the box."""
    if wp.min(wp.min(p0, p1), p2) > radius:
        return True
    if wp.max(wp.max(p0, p1), p2) < -radius:
        return True
    return False


@wp.func
def edge_axis_separates(
    axis: wp.vec3,
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    half: wp.vec3,
) -> wp.bool:
    """Separating axis test for one triangle-edge / box-axis cross product."""
    radius = half[0] * wp.abs(axis[0]) + half[1] * wp.abs(axis[1]) + half[2] * wp.abs(axis[2])
    return axis_separates(wp.dot(axis, v0), wp.dot(axis, v1), wp.dot(axis, v2), radius)


@wp.func
def triangle_box_overlap(
    v0: wp.vec3,
    v1: wp.vec3,
    v2: wp.vec3,
    half: wp.vec3,
) -> wp.bool:
    """Exact triangle vs axis-aligned box overlap test (Akenine-Moller SAT).

    The triangle must already be expressed in the box's local frame, with the box centred at
    the origin. All 13 separating axes are tested, so a triangle fully contained in the box
    is reported as overlapping.

    Args:
        v0: First triangle vertex in box-local frame.
        v1: Second triangle vertex in box-local frame.
        v2: Third triangle vertex in box-local frame.
        half: Box half-extents.

    Returns:
        True if the triangle and box overlap.
    """
    # 3 axes: box face normals.
    if axis_separates(v0[0], v1[0], v2[0], half[0]):
        return False
    if axis_separates(v0[1], v1[1], v2[1], half[1]):
        return False
    if axis_separates(v0[2], v1[2], v2[2], half[2]):
        return False

    e0 = v1 - v0
    e1 = v2 - v1
    e2 = v0 - v2

    # 1 axis: triangle face normal, i.e. box centre vs the triangle plane.
    normal = wp.cross(e0, e1)
    radius = (
        half[0] * wp.abs(normal[0]) + half[1] * wp.abs(normal[1]) + half[2] * wp.abs(normal[2])
    )
    if wp.abs(wp.dot(normal, v0)) > radius:
        return False

    # 9 axes: cross products of each triangle edge with each box axis.
    if edge_axis_separates(wp.vec3(0.0, -e0[2], e0[1]), v0, v1, v2, half):
        return False
    if edge_axis_separates(wp.vec3(e0[2], 0.0, -e0[0]), v0, v1, v2, half):
        return False
    if edge_axis_separates(wp.vec3(-e0[1], e0[0], 0.0), v0, v1, v2, half):
        return False

    if edge_axis_separates(wp.vec3(0.0, -e1[2], e1[1]), v0, v1, v2, half):
        return False
    if edge_axis_separates(wp.vec3(e1[2], 0.0, -e1[0]), v0, v1, v2, half):
        return False
    if edge_axis_separates(wp.vec3(-e1[1], e1[0], 0.0), v0, v1, v2, half):
        return False

    if edge_axis_separates(wp.vec3(0.0, -e2[2], e2[1]), v0, v1, v2, half):
        return False
    if edge_axis_separates(wp.vec3(e2[2], 0.0, -e2[0]), v0, v1, v2, half):
        return False
    if edge_axis_separates(wp.vec3(-e2[1], e2[0], 0.0), v0, v1, v2, half):
        return False

    return True


@wp.kernel(enable_backward=False)
def robot_cuboid_collision_kernel(
    robot: RobotMeshWarp,
    obs: CuboidDataWarp,
    env_query_idx: wp.array(dtype=wp.int32),
    in_collision: wp.array(dtype=wp.int32),
    contact_link: wp.array(dtype=wp.int32),
    batch_horizon: wp.int32,
    horizon: wp.int32,
    margin: wp.float32,
    use_multi_env: wp.uint8,
):
    """Exact triangle-box collision between robot links and cuboid obstacles.

    Args:
        robot: Robot collision triangles and per-link transforms.
        obs: Cuboid obstacle set.
        env_query_idx: Environment index per configuration.
        in_collision: Output flag per configuration, pre-zeroed.
        contact_link: Output contacting link index per configuration, pre-filled with -1.
        batch_horizon: Number of configurations (batch * horizon).
        margin: Clearance distance; inflates the box half-extents.
        use_multi_env: If 1, look up the environment from ``env_query_idx``.
    """
    tid = wp.tid()

    cfg_idx = tid / robot.total_tris
    tri_idx = tid - cfg_idx * robot.total_tris

    if cfg_idx >= batch_horizon:
        return
    if in_collision[cfg_idx] == 1:
        return

    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[cfg_idx / horizon]

    link_idx = robot.tri_link[tri_idx]
    # Reserved-but-unfilled slots carry a negative link, and hold no geometry.
    if link_idx < 0:
        return
    row = cfg_idx * robot.num_links + link_idx
    v0_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 0])
    v1_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 1])
    v2_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 2])

    n_obs = obs.n_per_env[env_idx]
    for o in range(n_obs):
        if not is_obs_enabled(obs, env_idx, o):
            continue

        flat = env_idx * obs.max_n + o
        inv_tf = load_obstacle_transform(obs, env_idx, o)

        a0 = wp.transform_point(inv_tf, v0_w)
        a1 = wp.transform_point(inv_tf, v1_w)
        a2 = wp.transform_point(inv_tf, v2_w)

        half = wp.vec3(
            obs.dims[flat, 0] * 0.5 + margin,
            obs.dims[flat, 1] * 0.5 + margin,
            obs.dims[flat, 2] * 0.5 + margin,
        )

        if triangle_box_overlap(a0, a1, a2, half):
            report_collision(in_collision, contact_link, cfg_idx, link_idx)
            return


# =============================================================================
# Voxel Obstacles: resolution-limited point sampling
# =============================================================================


@wp.kernel(enable_backward=False)
def robot_voxel_collision_kernel(
    robot: RobotMeshWarp,
    obs: VoxelDataWarp,
    env_query_idx: wp.array(dtype=wp.int32),
    in_collision: wp.array(dtype=wp.int32),
    contact_link: wp.array(dtype=wp.int32),
    batch_horizon: wp.int32,
    horizon: wp.int32,
    margin: wp.float32,
    use_multi_env: wp.uint8,
    max_samples_per_edge: wp.int32,
):
    """Sampled robot-triangle vs voxel ESDF collision.

    The triangle is sampled on a barycentric lattice fine enough that adjacent samples are
    at most half a voxel apart, then each sample is tested against the ESDF. This is
    **not exact**: geometry thinner than the voxel size can slip between samples.

    Args:
        robot: Robot collision triangles and per-link transforms.
        obs: Voxel grid obstacle set.
        env_query_idx: Environment index per configuration.
        in_collision: Output flag per configuration, pre-zeroed.
        contact_link: Output contacting link index per configuration, pre-filled with -1.
        batch_horizon: Number of configurations (batch * horizon).
        margin: Clearance distance; a sample collides when its ESDF value is <= margin.
        use_multi_env: If 1, look up the environment from ``env_query_idx``.
        max_samples_per_edge: Upper bound on lattice subdivisions, capping per-thread cost
            for large triangles or fine grids.
    """
    tid = wp.tid()

    cfg_idx = tid / robot.total_tris
    tri_idx = tid - cfg_idx * robot.total_tris

    if cfg_idx >= batch_horizon:
        return
    if in_collision[cfg_idx] == 1:
        return

    env_idx = wp.int32(0)
    if use_multi_env == wp.uint8(1):
        env_idx = env_query_idx[cfg_idx / horizon]

    link_idx = robot.tri_link[tri_idx]
    # Reserved-but-unfilled slots carry a negative link, and hold no geometry.
    if link_idx < 0:
        return
    row = cfg_idx * robot.num_links + link_idx
    v0_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 0])
    v1_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 1])
    v2_w = transform_point_link(robot.link_tf, row, robot.tri_verts[tri_idx * 3 + 2])

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

        # Choose a subdivision count so adjacent samples are at most voxel_size/2 apart.
        longest = wp.max(wp.length(a1 - a0), wp.max(wp.length(a2 - a1), wp.length(a0 - a2)))
        steps = wp.int32(wp.ceil(longest / wp.max(voxel_size * 0.5, 1.0e-6)))
        steps = wp.clamp(steps, 1, max_samples_per_edge)

        inv_steps = 1.0 / wp.float32(steps)
        for i in range(steps + 1):
            u = wp.float32(i) * inv_steps
            for j in range(steps + 1 - i):
                v = wp.float32(j) * inv_steps
                p = a0 + (a1 - a0) * u + (a2 - a0) * v
                if compute_local_sdf(obs, env_idx, o, p) <= margin:
                    report_collision(in_collision, contact_link, cfg_idx, link_idx)
                    return
