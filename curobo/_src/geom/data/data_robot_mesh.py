# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""GPU data storage for the robot's exact triangle collision geometry.

This module holds the robot's link collision meshes as a single flat triangle soup in
**link-local** coordinates, together with the map from each triangle to the link (body index)
that carries it. At query time the per-link forward-kinematics transforms are bridged in
directly from the FK cumulative-transform buffer, so no host-side pose conversion is needed.

This is the exact-geometry counterpart of the collision spheres. It is consumed only by the
forward-only mesh collision *constraint*; the differentiable sphere collision cost is
unaffected.
"""

from __future__ import annotations

# Standard Library
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

# Third Party
import numpy as np
import torch
import trimesh
import warp as wp

# CuRobo
from curobo._src.geom.types import Mesh
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_info, log_warn
from curobo._src.util.warp import init_warp

#: Default per-link triangle budget. Measured on an RTX 3090 Ti, query cost scales roughly
#: linearly with the total robot triangle count, so this is deliberately tight.
DEFAULT_MAX_TRIANGLES_PER_LINK = 256

#: Body index written into ``tri_link`` for a reserved-but-unfilled triangle slot. The
#: collision kernels skip these before any geometry is touched, so an empty reservation
#: costs one integer comparison per slot.
INACTIVE_TRI_LINK = -1

#: Per-link triangle budget: either one value for every link, or a per-link mapping. Links
#: missing from the mapping fall back to :data:`DEFAULT_MAX_TRIANGLES_PER_LINK`.
TriangleBudget = Union[None, int, Dict[str, int]]


def resolve_triangle_budget(link_name: str, budget: TriangleBudget) -> int:
    """Resolve the triangle budget that applies to one link.

    Args:
        link_name: Name of the link.
        budget: A single budget for all links, a per-link mapping, or None.

    Returns:
        The budget for this link, falling back to :data:`DEFAULT_MAX_TRIANGLES_PER_LINK`.
    """
    if budget is None:
        return DEFAULT_MAX_TRIANGLES_PER_LINK
    if isinstance(budget, dict):
        return int(budget.get(link_name, DEFAULT_MAX_TRIANGLES_PER_LINK))
    return int(budget)


# =============================================================================
# Warp Struct
# =============================================================================


@wp.struct
class RobotMeshWarp:
    """Warp struct for the robot's link collision triangles.

    Attributes:
        tri_verts: Triangle vertices in link-local frame, 3 consecutive entries per
            triangle. Shape: (total_tris * 3,).
        tri_link: Body index of the link owning each triangle. Shape: (total_tris,).
        link_tf: Row-major ``[R | t]`` per link, flattened to 12 floats.
            Shape: (batch * horizon * num_links, 12).
        num_links: Number of links in the kinematic tree (stride of ``link_tf``).
        total_tris: Total number of robot collision triangles.
        link_bounds: Link-local bounding sphere ``(x, y, z, radius)`` per body index; a
            negative radius marks a link without geometry. Shape: (num_links,).
    """

    tri_verts: wp.array(dtype=wp.vec3)
    tri_link: wp.array(dtype=wp.int32)
    link_tf: wp.array2d(dtype=wp.float32)
    link_bounds: wp.array(dtype=wp.vec4)
    num_links: wp.int32
    total_tris: wp.int32


@wp.func
def transform_point_link(
    link_tf: wp.array2d(dtype=wp.float32),
    row: wp.int32,
    p: wp.vec3,
) -> wp.vec3:
    """Apply a row-major ``[R | t]`` link transform to a link-local point.

    Args:
        link_tf: Flattened transform array, 12 floats per link.
        row: Flat index ``cfg_idx * num_links + link_idx``.
        p: Point in link-local frame.

    Returns:
        The point in world frame.
    """
    return wp.vec3(
        link_tf[row, 0] * p[0] + link_tf[row, 1] * p[1] + link_tf[row, 2] * p[2] + link_tf[row, 3],
        link_tf[row, 4] * p[0] + link_tf[row, 5] * p[1] + link_tf[row, 6] * p[2] + link_tf[row, 7],
        link_tf[row, 8] * p[0]
        + link_tf[row, 9] * p[1]
        + link_tf[row, 10] * p[2]
        + link_tf[row, 11],
    )


# =============================================================================
# Mesh Decimation
# =============================================================================


#: Surface deviation above which decimating a link mesh logs a warning, in meters.
DECIMATION_WARN_DEVIATION = 0.002

#: Quadric decimation settings tried by :func:`decimate_mesh`, as ``(aggressiveness,
#: preserve_border)``. Lower aggressiveness collapses edges more carefully; keeping the border
#: stops open boundaries, common in robot collision meshes, from being eaten away.
_DECIMATION_CANDIDATES = [(a, b) for b in (True, False) for a in (1.0, 3.0, 5.0, 7.0)]

#: Settings tried, in order, when the closest result is not searched for, e.g. for geometry
#: written at runtime: the first that reaches the budget is kept.
_DECIMATION_QUICK_CANDIDATES = [(3.0, True), (3.0, False)]


@wp.kernel(enable_backward=False)
def _surface_distance_kernel(
    mesh_id: wp.uint64,
    points: wp.array(dtype=wp.vec3),
    max_dist: wp.float32,
    distance: wp.array(dtype=wp.float32),
):
    """Distance from each point to the closest point of a mesh, capped at ``max_dist``."""
    tid = wp.tid()
    query = wp.mesh_query_point_no_sign(mesh_id, points[tid], max_dist)
    if query.result:
        closest = wp.mesh_eval_position(mesh_id, query.face, query.u, query.v)
        distance[tid] = wp.length(closest - points[tid])
    else:
        distance[tid] = max_dist


def mesh_surface_deviation(a: trimesh.Trimesh, b: trimesh.Trimesh, samples: int = 10000) -> float:
    """Approximate two-sided surface distance (Hausdorff distance) between two meshes.

    Points sampled on each surface are measured exactly against the other surface with a
    BVH query on the CPU. The sampling is seeded, so the result is deterministic; it can only
    underestimate the true distance, by missing the worst point between samples.

    Args:
        a: First mesh.
        b: Second mesh.
        samples: Points sampled on each surface.

    Returns:
        The largest distance from a sampled point of either surface to the other surface, in
        the units of the meshes.
    """
    device = "cpu"
    cap = float(np.linalg.norm(np.ptp(np.vstack([a.bounds, b.bounds]), axis=0))) + 1.0

    def one_sided(src: trimesh.Trimesh, dst: trimesh.Trimesh, seed: int) -> float:
        points = trimesh.sample.sample_surface(src, samples, seed=seed)[0]
        vertices = np.asarray(dst.vertices, dtype=np.float32)
        faces = np.asarray(dst.faces, dtype=np.int32).reshape(-1)
        dst_mesh = wp.Mesh(
            points=wp.array(vertices, dtype=wp.vec3, device=device),
            indices=wp.array(faces, device=device),
        )
        distance = wp.zeros(len(points), dtype=wp.float32, device=device)
        wp.launch(
            _surface_distance_kernel,
            dim=len(points),
            inputs=[
                dst_mesh.id,
                wp.array(np.asarray(points, dtype=np.float32), dtype=wp.vec3, device=device),
                cap,
                distance,
            ],
            device=device,
        )
        return float(distance.numpy().max())

    return max(one_sided(a, b, 0), one_sided(b, a, 1))


def _quadric_decimate(
    mesh: trimesh.Trimesh, max_triangles: int, aggressiveness: float, preserve_border: bool
) -> trimesh.Trimesh:
    """Quadric decimation that raises the aggressiveness whenever a pass stops short."""
    from fast_simplification import simplify

    vertices, faces = mesh.vertices, mesh.faces
    for agg in [a for a in (1.0, 3.0, 5.0, 7.0, 9.0, 10.0) if a >= aggressiveness]:
        if len(faces) <= max_triangles:
            break
        vertices, faces = simplify(
            vertices,
            faces,
            target_count=max_triangles,
            agg=agg,
            preserve_border=preserve_border,
        )
    return trimesh.Trimesh(vertices=vertices, faces=faces, process=False)


def decimate_mesh(
    mesh: trimesh.Trimesh,
    max_triangles: int,
    link_name: Optional[str] = None,
    thorough: bool = True,
    warn_deviation: float = DECIMATION_WARN_DEVIATION,
) -> trimesh.Trimesh:
    """Reduce a link mesh to at most ``max_triangles`` triangles, staying close to its surface.

    The mesh is first cleaned: duplicate vertices are merged, and holes are filled when that
    closes it. Quadric decimation (``fast_simplification``) then runs with several settings,
    and the result closest to the original surface (:func:`mesh_surface_deviation`) is kept.
    Most robot collision meshes are open, and keeping their open boundary is what keeps
    decimation from eating into them: on Franka's links it cuts the deviation from up to
    2.3 cm to about 1 mm.

    A warning reports the deviation when it exceeds ``warn_deviation``. If no setting reaches
    the budget, the closest of the results over it is kept, with a warning: collision queries
    on it are slower. If decimation fails altogether, this falls back to the convex hull,
    which fills concave regions such as the gap between gripper fingers.

    Args:
        mesh: Input link mesh.
        max_triangles: Triangle budget. Values <= 0 disable decimation.
        link_name: Name of the link, used only to make messages actionable.
        thorough: Try every decimation setting and keep the closest result. False keeps the
            first setting that reaches the budget, for geometry written at runtime.
        warn_deviation: Surface deviation in meters above which a warning is logged.

    Returns:
        A mesh with at most ``max_triangles`` triangles where possible.
    """
    if max_triangles <= 0 or len(mesh.faces) <= max_triangles:
        return mesh

    label = f"link '{link_name}'" if link_name is not None else "link mesh"
    base = mesh.copy()
    base.merge_vertices()
    closed = base.copy()
    trimesh.repair.fill_holes(closed)
    if closed.is_watertight:
        base = closed

    candidates = _DECIMATION_CANDIDATES if thorough else _DECIMATION_QUICK_CANDIDATES
    # Closest result within the budget, and closest result over it in case none fits.
    best, best_deviation = None, np.inf
    over, over_deviation = None, np.inf
    try:
        for aggressiveness, preserve_border in candidates:
            simplified = _quadric_decimate(base, max_triangles, aggressiveness, preserve_border)
            if len(simplified.faces) == 0:
                continue
            deviation = mesh_surface_deviation(mesh, simplified)
            if len(simplified.faces) > max_triangles:
                if deviation < over_deviation:
                    over, over_deviation = simplified, deviation
                continue
            if deviation < best_deviation:
                best, best_deviation = simplified, deviation
            if not thorough:
                break
    except Exception as e:  # fast_simplification raises various errors on bad input
        log_warn(
            f"quadric decimation of {label} failed ({e}); falling back to its convex hull, "
            "which fills concave regions. Check that 'fast_simplification' is installed."
        )
        best, over = None, None

    if best is None:
        if over is not None:
            best, best_deviation = over, over_deviation
        else:
            best = mesh.convex_hull
            best_deviation = mesh_surface_deviation(mesh, best)
    if len(best.faces) > max_triangles:
        log_warn(
            f"could not reduce {label} below {max_triangles} triangles "
            f"(got {len(best.faces)}); collision queries on it will be slower"
        )

    message = (
        f"{label} has {len(mesh.faces)} triangles, over its budget of {max_triangles}; "
        f"decimated to {len(best.faces)}, its surface moved by up to "
        f"{1000.0 * best_deviation:.1f} mm"
    )
    if best_deviation > warn_deviation:
        log_warn(
            message + ". Raise max_triangles_per_link for this link, or add the deviation to "
            "the mesh collision margin."
        )
    else:
        log_info(message)
    return best


def _as_triangle_soup(
    triangles: Union[torch.Tensor, np.ndarray, Mesh, trimesh.Trimesh],
    capacity: int,
    link_name: str,
) -> torch.Tensor:
    """Normalize runtime geometry to a ``[n, 3, 3]`` triangle soup that fits ``capacity``.

    Args:
        triangles: Triangle soup or mesh in link-local frame.
        capacity: Number of triangle slots available.
        link_name: Name of the link, used only to make warnings actionable.

    Returns:
        Triangle soup with at most ``capacity`` triangles.
    """
    if isinstance(triangles, Mesh):
        triangles = triangles.get_trimesh_mesh(transform_with_pose=True)
    if isinstance(triangles, trimesh.Trimesh):
        triangles = decimate_mesh(triangles, capacity, link_name, thorough=False)
        soup = np.asarray(triangles.triangles, dtype=np.float32)
        return torch.as_tensor(soup)

    soup = torch.as_tensor(triangles)
    if soup.ndim != 3 or soup.shape[1] != 3 or soup.shape[2] != 3:
        log_and_raise(
            f"triangles for link '{link_name}' must have shape [n, 3, 3], got "
            f"{tuple(soup.shape)}"
        )
    if soup.shape[0] > capacity:
        # A triangle soup carries no connectivity, so rebuild a mesh to decimate it.
        mesh = trimesh.Trimesh(
            vertices=soup.reshape(-1, 3).detach().cpu().numpy(),
            faces=np.arange(soup.shape[0] * 3).reshape(-1, 3),
            process=True,
        )
        mesh = decimate_mesh(mesh, capacity, link_name, thorough=False)
        soup = torch.as_tensor(np.asarray(mesh.triangles, dtype=np.float32))
    if soup.shape[0] > capacity:
        log_and_raise(
            f"link '{link_name}' was given {soup.shape[0]} triangles but only {capacity} "
            "slots are reserved; raise its extra_collision_meshes budget"
        )
    return soup


# =============================================================================
# Self-Collision Pairs
# =============================================================================


def mesh_self_collision_pairs(
    link_names: Sequence[str],
    link_name_to_idx_map: Dict[str, int],
    parent_body_indices: Sequence[int],
    fixed_joints: Sequence[bool],
    ignore: Optional[Dict[str, Sequence[str]]] = None,
) -> List[Tuple[int, int]]:
    """Select the link pairs that the mesh self-collision check tests.

    Links joined by fixed joints, including locked joints, move as one rigid body. A pair is
    skipped when both links belong to the same rigid body or to two rigid bodies that a single
    joint connects, since the meshes of such links usually touch at the joint. Pairs listed in
    ``ignore`` are skipped as well.

    Args:
        link_names: Links with mesh geometry, including reserved runtime slots.
        link_name_to_idx_map: Mapping from link name to kinematic body index.
        parent_body_indices: Parent body index of every body, indexed by body index. The
            root's entry is ignored.
        fixed_joints: Whether the joint between a body and its parent is fixed, per body.
        ignore: Additional pairs to skip, as ``{link: [links]}``, in either direction.

    Returns:
        Sorted ``(body_a, body_b)`` pairs with ``body_a < body_b``.
    """
    known = set(link_name_to_idx_map)
    ignored = set()
    for link, others in (ignore or {}).items():
        unknown = sorted({link, *others} - known)
        if unknown:
            log_warn(f"mesh_self_collision_ignore names unknown links: {unknown}")
        for other in others:
            if link in known and other in known:
                ignored.add(frozenset((link_name_to_idx_map[link], link_name_to_idx_map[other])))

    def rigid_root(body: int) -> int:
        while body > 0 and fixed_joints[body]:
            body = int(parent_body_indices[body])
        return body

    def connected(root_a: int, root_b: int) -> bool:
        return (root_a > 0 and rigid_root(int(parent_body_indices[root_a])) == root_b) or (
            root_b > 0 and rigid_root(int(parent_body_indices[root_b])) == root_a
        )

    bodies = sorted({link_name_to_idx_map[name] for name in link_names})
    pairs = []
    for i, a in enumerate(bodies):
        for b in bodies[i + 1 :]:
            root_a, root_b = rigid_root(a), rigid_root(b)
            if root_a == root_b or connected(root_a, root_b):
                continue
            if frozenset((a, b)) in ignored:
                continue
            pairs.append((a, b))
    return pairs


# =============================================================================
# Python Dataclass
# =============================================================================


@dataclass
class RobotMeshData:
    """GPU storage for the robot's link collision triangles in link-local frame.

    Tensor layouts:
        - tri_verts: (total_tris * 3, 3) float32 - link-local triangle vertices
        - tri_link: (total_tris,) int32 - body index per triangle
    """

    #: Link-local triangle vertices, 3 rows per triangle.
    tri_verts: torch.Tensor

    #: Body index of the link owning each triangle.
    tri_link: torch.Tensor

    #: Number of links in the kinematic tree.
    num_links: int

    #: Total number of robot collision triangles.
    total_tris: int

    #: Names of the links included, in the order they were packed.
    link_names: List[str]

    #: Triangle count per included link, aligned with :attr:`link_names`.
    link_triangle_counts: List[int]

    #: Device configuration.
    device_cfg: DeviceCfg

    #: Kinematic body index of each included link, aligned with :attr:`link_names`. These are
    #: indices into the kinematic tree, *not* positions in :attr:`link_names`, and the two
    #: differ whenever the tree holds links without collision meshes. Recovered from
    #: :attr:`tri_link` when not supplied.
    link_body_indices: Optional[List[int]] = None

    #: :attr:`tri_link` as last written by packing or :meth:`update_link_mesh`, before any
    #: :meth:`disable_link_mesh`. :meth:`enable_link_mesh` restores slots from it.
    reference_tri_link: Optional[torch.Tensor] = None

    #: Link-local bounding sphere ``(x, y, z, radius)`` of each link's geometry, indexed by body
    #: index, shape ``[num_links, 4]``. A negative radius marks a link without geometry. The
    #: kernels skip a link whose sphere is far from every obstacle.
    link_bounds: Optional[torch.Tensor] = None

    #: Body index pairs ``[num_pairs, 2]`` (int32) tested by the mesh self-collision check,
    #: set with :meth:`set_self_collision_pairs`. In each pair the first link has no more
    #: triangle slots than the second: the kernel iterates its triangles and queries the
    #: second link's BVH. None when no pairs are tested.
    self_collision_pairs: Optional[torch.Tensor] = None

    def __post_init__(self):
        """Recover the body indices, the reference slot owners and the link bounds."""
        if self.link_body_indices is None:
            offsets = np.cumsum([0] + list(self.link_triangle_counts[:-1]))
            self.link_body_indices = [int(self.tri_link[int(o)].item()) for o in offsets]
        if self.reference_tri_link is None:
            self.reference_tri_link = self.tri_link.clone()
        if self.link_bounds is None:
            self.link_bounds = torch.full(
                (self.num_links, 4), -1.0, dtype=torch.float32, device=self.tri_verts.device
            )
            active = (self.reference_tri_link >= 0).cpu().numpy()
            for link_name in self.link_names:
                span = self.get_triangle_slice(link_name)
                count = int(active[span].sum())
                # Packing fills a link's slots from the start, so active slots come first.
                tris = self.tri_verts[span.start * 3 : (span.start + count) * 3].view(-1, 3, 3)
                self._write_link_bounds(link_name, tris)

    @property
    def body_idx_to_link_name(self) -> Dict[int, str]:
        """Map each kinematic body index to the name of the link carrying its mesh."""
        return dict(zip(self.link_body_indices, self.link_names))

    def contact_link_names(self, contact_link: torch.Tensor) -> List[Optional[str]]:
        """Resolve reported contact body indices to link names.

        Args:
            contact_link: Body indices as returned by the collision query, where -1 marks a
                collision-free configuration.

        Returns:
            One link name per entry, flattened, with ``None`` where there was no collision.
        """
        lookup = self.body_idx_to_link_name
        return [lookup.get(int(i)) for i in contact_link.flatten().tolist()]

    def get_triangle_slice(self, link_name: str) -> slice:
        """Return the range of triangle slots owned by a link.

        Args:
            link_name: Name of a link that was packed or reserved.

        Returns:
            Slice into the triangle dimension, i.e. into :attr:`tri_link`.
        """
        if link_name not in self.link_names:
            log_and_raise(
                f"link '{link_name}' has no mesh collision triangles; reserve slots for it "
                "with extra_collision_meshes to update it at runtime"
            )
        pos = self.link_names.index(link_name)
        start = int(np.sum(self.link_triangle_counts[:pos]))
        return slice(start, start + self.link_triangle_counts[pos])

    def get_triangle_capacity(self, link_name: str) -> int:
        """Return how many triangle slots a link owns.

        Args:
            link_name: Name of a link that was packed or reserved.
        """
        return self.link_triangle_counts[self.link_names.index(link_name)]

    def set_self_collision_pairs(self, pairs: Sequence[Tuple[int, int]]) -> None:
        """Prepare the link pairs that the mesh self-collision check tests.

        Builds a BVH over each paired link's triangle slots, in its link frame and sharing
        memory with :attr:`tri_verts`, and a flat work list with one entry per triangle slot of
        the iterated link of each pair.

        Args:
            pairs: Body index pairs, e.g. from :func:`mesh_self_collision_pairs`.
        """
        body_to_name = self.body_idx_to_link_name
        ordered = []
        for a, b in pairs:
            if self.get_triangle_capacity(body_to_name[a]) > self.get_triangle_capacity(
                body_to_name[b]
            ):
                a, b = b, a
            ordered.append((a, b))
        if not ordered:
            self.self_collision_pairs = None
            return
        device = self.tri_verts.device
        self.self_collision_pairs = torch.tensor(ordered, dtype=torch.int32, device=device)

        self.link_tri_start = torch.zeros(self.num_links, dtype=torch.int32, device=device)
        self.link_mesh_ids = torch.zeros(self.num_links, dtype=torch.int64, device=device)
        self._link_bvh: Dict[str, wp.Mesh] = {}
        wp_device = wp.device_from_torch(device)
        for body in sorted({b for pair in ordered for b in pair}):
            name = body_to_name[body]
            span = self.get_triangle_slice(name)
            self.link_tri_start[body] = span.start
            points = wp.from_torch(
                self.tri_verts[span.start * 3 : span.stop * 3], dtype=wp.vec3
            )
            indices = wp.array(
                np.arange((span.stop - span.start) * 3, dtype=np.int32), device=wp_device
            )
            self._link_bvh[name] = wp.Mesh(points=points, indices=indices)
            self.link_mesh_ids[body] = self._link_bvh[name].id

        work_tri, work_pair = [], []
        for p, (a, _) in enumerate(ordered):
            span = self.get_triangle_slice(body_to_name[a])
            work_tri.extend(range(span.start, span.stop))
            work_pair.extend([p] * (span.stop - span.start))
        self.self_work_tri = torch.tensor(work_tri, dtype=torch.int32, device=device)
        self.self_work_pair = torch.tensor(work_pair, dtype=torch.int32, device=device)

    def self_collision_pair_names(
        self, contact_pair: torch.Tensor
    ) -> List[Optional[Tuple[str, str]]]:
        """Resolve reported self-collision pair indices to link names.

        Args:
            contact_pair: Pair indices into :attr:`self_collision_pairs`, as returned by the
                self-collision query, where -1 marks a configuration without self-collision.

        Returns:
            One ``(link_a, link_b)`` tuple per entry, flattened, with ``None`` where there was
            no self-collision.
        """
        lookup = self.body_idx_to_link_name
        pairs = [] if self.self_collision_pairs is None else self.self_collision_pairs.tolist()
        names = [(lookup[a], lookup[b]) for a, b in pairs]
        return [names[i] if i >= 0 else None for i in contact_pair.flatten().tolist()]

    def _refit_link_bvh(self, link_name: str) -> None:
        """Refit a link's self-collision BVH after its vertices changed in place."""
        bvh = getattr(self, "_link_bvh", {}).get(link_name)
        if bvh is not None:
            bvh.refit()

    def update_link_mesh(
        self,
        link_name: str,
        triangles: Union[torch.Tensor, np.ndarray, Mesh, trimesh.Trimesh],
    ):
        """Write link-local collision geometry into a link's reserved triangle slots.

        This is the exact-mesh counterpart of
        :meth:`~curobo._src.robot.types.kinematics_params.KinematicsParams.update_link_spheres`.
        The write is in-place, so the GPU buffers are never reallocated and any captured CUDA
        graph stays valid. Slots beyond the supplied geometry are left inactive.

        Args:
            link_name: Name of the link to update. Its slots must have been reserved with
                ``extra_collision_meshes``, or filled from a static collision mesh.
            triangles: Link-local geometry, either a ``[n, 3, 3]`` triangle soup (triangle,
                vertex, xyz) or a mesh. Geometry over the link's capacity is decimated to fit.
        """
        span = self.get_triangle_slice(link_name)
        capacity = span.stop - span.start
        body_idx = self.link_body_indices[self.link_names.index(link_name)]

        tris = _as_triangle_soup(triangles, capacity, link_name)
        n = tris.shape[0]

        verts = tris.reshape(-1, 3).to(dtype=self.tri_verts.dtype, device=self.tri_verts.device)
        self.tri_verts[span.start * 3 : (span.start + n) * 3] = verts
        self.reference_tri_link[span.start : span.start + n] = body_idx
        if n < capacity:
            self.reference_tri_link[span.start + n : span.stop] = INACTIVE_TRI_LINK
        self.tri_link[span] = self.reference_tri_link[span]
        self._write_link_bounds(link_name, verts.view(-1, 3, 3))
        self._refit_link_bvh(link_name)

    def _write_link_bounds(self, link_name: str, tris: torch.Tensor):
        """Store the bounding sphere of a link's triangles ``[n, 3, 3]``; none if ``n == 0``."""
        body_idx = self.link_body_indices[self.link_names.index(link_name)]
        if tris.shape[0] == 0:
            self.link_bounds[body_idx, 3] = -1.0
            return
        verts = tris.reshape(-1, 3).to(self.link_bounds)
        center = 0.5 * (verts.min(dim=0).values + verts.max(dim=0).values)
        self.link_bounds[body_idx, :3] = center
        self.link_bounds[body_idx, 3] = (verts - center).norm(dim=-1).max()

    def disable_link_mesh(self, link_name: str):
        """Deactivate every triangle slot of a link, so it can no longer report a collision.

        The geometry is left in place; :meth:`enable_link_mesh` or :meth:`update_link_mesh`
        reactivates the slots.

        Args:
            link_name: Name of the link to deactivate.
        """
        span = self.get_triangle_slice(link_name)
        self.tri_link[span] = INACTIVE_TRI_LINK

    def enable_link_mesh(self, link_name: str):
        """Reactivate a link's triangles after :meth:`disable_link_mesh`.

        Restores the geometry last packed or written with :meth:`update_link_mesh`, so a
        reserved link that holds no geometry stays inactive. The write is in-place.

        Args:
            link_name: Name of the link to reactivate.
        """
        span = self.get_triangle_slice(link_name)
        self.tri_link[span] = self.reference_tri_link[span]

    def clear_link_mesh(self, link_name: str):
        """Remove a link's geometry, so that :meth:`enable_link_mesh` keeps it inactive.

        Use this when runtime geometry such as an attached object is released; unlike
        :meth:`disable_link_mesh`, the geometry is not restored later. The write is in-place.

        Args:
            link_name: Name of the link to clear.
        """
        span = self.get_triangle_slice(link_name)
        self.reference_tri_link[span] = INACTIVE_TRI_LINK
        self.tri_link[span] = INACTIVE_TRI_LINK
        self._write_link_bounds(link_name, self.tri_verts[:0].view(0, 3, 3))

    def has_link_mesh(self, link_name: str) -> bool:
        """Whether a link owns triangle slots, packed or reserved.

        Args:
            link_name: Name of the link.
        """
        return link_name in self.link_names

    @property
    def num_active_triangles(self) -> int:
        """Number of triangle slots currently holding active geometry."""
        return int((self.tri_link >= 0).sum().item())

    @classmethod
    def from_kinematics(
        cls,
        kinematics,
        max_triangles_per_link: TriangleBudget = None,
        ignored_link_names: Optional[Sequence[str]] = None,
        extra_link_triangles: Optional[Dict[str, int]] = None,
    ) -> RobotMeshData:
        """Build robot collision triangle data from a loaded robot model.

        Args:
            kinematics: A :class:`~curobo._src.robot.kinematics.kinematics.Kinematics`
                instance loaded with ``load_collision_meshes=True``.
            max_triangles_per_link: Per-link triangle budget. See :func:`decimate_mesh`.
            ignored_link_names: Links to leave out of the mesh constraint entirely. See
                :meth:`from_link_meshes`.
            extra_link_triangles: Triangle slots to reserve per link. See
                :meth:`from_link_meshes`.

        Returns:
            Packed robot collision triangle data on the GPU.
        """
        init_warp(device_cfg=kinematics.device_cfg)

        link_meshes: Dict[str, Mesh] = kinematics.get_robot_collision_link_meshes()
        kinematics_config = kinematics.config.kinematics_config
        name_to_idx = kinematics_config.link_name_to_idx_map

        return cls.from_link_meshes(
            link_meshes=link_meshes,
            link_name_to_idx_map=name_to_idx,
            num_links=len(name_to_idx),
            device_cfg=kinematics.device_cfg,
            max_triangles_per_link=max_triangles_per_link,
            ignored_link_names=ignored_link_names,
            extra_link_triangles=extra_link_triangles,
        )

    @classmethod
    def from_link_meshes(
        cls,
        link_meshes: Dict[str, Mesh],
        link_name_to_idx_map: Dict[str, int],
        num_links: int,
        device_cfg: DeviceCfg,
        max_triangles_per_link: TriangleBudget = None,
        ignored_link_names: Optional[Sequence[str]] = None,
        extra_link_triangles: Optional[Dict[str, int]] = None,
    ) -> RobotMeshData:
        """Pack a set of link-local meshes into flat GPU triangle arrays.

        Args:
            link_meshes: Mapping from link name to its link-local collision mesh.
            link_name_to_idx_map: Mapping from link name to internal body index.
            num_links: Number of links in the kinematic tree.
            device_cfg: Device and dtype configuration.
            max_triangles_per_link: Per-link triangle budget, either a single value or a
                per-link mapping. See :func:`resolve_triangle_budget`.
            ignored_link_names: Links whose geometry is left out of the constraint entirely,
                so they can never report a collision against any obstacle. Use this for links
                that are in permanent contact with the scene by design, such as a base link
                bolted to its mounting plate; an exact triangle test resolves such a flush,
                coplanar contact as a collision even though penetration is zero.
            extra_link_triangles: Triangle slots to reserve per link name, for geometry that
                is only known at runtime such as a gripper or an attached object. The slots
                start inactive and are filled with :meth:`update_link_mesh`. The reserved
                count is also that link's triangle budget. This is the exact-mesh counterpart
                of ``extra_collision_spheres``.

        Returns:
            Packed robot collision triangle data on the GPU.
        """
        ignored = set(ignored_link_names or ())
        unknown = ignored.difference(link_meshes.keys())
        if unknown:
            log_warn(
                f"ignored_link_names contains links with no collision mesh: {sorted(unknown)}"
            )

        vert_blocks: List[np.ndarray] = []
        link_blocks: List[np.ndarray] = []
        link_names: List[str] = []
        link_triangle_counts: List[int] = []
        link_body_indices: List[int] = []

        for link_name in sorted(link_meshes.keys()):
            if link_name in ignored:
                continue
            if link_name not in link_name_to_idx_map:
                log_warn(f"link '{link_name}' is not in the kinematic tree, skipping its mesh")
                continue
            body_idx = link_name_to_idx_map[link_name]

            tri_mesh = link_meshes[link_name].get_trimesh_mesh(transform_with_pose=True)
            tri_mesh = decimate_mesh(
                tri_mesh, resolve_triangle_budget(link_name, max_triangles_per_link), link_name
            )

            faces = np.asarray(tri_mesh.faces, dtype=np.int64)
            if faces.size == 0:
                log_warn(f"link '{link_name}' has no triangles after decimation, skipping")
                continue
            vertices = np.asarray(tri_mesh.vertices, dtype=np.float32)

            # Expand indexed faces into a flat triangle soup: 3 vertices per triangle.
            vert_blocks.append(vertices[faces.reshape(-1)])
            link_blocks.append(np.full((len(faces),), body_idx, dtype=np.int32))
            link_names.append(link_name)
            link_triangle_counts.append(len(faces))
            link_body_indices.append(body_idx)

        # Reserve inactive slots for links whose geometry only exists at runtime. They are
        # appended last so that the offsets of the static links stay stable.
        for link_name, count in sorted((extra_link_triangles or {}).items()):
            if link_name in ignored:
                continue
            if link_name in link_names:
                log_and_raise(
                    f"link '{link_name}' has both a static collision mesh and a reservation in "
                    "extra_collision_meshes; remove one of them"
                )
            if link_name not in link_name_to_idx_map:
                log_warn(f"link '{link_name}' is not in the kinematic tree, skipping reservation")
                continue
            if count <= 0:
                continue
            vert_blocks.append(np.zeros((count * 3, 3), dtype=np.float32))
            link_blocks.append(np.full((count,), INACTIVE_TRI_LINK, dtype=np.int32))
            link_names.append(link_name)
            link_triangle_counts.append(int(count))
            link_body_indices.append(link_name_to_idx_map[link_name])

        if not vert_blocks:
            log_and_raise("no usable robot collision triangles were found")
        tri_verts_np = np.concatenate(vert_blocks, axis=0)
        tri_link_np = np.concatenate(link_blocks, axis=0)

        tri_verts = torch.as_tensor(
            tri_verts_np, dtype=torch.float32, device=device_cfg.device
        ).contiguous()
        tri_link = torch.as_tensor(
            tri_link_np, dtype=torch.int32, device=device_cfg.device
        ).contiguous()

        return cls(
            tri_verts=tri_verts,
            tri_link=tri_link,
            num_links=num_links,
            total_tris=int(tri_link_np.shape[0]),
            link_names=link_names,
            link_triangle_counts=link_triangle_counts,
            device_cfg=device_cfg,
            link_body_indices=link_body_indices,
        )

    def to_warp(self, link_transforms: torch.Tensor) -> RobotMeshWarp:
        """Convert to a Warp struct for kernel launches.

        Args:
            link_transforms: Per-link transforms ``[batch, horizon, num_links, 3, 4]`` from
                :attr:`~curobo._src.robot.kinematics.kinematics_state.KinematicsState.link_transforms`.

        Returns:
            RobotMeshWarp struct for use in Warp kernels.
        """
        if link_transforms.shape[-3] != self.num_links:
            log_and_raise(
                f"link_transforms has {link_transforms.shape[-3]} links, "
                f"expected {self.num_links}"
            )

        s = RobotMeshWarp()
        s.tri_verts = wp.from_torch(self.tri_verts, dtype=wp.vec3)
        s.tri_link = wp.from_torch(self.tri_link, dtype=wp.int32)
        s.link_tf = wp.from_torch(
            link_transforms.reshape(-1, 12).contiguous(), dtype=wp.float32
        )
        s.num_links = self.num_links
        s.total_tris = self.total_tris
        s.link_bounds = wp.from_torch(self.link_bounds, dtype=wp.vec4)
        return s
