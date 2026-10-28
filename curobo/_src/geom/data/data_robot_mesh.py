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
from typing import Dict, List, Optional, Sequence, Union

# Third Party
import numpy as np
import torch
import trimesh
import warp as wp

# CuRobo
from curobo._src.geom.types import Mesh
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_warn
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
    """

    tri_verts: wp.array(dtype=wp.vec3)
    tri_link: wp.array(dtype=wp.int32)
    link_tf: wp.array2d(dtype=wp.float32)
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


def decimate_mesh(
    mesh: trimesh.Trimesh, max_triangles: int, link_name: Optional[str] = None
) -> trimesh.Trimesh:
    """Reduce a link mesh to at most ``max_triangles`` triangles.

    Quadric decimation is used because it preserves concavities. It requires the optional
    ``fast_simplification`` package; without it this falls back to the convex hull, which
    may still exceed the budget and which closes openings such as the gap between gripper
    fingers. Query cost scales roughly linearly with the total triangle count, so exceeding
    the budget is a performance problem worth warning about.

    Note that decimation shrinks the mesh by a sub-millimetre amount, which makes the
    constraint very slightly optimistic. Use a small positive ``margin`` on the constraint
    to compensate when that matters.

    Args:
        mesh: Input link mesh.
        max_triangles: Triangle budget. Values <= 0 disable decimation.
        link_name: Name of the link, used only to make warnings actionable.

    Returns:
        A mesh with at most ``max_triangles`` triangles where possible.
    """
    if max_triangles <= 0 or len(mesh.faces) <= max_triangles:
        return mesh

    label = f"link '{link_name}'" if link_name is not None else "link mesh"
    log_warn(
        f"{label} has {len(mesh.faces)} triangles, over its budget of {max_triangles}; "
        "decimating. Raise max_triangles_per_link for this link to keep full fidelity."
    )

    try:
        simplified = mesh.simplify_quadric_decimation(face_count=max_triangles)
    except Exception as e:  # trimesh raises various backend-specific errors
        log_warn(f"quadric decimation failed ({e}), falling back to convex hull")
        simplified = None

    if simplified is None or len(simplified.faces) == 0:
        simplified = mesh.convex_hull

    if len(simplified.faces) > max_triangles:
        log_warn(
            f"could not reduce link mesh below {max_triangles} triangles "
            f"(got {len(simplified.faces)}); install 'fast_simplification' to enable "
            "quadric decimation and keep collision queries fast"
        )
    return simplified


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
        triangles = decimate_mesh(triangles, capacity, link_name)
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
        mesh = decimate_mesh(mesh, capacity, link_name)
        soup = torch.as_tensor(np.asarray(mesh.triangles, dtype=np.float32))
    if soup.shape[0] > capacity:
        log_and_raise(
            f"link '{link_name}' was given {soup.shape[0]} triangles but only {capacity} "
            "slots are reserved; raise its extra_collision_meshes budget"
        )
    return soup


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

    def __post_init__(self):
        """Recover the body index of each packed link and the reference slot owners."""
        if self.link_body_indices is None:
            offsets = np.cumsum([0] + list(self.link_triangle_counts[:-1]))
            self.link_body_indices = [int(self.tri_link[int(o)].item()) for o in offsets]
        if self.reference_tri_link is None:
            self.reference_tri_link = self.tri_link.clone()

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
        return s
