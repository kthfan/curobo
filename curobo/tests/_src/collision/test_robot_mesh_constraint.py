# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the exact robot-mesh collision constraint."""

# Third Party
import numpy as np
import pytest
import torch
import trimesh

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_collision import check_robot_mesh_collision
from curobo._src.geom.data.data_robot_mesh import RobotMeshData
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp

DEVICE_CFG = DeviceCfg()


@pytest.fixture(scope="module", autouse=True)
def _warp():
    init_warp(device_cfg=DEVICE_CFG)


def _box_mesh(name: str, extents, pose=None) -> Mesh:
    box = trimesh.creation.box(extents=extents)
    return Mesh(
        name=name,
        vertices=np.asarray(box.vertices),
        faces=np.asarray(box.faces),
        pose=pose if pose is not None else [0, 0, 0, 1, 0, 0, 0],
    )


@pytest.fixture(scope="module")
def robot_mesh() -> RobotMeshData:
    """Single-link robot: a 0.2 m cube centred on the link origin."""
    return RobotMeshData.from_link_meshes(
        link_meshes={"l0": _box_mesh("l0", (0.2, 0.2, 0.2))},
        link_name_to_idx_map={"l0": 0},
        num_links=1,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
    )


def _link_transforms(x_positions) -> torch.Tensor:
    """Identity-rotation transforms translating the single link along +x."""
    batch = len(x_positions)
    tf = torch.zeros(batch, 1, 1, 3, 4, device=DEVICE_CFG.device, dtype=torch.float32)
    tf[..., 0, 0] = 1.0
    tf[..., 1, 1] = 1.0
    tf[..., 2, 2] = 1.0
    for i, x in enumerate(x_positions):
        tf[i, 0, 0, 0, 3] = x
    return tf


def _check(robot_mesh, scene, tf, env_query_idx=None, **kwargs):
    use_multi_env = env_query_idx is not None
    if env_query_idx is None:
        env_query_idx = torch.zeros(tf.shape[0], dtype=torch.int32, device=DEVICE_CFG.device)
    return check_robot_mesh_collision(
        robot_mesh=robot_mesh,
        link_transforms=tf,
        scene=scene,
        env_query_idx=env_query_idx,
        use_multi_env=use_multi_env,
        **kwargs,
    )


# =============================================================================
# Cuboid obstacles
# =============================================================================


def test_cuboid_boundary(robot_mesh):
    """Link half-width 0.1 plus box half-width 0.15 -> contact for |x| < 0.25."""
    scene = SceneData.from_scene_cfg(
        SceneCfg(cuboid=[Cuboid(name="c", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.3, 0.3, 0.3])]),
        device_cfg=DEVICE_CFG,
    )
    tf = _link_transforms([0.0, 0.20, 0.24, 0.26, 0.5])
    coll, _ = _check(robot_mesh, scene, tf)
    assert coll.tolist() == [1, 1, 1, 0, 0]


def test_cuboid_containment_is_inherent(robot_mesh):
    """A link entirely inside a large box is detected without a containment pass."""
    scene = SceneData.from_scene_cfg(
        SceneCfg(cuboid=[Cuboid(name="c", pose=[0, 0, 0, 1, 0, 0, 0], dims=[2.0, 2.0, 2.0])]),
        device_cfg=DEVICE_CFG,
    )
    coll, _ = _check(robot_mesh, scene, _link_transforms([0.0]))
    assert coll.tolist() == [1]


def test_cuboid_margin_expands_collision(robot_mesh):
    """A positive margin flags configurations that are otherwise clear."""
    scene = SceneData.from_scene_cfg(
        SceneCfg(cuboid=[Cuboid(name="c", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.3, 0.3, 0.3])]),
        device_cfg=DEVICE_CFG,
    )
    tf = _link_transforms([0.30])
    assert _check(robot_mesh, scene, tf, margin=0.0)[0].tolist() == [0]
    assert _check(robot_mesh, scene, tf, margin=0.1)[0].tolist() == [1]


def test_contact_link_reported(robot_mesh):
    """contact_link holds the body index on collision and -1 otherwise."""
    scene = SceneData.from_scene_cfg(
        SceneCfg(cuboid=[Cuboid(name="c", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.3, 0.3, 0.3])]),
        device_cfg=DEVICE_CFG,
    )
    coll, link = _check(robot_mesh, scene, _link_transforms([0.0, 5.0]))
    assert coll.tolist() == [1, 0]
    assert link.tolist() == [0, -1]


# =============================================================================
# Mesh obstacles
# =============================================================================


def test_mesh_thin_plate_boundary(robot_mesh):
    """A thin plate is crossed exactly while the link overlaps it.

    The plate spans x in [0.5, 0.51]; the link half-width is 0.1, so surfaces touch for
    link x in [0.40, 0.61].
    """
    plate = trimesh.creation.box(extents=(0.01, 1.0, 1.0))
    scene = SceneData.from_scene_cfg(
        SceneCfg(
            mesh=[
                Mesh(
                    name="plate",
                    vertices=np.asarray(plate.vertices),
                    faces=np.asarray(plate.faces),
                    pose=[0.505, 0, 0, 1, 0, 0, 0],
                )
            ]
        ),
        device_cfg=DEVICE_CFG,
    )
    tf = _link_transforms([0.30, 0.45, 0.505, 0.58, 0.70])
    coll, _ = _check(robot_mesh, scene, tf, check_containment=False)
    assert coll.tolist() == [0, 1, 1, 1, 0]


def test_mesh_containment_requires_the_flag(robot_mesh):
    """A link fully inside a hollow obstacle crosses no triangle."""
    big = trimesh.creation.box(extents=(2.0, 2.0, 2.0))
    scene = SceneData.from_scene_cfg(
        SceneCfg(
            mesh=[
                Mesh(
                    name="shell",
                    vertices=np.asarray(big.vertices),
                    faces=np.asarray(big.faces),
                    pose=[0, 0, 0, 1, 0, 0, 0],
                )
            ]
        ),
        device_cfg=DEVICE_CFG,
    )
    tf = _link_transforms([0.0])
    assert _check(robot_mesh, scene, tf, check_containment=False)[0].tolist() == [0]
    assert _check(robot_mesh, scene, tf, check_containment=True)[0].tolist() == [1]


def _mesh_scene(name: str, tri_mesh: trimesh.Trimesh, pose=None) -> SceneData:
    return SceneData.from_scene_cfg(
        SceneCfg(
            mesh=[
                Mesh(
                    name=name,
                    vertices=np.asarray(tri_mesh.vertices),
                    faces=np.asarray(tri_mesh.faces),
                    pose=pose if pose is not None else [0, 0, 0, 1, 0, 0, 0],
                )
            ]
        ),
        device_cfg=DEVICE_CFG,
    )


@pytest.mark.parametrize("check_containment", [False, True])
def test_mesh_margin_expands_collision(robot_mesh, check_containment):
    """Face-to-face clearance: the link face at x + 0.1 faces the plate face at x = 0.5.

    With the link at x = 0.35 the gap is 0.05, so only a margin above 0.05 collides.
    """
    plate = trimesh.creation.box(extents=(0.01, 1.0, 1.0))
    scene = _mesh_scene("plate", plate, pose=[0.505, 0, 0, 1, 0, 0, 0])
    tf = _link_transforms([0.35])
    for margin, expected in [(0.0, 0), (0.04, 0), (0.06, 1)]:
        coll, _ = _check(robot_mesh, scene, tf, margin=margin, check_containment=check_containment)
        assert coll.tolist() == [expected], margin


def test_mesh_margin_vertex_to_face(robot_mesh):
    """An obstacle apex 0.015 from the interior of a link face, away from any link edge.

    Sampling only the link's own vertices or edges would miss this clearance violation.
    """
    gap = 0.015
    apex = np.array([0.1 + gap, 0.05, 0.01])
    spike = trimesh.Trimesh(
        vertices=np.vstack(
            [
                apex,
                apex + [0.2, 0.02, 0.02],
                apex + [0.2, -0.02, 0.02],
                apex + [0.2, -0.02, -0.02],
                apex + [0.2, 0.02, -0.02],
            ]
        ),
        faces=[[0, 2, 1], [0, 3, 2], [0, 4, 3], [0, 1, 4], [1, 2, 3], [1, 3, 4]],
    )
    scene = _mesh_scene("spike", spike)
    tf = _link_transforms([0.0])
    assert _check(robot_mesh, scene, tf, margin=0.01, check_containment=False)[0].tolist() == [0]
    assert _check(robot_mesh, scene, tf, margin=0.02, check_containment=False)[0].tolist() == [1]


def test_mesh_margin_edge_to_edge(robot_mesh):
    """A horizontal obstacle edge crosses the link's vertical edge at x = y = 0.1, 0.015 away.

    The closest points lie in the middle of both edges, so no vertex is involved.
    """
    gap = 0.015
    side = 0.05
    bar = trimesh.creation.box(extents=(0.5, side, side))
    # Turn a long edge of the bar towards -y, then align the bar with (1, -1, 0) so that edge
    # faces the link corner along w = (1, 1, 0) / sqrt(2).
    bar.apply_transform(trimesh.transformations.rotation_matrix(np.pi / 4, [1, 0, 0]))
    bar.apply_transform(trimesh.transformations.rotation_matrix(-np.pi / 4, [0, 0, 1]))
    w = np.array([1.0, 1.0, 0.0]) / np.sqrt(2.0)
    bar.apply_translation(np.array([0.1, 0.1, 0.0]) + w * (gap + side / np.sqrt(2.0)))
    scene = _mesh_scene("bar", bar)
    tf = _link_transforms([0.0])
    assert _check(robot_mesh, scene, tf, margin=0.01, check_containment=False)[0].tolist() == [0]
    assert _check(robot_mesh, scene, tf, margin=0.02, check_containment=False)[0].tolist() == [1]


# =============================================================================
# Voxel obstacles
# =============================================================================


def _box_esdf(voxel_size=0.02, grid_dims=(1.0, 1.0, 1.0), box_half=(0.15, 0.15, 0.15)):
    nx, ny, nz = (round(d / voxel_size) for d in grid_dims)
    ix = torch.arange(nx, device=DEVICE_CFG.device, dtype=torch.float32)
    iy = torch.arange(ny, device=DEVICE_CFG.device, dtype=torch.float32)
    iz = torch.arange(nz, device=DEVICE_CFG.device, dtype=torch.float32)
    gx, gy, gz = torch.meshgrid(ix, iy, iz, indexing="ij")

    wx = (gx - (nx - 1) / 2.0) * voxel_size
    wy = (gy - (ny - 1) / 2.0) * voxel_size
    wz = (gz - (nz - 1) / 2.0) * voxel_size

    dx = wx.abs() - box_half[0]
    dy = wy.abs() - box_half[1]
    dz = wz.abs() - box_half[2]
    outside = torch.sqrt(dx.clamp(min=0) ** 2 + dy.clamp(min=0) ** 2 + dz.clamp(min=0) ** 2)
    inside = torch.stack([dx, dy, dz], dim=-1).max(dim=-1).values.clamp(max=0)

    return VoxelGrid(
        name="esdf",
        pose=[0, 0, 0, 1, 0, 0, 0],
        dims=list(grid_dims),
        voxel_size=voxel_size,
        feature_tensor=(outside + inside).to(torch.float16),
        feature_dtype=torch.float16,
    )


def test_voxel_boundary_within_resolution(robot_mesh):
    """Link half-width 0.1 plus box half-width 0.15 -> contact for |x| < 0.25.

    The sampled ESDF test is resolution limited, so the boundary is only accurate to
    roughly one voxel.
    """
    voxel_size = 0.02
    scene = SceneData.from_scene_cfg(
        SceneCfg(voxel=[_box_esdf(voxel_size=voxel_size)]), device_cfg=DEVICE_CFG
    )
    coll, _ = _check(robot_mesh, scene, _link_transforms([0.0, 0.20]))
    assert coll.tolist() == [1, 1]

    coll, _ = _check(robot_mesh, scene, _link_transforms([0.25 + 3 * voxel_size]))
    assert coll.tolist() == [0]


def _large_triangle_robot() -> RobotMeshData:
    """One triangle with 0.64 m legs along x and y in the plane z = 0.

    On a 1 cm grid it needs 91 subdivisions, so the cap of 16 applies: samples are 4 cm apart
    along the legs, at x, y = -0.18 + 0.04 k, and the origin is the centre of a lattice cell.
    """
    corner = -0.18
    vertices = np.array(
        [[corner, corner, 0.0], [corner + 0.64, corner, 0.0], [corner, corner + 0.64, 0.0]]
    )
    return RobotMeshData.from_link_meshes(
        link_meshes={
            "l0": Mesh(
                name="l0",
                vertices=vertices,
                faces=np.array([[0, 1, 2]]),
                pose=[0, 0, 0, 1, 0, 0, 0],
            )
        },
        link_name_to_idx_map={"l0": 0},
        num_links=1,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
    )


def _translation(z: float) -> torch.Tensor:
    tf = torch.zeros(1, 1, 1, 3, 4, device=DEVICE_CFG.device, dtype=torch.float32)
    tf[..., 0, 0] = tf[..., 1, 1] = tf[..., 2, 2] = 1.0
    tf[..., 2, 3] = z
    return tf


def test_voxel_capped_sampling_still_catches_small_obstacles():
    """A 2 cm cube between the samples of a large triangle is still caught.

    The triangle cuts through the cube, but the nearest samples are 1.4 cm from it. The
    threshold widened for the coarse lattice covers that gap.
    """
    scene = SceneData.from_scene_cfg(
        SceneCfg(
            voxel=[_box_esdf(voxel_size=0.01, grid_dims=(0.2, 0.2, 0.2), box_half=(0.01,) * 3)]
        ),
        device_cfg=DEVICE_CFG,
    )
    robot = _large_triangle_robot()
    assert _check(robot, scene, _translation(0.0))[0].tolist() == [1]
    # The widening is bounded: with the cube 5 cm above the triangle, it is clear.
    assert _check(robot, scene, _translation(-0.06))[0].tolist() == [0]


# =============================================================================
# Scene handling
# =============================================================================


def test_empty_scene_reports_no_collision(robot_mesh):
    scene = SceneData.from_scene_cfg(SceneCfg(), device_cfg=DEVICE_CFG)
    coll, link = _check(robot_mesh, scene, _link_transforms([0.0, 1.0]))
    assert coll.tolist() == [0, 0]
    assert link.tolist() == [-1, -1]


def test_multi_env_uses_env_query_idx(robot_mesh):
    """Env 0 holds an obstacle at the origin; env 1 is empty."""
    scene = SceneData.from_batch_scene_cfg(
        [
            SceneCfg(cuboid=[Cuboid(name="c", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.3, 0.3, 0.3])]),
            SceneCfg(cuboid=[Cuboid(name="c", pose=[9, 9, 9, 1, 0, 0, 0], dims=[0.3, 0.3, 0.3])]),
        ],
        device_cfg=DEVICE_CFG,
    )
    tf = _link_transforms([0.0, 0.0])
    env_idx = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE_CFG.device)
    coll, _ = _check(robot_mesh, scene, tf, env_query_idx=env_idx)
    assert coll.tolist() == [1, 0]


def test_horizon_is_flattened(robot_mesh):
    """Each (batch, horizon) entry is evaluated independently."""
    scene = SceneData.from_scene_cfg(
        SceneCfg(cuboid=[Cuboid(name="c", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.3, 0.3, 0.3])]),
        device_cfg=DEVICE_CFG,
    )
    tf = torch.zeros(1, 3, 1, 3, 4, device=DEVICE_CFG.device, dtype=torch.float32)
    tf[..., 0, 0] = 1.0
    tf[..., 1, 1] = 1.0
    tf[..., 2, 2] = 1.0
    tf[0, 0, 0, 0, 3] = 0.0
    tf[0, 1, 0, 0, 3] = 5.0
    tf[0, 2, 0, 0, 3] = 0.1
    coll, _ = _check(robot_mesh, scene, tf)
    assert coll.tolist() == [1, 0, 1]


# =============================================================================
# Robot mesh packing
# =============================================================================


def test_decimation_respects_budget():
    pytest.importorskip(
        "fast_simplification",
        reason="quadric decimation backend required to honour the triangle budget",
    )
    sphere = trimesh.creation.icosphere(subdivisions=3)
    mesh = Mesh(
        name="l0",
        vertices=np.asarray(sphere.vertices),
        faces=np.asarray(sphere.faces),
        pose=[0, 0, 0, 1, 0, 0, 0],
    )
    data = RobotMeshData.from_link_meshes(
        link_meshes={"l0": mesh},
        link_name_to_idx_map={"l0": 0},
        num_links=1,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=64,
    )
    assert data.total_tris <= 64
    assert data.tri_verts.shape == (data.total_tris * 3, 3)
    assert data.tri_link.shape == (data.total_tris,)


def test_mesh_under_budget_is_left_untouched():
    """A mesh already within budget must not be altered."""
    data = RobotMeshData.from_link_meshes(
        link_meshes={"l0": _box_mesh("l0", (0.2, 0.2, 0.2))},
        link_name_to_idx_map={"l0": 0},
        num_links=1,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=64,
    )
    assert data.total_tris == 12
    assert data.tri_verts.shape == (36, 3)


def test_triangles_map_to_link_body_indices():
    data = RobotMeshData.from_link_meshes(
        link_meshes={
            "a": _box_mesh("a", (0.1, 0.1, 0.1)),
            "b": _box_mesh("b", (0.1, 0.1, 0.1)),
        },
        link_name_to_idx_map={"a": 2, "b": 5},
        num_links=8,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
    )
    assert data.total_tris == 24
    assert set(data.tri_link.tolist()) == {2, 5}
    assert data.num_links == 8


def test_to_warp_rejects_mismatched_link_count(robot_mesh):
    tf = torch.zeros(1, 1, 3, 3, 4, device=DEVICE_CFG.device, dtype=torch.float32)
    with pytest.raises(Exception):
        robot_mesh.to_warp(tf)


def _origin_cuboid_scene() -> SceneData:
    return SceneData.from_scene_cfg(
        SceneCfg(cuboid=[Cuboid(name="c", pose=[0, 0, 0, 1, 0, 0, 0], dims=[0.3, 0.3, 0.3])]),
        device_cfg=DEVICE_CFG,
    )


def _two_link_transforms(x0: float, x1: float) -> torch.Tensor:
    """Identity-rotation transforms for two links placed along +x."""
    tf = torch.zeros(1, 1, 2, 3, 4, device=DEVICE_CFG.device, dtype=torch.float32)
    tf[..., 0, 0] = 1.0
    tf[..., 1, 1] = 1.0
    tf[..., 2, 2] = 1.0
    tf[0, 0, 0, 0, 3] = x0
    tf[0, 0, 1, 0, 3] = x1
    return tf


def test_disable_and_enable_static_link_mesh():
    data = RobotMeshData.from_link_meshes(
        link_meshes={"l0": _box_mesh("l0", (0.2, 0.2, 0.2))},
        link_name_to_idx_map={"l0": 0},
        num_links=1,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
    )
    scene = _origin_cuboid_scene()
    tf = _link_transforms([0.0])
    assert _check(data, scene, tf)[0].tolist() == [1]

    data.disable_link_mesh("l0")
    assert data.num_active_triangles == 0
    assert _check(data, scene, tf)[0].tolist() == [0]

    data.enable_link_mesh("l0")
    assert data.num_active_triangles == 12
    assert _check(data, scene, tf)[0].tolist() == [1]


def test_enable_link_mesh_restores_runtime_geometry():
    """A reserved link comes back with the geometry last written, or stays empty."""
    data = RobotMeshData.from_link_meshes(
        link_meshes={"l0": _box_mesh("l0", (0.2, 0.2, 0.2))},
        link_name_to_idx_map={"l0": 0, "obj": 1},
        num_links=2,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
        extra_link_triangles={"obj": 16},
    )
    assert data.has_link_mesh("obj")
    assert not data.has_link_mesh("missing")
    scene = _origin_cuboid_scene()
    # The static link sits far away, the reserved link at the obstacle.
    tf = _two_link_transforms(5.0, 0.0)

    data.enable_link_mesh("obj")
    assert _check(data, scene, tf)[0].tolist() == [0]

    obj = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    data.update_link_mesh("obj", obj)
    assert _check(data, scene, tf)[0].tolist() == [1]

    data.disable_link_mesh("obj")
    assert _check(data, scene, tf)[0].tolist() == [0]
    data.enable_link_mesh("obj")
    assert data.num_active_triangles == 24
    assert _check(data, scene, tf)[0].tolist() == [1]

    # Writing while disabled reactivates the slots with the new geometry.
    data.disable_link_mesh("obj")
    data.update_link_mesh("obj", obj)
    assert _check(data, scene, tf)[0].tolist() == [1]


def test_clear_link_mesh_is_not_undone_by_enable():
    """Released geometry stays gone when the link's collision is re-enabled."""
    data = RobotMeshData.from_link_meshes(
        link_meshes={"l0": _box_mesh("l0", (0.2, 0.2, 0.2))},
        link_name_to_idx_map={"l0": 0, "obj": 1},
        num_links=2,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
        extra_link_triangles={"obj": 16},
    )
    scene = _origin_cuboid_scene()
    tf = _two_link_transforms(5.0, 0.0)
    data.update_link_mesh("obj", trimesh.creation.box(extents=(0.1, 0.1, 0.1)))
    assert _check(data, scene, tf)[0].tolist() == [1]

    data.clear_link_mesh("obj")
    assert _check(data, scene, tf)[0].tolist() == [0]
    data.enable_link_mesh("obj")
    assert data.num_active_triangles == 12
    assert _check(data, scene, tf)[0].tolist() == [0]


def test_ignored_links_are_not_packed():
    """An ignored link contributes no geometry, so it can never report a collision."""
    link_meshes = {
        "a": _box_mesh("a", (0.1, 0.1, 0.1)),
        "b": _box_mesh("b", (0.1, 0.1, 0.1)),
    }
    data = RobotMeshData.from_link_meshes(
        link_meshes=link_meshes,
        link_name_to_idx_map={"a": 2, "b": 5},
        num_links=8,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
        ignored_link_names=["a"],
    )
    assert data.link_names == ["b"]
    assert data.total_tris == 12
    assert set(data.tri_link.tolist()) == {5}
    assert data.body_idx_to_link_name == {5: "b"}


def test_ignored_link_reports_no_collision():
    """A link flush against an obstacle is silent once ignored."""
    scene = SceneData.from_scene_cfg(
        SceneCfg(mesh=[_box_mesh("wall", (1.0, 1.0, 1.0))]),
        num_envs=1,
        device_cfg=DEVICE_CFG,
    )
    tf = _link_transforms([0.0])
    kwargs = dict(
        link_transforms=tf,
        scene=scene,
        env_query_idx=torch.zeros(1, dtype=torch.int32, device=DEVICE_CFG.device),
        use_multi_env=False,
    )

    checked = RobotMeshData.from_link_meshes(
        link_meshes={"l0": _box_mesh("l0", (0.2, 0.2, 0.2))},
        link_name_to_idx_map={"l0": 0},
        num_links=1,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
    )
    in_collision, _ = check_robot_mesh_collision(robot_mesh=checked, **kwargs)
    assert in_collision[0].item() == 1

    with pytest.raises(Exception):
        RobotMeshData.from_link_meshes(
            link_meshes={"l0": _box_mesh("l0", (0.2, 0.2, 0.2))},
            link_name_to_idx_map={"l0": 0},
            num_links=1,
            device_cfg=DEVICE_CFG,
            max_triangles_per_link=0,
            ignored_link_names=["l0"],
        )


# =============================================================================
# Link-level culling
# =============================================================================


def _random_link_transforms(
    generator: torch.Generator, batch: int, num_links: int, low: float, high: float
) -> torch.Tensor:
    """Random rigid transforms ``[batch, 1, num_links, 3, 4]`` with positions in a cube."""
    q = torch.randn(batch * num_links, 4, generator=generator)
    q = q / q.norm(dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    rot = torch.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).view(-1, 3, 3)
    pos = low + (high - low) * torch.rand(batch * num_links, 3, 1, generator=generator)
    tf = torch.cat([rot, pos], dim=-1).view(batch, 1, num_links, 3, 4)
    return tf.to(device=DEVICE_CFG.device, dtype=torch.float32)


def _two_link_robot() -> RobotMeshData:
    plank = trimesh.creation.box(extents=(0.3, 0.04, 0.02))
    return RobotMeshData.from_link_meshes(
        link_meshes={
            "a": _box_mesh("a", (0.1, 0.1, 0.1)),
            "b": Mesh(
                name="b",
                vertices=np.asarray(plank.vertices),
                faces=np.asarray(plank.faces),
                pose=[0, 0, 0, 1, 0, 0, 0],
            ),
        },
        link_name_to_idx_map={"a": 0, "b": 1},
        num_links=2,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
    )


def _culling_scenes():
    plate = trimesh.creation.box(extents=(0.02, 0.6, 0.6))
    return {
        "cuboid": SceneData.from_scene_cfg(
            SceneCfg(
                cuboid=[
                    Cuboid(name="c", pose=[0.1, 0, 0, 1, 0, 0, 0], dims=[0.2, 0.3, 0.25]),
                ]
            ),
            device_cfg=DEVICE_CFG,
        ),
        "mesh": _mesh_scene("plate", plate, pose=[0.0, 0.05, 0, 1, 0, 0, 0]),
        "voxel": SceneData.from_scene_cfg(
            SceneCfg(voxel=[_box_esdf(voxel_size=0.02, grid_dims=(0.8, 0.8, 0.8))]),
            device_cfg=DEVICE_CFG,
        ),
    }


@pytest.mark.parametrize("scene_name", ["cuboid", "mesh", "voxel"])
@pytest.mark.parametrize("margin", [0.0, 0.02])
def test_link_culling_matches_unculled_results(scene_name, margin):
    """Skipping links far from obstacles never changes a result.

    Random poses span collisions, near misses and far links. Inflating every link's bounding
    sphere disables the culling and gives the reference.
    """
    scene = _culling_scenes()[scene_name]
    generator = torch.Generator().manual_seed(0)
    tf = _random_link_transforms(generator, 512, 2, -0.6, 0.6)

    robot = _two_link_robot()
    culled, culled_link = _check(robot, scene, tf, margin=margin)

    reference_robot = _two_link_robot()
    reference_robot.link_bounds[:, 3] = 1.0e3
    reference, reference_link = _check(reference_robot, scene, tf, margin=margin)

    assert 0 < int(reference.sum()) < reference.numel()
    assert torch.equal(culled, reference)
    assert torch.equal(culled_link, reference_link)


def test_link_culling_skips_only_far_links():
    """With one link at the obstacle and one far away, only the first is tested."""
    robot = _two_link_robot()
    scene = _culling_scenes()["cuboid"]
    tf = _two_link_transforms(0.1, 3.0)
    link_near = torch.zeros(3, 2, dtype=torch.int32, device=DEVICE_CFG.device)
    coll, _ = _check(robot, scene, tf, link_near=link_near)
    assert coll.tolist() == [1]
    assert link_near[0].tolist() == [1, 0]


def test_link_bounds_follow_runtime_geometry():
    data = RobotMeshData.from_link_meshes(
        link_meshes={"l0": _box_mesh("l0", (0.2, 0.2, 0.2))},
        link_name_to_idx_map={"l0": 0, "obj": 1},
        num_links=2,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
        extra_link_triangles={"obj": 16},
    )
    # Half the cube diagonal, centred at the origin.
    assert data.link_bounds[0].tolist() == pytest.approx([0, 0, 0, 0.1 * np.sqrt(3)], abs=1e-6)
    assert data.link_bounds[1, 3].item() < 0

    obj = trimesh.creation.box(extents=(0.1, 0.2, 0.3))
    obj.apply_translation([0.5, 0.0, 0.0])
    data.update_link_mesh("obj", obj)
    radius = 0.5 * np.sqrt(0.1**2 + 0.2**2 + 0.3**2)
    assert data.link_bounds[1].tolist() == pytest.approx([0.5, 0, 0, radius], abs=1e-6)

    data.clear_link_mesh("obj")
    assert data.link_bounds[1, 3].item() < 0


def test_voxel_culling_keeps_links_reaching_into_the_grid():
    """A link centred outside the grid can still reach an obstacle at the grid edge.

    The ESDF reads as free outside the grid, so the link centre alone would wrongly clear it.
    The grid spans |x| < 0.2 and the box |x| < 0.15; the plank spans x in [0.12, 0.42].
    """
    scene = SceneData.from_scene_cfg(
        SceneCfg(
            voxel=[
                _box_esdf(voxel_size=0.02, grid_dims=(0.4, 0.4, 0.4), box_half=(0.15, 0.15, 0.15))
            ]
        ),
        device_cfg=DEVICE_CFG,
    )
    tf = _two_link_transforms(3.0, 0.27)
    assert _check(_two_link_robot(), scene, tf)[0].tolist() == [1]
