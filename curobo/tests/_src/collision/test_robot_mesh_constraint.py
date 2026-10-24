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
        env_query_idx = torch.zeros(
            tf.shape[0], dtype=torch.int32, device=DEVICE_CFG.device
        )
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
