# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for the signed clearance between the robot's link meshes and the scene."""

# Third Party
import numpy as np
import pytest
import torch
import trimesh

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_collision import (
    check_robot_mesh_collision,
    compute_robot_mesh_distance,
)
from curobo._src.geom.data.data_robot_mesh import RobotMeshData
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Cuboid, Mesh, SceneCfg, VoxelGrid
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp

DEVICE_CFG = DeviceCfg()


@pytest.fixture(scope="module", autouse=True)
def _warp():
    init_warp(device_cfg=DEVICE_CFG)


def _mesh(name: str, tri_mesh: trimesh.Trimesh, pose=None) -> Mesh:
    return Mesh(
        name=name,
        vertices=np.asarray(tri_mesh.vertices),
        faces=np.asarray(tri_mesh.faces),
        pose=pose if pose is not None else [0, 0, 0, 1, 0, 0, 0],
    )


def _cube_robot() -> RobotMeshData:
    """Single-link robot: a 0.2 m cube centred on the link origin."""
    return RobotMeshData.from_link_meshes(
        link_meshes={"l0": _mesh("l0", trimesh.creation.box(extents=(0.2, 0.2, 0.2)))},
        link_name_to_idx_map={"l0": 0},
        num_links=1,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
    )


def _two_link_robot(ignored_link_names=None) -> RobotMeshData:
    return RobotMeshData.from_link_meshes(
        link_meshes={
            "a": _mesh("a", trimesh.creation.box(extents=(0.1, 0.1, 0.1))),
            "b": _mesh("b", trimesh.creation.box(extents=(0.3, 0.04, 0.02))),
        },
        link_name_to_idx_map={"a": 0, "b": 1},
        num_links=2,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
        ignored_link_names=ignored_link_names,
    )


def _translations(positions, num_links: int = 1) -> torch.Tensor:
    """Identity-rotation transforms ``[batch, 1, num_links, 3, 4]``, one position per batch."""
    batch = len(positions)
    tf = torch.zeros(batch, 1, num_links, 3, 4, device=DEVICE_CFG.device, dtype=torch.float32)
    tf[..., 0, 0] = 1.0
    tf[..., 1, 1] = 1.0
    tf[..., 2, 2] = 1.0
    tf[:, 0, :, :, 3] = torch.as_tensor(
        positions, device=DEVICE_CFG.device, dtype=torch.float32
    ).view(batch, -1, 3)
    return tf


def _random_transforms(seed: int, batch: int, num_links: int, low: float, high: float):
    """Random rigid transforms ``[batch, 1, num_links, 3, 4]`` with positions in a cube."""
    generator = torch.Generator().manual_seed(seed)
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


def _env(batch: int) -> torch.Tensor:
    return torch.zeros(batch, dtype=torch.int32, device=DEVICE_CFG.device)


def _distance(robot, scene, tf, env_query_idx=None, **kwargs) -> torch.Tensor:
    use_multi_env = env_query_idx is not None
    if env_query_idx is None:
        env_query_idx = _env(tf.shape[0])
    return compute_robot_mesh_distance(
        robot_mesh=robot,
        link_transforms=tf,
        scene=scene,
        env_query_idx=env_query_idx,
        use_multi_env=use_multi_env,
        **kwargs,
    )


def _cuboid_scene(dims, pose=None) -> SceneData:
    pose = pose if pose is not None else [0, 0, 0, 1, 0, 0, 0]
    return SceneData.from_scene_cfg(
        SceneCfg(cuboid=[Cuboid(name="c", pose=pose, dims=list(dims))]), device_cfg=DEVICE_CFG
    )


def _box_mesh_scene(dims, pose=None) -> SceneData:
    box = trimesh.creation.box(extents=dims)
    return SceneData.from_scene_cfg(SceneCfg(mesh=[_mesh("m", box, pose)]), device_cfg=DEVICE_CFG)


def _box_esdf(voxel_size: float, grid_dims, box_half) -> VoxelGrid:
    nx, ny, nz = (round(d / voxel_size) for d in grid_dims)
    axes = [
        (torch.arange(n, device=DEVICE_CFG.device, dtype=torch.float32) - (n - 1) / 2.0)
        * voxel_size
        for n in (nx, ny, nz)
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    q = grid.abs() - torch.as_tensor(box_half, device=DEVICE_CFG.device)
    outside = q.clamp(min=0).norm(dim=-1)
    inside = q.max(dim=-1).values.clamp(max=0)
    return VoxelGrid(
        name="esdf",
        pose=[0, 0, 0, 1, 0, 0, 0],
        dims=list(grid_dims),
        voxel_size=voxel_size,
        feature_tensor=(outside + inside).to(torch.float16),
        feature_dtype=torch.float16,
    )


def _voxel_scene(voxel_size=0.02, grid_dims=(1.0, 1.0, 1.0), box_half=(0.15, 0.15, 0.15)):
    return SceneData.from_scene_cfg(
        SceneCfg(voxel=[_box_esdf(voxel_size, grid_dims, box_half)]), device_cfg=DEVICE_CFG
    )


EXACT_SCENES = {"cuboid": _cuboid_scene, "mesh": _box_mesh_scene}


# =============================================================================
# Exact distances to cuboid and mesh obstacles
# =============================================================================


@pytest.mark.parametrize("scene_type", ["cuboid", "mesh"])
def test_face_to_face_distance(scene_type):
    """Link half-width 0.1 plus box half-width 0.15: the gap is x - 0.25."""
    scene = EXACT_SCENES[scene_type]((0.3, 0.3, 0.3))
    tf = _translations([[0.26, 0, 0], [0.3, 0, 0], [0.33, 0, 0], [0.5, 0, 0]])
    distance = _distance(_cube_robot(), scene, tf, max_distance=0.1)
    assert distance.shape == (4, 1)
    expected = torch.tensor([0.01, 0.05, 0.08, 0.1], device=DEVICE_CFG.device)
    assert torch.allclose(distance[:, 0], expected, atol=1.0e-5)


@pytest.mark.parametrize("scene_type", ["cuboid", "mesh"])
def test_edge_to_edge_and_corner_distance(scene_type):
    """Diagonal offsets make an edge pair, then a corner pair, the closest features."""
    scene = EXACT_SCENES[scene_type]((0.3, 0.3, 0.3))
    tf = _translations([[0.3, 0.3, 0.0], [0.3, 0.3, 0.3]])
    distance = _distance(_cube_robot(), scene, tf, max_distance=0.3)
    expected = torch.tensor([0.05 * 2**0.5, 0.05 * 3**0.5], device=DEVICE_CFG.device)
    assert torch.allclose(distance[:, 0], expected, atol=1.0e-5)


@pytest.mark.parametrize("scene_type", ["cuboid", "mesh"])
def test_rotated_link_distance_matches_a_sampled_reference(scene_type):
    """The exact distance is at most, and close to, a sampled reference.

    Random link poses; the reference is the box SDF at many points sampled on the link surface.
    """
    half = np.array([0.15, 0.1, 0.05])
    scene = EXACT_SCENES[scene_type](tuple(2 * half))
    robot = _two_link_robot()
    tf = _random_transforms(1, 64, 2, -0.35, 0.35)
    distance = _distance(robot, scene, tf, max_distance=0.2).cpu().numpy()

    for link, extents in ((0, (0.1, 0.1, 0.1)), (1, (0.3, 0.04, 0.02))):
        points = trimesh.creation.box(extents=extents).sample(20000, return_index=False)
        points = np.asarray(points, dtype=np.float32)
        for cfg in range(tf.shape[0]):
            pose = tf[cfg, 0, link].cpu().numpy()
            world = points @ pose[:, :3].T + pose[:, 3]
            q = np.abs(world) - half
            sdf = np.linalg.norm(np.maximum(q, 0.0), axis=-1) + np.minimum(q.max(axis=-1), 0.0)
            reference = min(float(sdf.min()), 0.2)
            if reference <= 0.0:
                assert distance[cfg, link] <= 1.0e-5
                continue
            assert distance[cfg, link] <= reference + 1.0e-5
            assert distance[cfg, link] >= reference - 6.0e-3


def test_cuboid_and_mesh_of_the_same_box_agree():
    """The cuboid and mesh kernels share no narrowphase code, so agreeing is a cross-check."""
    dims = (0.3, 0.2, 0.1)
    tf = _random_transforms(3, 256, 2, -0.35, 0.35)
    cuboid = _distance(_two_link_robot(), _cuboid_scene(dims), tf, max_distance=0.2)
    mesh = _distance(_two_link_robot(), _box_mesh_scene(dims), tf, max_distance=0.2)
    assert ((cuboid > 0) & (cuboid < 0.2)).sum() > 100 and (cuboid < 0).sum() > 10
    assert torch.allclose(cuboid, mesh, atol=1.0e-5)


@pytest.mark.parametrize("scene_type", ["cuboid", "mesh"])
def test_penetration_is_the_deepest_vertex(scene_type):
    """The link's face at x = 0.12 is 3 cm inside the box face at x = 0.15."""
    scene = EXACT_SCENES[scene_type]((0.3, 2.0, 2.0))
    distance = _distance(_cube_robot(), scene, _translations([[0.22, 0, 0]]))
    assert distance[0, 0].item() == pytest.approx(-0.03, abs=1.0e-4)


@pytest.mark.parametrize("scene_type", ["cuboid", "mesh"])
def test_contained_link_is_negative(scene_type):
    """A link swallowed whole: its vertices are 0.4 m inside the faces of a 1 m box."""
    scene = EXACT_SCENES[scene_type]((1.0, 1.0, 1.0))
    distance = _distance(_cube_robot(), scene, _translations([[0, 0, 0]]))
    assert distance[0, 0].item() == pytest.approx(-0.4, abs=1.0e-4)


def test_mesh_crossing_reports_zero_without_containment():
    """Open meshes have no inside: crossings report 0, and contained links look clear."""
    crossing = _distance(
        _cube_robot(),
        _box_mesh_scene((0.3, 2.0, 2.0)),
        _translations([[0.22, 0, 0]]),
        check_containment=False,
    )
    assert crossing[0, 0].item() == 0.0
    contained = _distance(
        _cube_robot(),
        _box_mesh_scene((1.0, 1.0, 1.0)),
        _translations([[0, 0, 0]]),
        check_containment=False,
    )
    assert contained[0, 0].item() == pytest.approx(0.1)


def test_nearest_of_several_obstacles():
    scene = SceneData.from_scene_cfg(
        SceneCfg(
            cuboid=[
                Cuboid(name="near", pose=[0.3, 0, 0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1]),
                Cuboid(name="far", pose=[-0.4, 0, 0, 1, 0, 0, 0], dims=[0.1, 0.1, 0.1]),
            ],
            mesh=[
                _mesh("m", trimesh.creation.box(extents=(0.1, 0.1, 0.1)), [0, 0.32, 0, 1, 0, 0, 0])
            ],
        ),
        device_cfg=DEVICE_CFG,
    )
    distance = _distance(_cube_robot(), scene, _translations([[0, 0, 0]]))
    # Near cuboid face at x = 0.25: 0.15 away; mesh face at y = 0.27: 0.17 away.
    assert distance[0, 0].item() == pytest.approx(0.1)
    distance = _distance(_cube_robot(), scene, _translations([[0, 0, 0]]), max_distance=0.5)
    assert distance[0, 0].item() == pytest.approx(0.15, abs=1.0e-5)


# =============================================================================
# Voxel obstacles
# =============================================================================


def test_voxel_distance_within_resolution():
    voxel_size = 0.02
    scene = _voxel_scene(voxel_size=voxel_size)
    tf = _translations([[0.3, 0, 0], [0.32, 0, 0], [0.2, 0, 0]])
    distance = _distance(_cube_robot(), scene, tf)[:, 0]
    expected = torch.tensor([0.05, 0.07, -0.05], device=DEVICE_CFG.device)
    assert torch.allclose(distance, expected, atol=voxel_size)


# =============================================================================
# Consistency with the collision check, culling and bookkeeping
# =============================================================================


def _consistency_scenes():
    plate = trimesh.creation.box(extents=(0.02, 0.6, 0.6))
    return {
        "cuboid": _cuboid_scene((0.2, 0.3, 0.25), pose=[0.1, 0, 0, 1, 0, 0, 0]),
        "mesh": SceneData.from_scene_cfg(
            SceneCfg(mesh=[_mesh("plate", plate, [0.0, 0.05, 0, 1, 0, 0, 0])]),
            device_cfg=DEVICE_CFG,
        ),
        "voxel": _voxel_scene(grid_dims=(0.8, 0.8, 0.8)),
    }


@pytest.mark.parametrize("scene_name", ["cuboid", "mesh", "voxel"])
def test_distance_agrees_with_the_collision_check(scene_name):
    """A configuration collides exactly where its clearance is within the margin.

    Cuboids are compared at margin 0, since the cuboid check inflates the box rather than
    measuring the exact clearance; a crossing without a vertex inside reports 0, hence
    ``<=``. The voxel check flags samples at or below its threshold.
    """
    margin = 0.0 if scene_name == "cuboid" else 0.02
    scene = _consistency_scenes()[scene_name]
    robot = _two_link_robot()
    tf = _random_transforms(0, 512, 2, -0.6, 0.6)
    collides, _ = check_robot_mesh_collision(
        robot, tf, scene, _env(512), use_multi_env=False, margin=margin
    )
    distance = _distance(robot, scene, tf).min(dim=-1).values
    assert 0 < int(collides.sum()) < collides.numel()
    within = distance < margin if scene_name == "mesh" else distance <= margin
    assert torch.equal(within.to(torch.int32), collides)


@pytest.mark.parametrize("scene_name", ["cuboid", "mesh", "voxel"])
def test_link_culling_does_not_change_the_distance(scene_name):
    scene = _consistency_scenes()[scene_name]
    tf = _random_transforms(2, 256, 2, -0.6, 0.6)
    culled = _distance(_two_link_robot(), scene, tf)
    reference_robot = _two_link_robot()
    reference_robot.link_bounds[:, 3] = 1.0e3
    reference = _distance(reference_robot, scene, tf)
    assert torch.equal(culled, reference)
    assert (culled < 0.1).any() and (culled == 0.1).any()


def test_ignored_link_reports_the_cap():
    """Both links sit inside a 0.4 m box; only the plank is tested, 5 cm deep."""
    scene = _cuboid_scene((0.4, 0.4, 0.4))
    tf = _translations([[0, 0, 0, 0, 0, 0]], num_links=2)
    distance = _distance(_two_link_robot(ignored_link_names=["a"]), scene, tf)
    assert distance[0, 0].item() == pytest.approx(0.1)
    assert distance[0, 1].item() == pytest.approx(-0.05, abs=1.0e-5)


def test_multi_env_uses_env_query_idx():
    """Env 0 holds a box 5 cm from the link; env 1 holds it far away."""
    scene = SceneData.from_batch_scene_cfg(
        [
            SceneCfg(
                cuboid=[Cuboid(name="c", pose=[0.3, 0, 0, 1, 0, 0, 0], dims=[0.3, 0.3, 0.3])]
            ),
            SceneCfg(cuboid=[Cuboid(name="c", pose=[9, 9, 9, 1, 0, 0, 0], dims=[0.3, 0.3, 0.3])]),
        ],
        device_cfg=DEVICE_CFG,
    )
    tf = _translations([[0, 0, 0], [0, 0, 0]])
    env = torch.tensor([0, 1], dtype=torch.int32, device=DEVICE_CFG.device)
    distance = _distance(_cube_robot(), scene, tf, env_query_idx=env)[:, 0]
    assert distance.tolist() == pytest.approx([0.05, 0.1], abs=1.0e-5)


def test_horizon_and_reused_buffers():
    scene = _cuboid_scene((0.3, 0.3, 0.3))
    tf = _translations([[0.3, 0, 0], [0.33, 0, 0]]).view(1, 2, 1, 3, 4)
    link_distance = torch.empty(2, dtype=torch.float32, device=DEVICE_CFG.device)
    link_near = torch.ones(3, 2, dtype=torch.int32, device=DEVICE_CFG.device)
    for _ in range(2):
        distance = _distance(
            _cube_robot(), scene, tf, link_distance=link_distance, link_near=link_near
        )
        assert distance.shape == (2, 1)
        assert distance[:, 0].tolist() == pytest.approx([0.05, 0.08], abs=1.0e-5)


def test_empty_scene_and_invalid_cap():
    scene = SceneData.from_scene_cfg(SceneCfg(), device_cfg=DEVICE_CFG)
    distance = _distance(_cube_robot(), scene, _translations([[0, 0, 0]]), max_distance=0.25)
    assert distance.tolist() == [[0.25]]
    with pytest.raises(ValueError, match="max_distance"):
        _distance(_cube_robot(), scene, _translations([[0, 0, 0]]), max_distance=0.0)
