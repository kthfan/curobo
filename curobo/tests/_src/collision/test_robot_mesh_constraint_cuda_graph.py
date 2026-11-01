# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""CUDA graph replays of the exact robot-mesh collision check follow in-place updates.

The check is captured once. The obstacles and the robot's link meshes are then changed the
way ``update_world``, the attachment manager and the link toggles change them at runtime, and
every replay must match an eager call on the same data.
"""

# Standard Library
from typing import Callable, Dict, List, Tuple

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
from curobo._src.types.pose import Pose
from curobo._src.util.cuda_graph_util import GraphExecutor
from curobo._src.util.warp import init_warp

DEVICE_CFG = DeviceCfg()

#: Voxel cache shared by every voxel grid in the tests.
VOXEL_SIZE = 0.02
VOXEL_GRID_DIMS = (0.8, 0.8, 0.8)

#: Number of random configurations checked at every step.
BATCH = 256


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


def _robot() -> RobotMeshData:
    """Links ``a`` (cube) and ``b`` (plank), and ``obj`` with slots for runtime geometry."""
    return RobotMeshData.from_link_meshes(
        link_meshes={
            "a": _mesh("a", trimesh.creation.box(extents=(0.1, 0.1, 0.1))),
            "b": _mesh("b", trimesh.creation.box(extents=(0.3, 0.04, 0.02))),
        },
        link_name_to_idx_map={"a": 0, "b": 1, "obj": 2},
        num_links=3,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
        extra_link_triangles={"obj": 64},
    )


def _scene() -> SceneData:
    """Fixed-capacity scene, as the solvers allocate it from their collision cache."""
    return SceneData.create_cache(
        num_envs=1,
        device_cfg=DEVICE_CFG,
        cuboid_cache=2,
        mesh_cache=2,
        voxel_cache={"layers": 1, "dims": list(VOXEL_GRID_DIMS), "voxel_size": VOXEL_SIZE},
    )


def _box_esdf(name: str, box_half: float) -> VoxelGrid:
    """ESDF of a cube of half extent ``box_half`` centred in the voxel cache's grid."""
    n = [round(d / VOXEL_SIZE) for d in VOXEL_GRID_DIMS]
    axes = [
        (torch.arange(k, device=DEVICE_CFG.device, dtype=torch.float32) - (k - 1) / 2.0)
        * VOXEL_SIZE
        for k in n
    ]
    grid = torch.stack(torch.meshgrid(*axes, indexing="ij"), dim=-1)
    d = grid.abs() - box_half
    outside = d.clamp(min=0).norm(dim=-1)
    inside = d.max(dim=-1).values.clamp(max=0)
    return VoxelGrid(
        name=name,
        pose=[0, 0, 0, 1, 0, 0, 0],
        dims=list(VOXEL_GRID_DIMS),
        voxel_size=VOXEL_SIZE,
        feature_tensor=(outside + inside).to(torch.float16),
        feature_dtype=torch.float16,
    )


def _random_link_transforms(seed: int, num_links: int = 3) -> torch.Tensor:
    """Random rigid transforms ``[BATCH, 1, num_links, 3, 4]`` with positions in a 1.2 m cube."""
    generator = torch.Generator().manual_seed(seed)
    q = torch.randn(BATCH * num_links, 4, generator=generator)
    w, x, y, z = (q / q.norm(dim=-1, keepdim=True)).unbind(-1)
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
    pos = -0.6 + 1.2 * torch.rand(BATCH * num_links, 3, 1, generator=generator)
    tf = torch.cat([rot, pos], dim=-1).view(BATCH, 1, num_links, 3, 4)
    return tf.to(device=DEVICE_CFG.device, dtype=torch.float32)


def _pose(position) -> Pose:
    return Pose.from_list([*position, 1.0, 0.0, 0.0, 0.0], device_cfg=DEVICE_CFG)


def _box(name: str = "box", position=(0.1, 0.0, 0.0)) -> Cuboid:
    return Cuboid(name=name, pose=[*position, 1, 0, 0, 0], dims=[0.2, 0.3, 0.25])


def _update_steps(
    robot: RobotMeshData, scene: SceneData, state: Dict[str, torch.Tensor]
) -> List[Tuple[str, Callable[[], None]]]:
    """Runtime updates, applied in order after the graph is captured.

    The scene holds only a cuboid at capture time, so the mesh and voxel kernels are
    captured while their obstacle types are empty.
    """
    plate = trimesh.creation.box(extents=(0.02, 0.6, 0.6))
    ball = trimesh.creation.icosphere(subdivisions=2, radius=0.2)
    attached = trimesh.creation.box(extents=(0.1, 0.2, 0.3))

    def set_tf():
        state["tf"] = _random_link_transforms(seed=1)

    return [
        ("new link transforms", set_tf),
        ("cuboid moved", lambda: scene.update_obstacle_pose("box", _pose((-0.2, 0.1, 0.0)))),
        ("cuboid added", lambda: scene.add_obstacle(_box("box2", (0.3, -0.3, 0.2)))),
        ("cuboid disabled", lambda: scene.enable_obstacle("box", False)),
        (
            "mesh loaded",
            lambda: scene.load_from_scene_cfg(SceneCfg(mesh=[_mesh("plate", plate)])),
        ),
        (
            "mesh replaced",
            lambda: scene.load_from_scene_cfg(
                SceneCfg(mesh=[_mesh("ball", ball, pose=[0.2, 0.2, 0.0, 1, 0, 0, 0])])
            ),
        ),
        ("mesh moved", lambda: scene.update_obstacle_pose("ball", _pose((-0.3, 0.0, 0.1)))),
        (
            "voxel grid loaded",
            lambda: scene.load_from_scene_cfg(SceneCfg(voxel=[_box_esdf("esdf", 0.15)])),
        ),
        (
            "voxel grid replaced",
            lambda: scene.load_from_scene_cfg(SceneCfg(voxel=[_box_esdf("esdf2", 0.3)])),
        ),
        ("cuboid scene loaded", lambda: scene.load_from_scene_cfg(SceneCfg(cuboid=[_box()]))),
        ("link mesh written", lambda: robot.update_link_mesh("obj", attached)),
        ("link mesh disabled", lambda: robot.disable_link_mesh("a")),
        ("link mesh enabled", lambda: robot.enable_link_mesh("a")),
        ("link mesh cleared", lambda: robot.clear_link_mesh("obj")),
    ]


def run_update_steps(use_cuda_graph: bool, margin: float) -> List[str]:
    """Replay one captured check after each update and compare it with an eager call.

    Args:
        use_cuda_graph: Capture the check in a CUDA graph; False runs it eagerly, so the
            comparison is trivial but the steps can be checked on a CPU.
        margin: Clearance passed to the check.

    Returns:
        Names of the steps that did not change the eager result. A step that changes nothing
        cannot reveal a stale graph.
    """
    robot = _robot()
    scene = _scene()
    scene.load_from_scene_cfg(SceneCfg(cuboid=[_box()]))
    state = {"tf": _random_link_transforms(seed=0)}

    num_cfgs = BATCH
    device = DEVICE_CFG.device
    env_query_idx = torch.zeros(BATCH, dtype=torch.int32, device=device)
    in_collision = torch.zeros(num_cfgs, dtype=torch.int32, device=device)
    contact_link = torch.zeros(num_cfgs, dtype=torch.int32, device=device)
    link_near = torch.zeros(3, num_cfgs * robot.num_links, dtype=torch.int32, device=device)

    def check(tf: torch.Tensor, buffers: bool = True) -> Tuple[torch.Tensor, torch.Tensor]:
        kwargs = (
            {"in_collision": in_collision, "contact_link": contact_link, "link_near": link_near}
            if buffers
            else {}
        )
        return check_robot_mesh_collision(
            robot_mesh=robot,
            link_transforms=tf,
            scene=scene,
            env_query_idx=env_query_idx,
            use_multi_env=False,
            margin=margin,
            check_containment=True,
            **kwargs,
        )

    executor = GraphExecutor(check, device, use_cuda_graph=use_cuda_graph, clone_outputs=True)

    unchanged = []
    previous = None
    for name, update in [("captured", lambda: None)] + _update_steps(robot, scene, state):
        update()
        replayed = executor(state["tf"])
        eager = check(state["tf"], buffers=False)
        assert torch.equal(replayed[0], eager[0]), f"in_collision differs after: {name}"
        assert torch.equal(replayed[1], eager[1]), f"contact_link differs after: {name}"
        if previous is not None and all(torch.equal(e, p) for e, p in zip(eager, previous)):
            unchanged.append(name)
        previous = [t.clone() for t in eager]
    return unchanged


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA graphs need CUDA")
@pytest.mark.parametrize("margin", [0.0, 0.02])
def test_replay_follows_in_place_updates(margin):
    unchanged = run_update_steps(use_cuda_graph=True, margin=margin)
    assert unchanged == []
