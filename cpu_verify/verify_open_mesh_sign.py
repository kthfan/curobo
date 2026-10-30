# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Measure the inside/outside classification of open obstacle meshes on the CPU.

Branch-local verification tool. Do not merge into ``dev/main``.

The mesh containment pass of the robot-mesh check and the sphere collision cost both read the
sign of the mesh SDF (``data_mesh.compute_local_sdf``), which Warp's ``mesh_query_point``
derives from the orientation of the closest face hit by rays along +x, +y and +z (majority
vote). This reports, for closed and open meshes, which regions read as inside, and checks that
the containment pass of ``check_robot_mesh_collision`` and the sphere collision check both flag
exactly the points the SDF reads as inside.

Usage (from the repository root)::

    PYTHONPATH=. python cpu_verify/verify_open_mesh_sign.py
"""

# Patch DeviceCfg to the CPU before any other CuRobo import.
import cpu_verify.cpu_device_plugin  # noqa: F401, I001

# Standard Library
from typing import Dict, Tuple

# Third Party
import numpy as np
import torch
import trimesh
import warp as wp

# CuRobo
from curobo._src.geom.collision.buffer_collision import CollisionBuffer
from curobo._src.geom.collision.collision_scene import SceneCollision, SceneCollisionCfg
from curobo._src.geom.collision.wp_robot_mesh_collision import check_robot_mesh_collision
from curobo._src.geom.data import data_mesh
from curobo._src.geom.data.data_mesh import MeshDataWarp
from curobo._src.geom.data.data_robot_mesh import RobotMeshData
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh, SceneCfg
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp

DEVICE_CFG = DeviceCfg()
_sdf = wp.func(data_mesh.compute_local_sdf, module=__name__)

#: Edge of the robot cube placed at each query point.
CUBE = 0.01


@wp.kernel
def _sdf_kernel(obs: MeshDataWarp, pts: wp.array(dtype=wp.vec3), out: wp.array(dtype=wp.float32)):
    i = wp.tid()
    out[i] = _sdf(obs, 0, 0, pts[i])


def _quad(normal_tilt_deg: float = 0.0) -> trimesh.Trimesh:
    """0.4 m square in the xy plane, normal +z, tilted about x."""
    v = np.array([[-0.2, -0.2, 0], [0.2, -0.2, 0], [0.2, 0.2, 0], [-0.2, 0.2, 0]], float)
    quad = trimesh.Trimesh(vertices=v, faces=[[0, 1, 2], [0, 2, 3]], process=False)
    quad.apply_transform(
        trimesh.transformations.rotation_matrix(np.radians(normal_tilt_deg), [1, 0, 0])
    )
    return quad


def _quad_facing(normal) -> trimesh.Trimesh:
    """0.4 m square through the origin with the given normal."""
    quad = _quad()
    quad.apply_transform(trimesh.geometry.align_vectors([0, 0, 1], np.asarray(normal, float)))
    return quad


def _meshes() -> Dict[str, trimesh.Trimesh]:
    box = trimesh.creation.box(extents=(0.4, 0.4, 0.4))
    top_up = box.face_normals[:, 2] > 0.9
    open_box = trimesh.Trimesh(box.vertices, box.faces[~top_up], process=False)
    flipped = box.copy()
    faces = flipped.faces.copy()
    side = flipped.face_normals[:, 0] > 0.9
    faces[side] = faces[side][:, ::-1]
    flipped = trimesh.Trimesh(flipped.vertices, faces, process=False)
    quad = _quad()
    double = trimesh.Trimesh(
        quad.vertices, np.vstack([quad.faces, quad.faces[:, ::-1]]), process=False
    )
    return {
        "closed box": box,
        "box without top face": open_box,
        "box with +x face flipped": flipped,
        "horizontal quad": quad,
        "quad tilted 45 deg": _quad(45.0),
        "double-sided quad": double,
        "quad facing (1, 1, 1)": _quad_facing((1.0, 1.0, 1.0)),
    }


#: Named query points for every mesh. Points within the cube's reach of a surface collide by
#: the surface test and are reported as "on surface".
POINTS: Dict[str, Tuple[float, float, float]] = {
    "centre": (0.0, 0.0, 0.0),
    "below centre": (0.0, 0.0, -0.1),
    "above centre": (0.0, 0.0, 0.1),
    "box interior near -x": (-0.12, 0.0, 0.0),
    "box interior near +x": (0.12, 0.0, 0.0),
    "box interior near top": (0.0, 0.0, 0.12),
    "outside, beside": (0.35, 0.0, 0.0),
    "outside, below": (0.0, 0.0, -0.35),
    "outside, above": (0.0, 0.0, 0.35),
    "behind (-1, -1, -1)": (-0.06, -0.06, -0.06),
    "in front (+1, +1, +1)": (0.06, 0.06, 0.06),
}


def _robot_cube() -> RobotMeshData:
    cube = trimesh.creation.box(extents=(CUBE, CUBE, CUBE))
    return RobotMeshData.from_link_meshes(
        link_meshes={
            "l0": Mesh(
                name="l0",
                vertices=np.asarray(cube.vertices),
                faces=np.asarray(cube.faces),
                pose=[0, 0, 0, 1, 0, 0, 0],
            )
        },
        link_name_to_idx_map={"l0": 0},
        num_links=1,
        device_cfg=DEVICE_CFG,
        max_triangles_per_link=0,
    )


def _sphere_collision(scene_cfg: SceneCfg, pts: torch.Tensor) -> list:
    """Whether a sphere of the cube's half edge at each point collides, by the sphere check."""
    collision = SceneCollision.from_config(
        SceneCollisionCfg(device_cfg=DEVICE_CFG, scene_model=scene_cfg)
    )
    spheres = torch.cat([pts, torch.full_like(pts[:, :1], CUBE / 2)], dim=-1).view(1, -1, 1, 4)
    buffer = CollisionBuffer.from_shape(spheres.shape, DEVICE_CFG)
    zero = torch.zeros(1, device=DEVICE_CFG.device)
    distance = collision.get_sphere_distance(
        KinematicsState(robot_spheres=spheres), buffer, zero + 1.0, zero
    )
    return (distance.view(-1) > 0).tolist()


def main() -> int:
    """Print the classification of every query point; return 1 on a mismatch."""
    init_warp(device_cfg=DEVICE_CFG)
    robot = _robot_cube()
    pts = torch.tensor(list(POINTS.values()), dtype=torch.float32, device=DEVICE_CFG.device)
    tf = torch.zeros(len(POINTS), 1, 1, 3, 4, device=DEVICE_CFG.device)
    tf[..., 0, 0] = tf[..., 1, 1] = tf[..., 2, 2] = 1.0
    tf[:, 0, 0, :, 3] = pts
    env = torch.zeros(len(POINTS), dtype=torch.int32, device=DEVICE_CFG.device)

    mismatches = 0
    for name, tri in _meshes().items():
        mesh = Mesh(
            name=name,
            vertices=np.asarray(tri.vertices),
            faces=np.asarray(tri.faces),
            pose=[0, 0, 0, 1, 0, 0, 0],
        )
        scene = SceneData.from_scene_cfg(SceneCfg(mesh=[mesh]), device_cfg=DEVICE_CFG)
        sdf = torch.zeros(len(POINTS), dtype=torch.float32, device=DEVICE_CFG.device)
        wp.launch(
            _sdf_kernel,
            dim=len(POINTS),
            inputs=[scene.meshes.to_warp(), wp.from_torch(pts, dtype=wp.vec3), wp.from_torch(sdf)],
        )
        coll, _ = check_robot_mesh_collision(robot, tf, scene, env, False, check_containment=True)
        sphere = _sphere_collision(SceneCfg(mesh=[mesh]), pts)
        print(
            f"\n{name} (watertight={tri.is_watertight}, winding consistent="
            f"{tri.is_winding_consistent})"
        )
        for (pname, _), d, c, sp in zip(POINTS.items(), sdf.tolist(), coll.tolist(), sphere):
            flags = f"mesh={'hit' if c else '-  '} sphere={'hit' if sp else '-'}"
            if abs(d) <= CUBE:
                print(f"  {pname:24s} sdf={d:+.3f}  on surface  {flags}")
                continue
            inside = d < 0
            mismatch = inside != bool(c) or inside != sp
            mismatches += mismatch
            print(
                f"  {pname:24s} sdf={d:+.3f}  {'INSIDE ' if inside else 'outside'}     {flags}"
                f"{'  <-- MISMATCH' if mismatch else ''}"
            )
    print(f"\ncontainment / SDF sign mismatches: {mismatches}")
    return int(mismatches > 0)


if __name__ == "__main__":
    raise SystemExit(main())
