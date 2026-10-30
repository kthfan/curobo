# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Measure Franka self-collision with exact link meshes against the sphere self-collision check.

Branch-local verification tool. Do not merge into ``dev/main``.

For every pair of collision links that ``self_collision_ignore`` does not exclude (or, with
``--ignored``, every pair it excludes), this samples
random joint configurations (URDF forward kinematics with yourdfpy), tests the decimated link
meshes against each other with the Warp robot-mesh kernel (link B as a mesh obstacle), and
compares with the sphere check (radius plus ``self_collision_buffer``). It reports pairs that
touch at the default configuration and how often the meshes collide while the spheres do not.

Usage (from the repository root)::

    PYTHONPATH=. python cpu_verify/verify_franka_self_collision.py --configs 2000
"""

# Patch DeviceCfg to the CPU before any other CuRobo import.
import cpu_verify.cpu_device_plugin  # noqa: F401, I001

# Standard Library
import argparse
import itertools
from typing import Dict, List

# Third Party
import numpy as np
import torch
import trimesh
import yaml
import yourdfpy

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_collision import check_robot_mesh_collision
from curobo._src.geom.data.data_robot_mesh import RobotMeshData, decimate_mesh
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh, SceneCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp

ROBOT_ROOT = "curobo/content/assets/robot/franka_description/"
ROBOT_YAML = "curobo/content/configs/robot/franka.yml"


def link_meshes(urdf: yourdfpy.URDF, links: List[str], budget: int) -> Dict[str, trimesh.Trimesh]:
    """Collision mesh of each link in its link frame, decimated to the triangle budget."""
    meshes = {}
    for name in links:
        collision = urdf.link_map[name].collisions[0]
        tri = trimesh.load(ROBOT_ROOT + collision.geometry.mesh.filename, force="mesh")
        if collision.geometry.mesh.scale is not None:
            tri.apply_scale(collision.geometry.mesh.scale)
        if collision.origin is not None:
            tri.apply_transform(collision.origin)
        meshes[name] = decimate_mesh(tri, budget, name)
    return meshes


def link_poses(urdf: yourdfpy.URDF, links: List[str], joint_names: List[str], q: np.ndarray):
    """World transform ``[n, 4, 4]`` of each link for every configuration."""
    poses = {name: np.zeros((len(q), 4, 4)) for name in links}
    for i, qi in enumerate(q):
        urdf.update_cfg(dict(zip(joint_names, qi)))
        for name in links:
            poses[name][i] = urdf.get_transform(name, "panda_link0")
    return poses


def sphere_collision(kin: dict, a: str, b: str, pa: np.ndarray, pb: np.ndarray) -> np.ndarray:
    """Whether any sphere of link ``a`` overlaps a sphere of link ``b``, with self buffers."""
    buffer = kin.get("self_collision_buffer", {})

    def world(link, poses):
        s = kin["collision_spheres"][link]
        c = np.array([x["center"] for x in s])
        r = np.array([x["radius"] for x in s]) + buffer.get(link, 0.0)
        return np.einsum("nij,kj->nki", poses[:, :3, :3], c) + poses[:, None, :3, 3], r

    ca, ra = world(a, pa)
    cb, rb = world(b, pb)
    d = np.linalg.norm(ca[:, :, None] - cb[:, None, :], axis=-1)
    return (d < ra[None, :, None] + rb[None, None, :]).any(axis=(1, 2))


def main() -> int:
    """Print per-pair mesh and sphere self-collision rates."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--configs", type=int, default=2000)
    parser.add_argument("--budget", type=int, default=256)
    parser.add_argument("--margin", type=float, default=0.0)
    parser.add_argument(
        "--ignored", action="store_true", help="check the pairs self_collision_ignore excludes"
    )
    args = parser.parse_args()

    device_cfg = DeviceCfg()
    init_warp(device_cfg=device_cfg)
    kin = yaml.safe_load(open(ROBOT_YAML))["robot_cfg"]["kinematics"]
    urdf = yourdfpy.URDF.load(ROBOT_ROOT + "franka_panda.urdf", load_meshes=False,
                              build_scene_graph=True)
    joint_names = kin["cspace"]["joint_names"]
    links = [n for n in kin["collision_link_names"] if n in urdf.link_map
             and urdf.link_map[n].collisions]
    ignore = {frozenset((a, b)) for a, bs in kin["self_collision_ignore"].items() for b in bs}
    pairs = [
        (a, b)
        for a, b in itertools.combinations(links, 2)
        if (frozenset((a, b)) in ignore) == args.ignored
    ]

    rng = np.random.default_rng(0)
    lower = np.array([urdf.joint_map[j].limit.lower for j in joint_names])
    upper = np.array([urdf.joint_map[j].limit.upper for j in joint_names])
    q = np.vstack([kin["cspace"]["default_joint_position"],
                   lower + (upper - lower) * rng.random((args.configs, len(joint_names)))])
    poses = link_poses(urdf, links, joint_names, q)
    meshes = link_meshes(urdf, links, args.budget)

    n = len(q)
    env = torch.zeros(n, dtype=torch.int32)
    any_mesh = np.zeros(n, bool)
    any_sphere = np.zeros(n, bool)
    print(f"{len(pairs)} checked pairs of {len(links)} links, {n - 1} random configs + default")
    print(f"{'pair':40s} {'default':>8s} {'mesh %':>7s} {'sphere %':>8s} {'mesh only %':>11s}")
    for a, b in pairs:
        ma, mb = meshes[a], meshes[b]
        robot = RobotMeshData.from_link_meshes(
            link_meshes={a: Mesh(name=a, vertices=ma.vertices, faces=ma.faces,
                                 pose=[0, 0, 0, 1, 0, 0, 0])},
            link_name_to_idx_map={a: 0}, num_links=1, device_cfg=device_cfg,
            max_triangles_per_link=0,
        )
        scene = SceneData.from_scene_cfg(
            SceneCfg(mesh=[Mesh(name=b, vertices=mb.vertices, faces=mb.faces,
                                pose=[0, 0, 0, 1, 0, 0, 0])]),
            device_cfg=device_cfg,
        )
        a_in_b = np.linalg.inv(poses[b]) @ poses[a]
        tf = torch.tensor(a_in_b[:, :3], dtype=torch.float32).view(n, 1, 1, 3, 4)
        mesh_hit, _ = check_robot_mesh_collision(robot, tf, scene, env, False,
                                                 margin=args.margin, check_containment=False)
        mesh_hit = mesh_hit.numpy().astype(bool)
        sphere_hit = sphere_collision(kin, a, b, poses[a], poses[b])
        any_mesh |= mesh_hit
        any_sphere |= sphere_hit
        r = mesh_hit[1:]
        s = sphere_hit[1:]
        print(f"{a + ' - ' + b:40s} {'HIT' if mesh_hit[0] else '-':>8s} {100 * r.mean():7.2f} "
              f"{100 * s.mean():8.2f} {100 * (r & ~s).mean():11.2f}")
    r, s = any_mesh[1:], any_sphere[1:]
    print(f"\nany pair: mesh {100 * r.mean():.2f} %, sphere {100 * s.mean():.2f} %, "
          f"mesh only {100 * (r & ~s).mean():.2f} %, sphere only {100 * (s & ~r).mean():.2f} %")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
