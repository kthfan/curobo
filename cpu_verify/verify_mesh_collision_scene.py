# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Check the scene of ``test_mesh_collision_end_to_end.py`` on the CPU.

Branch-local verification tool. Do not merge into ``dev/main``.

The end-to-end test needs a scene where, at the goal pose, the enabled spheres are clear of
the box (by more than the optimizer's 1 cm collision activation distance) while the gripper
meshes collide. This places the Franka links at the goal pose with URDF forward kinematics
(yourdfpy), then checks the spheres analytically and the meshes with the Warp kernel.

Usage (from the repository root)::

    PYTHONPATH=. python cpu_verify/verify_mesh_collision_scene.py
"""

# Patch DeviceCfg to the CPU before any other CuRobo import.
import cpu_verify.cpu_device_plugin  # noqa: F401, I001

# Third Party
import numpy as np
import torch
import trimesh
import yaml
import yourdfpy

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_collision import check_robot_mesh_collision
from curobo._src.geom.data.data_robot_mesh import RobotMeshData
from curobo._src.geom.data.data_scene import SceneData
from curobo._src.geom.types import Mesh
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.warp import init_warp
from curobo.tests._src.motion.test_mesh_collision_end_to_end import (
    GOAL_POSE,
    GRIPPER_LINKS,
    palm_box,
)

ROBOT_ROOT = "curobo/content/assets/robot/franka_description/"
ROBOT_YAML = "curobo/content/configs/robot/franka.yml"
ACTIVATION_DISTANCE = 0.01
EXPECTED_MESH_COLLISION = {
    "panda_hand": True,
    "panda_link7": False,
    "panda_link6": False,
    "panda_link5": False,
}


def link_poses_at_goal() -> tuple:
    """Place the robot with ``panda_hand`` at ``GOAL_POSE``.

    Returns:
        ``(poses, kinematics_yaml, urdf)``: world 4x4 transform per URDF link, the
        ``kinematics`` section of the robot YAML, and the loaded URDF.
    """
    kin = yaml.safe_load(open(ROBOT_YAML))["robot_cfg"]["kinematics"]
    urdf = yourdfpy.URDF.load(
        ROBOT_ROOT + "franka_panda.urdf", load_meshes=False, build_scene_graph=True
    )
    urdf.update_cfg(
        dict(zip(kin["cspace"]["joint_names"], kin["cspace"]["default_joint_position"]))
    )
    base = {name: urdf.get_transform(name, "panda_link0") for name in urdf.link_map}
    x, y, z, qw, qx, qy, qz = GOAL_POSE
    goal = trimesh.transformations.quaternion_matrix([qw, qx, qy, qz])
    goal[:3, 3] = [x, y, z]
    hand_inv = np.linalg.inv(base["panda_hand"])
    return {name: goal @ hand_inv @ tf for name, tf in base.items()}, kin, urdf


def main() -> int:
    """Print sphere clearances and mesh collisions; return non-zero if the scene is invalid."""
    device_cfg = DeviceCfg()
    init_warp(device_cfg=device_cfg)
    poses, kin, urdf = link_poses_at_goal()

    box = palm_box().cuboid[0]
    center = np.array(box.pose[:3])
    half = np.array(box.dims) / 2.0

    clearance, closest = np.inf, None
    for link, spheres in kin["collision_spheres"].items():
        if link in GRIPPER_LINKS or link not in poses:
            continue
        for sphere in spheres:
            c = (poses[link] @ np.r_[sphere["center"], 1.0])[:3]
            d = np.linalg.norm(np.maximum(np.abs(c - center) - half, 0.0)) - sphere["radius"]
            if d < clearance:
                clearance, closest = d, link
    print(f"closest enabled sphere: {closest}, clearance {clearance:.4f} m")
    ok = clearance > ACTIVATION_DISTANCE

    scene = SceneData.from_scene_cfg(palm_box(), device_cfg=device_cfg)
    for link, expected in EXPECTED_MESH_COLLISION.items():
        mesh = urdf.link_map[link].collisions[0].geometry.mesh.filename
        tri = trimesh.load(ROBOT_ROOT + mesh, force="mesh")
        robot_mesh = RobotMeshData.from_link_meshes(
            link_meshes={
                link: Mesh(
                    name=link, vertices=tri.vertices, faces=tri.faces, pose=[0, 0, 0, 1, 0, 0, 0]
                )
            },
            link_name_to_idx_map={link: 0},
            num_links=1,
            device_cfg=device_cfg,
            max_triangles_per_link=0,
        )
        tf = torch.tensor(poses[link][:3], dtype=torch.float32).view(1, 1, 1, 3, 4)
        coll, _ = check_robot_mesh_collision(
            robot_mesh, tf, scene, torch.zeros(1, dtype=torch.int32), use_multi_env=False
        )
        collides = bool(coll.item())
        print(f"{link}: mesh collides = {collides} (expected {expected})")
        ok = ok and collides == expected
    print("scene OK" if ok else "scene INVALID")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
