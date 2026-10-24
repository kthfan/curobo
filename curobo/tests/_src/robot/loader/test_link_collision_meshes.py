# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for loading exact triangle collision meshes for robot links."""

# Third Party
import numpy as np
import pytest

from curobo._src.geom.types import Mesh
from curobo._src.robot.kinematics.kinematics import Kinematics, KinematicsCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


def _robot_data():
    return load_yaml(join_path(get_robot_configs_path(), "franka.yml"))


@pytest.fixture(scope="module")
def cfg_with_collision_meshes():
    robot_data = _robot_data()
    robot_data["robot_cfg"]["kinematics"]["load_collision_meshes"] = True
    return KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])


@pytest.fixture(scope="module")
def cfg_without_collision_meshes():
    return KinematicsCfg.from_robot_yaml_file(_robot_data(), ["panda_hand"])


def test_collision_meshes_not_loaded_by_default(cfg_without_collision_meshes):
    assert cfg_without_collision_meshes.kinematics_config.link_collision_meshes is None


def test_get_robot_collision_link_meshes_raises_when_not_loaded(cfg_without_collision_meshes):
    kin = Kinematics(cfg_without_collision_meshes)
    with pytest.raises(Exception):
        kin.get_robot_collision_link_meshes()


def test_collision_meshes_loaded_for_collision_links(cfg_with_collision_meshes):
    meshes = cfg_with_collision_meshes.kinematics_config.link_collision_meshes
    assert meshes is not None
    assert len(meshes) > 0
    for name, mesh in meshes.items():
        assert isinstance(mesh, Mesh), name
        assert len(mesh.faces) > 0, name
        assert len(mesh.vertices) > 0, name


def test_collision_meshes_are_link_local_with_identity_pose(cfg_with_collision_meshes):
    """The URDF <origin> and scale must be baked into vertices, leaving an identity pose."""
    meshes = cfg_with_collision_meshes.kinematics_config.link_collision_meshes
    for name, mesh in meshes.items():
        np.testing.assert_allclose(mesh.pose, [0, 0, 0, 1, 0, 0, 0], atol=0, err_msg=name)
        # Mesh.__post_init__ applies scale to vertices and clears it; None == fully baked.
        assert mesh.scale is None, name


def test_collision_meshes_are_finite_and_bounded(cfg_with_collision_meshes):
    """Link-local geometry must be finite and near the link origin, not in world coordinates."""
    meshes = cfg_with_collision_meshes.kinematics_config.link_collision_meshes
    for name, mesh in meshes.items():
        verts = np.asarray(mesh.vertices, dtype=np.float64)
        assert np.isfinite(verts).all(), name
        assert np.abs(verts).max() < 5.0, f"{name} vertices look world-framed, not link-local"


def test_collision_mesh_differs_from_visual_mesh(cfg_with_collision_meshes):
    """Franka ships distinct (decimated) collision meshes; confirm we are not reading visuals."""
    kin = Kinematics(cfg_with_collision_meshes)
    collision_meshes = kin.get_robot_collision_link_meshes()

    differing = 0
    for name, collision_mesh in collision_meshes.items():
        visual_mesh = kin.get_link_mesh(name, use_collision_mesh=False)
        if visual_mesh is None:
            continue
        # get_link_mesh may return a lazy file-backed mesh; materialize the triangles.
        visual_faces = visual_mesh.get_trimesh_mesh(transform_with_pose=True).faces
        if len(visual_faces) != len(collision_mesh.faces):
            differing += 1
    assert differing > 0, "collision meshes are identical to visual meshes for every link"


def test_virtual_attachment_links_are_excluded():
    """Synthetic links such as "attached_object" are not in the URDF and must be skipped."""
    robot_data = _robot_data()
    robot_data["robot_cfg"]["kinematics"]["load_collision_meshes"] = True
    robot_data["robot_cfg"]["kinematics"]["extra_collision_spheres"] = {"attached_object": 100}
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])

    meshes = cfg.kinematics_config.link_collision_meshes
    assert "attached_object" not in meshes
    assert len(meshes) > 0


def test_load_collision_meshes_implies_load_meshes():
    from curobo._src.robot.loader.kinematics_loader_cfg import KinematicsLoaderCfg

    robot_data = _robot_data()["robot_cfg"]["kinematics"]
    robot_data["load_collision_meshes"] = True
    robot_data["load_meshes"] = False
    cfg = KinematicsLoaderCfg(**robot_data)
    assert cfg.load_meshes is True


def test_ignored_links_are_not_loaded(cfg_with_collision_meshes):
    """Links named in mesh_collision_ignored_link_names must carry no collision geometry."""
    baseline = cfg_with_collision_meshes.kinematics_config.link_collision_meshes
    ignored = sorted(baseline.keys())[0]

    robot_data = _robot_data()
    robot_data["robot_cfg"]["kinematics"]["load_collision_meshes"] = True
    robot_data["robot_cfg"]["kinematics"]["mesh_collision_ignored_link_names"] = [ignored]
    cfg = KinematicsCfg.from_robot_yaml_file(robot_data, ["panda_hand"])

    meshes = cfg.kinematics_config.link_collision_meshes
    assert ignored not in meshes
    assert set(meshes.keys()) == set(baseline.keys()) - {ignored}
    # Sphere-based collision checking must be untouched.
    assert ignored in cfg.kinematics_config.link_name_to_idx_map
