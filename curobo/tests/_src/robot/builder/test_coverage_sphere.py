# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Tests for SphereCoverage: measuring and covering where link meshes leave their spheres."""

# Third Party
import numpy as np
import pytest
import torch
import trimesh

# CuRobo
from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery
from curobo._src.robot.builder.coverage_sphere import SphereCoverage
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml

FRANKA = "franka.yml"
NUM_CONFIGS = 200
NUM_SURFACE_SAMPLES = 2000


def _franka_kinematics() -> dict:
    return load_yaml(join_path(get_robot_configs_path(), FRANKA))["robot_cfg"]["kinematics"]


def _coverage() -> SphereCoverage:
    return SphereCoverage(FRANKA, num_surface_samples=NUM_SURFACE_SAMPLES)


@pytest.fixture(scope="module")
def baseline():
    """Report of the shipped Franka configuration."""
    return _coverage().analyze(num_configs=NUM_CONFIGS)


def test_analyze_reports_every_mesh_link_and_the_ignored_pairs(baseline):
    spheres = _franka_kinematics()["collision_spheres"]
    meshes = {link.link_name for link in baseline.links}
    assert meshes == {f"panda_link{i}" for i in range(8)} | {
        "panda_hand",
        "panda_leftfinger",
        "panda_rightfinger",
    }
    expected = sum(1 for link in spheres.values() for s in link if s["radius"] > 0.0)
    assert sum(baseline.num_spheres.values()) == expected
    # The few large spheres of the shipped configuration leave parts of every link uncovered.
    assert all(link.max_protrusion > baseline.tolerance for link in baseline.links)
    # panda_link5 - panda_link7 is ignored by the sphere check while the meshes collide.
    assert baseline.missed_config_ratio > 0.0
    assert {"panda_link5", "panda_link7"} in [set(pair) for pair in baseline.suggested_checks]
    for pair in baseline.pairs:
        assert 0.0 <= pair.missed_ratio <= pair.mesh_collision_ratio


def test_cover_meshes_reduces_protrusion_with_spheres_inside_the_meshes(baseline):
    coverage = _coverage()
    original = {name: len(s) for name, s in coverage.collision_spheres.items()}
    tolerance = 0.01
    added = coverage.cover_meshes(tolerance=tolerance, max_new_spheres_per_link=4)
    after = coverage.analyze(tolerance=tolerance, num_configs=NUM_CONFIGS)

    assert added and all(1 <= n <= 4 for n in added.values())
    assert "panda_link0" not in added
    before = {link.link_name: link for link in baseline.links}
    moving = [link for link in after.links if link.link_name != "panda_link0"]
    for link in moving:
        assert link.p95_protrusion <= before[link.link_name].p95_protrusion + 1.0e-6
    assert sum(link.p95_protrusion for link in moving) < 0.6 * sum(
        before[link.link_name].p95_protrusion for link in moving
    )

    # Every new sphere sticks out of its link mesh by at most the tolerance.
    robot = load_yaml(join_path(get_robot_configs_path(), FRANKA))
    robot["robot_cfg"]["kinematics"]["load_collision_meshes"] = True
    kinematics = Kinematics(KinematicsCfg.from_robot_yaml_file(robot, device_cfg=DeviceCfg()))
    meshes = kinematics.get_robot_collision_link_meshes()
    device = DeviceCfg().device
    for name, count in added.items():
        mesh = meshes[name]
        query = WarpMeshQuery(trimesh.Trimesh(mesh.vertices, mesh.faces), device)
        new = coverage.collision_spheres[name][original[name] :]
        assert len(new) == count
        centers = torch.tensor([s["center"] for s in new], dtype=torch.float32, device=device)
        radii = torch.tensor([s["radius"] for s in new], dtype=torch.float32, device=device)
        sdf, _ = query.query_sdf(centers)
        assert torch.all(-sdf >= radii - tolerance - 1.0e-4)


def test_cover_meshes_skips_the_base_unless_named_and_checks_its_arguments():
    coverage = _coverage()
    with pytest.raises(ValueError):
        coverage.cover_meshes(tolerance=0.0)
    with pytest.raises(ValueError):
        coverage.cover_meshes(link_names=["attached_object"])
    # panda_link0 does not move, so only an explicit request covers it.
    assert coverage.cover_meshes(link_names=["panda_link0"], max_new_spheres_per_link=1) == {
        "panda_link0": 1
    }


def test_remove_self_collision_ignore_and_save_keep_the_rest_of_the_configuration(tmp_path):
    coverage = _coverage()
    coverage.cover_meshes(max_new_spheres_per_link=2)
    coverage.remove_self_collision_ignore([("panda_link7", "panda_link5")])
    report = coverage.analyze(num_configs=NUM_CONFIGS)
    pair = next(p for p in report.pairs if {p.link_a, p.link_b} == {"panda_link5", "panda_link7"})
    assert not pair.sphere_check_ignores

    path = str(tmp_path / "franka_covered.yml")
    coverage.save(path)
    saved = load_yaml(path)["robot_cfg"]["kinematics"]
    original = _franka_kinematics()
    assert saved["collision_spheres"] == coverage.collision_spheres
    assert "panda_link7" not in saved["self_collision_ignore"]["panda_link5"]
    for link, ignored in original["self_collision_ignore"].items():
        if link != "panda_link5":
            assert saved["self_collision_ignore"][link] == ignored
    for key, value in original.items():
        if key not in ("collision_spheres", "self_collision_ignore"):
            assert saved[key] == value, key

    reloaded = SphereCoverage(path, num_surface_samples=NUM_SURFACE_SAMPLES)
    assert reloaded.analyze(num_configs=NUM_CONFIGS).num_spheres == report.num_spheres


def test_cover_meshes_is_reproducible():
    first, second = _coverage(), _coverage()
    first.cover_meshes(max_new_spheres_per_link=2)
    second.cover_meshes(max_new_spheres_per_link=2)
    assert first.collision_spheres == second.collision_spheres


def test_report_markdown_compares_with_a_baseline(baseline):
    text = baseline.to_markdown()
    assert "## Links" in text and "## Self-collision pairs" in text and "->" not in text
    compared = baseline.to_markdown(baseline=baseline)
    assert "->" in compared
    assert np.isclose(baseline.tolerance, 0.01)
