# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Compare a robot's collision spheres with its link meshes, and cover what they miss.

The optimizers only see the collision spheres, while the exact mesh check tests the link
meshes. Where a mesh sticks out of its link's spheres, nothing steers the robot away from a
collision that only the mesh check reports. :class:`SphereCoverage` measures those gaps on a
robot configuration file, adds spheres that cover them, and writes the configuration back.
"""

# Standard Library
import copy
import os
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Union

# Third Party
import numpy as np
import torch
import trimesh
import yaml

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_collision import check_robot_mesh_self_collision
from curobo._src.geom.sphere_fit.wp_mesh_query import WarpMeshQuery
from curobo._src.robot.kinematics.kinematics import Kinematics
from curobo._src.robot.kinematics.kinematics_cfg import KinematicsCfg
from curobo._src.robot.kinematics.kinematics_state import KinematicsState
from curobo._src.state.state_joint import JointState
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.util.logging import log_and_raise, log_info, log_warn
from curobo._src.util.sampling.sample_buffer import SampleBuffer
from curobo._src.util_file import get_robot_configs_path, join_path, load_yaml


class _ConfigDumper(yaml.SafeDumper):
    """YAML dumper that keeps key order and writes lists of scalars inline."""


def _represent_list(dumper: yaml.SafeDumper, data: list) -> yaml.Node:
    inline = all(not isinstance(v, (dict, list, tuple)) for v in data)
    return dumper.represent_sequence("tag:yaml.org,2002:seq", data, flow_style=inline)


_ConfigDumper.add_representer(list, _represent_list)
_ConfigDumper.add_representer(tuple, _represent_list)


@dataclass
class LinkSphereCoverage:
    """How far one link's mesh sticks out of the union of its collision spheres."""

    #: Link name.
    link_name: str
    #: Number of collision spheres of the link.
    num_spheres: int
    #: Largest distance from a mesh surface point to the nearest sphere, in meters.
    max_protrusion: float
    #: 95th percentile of that distance over the mesh surface, in meters.
    p95_protrusion: float
    #: Fraction of the mesh surface farther than the tolerance from every sphere.
    uncovered_ratio: float


@dataclass
class PairSphereCoverage:
    """Mesh and sphere self-collision of one link pair over the sampled configurations."""

    #: First link of the pair.
    link_a: str
    #: Second link of the pair.
    link_b: str
    #: True when ``self_collision_ignore`` excludes the pair from the sphere check.
    sphere_check_ignores: bool
    #: Fraction of configurations where the two link meshes collide.
    mesh_collision_ratio: float
    #: Fraction of configurations where the two links' spheres overlap, self-collision
    #: buffers included, whether or not the sphere check ignores the pair.
    sphere_collision_ratio: float
    #: Fraction of configurations where the meshes collide and the spheres do not.
    missed_ratio: float
    #: Fraction of configurations where the spheres overlap and the meshes do not.
    false_alarm_ratio: float
    #: True when the two links' spheres overlap at the default joint configuration.
    spheres_overlap_at_default: bool


@dataclass
class SphereCoverageReport:
    """Result of :meth:`SphereCoverage.analyze`."""

    #: Distance below which a mesh point counts as covered, in meters.
    tolerance: float
    #: Number of sampled joint configurations behind the pair statistics.
    num_configs: int
    #: Per-link surface coverage, for links that have a mesh.
    links: List[LinkSphereCoverage]
    #: Per-pair self-collision statistics, for the pairs the mesh self-collision check tests
    #: whose meshes collided at least once, or whose spheres did while the sphere check
    #: tests the pair.
    pairs: List[PairSphereCoverage]
    #: Fraction of configurations where some link meshes collide while the sphere
    #: self-collision check, with its ignore list, reports nothing: the solver accepts them
    #: and only the mesh check rejects them.
    missed_config_ratio: float
    #: Fraction of configurations where the sphere self-collision check reports a collision
    #: while no pair of link meshes collides: spheres that are larger than the meshes make the
    #: solver avoid configurations that are free.
    false_alarm_config_ratio: float
    #: Pairs checked by the sphere check whose spheres overlap at the default joint
    #: configuration, as ``(link_a, link_b)``.
    default_config_collisions: List[Tuple[str, str]] = field(default_factory=list)
    #: Number of collision spheres per link.
    num_spheres: Dict[str, int] = field(default_factory=dict)

    @property
    def suggested_checks(self) -> List[Tuple[str, str]]:
        """Ignored pairs worth handing to the sphere check, as ``(link_a, link_b)``.

        These are pairs that ``self_collision_ignore`` excludes from the sphere check although
        their meshes collide in some sampled configurations, and whose spheres do not overlap
        at the default joint configuration. While ignored, only the mesh check sees their
        collisions, so the optimizer does not keep the links apart. Pass them to
        :meth:`SphereCoverage.remove_self_collision_ignore`.
        """
        return [
            (p.link_a, p.link_b)
            for p in self.pairs
            if p.sphere_check_ignores
            and p.mesh_collision_ratio > 0.0
            and not p.spheres_overlap_at_default
        ]

    def to_markdown(self, baseline: Optional["SphereCoverageReport"] = None) -> str:
        """Format the report as Markdown, next to ``baseline`` (e.g. before covering) if given.

        Args:
            baseline: Earlier report of the same robot; its values are shown as ``before ->``.

        Returns:
            Markdown text with a summary, a link table and a pair table.
        """

        def mm(value: float) -> str:
            return f"{1000.0 * value:.1f}"

        def pct(value: float) -> str:
            return f"{100.0 * value:.1f}"

        def pair_of(old: Optional[object], new: object, attr: str, fmt) -> str:
            if old is None:
                return fmt(getattr(new, attr))
            return f"{fmt(getattr(old, attr))} -> {fmt(getattr(new, attr))}"

        old_links = {lc.link_name: lc for lc in baseline.links} if baseline else {}
        old_pairs = {(p.link_a, p.link_b): p for p in baseline.pairs} if baseline else {}
        lines = [
            "## Summary",
            "",
            f"- Tolerance: {mm(self.tolerance)} mm. Sampled configurations: {self.num_configs}.",
            "- Configurations the sphere check accepts while the link meshes collide: "
            + (
                f"{pct(baseline.missed_config_ratio)} % -> {pct(self.missed_config_ratio)} %."
                if baseline
                else f"{pct(self.missed_config_ratio)} %."
            ),
            "- Configurations the sphere check rejects while the link meshes are free: "
            + (
                f"{pct(baseline.false_alarm_config_ratio)} % -> "
                f"{pct(self.false_alarm_config_ratio)} %."
                if baseline
                else f"{pct(self.false_alarm_config_ratio)} %."
            ),
            "- Collision spheres: "
            + (
                f"{sum(baseline.num_spheres.values())} -> {sum(self.num_spheres.values())}."
                if baseline
                else f"{sum(self.num_spheres.values())}."
            ),
        ]
        if self.default_config_collisions:
            names = ", ".join(f"{a} - {b}" for a, b in self.default_config_collisions)
            lines.append(
                f"- **Warning:** spheres overlap at the default joint configuration: {names}."
            )
        lines += [
            "",
            "## Links",
            "",
            "Distance from the link mesh to its own spheres (scene buffer included).",
            "",
            "| link | spheres | max out (mm) | p95 out (mm) | surface out (%) |",
            "|---|---|---|---|---|",
        ]
        for lc in self.links:
            old = old_links.get(lc.link_name)
            lines.append(
                f"| {lc.link_name} | {pair_of(old, lc, 'num_spheres', str)} "
                f"| {pair_of(old, lc, 'max_protrusion', mm)} "
                f"| {pair_of(old, lc, 'p95_protrusion', mm)} "
                f"| {pair_of(old, lc, 'uncovered_ratio', pct)} |"
            )
        lines += [
            "",
            "## Self-collision pairs",
            "",
            "Pairs whose meshes, or whose spheres where the sphere check tests the pair, "
            "collided in a sampled configuration. *Missed*: the meshes "
            "collide and the spheres do not. *False alarm*: the spheres overlap and the meshes "
            "do not.",
            "",
            "| pair | sphere check | mesh (%) | spheres (%) | missed (%) | false alarm (%) |",
            "|---|---|---|---|---|---|",
        ]
        for p in self.pairs:
            old = old_pairs.get((p.link_a, p.link_b))
            status = "ignored" if p.sphere_check_ignores else "checked"
            lines.append(
                f"| {p.link_a} - {p.link_b} | {status} "
                f"| {pct(p.mesh_collision_ratio)} "
                f"| {pair_of(old, p, 'sphere_collision_ratio', pct)} "
                f"| {pair_of(old, p, 'missed_ratio', pct)} "
                f"| {pair_of(old, p, 'false_alarm_ratio', pct)} |"
            )
        suggested = self.suggested_checks
        if suggested:
            lines += [
                "",
                "Pairs the sphere check ignores (`self_collision_ignore`) although their meshes "
                "collide are left to the mesh check, so the optimizer does not keep the links "
                "apart. Their spheres do not overlap at the default joint configuration, so "
                "they can be removed from the ignore list: "
                + ", ".join(f"{a} - {b}" for a, b in suggested)
                + ".",
            ]
        return "\n".join(lines) + "\n"


class SphereCoverage:
    """Measure and cover where a robot's link meshes stick out of its collision spheres.

    Load a robot configuration, compare it with :meth:`analyze`, add spheres with
    :meth:`cover_meshes`, and write the result with :meth:`save`:

    .. code-block:: python

        coverage = SphereCoverage("franka.yml")
        before = coverage.analyze()
        coverage.cover_meshes()
        after = coverage.analyze()
        print(after.to_markdown(baseline=before))
        coverage.save("franka_covered.yml")

    The meshes are the ones the mesh collision check uses: each link's ``<collision>``
    geometry, or its visual geometry when it has none. Existing spheres are kept.
    """

    def __init__(
        self,
        robot_config: Union[str, Dict],
        device_cfg: Optional[DeviceCfg] = None,
        num_surface_samples: int = 10000,
        seed: int = 0,
    ):
        """Load the robot, its link meshes and its collision spheres.

        Args:
            robot_config: Robot configuration file, as a path, a file name under the robot
                configuration folder, or the loaded dictionary.
            device_cfg: Device configuration. Defaults to CUDA:0.
            num_surface_samples: Points sampled on each link mesh to measure coverage.
            seed: Seed of the surface samples and of the candidates of :meth:`cover_meshes`.
        """
        self.device_cfg = device_cfg or DeviceCfg()
        if isinstance(robot_config, str) and not os.path.exists(robot_config):
            robot_config = join_path(get_robot_configs_path(), robot_config)
        self._config = copy.deepcopy(load_yaml(robot_config))
        kinematics = self._kinematics_data(self._config)
        spheres = kinematics["collision_spheres"]
        if isinstance(spheres, str):
            spheres = load_yaml(join_path(get_robot_configs_path(), spheres))
            spheres = spheres.get("collision_spheres", spheres)
        self._spheres: Dict[str, List[Dict]] = copy.deepcopy(spheres)
        self._scene_buffer = kinematics.get("collision_sphere_buffer", 0.0) or 0.0
        self._self_buffer: Dict[str, float] = dict(kinematics.get("self_collision_buffer") or {})
        self._self_ignore: Dict[str, List[str]] = copy.deepcopy(
            dict(kinematics.get("self_collision_ignore") or {})
        )

        loaded = copy.deepcopy(self._config)
        self._kinematics_data(loaded)["load_collision_meshes"] = True
        self._kinematics = Kinematics(
            KinematicsCfg.from_robot_yaml_file(loaded, device_cfg=self.device_cfg),
            compute_link_transforms=True,
        )
        self._mesh_data = self._kinematics.config.kinematics_config.get_robot_mesh_data()
        rng = np.random.default_rng(seed)
        self._generator = torch.Generator(device=self.device_cfg.device).manual_seed(seed)
        self._surfaces: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}
        self._queries: Dict[str, WarpMeshQuery] = {}
        for name, mesh in self._kinematics.get_robot_collision_link_meshes().items():
            if name not in self._spheres:
                continue
            tri = trimesh.Trimesh(np.asarray(mesh.vertices), np.asarray(mesh.faces))
            points, faces = trimesh.sample.sample_surface(
                tri, num_surface_samples, seed=int(rng.integers(1 << 31))
            )
            points = np.concatenate([points, tri.vertices])
            normals = np.concatenate([tri.face_normals[faces], tri.vertex_normals])
            self._surfaces[name] = (self._tensor(points), self._tensor(normals))
            self._queries[name] = WarpMeshQuery(tri, self.device_cfg.device)

    @property
    def collision_spheres(self) -> Dict[str, List[Dict]]:
        """Current collision spheres per link, as in the configuration file."""
        return self._spheres

    def analyze(
        self, tolerance: float = 0.01, num_configs: int = 2000, seed: int = 345
    ) -> SphereCoverageReport:
        """Measure how well the current spheres cover the link meshes.

        Each link's mesh surface is compared with its own spheres, enlarged by the scene
        ``collision_sphere_buffer``. Self-collision is sampled: the default joint
        configuration and ``num_configs`` Halton samples within the joint limits are tested
        with the mesh self-collision check, pair by pair, and with sphere overlap, enlarged by
        ``self_collision_buffer``.

        Args:
            tolerance: Distance below which a mesh point counts as covered, in meters.
            num_configs: Number of sampled joint configurations.
            seed: Seed of the joint samples.

        Returns:
            The coverage report.
        """
        links = []
        for name, (points, _) in self._surfaces.items():
            centers, radii = self._link_spheres(name, self._scene_buffer_of(name))
            gap = self._gap(points, centers, radii).clamp(min=0.0)
            links.append(
                LinkSphereCoverage(
                    link_name=name,
                    num_spheres=int(centers.shape[0]),
                    max_protrusion=float(gap.max()),
                    p95_protrusion=float(torch.quantile(gap, 0.95)),
                    uncovered_ratio=float((gap > tolerance).float().mean()),
                )
            )
        pairs, missed, false_alarm, at_default = self._sample_self_collision(num_configs, seed)
        return SphereCoverageReport(
            tolerance=tolerance,
            num_configs=num_configs,
            links=links,
            pairs=pairs,
            missed_config_ratio=missed,
            false_alarm_config_ratio=false_alarm,
            default_config_collisions=at_default,
            num_spheres={name: len(self._link_spheres(name, 0.0)[1]) for name in self._spheres},
        )

    def cover_meshes(
        self,
        tolerance: float = 0.01,
        max_new_spheres_per_link: int = 16,
        min_radius: float = 0.005,
        num_candidates: int = 64,
        link_names: Optional[List[str]] = None,
    ) -> Dict[str, int]:
        """Add spheres where a link mesh sticks out of its spheres by more than ``tolerance``.

        Spheres are added one at a time, greedily. Each candidate touches an uncovered surface
        point from inside the mesh, with the largest radius at which it sticks out of the
        mesh by at most ``tolerance``; the candidate that covers the most uncovered surface,
        weighted by how far it was uncovered, is kept. Existing spheres are not changed.

        Links that no joint moves, such as the base link, are skipped unless named in
        ``link_names``: the optimizer cannot steer them, and their spheres are often kept
        small on purpose so they stay clear of the surface the robot is mounted on.

        Args:
            tolerance: Distance below which a mesh point counts as covered, and how far a new
                sphere may stick out of the mesh, in meters.
            max_new_spheres_per_link: Most spheres added to one link.
            min_radius: Smallest radius of a new sphere, in meters.
            num_candidates: Uncovered surface points tried per added sphere.
            link_names: Links to cover. Defaults to every link with a mesh that a joint moves.

        Returns:
            Number of spheres added per link.
        """
        if tolerance <= 0.0:
            log_and_raise(f"tolerance must be positive, got {tolerance}")
        if link_names is None:
            static = self._static_links()
            if static:
                log_info(f"Not covering links that no joint moves: {sorted(static)}")
            link_names = [name for name in self._surfaces if name not in static]
        unknown = set(link_names).difference(self._surfaces)
        if unknown:
            log_and_raise(f"No collision mesh and spheres to cover for links {sorted(unknown)}")
        added: Dict[str, int] = {}
        for name in link_names:
            points, normals = self._surfaces[name]
            buffer = self._scene_buffer_of(name)
            centers, radii = self._link_spheres(name, buffer)
            gap = self._gap(points, centers, radii)
            radius_grid = self._radius_grid(points, max(min_radius, tolerance))
            count = 0
            while count < max_new_spheres_per_link:
                uncovered = gap > tolerance
                if not uncovered.any():
                    break
                center, radius = self._best_sphere(
                    name, points, normals, gap, uncovered, radius_grid, tolerance, num_candidates
                )
                if center is None:
                    break
                gap = torch.minimum(gap, torch.linalg.norm(points - center, dim=-1) - radius)
                self._spheres[name].append(
                    {
                        "center": [round(float(c), 4) for c in center],
                        "radius": round(float(radius) - buffer, 4),
                    }
                )
                count += 1
            if count > 0:
                added[name] = count
                left = float(gap.clamp(min=0.0).max())
                log_info(f"{name}: added {count} spheres, mesh now within {1000 * left:.1f} mm")
        return added

    def remove_self_collision_ignore(self, pairs: List[Tuple[str, str]]) -> None:
        """Hand link pairs back to the sphere self-collision check.

        The pairs are removed from ``self_collision_ignore``, in both directions, for later
        :meth:`analyze` calls and for :meth:`save`. See
        :attr:`SphereCoverageReport.suggested_checks` for candidates.

        Args:
            pairs: Link pairs as ``(link_a, link_b)``.
        """
        for link_a, link_b in pairs:
            for first, second in ((link_a, link_b), (link_b, link_a)):
                if second in self._self_ignore.get(first, []):
                    self._self_ignore[first] = [n for n in self._self_ignore[first] if n != second]
            log_info(f"The sphere self-collision check now tests {link_a} - {link_b}")

    def save(self, output_path: str) -> None:
        """Write the robot configuration with the current collision spheres.

        Everything else in the configuration is kept as loaded, except ``self_collision_ignore``
        after :meth:`remove_self_collision_ignore`. The spheres are written inline, also when
        the original configuration referenced a separate sphere file.

        Args:
            output_path: Path of the robot configuration file to write.
        """
        config = copy.deepcopy(self._config)
        kinematics = self._kinematics_data(config)
        kinematics["collision_spheres"] = copy.deepcopy(self._spheres)
        if "self_collision_ignore" in kinematics:
            kinematics["self_collision_ignore"] = copy.deepcopy(self._self_ignore)
        folder = os.path.dirname(output_path)
        if folder:
            os.makedirs(folder, exist_ok=True)
        with open(output_path, "w") as file:
            yaml.dump(config, file, Dumper=_ConfigDumper, sort_keys=False)
        log_info(f"Saved robot configuration to {output_path}")

    # ------------------------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------------------------

    @staticmethod
    def _kinematics_data(config: Dict) -> Dict:
        """The ``kinematics`` section of a robot configuration, with or without ``robot_cfg``."""
        return config["robot_cfg"]["kinematics"] if "robot_cfg" in config else config["kinematics"]

    def _tensor(self, values: np.ndarray) -> torch.Tensor:
        return torch.as_tensor(values, dtype=torch.float32, device=self.device_cfg.device)

    def _scene_buffer_of(self, name: str) -> float:
        if isinstance(self._scene_buffer, dict):
            return float(self._scene_buffer.get(name, 0.0))
        return float(self._scene_buffer)

    def _link_spheres(self, name: str, buffer: float) -> Tuple[torch.Tensor, torch.Tensor]:
        """Centers ``[n, 3]`` and radii ``[n]`` of a link's spheres, enlarged by ``buffer``.

        Spheres whose enlarged radius is not positive (disabled placeholders) are left out.
        """
        spheres = [s for s in self._spheres.get(name, []) if s["radius"] + buffer > 0.0]
        centers = self._tensor(np.array([s["center"] for s in spheres]).reshape(-1, 3))
        radii = self._tensor(np.array([s["radius"] + buffer for s in spheres]))
        return centers, radii

    @staticmethod
    def _gap(points: torch.Tensor, centers: torch.Tensor, radii: torch.Tensor) -> torch.Tensor:
        """Distance from each point to the nearest sphere surface, negative inside a sphere."""
        if centers.shape[0] == 0:
            return torch.full_like(points[:, 0], float("inf"))
        return (torch.cdist(points, centers) - radii).min(dim=-1).values

    def _radius_grid(self, points: torch.Tensor, min_radius: float) -> torch.Tensor:
        """Candidate radii from ``min_radius`` to half the link's largest extent."""
        extent = float((points.max(dim=0).values - points.min(dim=0).values).max())
        max_radius = max(0.5 * extent, min_radius)
        return self._tensor(np.geomspace(min_radius, max_radius, 24))

    def _best_sphere(
        self,
        name: str,
        points: torch.Tensor,
        normals: torch.Tensor,
        gap: torch.Tensor,
        uncovered: torch.Tensor,
        radius_grid: torch.Tensor,
        tolerance: float,
        num_candidates: int,
    ) -> Tuple[Optional[torch.Tensor], float]:
        """Pick the new sphere that covers the most uncovered surface.

        Candidates are the farthest uncovered points plus a random sample of the others, each
        stepped inward along its normal, and outward too in case the normal is flipped. A
        candidate radius is kept while the sphere sticks out of the mesh by at most
        ``tolerance``: its center lies inside the mesh, at least ``radius - tolerance`` deep.
        The first radius that fails ends the search for that candidate.
        """
        idx = torch.nonzero(uncovered).view(-1)
        worst = idx[torch.argsort(gap[idx], descending=True)[: num_candidates // 2]]
        order = torch.randperm(idx.numel(), device=idx.device, generator=self._generator)
        others = idx[order[: num_candidates // 2]]
        idx = torch.unique(torch.cat([worst, others]))
        base = points[idx]
        direction = torch.cat([-normals[idx], normals[idx]])
        base = torch.cat([base, base])
        # Candidate centers [num_candidates, num_radii, 3].
        centers = base[:, None, :] + direction[:, None, :] * radius_grid[None, :, None]
        sdf, _ = self._queries[name].query_sdf(centers.reshape(-1, 3))
        depth = -sdf.view(centers.shape[:2])
        fits = (depth >= radius_grid[None, :] - tolerance).int().cumprod(dim=1).bool()
        if not fits.any():
            return None, 0.0
        # Weighted surface each candidate sphere covers, among the uncovered points.
        weight = torch.where(uncovered, gap, torch.zeros_like(gap))
        dist = torch.cdist(centers.reshape(-1, 3), points)
        covered = dist - radius_grid.repeat(centers.shape[0])[:, None] <= tolerance
        gain = (covered.float() * weight[None, :]).sum(dim=-1).view(centers.shape[:2])
        gain = torch.where(fits, gain, torch.full_like(gain, -1.0))
        best = int(torch.argmax(gain))
        if float(gain.view(-1)[best]) <= 0.0:
            return None, 0.0
        row, col = divmod(best, radius_grid.shape[0])
        return centers[row, col], float(radius_grid[col])

    def _sample_kinematics(self, num_configs: int, seed: int) -> KinematicsState:
        """Kinematics state of the default joint configuration followed by Halton samples.

        Returns:
            The state of ``num_configs + 1`` configurations, with link transforms.
        """
        kinematics = self._kinematics
        limits = kinematics.get_joint_limits()
        samples = SampleBuffer.create_halton_sample_buffer(
            kinematics.get_dof(),
            device_cfg=self.device_cfg,
            low_bounds=limits.position_lower_limits,
            up_bounds=limits.position_upper_limits,
            seed=seed,
            store_buffer=None,
        ).get_samples(num_configs, bounded=True)
        default = kinematics.default_joint_state.position.view(1, -1)
        q = torch.cat([default, samples.to(default)])
        return kinematics.compute_kinematics(
            JointState.from_position(q.unsqueeze(1), joint_names=kinematics.joint_names)
        )

    def _static_links(self) -> List[str]:
        """Links with a mesh whose pose relative to the base no joint changes."""
        link_tf = self._sample_kinematics(64, seed=0).link_transforms.view(65, -1, 3, 4)
        spread = (link_tf - link_tf[:1]).abs().amax(dim=(0, 2, 3))
        name_to_idx = {v: k for k, v in self._mesh_data.body_idx_to_link_name.items()}
        return [name for name in self._surfaces if float(spread[name_to_idx[name]]) < 1.0e-6]

    def _pair_ignored(self, link_a: str, link_b: str) -> bool:
        return link_b in self._self_ignore.get(link_a, []) or link_a in self._self_ignore.get(
            link_b, []
        )

    def _sample_self_collision(
        self, num_configs: int, seed: int
    ) -> Tuple[List[PairSphereCoverage], float, float, List[Tuple[str, str]]]:
        """Test mesh and sphere self-collision per pair on the default and sampled configs.

        Returns:
            Per-pair statistics over the samples, the fractions of samples the sphere check
            accepts while the meshes collide and rejects while they are free, and the checked
            pairs whose spheres overlap at the default configuration.
        """
        data = self._mesh_data
        if data.self_collision_pairs is None:
            return [], 0.0, 0.0, []
        state = self._sample_kinematics(num_configs, seed)
        link_tf = state.link_transforms.view(num_configs + 1, -1, 3, 4)
        names = data.body_idx_to_link_name
        name_to_idx = {v: k for k, v in names.items()}

        def world_spheres(name: str) -> Tuple[torch.Tensor, torch.Tensor]:
            centers, radii = self._link_spheres(name, self._self_buffer.get(name, 0.0))
            tf = link_tf[:, name_to_idx[name]]
            return centers @ tf[:, :, :3].transpose(1, 2) + tf[:, None, :, 3], radii

        all_pairs = data.self_collision_pairs.tolist()
        mesh_any = torch.zeros(num_configs + 1, dtype=torch.bool, device=link_tf.device)
        sphere_any = torch.zeros_like(mesh_any)
        results, at_default = [], []
        try:
            for a, b in all_pairs:
                name_a, name_b = names[a], names[b]
                data.set_self_collision_pairs([(a, b)])
                mesh_hit, _ = check_robot_mesh_self_collision(data, state.link_transforms)
                mesh_hit = mesh_hit.bool()
                sphere_hit = torch.zeros_like(mesh_hit)
                if name_a in self._spheres and name_b in self._spheres:
                    ca, ra = world_spheres(name_a)
                    cb, rb = world_spheres(name_b)
                    if ra.numel() > 0 and rb.numel() > 0:
                        overlap = torch.cdist(ca, cb) - ra[None, :, None] - rb[None, None, :]
                        sphere_hit = overlap.flatten(1).min(dim=1).values < 0.0
                ignored = self._pair_ignored(name_a, name_b)
                mesh_any |= mesh_hit
                if not ignored:
                    sphere_any |= sphere_hit
                    if bool(sphere_hit[0]):
                        at_default.append((name_a, name_b))
                hits, spheres = mesh_hit[1:], sphere_hit[1:]
                if hits.any() or (spheres.any() and not ignored):
                    results.append(
                        PairSphereCoverage(
                            link_a=name_a,
                            link_b=name_b,
                            sphere_check_ignores=ignored,
                            mesh_collision_ratio=float(hits.float().mean()),
                            sphere_collision_ratio=float(spheres.float().mean()),
                            missed_ratio=float((hits & ~spheres).float().mean()),
                            false_alarm_ratio=float((spheres & ~hits).float().mean()),
                            spheres_overlap_at_default=bool(sphere_hit[0]),
                        )
                    )
        finally:
            data.set_self_collision_pairs(all_pairs)
        missed = (mesh_any & ~sphere_any)[1:]
        false_alarm = (sphere_any & ~mesh_any)[1:]
        if bool(mesh_any[0]):
            log_warn("link meshes collide at the default joint configuration")
        return results, float(missed.float().mean()), float(false_alarm.float().mean()), at_default
