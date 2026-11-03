#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Add collision spheres where a robot's link meshes stick out of them.

The optimizers steer the robot with its collision spheres only, so a mesh that sticks out of
its spheres can collide without anything pushing it away; the exact mesh check then rejects
the result. This script measures where that happens, adds spheres that cover the meshes, and
writes a report and the updated robot configuration:

.. code-block:: bash

   python -m curobo.examples.guides.cover_link_meshes_with_spheres --robot franka.yml

Output files are written to ``~/.cache/curobo/examples/cover_link_meshes_with_spheres/``
unless ``--output`` and ``--report`` give other paths.
"""

# Standard Library
import argparse
from pathlib import Path

# CuRobo
from curobo import runtime
from curobo.robot_builder import SphereCoverage


def main() -> None:
    """Analyze, cover and save the robot given on the command line."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot", default="franka.yml", help="Robot configuration file.")
    parser.add_argument("--output", default=None, help="Robot configuration to write.")
    parser.add_argument("--report", default=None, help="Markdown report to write.")
    parser.add_argument("--tolerance", type=float, default=0.005, help="Allowed gap in meters.")
    parser.add_argument("--max_new_spheres_per_link", type=int, default=16)
    parser.add_argument("--num_configs", type=int, default=2000)
    parser.add_argument(
        "--check_suggested_pairs",
        action="store_true",
        help="Also remove the suggested pairs from self_collision_ignore.",
    )
    args = parser.parse_args()

    out_dir = Path(runtime.cache_dir) / "examples" / "cover_link_meshes_with_spheres"
    stem = Path(args.robot).stem
    output = args.output or str(out_dir / f"{stem}_covered.yml")
    report_path = Path(args.report or out_dir / f"{stem}_covered.md")

    coverage = SphereCoverage(args.robot)
    before = coverage.analyze(tolerance=args.tolerance, num_configs=args.num_configs)
    added = coverage.cover_meshes(
        tolerance=args.tolerance, max_new_spheres_per_link=args.max_new_spheres_per_link
    )
    after = coverage.analyze(tolerance=args.tolerance, num_configs=args.num_configs)
    checked = after.suggested_checks if args.check_suggested_pairs else []
    if checked:
        coverage.remove_self_collision_ignore(checked)
        after = coverage.analyze(tolerance=args.tolerance, num_configs=args.num_configs)
    coverage.save(output)

    report = f"# Sphere coverage of {args.robot}\n\n"
    report += "Added spheres: " + (
        ", ".join(f"{name} +{n}" for name, n in added.items()) or "none"
    )
    if checked:
        report += ".\n\nRemoved from `self_collision_ignore`: " + ", ".join(
            f"{a} - {b}" for a, b in checked
        )
    report += f".\n\nConfiguration written to `{output}`.\n\n" + after.to_markdown(before)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(report)
    print(report)
    print(f"Report written to {report_path}")


if __name__ == "__main__":
    main()
