# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Illustrate candidate EEF corridor geometries (numpy + matplotlib only, no cuRobo).

NOT for main: see ``cpu_verify/README.md``.

Run::

    python cpu_verify/eef_bound/illustrate_corridor_options.py --out cpu_verify/eef_bound/out
"""

from __future__ import annotations

import argparse
import math
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Polygon, Rectangle

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
REGION = "#dcdad3"

MARGIN = 0.02


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.grid": True,
            "grid.color": "#e2e1dc",
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
            "lines.linewidth": 2.0,
        }
    )


def _segment_and_ends(ax, start: np.ndarray) -> None:
    ax.plot([start[0], 0.0], [start[1], 0.0], color=INK_2, linewidth=1, linestyle="--")
    ax.plot(*start, "s", color=INK, label="start")
    ax.plot(0.0, 0.0, "*", color=INK, markersize=12, label="goal")


def _goal_box(ax, start: np.ndarray, title: str) -> None:
    lo = np.minimum(start, 0.0) - MARGIN
    hi = np.maximum(start, 0.0) + MARGIN
    ax.add_patch(Rectangle(lo, *(hi - lo), color=REGION, zorder=0, label="allowed region"))
    _segment_and_ends(ax, start)
    # Worst allowed point: the box corner farthest from the straight segment.
    corners = np.array([[lo[0], lo[1]], [lo[0], hi[1]], [hi[0], lo[1]], [hi[0], hi[1]]])
    d = start / np.linalg.norm(start)
    lateral = corners - np.outer(corners @ d, d)
    k = int(np.argmax(np.linalg.norm(lateral, axis=1)))
    foot = corners[k] - lateral[k]
    ax.plot(*corners[k], "o", color=ORANGE, markersize=7)
    ax.plot([corners[k][0], foot[0]], [corners[k][1], foot[1]], color=ORANGE, linewidth=1.5,
            label=f"max deviation {np.linalg.norm(lateral[k]) * 100:.0f} cm")
    ax.set_title(title, fontsize=10)


def _segment_box(ax, start: np.ndarray, title: str) -> None:
    length = np.linalg.norm(start)
    u = start / length
    v = np.array([-u[1], u[0]])
    corners = [
        -MARGIN * u - MARGIN * v,
        (length + MARGIN) * u - MARGIN * v,
        (length + MARGIN) * u + MARGIN * v,
        -MARGIN * u + MARGIN * v,
    ]
    ax.add_patch(Polygon(corners, color=REGION, zorder=0, label="allowed region"))
    _segment_and_ends(ax, start)
    ax.annotate("", xy=0.12 * u, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=BLUE, lw=2))
    ax.annotate("", xy=0.08 * v, xytext=(0, 0), arrowprops=dict(arrowstyle="->", color=AQUA, lw=2))
    ax.text(*(0.13 * u + [0.0, -0.035]), "u (along)", color=BLUE, fontsize=9)
    ax.text(*(0.09 * v + [-0.07, 0.01]), "v (lateral)", color=AQUA, fontsize=9)
    ax.plot([], [], color=ORANGE, label=f"max deviation {MARGIN * 100:.0f} cm")
    ax.set_title(title, fontsize=10)


def _angle_cap(ax) -> None:
    t = np.linspace(0.0, 1.0, 101)
    theta_start = math.radians(60.0)
    margin = math.radians(5.0)
    slerp = theta_start * (1.0 - t)
    wobble = slerp + math.radians(30.0) * np.sin(math.pi * np.clip(t / 0.6, 0.0, 1.0))
    cap = theta_start + margin
    ax.axhspan(0.0, math.degrees(cap), color=REGION, label="allowed: θ ≤ θ_start + margin")
    ax.plot(t, np.degrees(slerp), color=BLUE, label="slerp path")
    ax.plot(t, np.degrees(wobble), color=ORANGE, label="path that swings away")
    over = wobble > cap
    ax.fill_between(t, math.degrees(cap), np.degrees(wobble), where=over, color=ORANGE, alpha=0.25,
                    label="hinge(θ − θ_max)")
    ax.set_xlabel("normalized time")
    ax.set_ylabel("θ = angle(goal⁻¹ · current) [deg]")
    ax.set_ylim(0, 95)
    ax.set_title("(4) angle cap relative to goal", fontsize=10)
    ax.legend(fontsize=7.5, loc="upper right")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    _style()

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.6))
    approach = np.array([0.0, 0.35])
    diagonal = np.array([0.4, 0.4])
    _goal_box(axes[0], approach, "(1) goal frame box, start on a goal axis")
    _goal_box(axes[1], diagonal, "(2) goal frame box, diagonal start")
    _segment_box(axes[2], diagonal, "(3) segment frame box, same diagonal start")
    for ax in axes[:3]:
        ax.set_xlim(-0.08, 0.5)
        ax.set_ylim(-0.08, 0.5)
        ax.set_aspect("equal")
        ax.set_xlabel("goal-frame x [m]")
        ax.legend(fontsize=7.5, loc="lower right")
    axes[0].set_ylabel("goal-frame y [m]")
    _angle_cap(axes[3])
    fig.suptitle(f"Position corridors (margin {MARGIN * 100:.0f} cm on every side) and rotation cap",
                 fontsize=11)
    fig.tight_layout()
    path = os.path.join(args.out, "corridor_options.png")
    fig.savefig(path, dpi=130)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
