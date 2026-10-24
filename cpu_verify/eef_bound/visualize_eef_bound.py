# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render diagnostic plots of :class:`EEFBoundCost` on CPU (headless, matplotlib Agg).

NOT for main: see ``cpu_verify/README.md``.

Run from the repository root::

    python cpu_verify/eef_bound/visualize_eef_bound.py --out cpu_verify/eef_bound/out

Figures:

* ``corridor_1d.png``: cost and gradient along each goal-frame axis, kernel vs finite
  differences, with the corridor shaded.
* ``trajectory_xy.png``: a trajectory in the goal frame against the corridor box, coloured by
  cost, with the descent direction the kernel reports.
* ``rotation_gradient.png``: alignment of the kernel rotation gradient with world-frame finite
  differences versus goal yaw (defect D1), and the residual error after rotating it to world
  frame versus rotation angle (log-map Jacobian approximation).
* ``warp_vs_torch.png``: Warp kernel vs PyTorch reference cost on random inputs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys
from typing import Callable, Tuple

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cpu_adapter import (
    EEFBoundCost,
    EEFBoundCostPyTorch,
    finite_difference_world,
    init,
    make_cost,
    make_poses,
    quat_axis_angle,
    quat_grad_to_world_omega,
    random_quat,
    single_link_problem,
)
from curobo._src.cost.cost_eef_bound_py import _rotate_vector_by_quat_wxyz, quat_mul_wxyz

# Reference palette (dataviz skill, light mode).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
CORRIDOR = "#e6e5e0"

IDENTITY = quat_axis_angle([1.0, 0.0, 0.0], 0.0)
ORIGIN = torch.zeros(3)


def _style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "axes.edgecolor": INK_2,
            "axes.labelcolor": INK,
            "axes.grid": True,
            "grid.color": "#e2e1dc",
            "grid.linewidth": 0.6,
            "xtick.color": INK_2,
            "ytick.color": INK_2,
            "text.color": INK,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )


def _cost_and_grad(
    cost, start_p: torch.Tensor, start_q: torch.Tensor, position: torch.Tensor,
    quaternion: torch.Tensor, goal_p: torch.Tensor, goal_q: torch.Tensor,
) -> Tuple[float, torch.Tensor, torch.Tensor]:
    """Return cost, position gradient and world-frame angular gradient at ``t=1``."""
    tp, gp = single_link_problem(start_p, start_q, position, quaternion, goal_p, goal_q)
    tp.position.requires_grad_(True)
    tp.quaternion.requires_grad_(True)
    value = cost.forward(tp, gp)
    value.sum().backward()
    omega = quat_grad_to_world_omega(tp.quaternion.detach()[0, 1, 0], tp.quaternion.grad[0, 1, 0])
    return value[0, 1, 0].item(), tp.position.grad[0, 1, 0].clone(), omega


def _cost_fn(cost, start_p, start_q, goal_p, goal_q) -> Callable[[torch.Tensor, torch.Tensor], float]:
    def fn(position: torch.Tensor, quaternion: torch.Tensor) -> float:
        tp, gp = single_link_problem(start_p, start_q, position, quaternion, goal_p, goal_q)
        return cost.forward(tp, gp)[0, 1, 0].item()

    return fn


def plot_corridor_1d(out_dir: str) -> None:
    margin = 0.05
    margins = torch.full((4, 3), -1.0)
    margins[:2] = margin
    cost = make_cost(EEFBoundCost, margins)
    start = torch.tensor([0.4, -0.2, 0.1])
    fn = _cost_fn(cost, start, IDENTITY, ORIGIN, IDENTITY)
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for k, (ax, name) in enumerate(zip(axes, "xyz")):
        lo = min(start[k].item(), 0.0) - margin
        hi = max(start[k].item(), 0.0) + margin
        sweep = torch.linspace(lo - 0.3, hi + 0.3, 121)
        costs, grads, fds = [], [], []
        for s in sweep:
            p = 0.5 * start.clone()
            p[k] = s
            c, g, _ = _cost_and_grad(cost, start, IDENTITY, p, IDENTITY, ORIGIN, IDENTITY)
            fd, _ = finite_difference_world(fn, p, IDENTITY)
            costs.append(c)
            grads.append(g[k].item())
            fds.append(fd[k].item())
        ax.axvspan(lo, hi, color=CORRIDOR, label="corridor")
        ax.axvline(start[k].item(), color=INK_2, linewidth=1, linestyle=":")
        ax.axvline(0.0, color=INK_2, linewidth=1, linestyle="--")
        ax.plot(sweep, costs, color=BLUE, label="cost")
        ax.plot(sweep, grads, color=ORANGE, label="kernel dC/d" + name)
        ax.plot(sweep[::6], fds[::6], "o", color=AQUA, markersize=5, label="finite diff.")
        ax.set_title(f"goal-frame {name}  (dotted: start, dashed: goal)", fontsize=10)
        ax.set_xlabel(f"{name} [m]")
    axes[0].set_ylabel("cost / gradient")
    axes[0].legend(loc="upper center", fontsize=8)
    fig.suptitle("Position corridor, one axis swept at a time", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "corridor_1d.png"), dpi=130)
    plt.close(fig)


def plot_trajectory_xy(out_dir: str) -> None:
    margin = 0.05
    margins = torch.full((4, 3), -1.0)
    margins[:2] = margin
    cost = make_cost(EEFBoundCost, margins)
    start = torch.tensor([0.5, 0.2, 0.0])
    H = 25
    t = torch.linspace(0.0, 1.0, H)
    pos = torch.zeros(1, H, 1, 3)
    pos[0, :, 0, 0] = start[0] * (1 - t) + 0.15 * torch.sin(math.pi * t)
    pos[0, :, 0, 1] = start[1] * (1 - t) - 0.25 * torch.sin(math.pi * t)
    quat = IDENTITY.expand(1, H, 1, 4).contiguous()
    tp, gp = make_poses(pos, quat, torch.zeros(1, 1, 1, 1, 3), IDENTITY.view(1, 1, 1, 1, 4))
    tp.position.requires_grad_(True)
    values = cost.forward(tp, gp)
    values.sum().backward()
    c = values[0, :, 0].detach()
    g = tp.position.grad[0, :, 0]
    xy = pos[0, :, 0, :2].detach()

    fig, ax = plt.subplots(figsize=(6.4, 5.2))
    lo = torch.minimum(start[:2], torch.zeros(2)) - margin
    hi = torch.maximum(start[:2], torch.zeros(2)) + margin
    ax.add_patch(
        plt.Rectangle(lo.tolist(), *(hi - lo).tolist(), color=CORRIDOR, zorder=0, label="corridor (x, y)")
    )
    ax.plot(xy[:, 0], xy[:, 1], color=INK_2, linewidth=1, zorder=1)
    sc = ax.scatter(xy[:, 0], xy[:, 1], c=c, cmap="Blues", vmin=0.0, s=40, zorder=2,
                    edgecolors=INK_2, linewidths=0.5)
    ax.quiver(xy[:, 0], xy[:, 1], -g[:, 0], -g[:, 1], color=ORANGE, scale=12, width=0.004,
              zorder=3, label="-gradient")
    ax.plot(*start[:2].tolist(), "s", color=INK, label="start (t=0)")
    ax.plot(0.0, 0.0, "*", color=INK, markersize=12, label="goal")
    ax.set_aspect("equal")
    ax.set_xlabel("goal-frame x [m]")
    ax.set_ylabel("goal-frame y [m]")
    ax.legend(loc="upper left", fontsize=8)
    fig.colorbar(sc, ax=ax, label="cost per waypoint")
    ax.set_title("Trajectory vs. corridor (top-down)", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectory_xy.png"), dpi=130)
    plt.close(fig)


def plot_rotation_gradient(out_dir: str) -> None:
    margins = torch.full((4, 3), -1.0)
    margins[2:] = 0.02
    cost = make_cost(EEFBoundCost, margins)

    def cos_sim(a: torch.Tensor, b: torch.Tensor) -> float:
        return torch.nn.functional.cosine_similarity(a, b, dim=0).item()

    yaws = torch.linspace(0.0, math.pi, 25)
    kernel_cos, fixed_cos = [], []
    for yaw in yaws:
        goal_q = quat_axis_angle([0.0, 0.0, 1.0], yaw.item())
        q = quat_mul_wxyz(goal_q, quat_axis_angle([1.0, 0.0, 0.0], 0.3))
        _, _, omega = _cost_and_grad(cost, ORIGIN, goal_q, ORIGIN, q, ORIGIN, goal_q)
        _, fd = finite_difference_world(_cost_fn(cost, ORIGIN, goal_q, ORIGIN, goal_q), ORIGIN, q)
        kernel_cos.append(cos_sim(omega, fd))
        # What-if: rotate the goal-frame gradient to world frame (the fix for D1).
        fixed_cos.append(cos_sim(_rotate_vector_by_quat_wxyz(goal_q, omega), fd))

    angles = torch.linspace(0.05, 2.8, 25)
    approx_cos = []
    goal_q = quat_axis_angle([0.0, 0.0, 1.0], 0.7)
    for angle in angles:
        # Off-axis rotation (axis 1,1,0.3 in goal frame): the log-map Jacobian now matters.
        q = quat_mul_wxyz(goal_q, quat_axis_angle([1.0, 1.0, 0.3], angle.item()))
        _, _, omega = _cost_and_grad(cost, ORIGIN, goal_q, ORIGIN, q, ORIGIN, goal_q)
        _, fd = finite_difference_world(_cost_fn(cost, ORIGIN, goal_q, ORIGIN, goal_q), ORIGIN, q)
        approx_cos.append(cos_sim(_rotate_vector_by_quat_wxyz(goal_q, omega), fd))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    ax = axes[0]
    ax.plot(yaws, kernel_cos, color=ORANGE, label="kernel as written")
    ax.plot(yaws, fixed_cos, color=BLUE, linestyle="--", label="rotated to world frame")
    ax.set_xlabel("goal yaw [rad]")
    ax.set_ylabel("cos(kernel grad, finite diff.)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("D1: rotation gradient frame", fontsize=10)
    ax.legend(fontsize=8)
    ax = axes[1]
    ax.plot(angles, approx_cos, color=BLUE)
    ax.set_xlabel("rotation from goal, off-axis [rad]")
    ax.set_ylabel("cos(world grad, finite diff.)")
    ax.set_ylim(-1.1, 1.1)
    ax.set_title("After the frame fix: log-map Jacobian approximation", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rotation_gradient.png"), dpi=130)
    plt.close(fig)


def plot_warp_vs_torch(out_dir: str) -> None:
    gen = torch.Generator().manual_seed(0)
    B, H, L, G = 16, 8, 2, 3
    margins = torch.rand(L, 4, 3, generator=gen) * 0.2
    margins[torch.rand(L, 4, 3, generator=gen) < 0.3] = -1.0
    frames = [f"link_{i}" for i in range(L)]
    wp_cost = make_cost(EEFBoundCost, margins, tool_frames=frames)
    pt_cost = make_cost(EEFBoundCostPyTorch, margins, tool_frames=frames)
    pos = torch.randn(B, H, L, 3, generator=gen) * 0.3
    quat = random_quat(B * H * L, gen).view(B, H, L, 4)
    goal_pos = torch.randn(B, 1, L, G, 3, generator=gen) * 0.3
    goal_quat = random_quat(B * L * G, gen).view(B, 1, L, G, 4)
    tp, gp = make_poses(pos, quat, goal_pos, goal_quat)
    a = wp_cost.forward(tp, gp).flatten()
    b = pt_cost.forward(tp, gp).flatten()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    top = max(a.max().item(), b.max().item()) * 1.05
    ax.plot([0, top], [0, top], color=INK_2, linewidth=1, linestyle="--")
    ax.plot(b, a, "o", color=BLUE, markersize=5, alpha=0.8)
    ax.set_xlabel("PyTorch reference cost")
    ax.set_ylabel("Warp kernel cost (CPU)")
    ax.set_title(f"Forward agreement, max |diff| = {(a - b).abs().max().item():.1e}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "warp_vs_torch.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    init()
    _style()
    plot_corridor_1d(args.out)
    plot_trajectory_xy(args.out)
    plot_rotation_gradient(args.out)
    plot_warp_vs_torch(args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
