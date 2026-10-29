# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Render diagnostic plots of the tool pose bound cost on CPU (headless, matplotlib Agg).

NOT for main: see ``cpu_verify/README.md``.

Run from the repository root::

    python cpu_verify/eef_bound/visualize_eef_bound.py --out cpu_verify/eef_bound/out

Figures:

* ``bound_1d.png``: cost and gradient along each reference axis of a line bound anchored to a
  rotated goal, kernel vs base-frame finite differences.
* ``trajectory_plane.png``: a curved trajectory against a plane bound, side and top views,
  coloured by cost, with the descent direction the kernel reports.
* ``rotation_gradient.png``: alignment of the orientation gradient with finite differences vs
  goal yaw and vs rotation angle (both should be 1).
* ``tilt_yaw_roll.png``: tilt/yaw/roll vs rotation-vector components while a tool pitched up
  20 degrees yaws about the world z axis and rolls about its own axis.
* ``kernel_vs_reference.png``: Warp kernel vs PyTorch evaluation on random inputs.
"""

from __future__ import annotations

import argparse
import math
import os
import sys

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from cpu_adapter import (
    IDENTITY,
    analytic_gradient,
    evaluate,
    finite_difference_world,
    init,
    make_cost,
    quat_axis_angle,
    random_quat,
)
from curobo._src.cost.tool_pose_bound import (
    ToolPoseBound,
    _quat_mul,
    _quat_rotate,
    tool_pose_bound_components,
    tool_pose_bound_cost,
)
from curobo._src.types.tool_pose import GoalToolPose, ToolPose

# Reference palette (dataviz skill, light mode).
BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
INK = "#0b0b0b"
INK_2 = "#52514e"
SURFACE = "#fcfcfb"
REGION = "#e6e5e0"
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


def plot_bound_1d(out_dir: str) -> None:
    lower, upper = -0.3, 0.0
    cost = make_cost({"ee": ToolPoseBound.line(min_along=lower, max_along=upper)})
    goal_q = quat_axis_angle([0.0, 0.0, 1.0], 0.6)
    axes_world = [_quat_rotate(goal_q, torch.eye(3)[k]) for k in range(3)]

    def fn(p, q):
        return evaluate(cost, p, q, ORIGIN, goal_q)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), sharey=True)
    for k, (ax, name) in enumerate(zip(axes, "xyz")):
        if k == 2:
            base, lo, hi = ORIGIN, lower, upper
        else:
            base, lo, hi = -0.1 * axes_world[2], 0.0, 0.0
        sweep = torch.linspace(lo - 0.25, hi + 0.25, 101)
        costs, grads, fds = [], [], []
        for s in sweep:
            p = base + s.item() * axes_world[k]
            costs.append(fn(p, IDENTITY))
            g, _ = analytic_gradient(cost, p, IDENTITY, ORIGIN, goal_q)
            fd, _ = finite_difference_world(fn, p, IDENTITY)
            grads.append(torch.dot(g, axes_world[k]).item())
            fds.append(torch.dot(fd, axes_world[k]).item())
        if hi > lo:
            ax.axvspan(lo, hi, color=REGION, label="allowed interval")
        else:
            ax.axvline(lo, color=INK_2, linewidth=1, linestyle=":", label="allowed value")
        ax.plot(sweep, costs, color=BLUE, label="cost (squared hinge)")
        ax.plot(sweep, grads, color=ORANGE, label="kernel gradient")
        ax.plot(sweep[::5], fds[::5], "o", color=AQUA, markersize=5, label="finite diff.")
        ax.set_title(f"reference {name} (goal yawed 0.6 rad)", fontsize=10)
        ax.set_xlabel(f"{name} in reference frame [m]")
    axes[0].set_ylabel("cost / gradient")
    axes[0].legend(fontsize=8, loc="upper center")
    fig.suptitle("Line bound along the goal z axis, z in [-0.3, 0]", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "bound_1d.png"), dpi=130)
    plt.close(fig)


def plot_trajectory_plane(out_dir: str) -> None:
    cost = make_cost({"ee": ToolPoseBound.plane(anchor="base", tolerance=0.005)})
    horizon = 40
    t = torch.linspace(0.0, 1.0, horizon)
    position = torch.zeros(1, horizon, 1, 3)
    position[0, :, 0, 0] = 0.6 * t
    position[0, :, 0, 1] = 0.25 * torch.sin(2.0 * math.pi * t)
    position[0, :, 0, 2] = (
        0.06 * torch.clamp(torch.sin(math.pi * (t - 0.4) / 0.4), min=0.0) * (t > 0.4)
    )
    quaternion = IDENTITY.expand(1, horizon, 1, 4).contiguous()
    position.requires_grad_(True)
    tool = ToolPose(["ee"], position, quaternion)
    values = cost.forward(tool)
    values.sum().backward()
    c = values[0, :, 0].detach()
    g = position.grad[0, :, 0]
    p = position.detach()[0, :, 0]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    ax = axes[0]
    ax.axhspan(-0.005, 0.005, color=REGION, label="plane ± tolerance")
    ax.plot(p[:, 0], p[:, 2], color=INK_2, linewidth=1)
    sc = ax.scatter(
        p[:, 0],
        p[:, 2],
        c=c,
        cmap="Blues",
        vmin=0.0,
        s=36,
        edgecolors=INK_2,
        linewidths=0.5,
        zorder=3,
    )
    ax.quiver(
        p[:, 0],
        p[:, 2],
        -g[:, 0],
        -g[:, 2],
        color=ORANGE,
        scale=0.6,
        width=0.004,
        zorder=4,
        label="-gradient",
    )
    ax.set_xlabel("x [m]")
    ax.set_ylabel("z [m]")
    ax.set_title("side view: only leaving the plane costs", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    fig.colorbar(sc, ax=ax, label="cost per waypoint")
    ax = axes[1]
    ax.plot(p[:, 0], p[:, 1], color=INK_2, linewidth=1)
    ax.scatter(
        p[:, 0],
        p[:, 1],
        c=c,
        cmap="Blues",
        vmin=0.0,
        s=36,
        edgecolors=INK_2,
        linewidths=0.5,
        zorder=3,
    )
    ax.set_aspect("equal")
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("top view: in-plane curve is free", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "trajectory_plane.png"), dpi=130)
    plt.close(fig)


def plot_rotation_gradient(out_dir: str) -> None:
    box = make_cost({"ee": ToolPoseBound.orientation_box(0.02, 0.02, 0.05)})
    cone = make_cost({"ee": ToolPoseBound.axis_tilt(max_angle=0.03)})

    def alignment(goal_q, quaternion, cost=box):
        def fn(p, q):
            return evaluate(cost, p, q, ORIGIN, goal_q)

        _, fd = finite_difference_world(fn, ORIGIN, quaternion)
        _, omega = analytic_gradient(cost, ORIGIN, quaternion, ORIGIN, goal_q)
        return torch.nn.functional.cosine_similarity(omega, fd, dim=0).item()

    yaws = torch.linspace(0.0, math.pi, 25)
    by_yaw = []
    for yaw in yaws:
        goal_q = quat_axis_angle([0.0, 0.0, 1.0], yaw.item())
        by_yaw.append(alignment(goal_q, _quat_mul(goal_q, quat_axis_angle([1.0, 0.4, 0.2], 0.5))))
    angles = torch.linspace(0.1, 2.8, 25)
    goal_q = quat_axis_angle([0.0, 0.0, 1.0], 0.7)
    by_angle = [
        alignment(goal_q, _quat_mul(goal_q, quat_axis_angle([1.0, 1.0, 0.3], a.item())), cone)
        for a in angles
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(yaws, by_yaw, color=BLUE)
    axes[0].set_xlabel("goal yaw [rad]")
    axes[0].set_title("orientation_box gradient vs goal orientation (D1)", fontsize=10)
    axes[1].plot(angles, by_angle, color=BLUE)
    axes[1].set_xlabel("rotation from goal, off-axis [rad]")
    axes[1].set_title("axis_tilt gradient vs rotation size (exact)", fontsize=10)
    for ax in axes:
        ax.set_ylim(-1.1, 1.1)
        ax.set_ylabel("cos(kernel gradient, finite diff.)")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "rotation_gradient.png"), dpi=130)
    plt.close(fig)


def plot_tilt_yaw_roll(out_dir: str) -> None:
    bound = ToolPoseBound(anchor="base", tilt=(0.5, 1.5))
    pitched = quat_axis_angle([0.0, 1.0, 0.0], math.radians(70))  # tool z 20 deg above horizon
    sweep = torch.linspace(-math.radians(150), math.radians(150), 61)
    quats = torch.stack(
        [
            _quat_mul(
                _quat_mul(quat_axis_angle([0.0, 0.0, 1.0], a.item()), pitched),
                quat_axis_angle([0.0, 0.0, 1.0], -0.5 * a.item()),
            )
            for a in sweep
        ]
    )
    components, _ = tool_pose_bound_components(
        bound, torch.zeros(len(sweep), 3), quats, ORIGIN, IDENTITY
    )
    half = torch.atan2(quats[:, 1:].norm(dim=-1), quats[:, 0])
    rotvec = quats[:, 1:] * (2.0 * half / torch.clamp(torch.sin(half), min=1e-6)).unsqueeze(-1)

    deg = torch.rad2deg
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    ax.plot(deg(sweep), deg(components[:, 3]), color=BLUE, label="tilt")
    ax.plot(deg(sweep), deg(components[:, 4]), color=AQUA, label="yaw")
    ax.plot(deg(sweep), deg(components[:, 5]), color=ORANGE, label="roll")
    ax.plot(deg(sweep), deg(rotvec[:, 1]), color=YELLOW, linestyle="--", label="rotvec y")
    ax.set_xlabel("yaw about world z [deg] (roll about the tool axis = -yaw / 2)")
    ax.set_ylabel("component [deg]")
    ax.set_title("Tool pitched up 20°: tilt stays 70°, yaw and roll separate", fontsize=10)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tilt_yaw_roll.png"), dpi=130)
    plt.close(fig)


def plot_kernel_vs_reference(out_dir: str) -> None:
    gen = torch.Generator().manual_seed(0)
    bound = ToolPoseBound(
        anchor="goal",
        reference_rotation=[0.8, 0.2, 0.4, 0.4],
        tool_rotation=[0.9, 0.3, -0.2, 0.25],
        x=(-0.02, 0.03),
        y=(None, 0.01),
        z=(-0.3, None),
        tilt=(0.5, 1.4),
        yaw=(-0.3, 0.4),
        roll=(2.8, 3.6),
    )
    cost = make_cost({"ee": bound})
    batch, horizon = 16, 8
    position = torch.randn(batch, horizon, 1, 3, generator=gen) * 0.3
    quaternion = random_quat(batch * horizon, gen).view(batch, horizon, 1, 4)
    goal_position = torch.randn(batch, 1, 1, 1, 3, generator=gen) * 0.3
    goal_quaternion = random_quat(batch, gen).view(batch, 1, 1, 1, 4)
    idxs = torch.arange(batch, dtype=torch.int32)
    kernel = cost.forward(
        ToolPose(["ee"], position, quaternion),
        GoalToolPose(["ee"], goal_position, goal_quaternion),
        idxs,
    ).flatten()
    reference = tool_pose_bound_cost(
        bound,
        position[:, :, 0],
        quaternion[:, :, 0],
        goal_position[:, 0, 0, 0].unsqueeze(1),
        goal_quaternion[:, 0, 0, 0].unsqueeze(1),
    ).flatten()
    fig, ax = plt.subplots(figsize=(5, 4.6))
    top = max(kernel.max().item(), reference.max().item()) * 1.05
    ax.plot([0, top], [0, top], color=INK_2, linewidth=1, linestyle="--")
    ax.plot(reference, kernel, "o", color=BLUE, markersize=5, alpha=0.8)
    ax.set_xlabel("PyTorch evaluation")
    ax.set_ylabel("Warp kernel (CPU)")
    diff = (kernel - reference).abs().max().item()
    ax.set_title(f"Forward agreement, max |diff| = {diff:.1e}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "kernel_vs_reference.png"), dpi=130)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--out", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
    )
    args = parser.parse_args()
    os.makedirs(args.out, exist_ok=True)
    init()
    _style()
    plot_bound_1d(args.out)
    plot_trajectory_plane(args.out)
    plot_rotation_gradient(args.out)
    plot_tilt_yaw_roll(args.out)
    plot_kernel_vs_reference(args.out)
    print(f"wrote figures to {args.out}")


if __name__ == "__main__":
    main()
