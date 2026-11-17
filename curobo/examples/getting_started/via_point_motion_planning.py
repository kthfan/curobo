# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Move the tool through several poses in one continuous motion.

A pick usually takes more than one pose: move above the part, then go straight
down onto it. The via-point planner takes an *ordered list* of Cartesian
waypoints and returns one trajectory that passes through all of them without
stopping. The motion that ends at a waypoint, its segment, can have its own
constraints: here the last segment moves straight down while the tool stays
upright.

By the end of this tutorial you will have:

- Created and warmed up a via-point planner for the Franka Panda robot
- Built three Cartesian waypoints from the robot's current tool pose
- Constrained the last segment to a straight, upright descent
- Planned one trajectory through all waypoints and saved a plot

Step 1: Run the tutorial
------------------------

.. code-block:: bash

   python -m curobo.examples.getting_started.via_point_motion_planning

Step 2: Check the output
------------------------

When the tutorial finishes successfully you will see output like::

    === Via-Point Motion Planning ===
    ✓ Via-point planning succeeded in 1 attempt(s), 0.21 s
      Waypoint 0: 0.4 mm, 0.002 rad
      Waypoint 1: 0.3 mm, 0.001 rad
      Waypoint 2: 0.2 mm, 0.001 rad
      Trajectory: 2.31 s, 116 steps of 0.020 s, 7 joints
    Trajectory plot saved to: ~/.cache/curobo/examples/via_point_motion_planning/via_point_plan.pdf

The errors, the planning time and the duration vary between runs and GPUs. The
first plan after construction compiles kernels and captures CUDA graphs, which
``warmup()`` does before the timed plan. Output files are written to
``~/.cache/curobo/`` by default; override with ``--output-dir``.

Step 3: Understand the pipeline
-------------------------------

1. **Create the planner once**: ``ViaPointMotionPlannerCfg.create`` takes the
   robot, the scene and the via-point settings, such as ``waypoint_tolerance``
   and ``chain_metric``. Other keyword arguments, such as ``collision_cache``,
   are forwarded to ``MotionPlannerCfg.create``. Call ``warmup()`` before the
   first plan and reuse the planner for every request.

2. **Build waypoints**: The waypoints are one ``GoalToolPose`` of shape
   ``[1, num_waypoints, tool frames, 1, 3/4]`` in the robot base frame,
   without the current pose. The tutorial offsets the current tool position
   and keeps its orientation, so every waypoint is reachable.

3. **Constrain segments**: ``plan_waypoints`` takes one entry per waypoint.
   Entry ``k`` constrains segment ``k``, the motion that ends at waypoint
   ``k``; ``None`` reaches the waypoint and leaves the path free. The last
   entry combines ``ToolPoseCriteria.linear_motion("z")`` with
   ``enforce_path_tolerance``, so the descent fails rather than leaving its
   line by more than 5 mm, and a ``ToolPoseBound.axis_tilt`` bound that keeps
   the tool upright within 10°.

4. **Plan and execute**: ``plan_waypoints`` returns a ``ViaPointPlanResult``.
   Check ``success`` before using the trajectory; ``status`` explains a
   failure. ``get_interpolated_plan()`` returns the executable trajectory,
   trimmed to its valid steps, with its ``dt`` and ``joint_names``.
   ``waypoint_position_error`` and ``waypoint_rotation_error`` report how
   closely it passes each waypoint.

.. note::

   The trajectory starts at ``start_state``, which must be at rest. In an
   application, build it from the measured joint positions and reorder it to
   ``planner.joint_names``.

To learn how the planner chooses one IK solution per waypoint and why the
whole path is optimized at once, see :ref:`via_point_planner_note`. For more
segment constraints, see :ref:`guide_via_point_planning`.
"""

import argparse
import math
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import torch

import curobo.runtime as runtime
from curobo.types import GoalToolPose, JointState, ToolPoseBound, ToolPoseCriteria
from curobo.via_point_planner import ViaPointMotionPlanner, ViaPointMotionPlannerCfg


def _get_output_dir() -> Path:
    """Return the example output directory, creating it if needed."""
    out = Path(runtime.cache_dir) / "examples" / "via_point_motion_planning"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _waypoints_from_offsets(
    planner: ViaPointMotionPlanner, start: JointState, offsets: torch.Tensor
) -> GoalToolPose:
    """Offset the current tool position, keeping its orientation.

    Args:
        planner: The via-point planner.
        start: Start joint state ``[1, dof]``.
        offsets: Position offsets ``[num_waypoints, 3]`` in meters, in the base frame.

    Returns:
        Waypoints of shape ``[1, num_waypoints, tool frames, 1, 3/4]``.
    """
    home = planner.compute_kinematics(start).tool_poses  # [1, 1, frames, 3/4]
    num_waypoints = offsets.shape[0]
    position = home.position[:, 0:1] + offsets.to(home.position).view(1, num_waypoints, 1, 3)
    quaternion = home.quaternion[:, 0:1].expand(-1, num_waypoints, -1, -1)
    return GoalToolPose(
        planner.tool_frames,
        position.unsqueeze(3).contiguous(),
        quaternion.unsqueeze(3).contiguous(),
    )


def _plot_trajectory(
    positions: torch.Tensor,
    joint_names: List[str],
    dt: float,
    waypoint_times: List[float],
    save_path: str,
) -> None:
    """Plot joint positions over time, mark the waypoints and save to *save_path*."""
    import matplotlib.pyplot as plt

    pos_np = positions.cpu().numpy()
    t = [i * dt for i in range(pos_np.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(joint_names):
        ax.plot(t, pos_np[:, j], label=name)
    for k, time in enumerate(waypoint_times):
        ax.axvline(time, color="gray", linestyle="--", linewidth=0.8)
        ax.text(time, ax.get_ylim()[1], f" wp {k}", va="top", fontsize=8, color="gray")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")
    ax.set_title("Via-Point Trajectory")
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Trajectory plot saved to: {save_path}")


def via_point_planning_example(output_dir: Optional[Path] = None) -> bool:
    """Plan through three waypoints, the last one reached by a straight, upright descent.

    Args:
        output_dir: Where to save output files. Defaults to
            ``<runtime.cache_dir>/examples/via_point_motion_planning/``.

    Returns:
        True if planning succeeded.
    """
    if output_dir is None:
        output_dir = _get_output_dir()

    # Create and warm up the planner once; reuse it for every request.
    planner = ViaPointMotionPlanner(
        ViaPointMotionPlannerCfg.create(
            robot="franka.yml",
            scene_model="collision_table.yml",
            collision_cache={"cuboid": 8, "mesh": 2},
            use_cuda_graph=True,
        )
    )
    planner.warmup()

    start = JointState.from_position(
        planner.default_joint_state.position.unsqueeze(0), joint_names=planner.joint_names
    )

    # Forward, sideways, then 20 cm straight down.
    offsets = torch.tensor([[0.15, 0.0, 0.0], [0.15, 0.2, 0.0], [0.15, 0.2, -0.2]])
    waypoints = _waypoints_from_offsets(planner, start, offsets)

    frame = planner.tool_frames[0]
    descent = replace(
        ToolPoseCriteria.linear_motion("z"),
        # 5 mm off the line, 0.05 rad of rotation (expressed as 2 * sin(angle / 2)).
        path_tolerance=(0.005, 2.0 * math.sin(0.05 / 2.0)),
        enforce_path_tolerance=True,
        bounds=[ToolPoseBound.axis_tilt(max_angle=math.radians(10.0), anchor="base")],
    )
    # One entry per waypoint: entry k constrains the motion that ends at waypoint k.
    segment_criteria = [None, None, {frame: descent}]

    result = planner.plan_waypoints(waypoints, start, segment_criteria)
    if not bool(result.success.item()):
        print(f"✗ Planning failed: {result.status}")
        planner.destroy()
        return False

    print(
        f"✓ Via-point planning succeeded in {result.attempts} attempt(s), "
        f"{result.planning_time:.2f} s"
    )
    position_error = result.waypoint_position_error.view(-1).tolist()
    rotation_error = result.waypoint_rotation_error.view(-1).tolist()
    for k, (p_err, r_err) in enumerate(zip(position_error, rotation_error)):
        print(f"  Waypoint {k}: {p_err * 1000.0:.1f} mm, {r_err:.3f} rad")

    trajectory = result.get_interpolated_plan()
    dt = float(trajectory.dt.reshape(-1)[0].item())
    positions = trajectory.position.reshape(-1, trajectory.position.shape[-1])
    duration = (positions.shape[0] - 1) * dt
    print(
        f"  Trajectory: {duration:.2f} s, {positions.shape[0]} steps of {dt:.3f} s, "
        f"{positions.shape[-1]} joints"
    )

    # Waypoint indices refer to the optimized trajectory's samples, spaced by its own dt.
    solution_dt = float(result.trajopt_result.js_solution.dt.reshape(-1)[0].item())
    waypoint_times = [i * solution_dt for i in result.waypoint_indices]
    _plot_trajectory(
        positions,
        trajectory.joint_names,
        dt,
        waypoint_times,
        save_path=str(output_dir / "via_point_plan.pdf"),
    )
    planner.destroy()
    return True


def main() -> None:
    """Run the tutorial."""
    parser = argparse.ArgumentParser(description="Via-point motion planning with cuRobo")
    parser.add_argument("--output-dir", type=str, default=None, help="Directory for output files.")
    parser.add_argument("--test", action="store_true", help="Run as self-test with assertions")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this example.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else _get_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    print("=== Via-Point Motion Planning ===")
    success = via_point_planning_example(output_dir)
    if args.test:
        assert success, "Via-point planning failed"
        assert (output_dir / "via_point_plan.pdf").exists(), "via_point_plan.pdf not created"


if __name__ == "__main__":
    main()
