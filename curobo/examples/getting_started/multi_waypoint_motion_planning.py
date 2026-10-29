# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Plan one trajectory through several Cartesian waypoints.

Pose-to-pose motion planning moves the tool to one goal. Many tasks are a
sequence of goals instead: move above a part, descend onto it, slide it along
a surface. The multi-waypoint planner takes an *ordered list* of Cartesian
waypoints and returns a single joint trajectory that passes through all of
them. Each segment, the motion between two consecutive waypoints, can use its
own tool pose criteria, for example free motion for a transfer and a
straight line for an approach.

By the end of this tutorial you will have:

- Created a multi-waypoint planner for the Franka Panda robot
- Built three reachable Cartesian waypoints
- Set tool pose criteria for each segment
- Planned one stitched trajectory through all waypoints and saved a plot

Step 1: Run the tutorial
------------------------

.. code-block:: bash

   python -m curobo.examples.getting_started.multi_waypoint_motion_planning

Step 2: Check the output
------------------------

When the tutorial finishes successfully you will see output like::

    === Multi-Waypoint Motion Planning ===
    ✓ Multi-waypoint planning succeeded!
      Attempts: 1
      Chosen IK node per waypoint: [0, 3, 1]
      Segments: 2
      Composed trajectory has 180 waypoints
    Trajectory plot saved to: ~/.cache/curobo/examples/multi_waypoint_motion_planning/multi_waypoint_plan.pdf

The chosen IK indices and the trajectory length vary between runs. Output
files are written to ``~/.cache/curobo/`` by default; override with
``--output-dir``.

Step 3: Understand the pipeline
-------------------------------

1. **Create the planner**: ``MultiWaypointMotionPlannerCfg.create`` takes the
   robot and the multi-waypoint settings, such as ``ik_return_seeds`` (IK
   solutions kept per waypoint). Any other keyword argument, such as
   ``scene_model``, ``num_ik_seeds``, or ``mesh_collision_check``, is
   forwarded to ``MotionPlannerCfg.create``.

2. **Build waypoints**: Each waypoint is a ``GoalToolPose``. The tutorial
   derives them with forward kinematics from joint configurations near the
   default configuration, so every waypoint is reachable.

3. **Set per-segment criteria**: ``segment_tool_pose_criteria`` holds one
   dictionary per segment, so ``num_waypoints - 1`` in total, mapping tool
   frame names to ``ToolPoseCriteria``. Both segments here track position and
   orientation. Use ``ToolPoseCriteria.linear_motion("z")`` for a segment
   that must move in a straight line along the goal z axis, or set
   ``bounds`` on the criteria to keep the tool within limits
   (:ref:`tool_pose_bounds_guide`). An empty dictionary tracks position and
   orientation.

4. **Plan**: ``plan_waypoints`` returns a ``MultiWaypointPlanResult``:

   - ``success`` and ``status``, which names the failing waypoint or segment
     when planning fails.
   - ``composed_trajectory``, the stitched joint trajectory, sampled at the
     trajectory optimizer's ``interpolation_dt``.
   - ``segment_results``, the trajectory optimization result of each segment.
   - ``chosen_ik_indices``, ``attempts``, and ``planning_time``.

.. note::

   The trajectory starts at an IK solution of the **first waypoint**, not at
   ``start_state``, which only seeds IK. To execute it from the robot's
   current configuration, first move to
   ``result.composed_trajectory.position[0]``, for example with
   ``planner.motion_planner.plan_cspace``.

To learn how the planner chooses IK solutions and merges the criteria of
neighbouring segments, see :ref:`multi_waypoint_planner_note`.
"""

import argparse
from pathlib import Path
from typing import List, Optional

import torch

import curobo.runtime as runtime
from curobo.multi_waypoint_planner import (
    MultiWaypointMotionPlanner,
    MultiWaypointMotionPlannerCfg,
)
from curobo.types import GoalToolPose, JointState, ToolPoseCriteria


def _get_output_dir() -> Path:
    """Return the example output directory, creating it if needed."""
    out = Path(runtime.cache_dir) / "examples" / "multi_waypoint_motion_planning"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _waypoint_from_joint_config(
    planner: MultiWaypointMotionPlanner, q: torch.Tensor
) -> GoalToolPose:
    """Build a Cartesian waypoint by running FK on a joint configuration.

    Deriving waypoints from valid joint configurations guarantees that IK
    solutions exist, keeping the example deterministic.

    Args:
        planner: The multi-waypoint planner (provides kinematics + frames).
        q: Joint configuration of shape ``[dof]``.

    Returns:
        A single-goal :class:`GoalToolPose` for the tool frames.
    """
    js = JointState.from_position(q.unsqueeze(0), joint_names=planner.joint_names)
    kinematics_state = planner.compute_kinematics(js)
    pose_dict = kinematics_state.tool_poses.to_dict()
    return GoalToolPose.from_poses(
        pose_dict, ordered_tool_frames=planner.tool_frames, num_goalset=1
    )


def _plot_trajectory(
    positions: torch.Tensor,
    joint_names: List[str],
    dt: float,
    save_path: str,
    title: str = "Multi-Waypoint Trajectory",
):
    """Plot joint positions over time and save to *save_path*."""
    import matplotlib.pyplot as plt

    pos_np = positions.cpu().numpy()
    t = [i * dt for i in range(pos_np.shape[0])]

    fig, ax = plt.subplots(figsize=(10, 5))
    for j, name in enumerate(joint_names):
        ax.plot(t, pos_np[:, j], label=name)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Joint position (rad)")
    ax.set_title(title)
    ax.legend(fontsize=7, ncol=2, loc="upper right")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    print(f"Trajectory plot saved to: {save_path}")


def multi_waypoint_planning_example(output_dir: Optional[Path] = None) -> bool:
    """Plan a trajectory through three Cartesian waypoints.

    Args:
        output_dir: Where to save output files. Defaults to
            ``<runtime.cache_dir>/examples/multi_waypoint_motion_planning/``.

    Returns:
        True if planning succeeded.
    """
    if output_dir is None:
        output_dir = _get_output_dir()

    config = MultiWaypointMotionPlannerCfg.create(
        robot="franka.yml",
        ik_return_seeds=12,
        num_ik_seeds=16,
        num_trajopt_seeds=2,
        use_cuda_graph=False,
    )
    planner = MultiWaypointMotionPlanner(config)

    default_q = planner.default_joint_state.position
    start_state = JointState.from_position(
        default_q.unsqueeze(0), joint_names=planner.joint_names
    )

    # Three reachable waypoints derived from joint-space offsets of the start.
    offsets = [
        torch.tensor([0.4, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0], device=default_q.device),
        torch.tensor([0.0, 0.3, 0.0, -0.3, 0.0, 0.3, 0.0], device=default_q.device),
        torch.tensor([-0.4, 0.0, 0.2, 0.0, 0.3, 0.0, 0.0], device=default_q.device),
    ]
    waypoints = [_waypoint_from_joint_config(planner, default_q + o) for o in offsets]

    # One criteria dict per segment (num_waypoints - 1 = 2). Both segments here
    # track full position + orientation; swap in
    # ``ToolPoseCriteria.linear_motion("z")`` for a straight-line segment.
    segment_criteria = [
        {f: ToolPoseCriteria.track_position_and_orientation() for f in planner.tool_frames}
        for _ in range(len(waypoints) - 1)
    ]

    result = planner.plan_waypoints(
        waypoints, start_state, segment_tool_pose_criteria=segment_criteria
    )

    if not bool(result.success.item()):
        print(f"✗ Planning failed: {result.status}")
        return False

    print("✓ Multi-waypoint planning succeeded!")
    print(f"  Attempts: {result.attempts}")
    print(f"  Chosen IK node per waypoint: {result.chosen_ik_indices}")
    print(f"  Segments: {len(result.segment_results)}")
    n_waypoints = result.composed_trajectory.position.shape[0]
    print(f"  Composed trajectory has {n_waypoints} waypoints")

    interp_dt = planner.trajopt_solver.config.interpolation_dt
    _plot_trajectory(
        result.composed_trajectory.position,
        planner.joint_names,
        dt=interp_dt,
        save_path=str(output_dir / "multi_waypoint_plan.pdf"),
    )
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=str, default=None, help="Directory for output files."
    )
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("CUDA is required for this example.")
        return

    output_dir = Path(args.output_dir) if args.output_dir else None
    print("=== Multi-Waypoint Motion Planning ===")
    multi_waypoint_planning_example(output_dir)


if __name__ == "__main__":
    main()
