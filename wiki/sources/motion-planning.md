---
tags: [motion-planning, trajectory-optimization, batch, graph-planner, PRM]
date_created: 2026-04-23
date_updated: 2026-04-23
source_count: 3
---

# Motion Planning

cuRobo provides three motion planning surfaces: single-problem planning, batch multi-problem planning, and low-level trajectory optimization.

## Modules

- `curobo/motion_planner.py` → `curobo._src.motion.motion_planner`
- `curobo/batch_motion_planner.py` → `curobo._src.motion.motion_planner_batch`
- `curobo/trajectory_optimizer.py` → `curobo._src.solver.solver_trajopt`

---

## MotionPlanner

High-level planner combining PRM graph search + trajectory optimization.

### Key Classes

| Class | Role |
|-------|------|
| `MotionPlanner` | Orchestrates graph seed + TrajOpt refinement |
| `MotionPlannerConfig` | Full configuration (robot, scene, solver params) |
| `MotionPlannerResult` | Output trajectory + success flags |

### Workflow

```mermaid
flowchart LR
    A[goal_tool_poses\ncurrent_state] --> B[PRM Graph Search\ngraph_planner_prm.py]
    B -->|collision-free seed| C[TrajectoryOptimizer\nsolver_trajopt.py]
    C --> D[MotionPlannerResult\ntrajectory + metrics]
```

### Key Method

```python
result = planner.plan_pose(goal_tool_poses, current_state)
```

---

## BatchMotionPlanner

Solves **multiple independent planning problems** in a single IK + TrajOpt GPU pass.

### Key Parameter

| Parameter | Description |
|-----------|-------------|
| `max_batch_size: int` | Controls GPU parallelism (number of problems per pass) |

### Key Method

```python
result = planner.plan_pose(goal_tool_poses, current_states)
```

---

## TrajectoryOptimizer

Lower-level collision-aware trajectory generation without graph search. Best used when a good seed trajectory is already available (e.g., warm-started replanning).

### Key Classes

| Class | Role |
|-------|------|
| `TrajectoryOptimizer` | Runs numerical optimization on a seed trajectory |
| `TrajOptConfig` | Config dataclass |
| `TrajOptResult` | Output |

### Key Method

```python
result = trajopt.solve_pose(goal_tool_poses)
```

---

## Graph Planner Internals (`_src/graph_planner/`)

| Component | Description |
|-----------|-------------|
| `graph_planner_prm.py` | Probabilistic Roadmap (PRM) builder |
| `graph/` | Node manager, sampling, distance, lazy-collision check |
| `search/` | NetworkX path finder + pruner |
| `result.py` | Graph planner result type |

## Related Pages

- [[curobo-overview]]
- [[inverse-kinematics]]
- [[optimizers]]
- [[collision-checking]]
