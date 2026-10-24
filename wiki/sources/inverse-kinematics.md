---
tags: [inverse-kinematics, IK, numerical-optimization, multi-seed]
date_created: 2026-04-23
date_updated: 2026-04-23
source_count: 1
---

# Inverse Kinematics

`curobo/inverse_kinematics.py` → `curobo._src.solver.solver_ik`

Collision-aware IK using numerical optimization with **multiple random seeds** run in parallel on GPU.

## Key Classes

| Class | Role |
|-------|------|
| `IKSolver` | Main solver — runs multi-seed parallel IK |
| `IKSolverConfig` | Configuration dataclass |
| `IKResult` | Output: joint solutions + quality metrics |

## Key Method

```python
result = ik.solve_pose(goal_tool_poses)
```

## Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `robot_config: str` | Path to robot YAML/XRDF config |
| `num_seeds: int` | Number of random restarts for robustness |
| `collision_activation_distance: float` | Safety margin for collision costs |

## Multi-Seed Strategy

```mermaid
flowchart LR
    G[goal_tool_poses] --> S1[Seed 1]
    G --> S2[Seed 2]
    G --> SN[Seed N]
    S1 & S2 & SN -->|parallel GPU optimization| R[IKResult\nbest solution + metrics]
```

Seeds are generated from `_src/solver/seed_ik/` using Halton sequences and random sampling. The best solution across all seeds is returned.

## Seed Generation (`_src/solver/seed_ik/`)

Halton sequences and random samplers from `_src/util/sampling/` provide diverse starting configurations to avoid local minima.

## Related Pages

- [[curobo-overview]]
- [[kinematics]]
- [[motion-planning]]
- [[types-and-config]]
