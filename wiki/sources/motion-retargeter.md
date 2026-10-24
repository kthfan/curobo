---
tags: [motion-retargeting, humanoid, IK, MPC, multi-link]
date_created: 2026-04-23
date_updated: 2026-04-23
source_count: 1
---

# Motion Retargeter

`curobo/motion_retargeter.py` → `curobo._src.motion.motion_retargeter`

IK + MPC-based motion retargeting — converts per-frame tool pose targets to joint trajectories. Especially suited for **humanoid robots** with multi-link tracking requirements (e.g., pelvis + both hands).

## Key Classes

| Class | Role |
|-------|------|
| `MotionRetargeter` | Main retargeter |
| `MotionRetargeterConfig` | Configuration dataclass |
| `MotionRetargeterResult` | Output joint trajectory |
| `ToolPoseCriteria` | Per-link tracking weight specification |
| `ToolPoseCriteriaConfig` | Config for `ToolPoseCriteria` |
| `RetargetingFrame` | Specifies a single link's target pose + weights |

## Key Method

```python
result = retargeter.solve_frame(goal_tool_pose)
```

## ToolPoseCriteria

Separates **position** and **orientation** tracking weight specification per link:

| Field | Description |
|-------|-------------|
| `xyz_weight` | Weight for position tracking per axis |
| `rpy_weight` | Weight for orientation tracking per axis |
| `link_name` | Which link to track |

## Multi-Link Retargeting

```mermaid
flowchart LR
    T1[Pelvis target] --> RC[MotionRetargeter]
    T2[Left hand target] --> RC
    T3[Right hand target] --> RC
    RC -->|IK + MPC| JS[JointState trajectory]
```

Cost function (`_src/cost/cost_support_polygon.py`) enforces support polygon constraints for balance during retargeting.

## Related Pages

- [[curobo-overview]]
- [[mpc]]
- [[inverse-kinematics]]
- [[kinematics]]
