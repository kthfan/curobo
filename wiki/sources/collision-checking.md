---
tags: [collision, collision-checking, differentiable, GPU, scene]
date_created: 2026-04-23
date_updated: 2026-04-23
source_count: 1
---

# Collision Checking

`curobo/collision_checking.py` → `curobo._src.collision.collision_robot_scene`

Standalone robot-scene collision checker for use in custom pipelines outside the main solvers. Returns **differentiable distance tensors** — gradients flow through for custom optimization loops.

## Key Classes

| Class | Alias of | Role |
|-------|----------|------|
| `RobotSceneCollision` | `_src.collision.collision_robot_scene.RobotSceneCollision` | Main collision checker |
| `RobotSceneCollisionConfig` | `...CollisionConfig` | Configuration |

## Key Methods

| Method | Returns |
|--------|---------|
| `get_scene_self_collision_distance_from_joints(q)` | `(scene_dist, self_dist)` — differentiable tensors |
| `sample(num_samples, mask_valid)` | Sampled collision-free configurations |
| `validate(q)` | Boolean validity tensor |

## Configuration Parameters

| Parameter | Description |
|-----------|-------------|
| `robot_config: str` | Path to robot YAML/XRDF |
| `scene_model: str` | Scene description |
| `collision_activation_distance: float` | Safety margin |

## Differentiability

All distance outputs are PyTorch tensors with `requires_grad=True`. This enables **gradient-based custom optimization**:

```python
dist_scene, dist_self = checker.get_scene_self_collision_distance_from_joints(q)
loss = -dist_scene.clamp(max=0).sum()  # penalize penetration
loss.backward()  # gradients flow back to q
```

## Internal Architecture (`_src/collision/`)

| File | Role |
|------|------|
| `collision_robot_scene.py` | Main implementation |
| `collision_robot_scene_cfg.py` | Config dataclass |
| `attachment_manager.py` | Manages attached objects (gripper payloads, etc.) |

Scene-side kernels live in `_src/geom/collision/` (Warp-based).

## Related Pages

- [[curobo-overview]]
- [[scene]]
- [[kinematics]]
- [[motion-planning]]
