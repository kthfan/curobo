---
tags: [kinematics, forward-kinematics, jacobian, CUDA, differentiable]
date_created: 2026-04-23
date_updated: 2026-04-23
source_count: 1
---

# Kinematics

`curobo/kinematics.py` → `curobo._src.robot.kinematics`

Differentiable **forward kinematics** with Jacobian computation, center-of-mass (CoM), and collision sphere generation. Backed by custom CUDA kernels.

## Key Classes

| Class | Role |
|-------|------|
| `CudaRobotModel` | Main FK model |
| `CudaRobotModelConfig` | Configuration (URDF path, link names, etc.) |
| `CudaRobotModelState` | Output state (link poses, Jacobians, spheres) |

## Typical Workflow

```python
config = CudaRobotModelConfig.from_robot_yaml("robot.yml")
model = CudaRobotModel(config)
state = model.compute_kinematics(joint_state)
# state.ee_pose, state.jacobian, state.link_spheres ...
```

## CUDA Kernel Architecture (`_src/curobolib/kernels/kinematics/`)

| Kernel | Description |
|--------|-------------|
| FK forward kernel | Computes link poses via chain rule over joint transforms |
| FK backward kernel | Backpropagates gradients through the kinematic chain |
| Jacobian kernel | Analytical Jacobian computation in parallel |

All kernels are exposed via pybind11 (`_src/curobolib/backends/pybind/`) and wrapped as PyTorch autograd functions for seamless `.backward()` support.

## Outputs

| Field | Description |
|-------|-------------|
| `ee_pose` | Tool/end-effector SE(3) pose |
| `jacobian` | 6×n Jacobian at the end-effector |
| `link_poses` | All tracked link SE(3) poses |
| `link_spheres` | Collision sphere positions (used by collision checker) |
| `com` | Center of mass (for humanoid balance costs) |

## Related Pages

- [[curobo-overview]]
- [[inverse-kinematics]]
- [[collision-checking]]
- [[types-and-config]]
