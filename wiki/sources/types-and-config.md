---
tags: [types, config-io, runtime, data-types, GPU, YAML, XRDF]
date_created: 2026-04-23
date_updated: 2026-04-23
source_count: 3
---

# Types, Config I/O & Runtime

Covers three utility modules:
- `curobo/types.py` → `curobo._src.types`
- `curobo/config_io.py` → `curobo._src.util.config_io`
- `curobo/runtime.py` → `curobo._src.runtime`

---

## Types (`types.py`)

Shared GPU tensor-backed dataclasses used across all cuRobo modules.

| Type | Description |
|------|-------------|
| `JointState` | Robot joint positions / velocities / accelerations; factory: `JointState.from_position(q)` |
| `RobotState` | Full robot state (joints + EE pose + spheres) |
| `Pose` | SE(3) pose — position + quaternion (w, x, y, z) |
| `ToolPose` | Pose + tracking criteria for a specific tool link |
| `CameraObservation` | Depth image + camera intrinsics + camera pose |
| `ContentPath` | Path resolver for cuRobo content assets |
| `DeviceCfg` | `device: str` + `dtype: torch.dtype` |

All types are designed to live on GPU (`cuda`) tensors and support batching.

### Pose Convention
cuRobo uses **quaternion order (w, x, y, z)** throughout.

---

## Config I/O (`config_io.py`)

Pure utility module for YAML and path manipulation.

| Function | Description |
|----------|-------------|
| `load_yaml(path)` | Load a `.yml` config file |
| `save_yaml(data, path)` | Save config dict to YAML |
| `get_robot_configs_path()` | Resolve bundled robot config directory |
| `get_scene_configs_path()` | Resolve bundled scene config directory |
| `get_content_path()` | Resolve cuRobo content root |
| `load_xrdf(path)` | Load XRDF robot description |

---

## Runtime Flags (`runtime.py`)

Module-level boolean singletons that globally gate features:

| Flag | Description |
|------|-------------|
| `CUDA_GRAPH_ENABLED` | Enables CUDA graph capture for solver loops |
| `JIT_ENABLED` | Enables JIT compilation of kernels |
| `TIMER_ENABLED` | Enables GPU event timers (profiling) |
| `DEBUG_MODE` | Enables extra logging / assertions |

Setting `CUDA_GRAPH_ENABLED = False` globally disables CUDA graph capture, useful for debugging.

---

## Related Pages

- [[curobo-overview]]
- [[kinematics]]
- [[robot-builder]]
