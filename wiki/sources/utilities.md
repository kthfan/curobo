---
tags: [sphere-fitting, rollout, profiling, viewer, logging, utilities]
date_created: 2026-04-23
date_updated: 2026-04-23
source_count: 5
---

# Utilities

Covers: `sphere_fit.py`, `rollout.py`, `profiling.py`, `viewer.py`, `logging.py`

---

## Sphere Fitting (`sphere_fit.py`)

`curobo/sphere_fit.py` → `curobo._src.geom.sphere_fit`

Approximates mesh geometry with spheres for fast GPU collision checking.

| Class / Function | Description |
|-----------------|-------------|
| `SphereFitType` | Enum: `VOXEL`, `MORPHIT`, etc. |
| `SphereFitResult` | `.centers`, `.radii`, `.metrics` |
| `fit_spheres_to_mesh(mesh, type, num_spheres)` | Main entry point → `SphereFitResult` |
| `compute_sphere_count(mesh, density)` | Auto-size sphere count from mesh bounding volume |

Multiple fitting strategies are selectable via `SphereFitType`. `compute_sphere_count` uses a bounding volume heuristic for auto-sizing.

---

## Rollout (`rollout.py`)

`curobo/rollout.py` → `curobo._src.rollout`

Exposes rollout primitives for standalone optimization experiments.

| Class | Description |
|-------|-------------|
| `RosenbrockRollout` | Canonical non-convex Rosenbrock benchmark rollout |
| `RosenbrockRolloutConfig` | Config for benchmark |

> [!WARNING]
> Internal robot rollouts (`rollout_robot.py`) are **not** exposed here. Only the benchmark rollout is public. Use `_src.rollout.rollout_robot` directly for advanced custom rollouts.

All rollouts conform to `rollout_protocol.py` structural subtype.

---

## Profiling (`profiling.py`)

`curobo/profiling.py` → `curobo._src.util.cuda_event_timer`

Lightweight GPU timing utilities.

| Class | Usage |
|-------|-------|
| `CudaEventTimer` | `timer.start()` / `timer.stop()` → elapsed seconds |

Gated by `TIMER_ENABLED` runtime flag — when disabled, returns `0.0` as a no-op for zero production overhead.

---

## Viewer (`viewer.py`)

`curobo/viewer.py` → `curobo._src.util.viser_visualizer` + `usd_writer`

Visualization backends with **lazy optional imports**.

| Function | Backend | Description |
|----------|---------|-------------|
| `create_viewer()` | Viser (web 3D) | Interactive real-time robot visualization |
| `create_usd_writer()` | USD (`usd-core`) | Offline trajectory export to USD format |

**Key methods:**
```python
viz.set_joint_state(joint_state)   # live update
writer.write(joint_trajectory)      # export trajectory
```

If optional dependencies (`viser`, `usd-core`) are missing, factory functions raise `ImportError` with install instructions.

---

## Logging (`logging.py`)

`curobo/logging.py` → `curobo._src.util.logging`

| Function | Description |
|----------|-------------|
| `get_logger(name)` | Get a named logger |
| `set_log_level(level)` | Set global log level |
| `log_info(msg)` | Info log |
| `log_warn(msg)` | Warning log |
| `log_error(msg)` | Error log |

---

## Related Pages

- [[curobo-overview]]
- [[collision-checking]]
- [[robot-builder]]
- [[types-and-config]]
