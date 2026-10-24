# Wiki Log

Append-only chronological record of all wiki operations.

---

## [2026-04-23] init | Wiki initialized

Created initial wiki structure: `wiki/index.md`, `wiki/log.md`, `wiki/sources/`.

---

## [2026-04-23] ingest | cuRobo Library (all 23 public modules)

Ingested all source files from `curobo/`. Created 13 wiki pages in `wiki/sources/`:
- `curobo-overview.md` — full architecture overview and design patterns
- `motion-planning.md` — MotionPlanner, BatchMotionPlanner, TrajectoryOptimizer
- `inverse-kinematics.md` — multi-seed collision-aware IK
- `kinematics.md` — differentiable FK + Jacobian CUDA kernels
- `collision-checking.md` — standalone differentiable collision checker
- `optimizers.md` — MPPI, ES, STOMP, L-BFGS, CG, multi-stage optimizer
- `scene.md` — obstacle primitives and BloxMap integration
- `perception.md` — TSDF/ESDF mapping, segmentation, pose estimation
- `mpc.md` — real-time MPC with warm-start
- `motion-retargeter.md` — humanoid multi-link retargeting
- `robot-builder.md` — URDF→config builder + debugger, XRDF export
- `types-and-config.md` — shared GPU types, YAML I/O, runtime flags
- `utilities.md` — sphere fitting, rollout, profiling, viewer, logging

Updated `wiki/index.md` with all 13 source pages.
