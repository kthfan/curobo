# CPU-only verification harness (NOT for main)

This directory exists only on verification branches (currently
`claude/brave-bardeen-di9ty0`, stacked on `dev/eef_cost`). It must never be merged into
`main`, and it must not be merged into `dev/eef_cost` either, because `dev/eef_cost` is
headed for `main`.

## How main stays clean

| Layer | What it does |
|---|---|
| Location | Everything is under top-level `cpu_verify/`, outside the `curobo` package. There is no `__init__.py`, so setuptools does not package it, and the repo's pytest `testpaths` (`curobo/tests`) does not collect it. |
| No library hooks | Nothing in `curobo/` imports `cpu_verify`, and the harness never monkeypatches library code. Warp already compiles the kernel for the `cpu` device, so production code needs no CPU-specific branch. |
| Marker | Every harness file carries `NOT for main: see cpu_verify/README.md`. |
| CI guard | `.github/workflows/cpu-verify-guard.yml` fails any PR into `main` (and any push to `main`) that contains `cpu_verify/`, `.github/workflows/cpu-verify.yml`, or the marker anywhere under `curobo/`. The guard travels with the harness, so a PR that carries the harness also carries the check. |
| Branch flow | To pick up new library work, merge `dev/eef_cost` **into** this branch. Fixes found here are ported to `dev/eef_cost` as separate commits that touch only `curobo/`, never by merging this branch. |

## Run

Requirements: CPU builds of `torch` and `warp-lang`, plus `numpy scipy pyyaml setuptools_scm
matplotlib pytest`. No GPU, CUDA driver, or curobolib build is needed.

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install warp-lang numpy scipy pyyaml setuptools_scm matplotlib pytest pytest-xdist

# Library unit tests for the bound cost, on CPU
PYTHONPATH=$PWD python -m pytest -n0 -p no:cacheprovider \
  curobo/tests/_src/cost/test_tool_pose_bound.py curobo/tests/_src/cost/test_cost_eef_bound.py

# Verification harness (its own pytest.ini, single process)
python -m pytest cpu_verify

# Figures (written to cpu_verify/eef_bound/out/, which is git-ignored)
python cpu_verify/eef_bound/visualize_eef_bound.py
```

`.github/workflows/cpu-verify.yml` runs the same three steps on a GitHub-hosted CPU runner
and uploads the figures as an artifact.

## Files

| File | Purpose |
|---|---|
| `conftest.py`, `pytest.ini` | Hide GPUs, put the repo root on `sys.path`, override the repo pytest options. |
| `eef_bound/cpu_adapter.py` | Builders on CPU; a Python mirror of the CUDA quaternion-gradient → angular-gradient map; base-frame finite differences. |
| `eef_bound/test_eef_bound_cpu.py` | Regression checks for the review defects (D1, D4) on the redesigned cost. |
| `eef_bound/visualize_eef_bound.py` | Headless matplotlib diagnostics of the bound cost. |
| `eef_bound/illustrate_corridor_options.py` | Geometry sketch used during the design discussion. |
| `eef_bound/gpu_check.py` | End-to-end planner and MPC checks on a CUDA GPU. |
| `eef_bound/gpu_weight_sweep.py` | Sweep of the `eef_bound` weight on a CUDA GPU (planner, optionally MPC). |
| `eef_bound/diagnose_crash.py` | Runs each suspect step of the tests in its own subprocess to locate crashes or hangs. |
| `eef_bound/REVIEW.md` | Review findings, the verification plan, and the redesign status. |
