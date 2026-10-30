# cpu_verify

Branch-local tools for checking GPU code paths on a machine without a GPU. This directory
exists only on `dev/improve_mesh`. **Delete it before merging into `dev/main`** (tracked in
`todo.md`).

These tools do not replace the GPU test run: CUDA graphs, custom CUDA extensions and
GPU-specific numerics are not covered.

## Setup

```bash
python -m venv cpu_verify/.venv
cpu_verify/.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
cpu_verify/.venv/bin/pip install warp-lang numpy trimesh scipy pytest pyyaml importlib_resources \
    networkx numpy-quaternion tqdm yourdfpy
```

## Tools

- `cpu_device_plugin.py`: pytest plugin that makes `DeviceCfg()` default to the CPU, so
  tests that only need torch and Warp kernels run on the CPU:

  ```bash
  PYTHONPATH=. cpu_verify/.venv/bin/pytest -p cpu_verify.cpu_device_plugin \
      curobo/tests/_src/collision/test_robot_mesh_constraint.py
  ```

- `verify_triangle_distance.py`: compares the Warp `triangles_within` predicate of the mesh
  collision kernel with a SciPy reference distance on random triangle pairs (including
  near-parallel, sliver and degenerate triangles):

  ```bash
  PYTHONPATH=. cpu_verify/.venv/bin/python cpu_verify/verify_triangle_distance.py --pairs 2000
  ```

- `verify_mesh_collision_scene.py`: checks the scene of
  `curobo/tests/_src/motion/test_mesh_collision_end_to_end.py` with URDF forward kinematics:
  the enabled spheres must stay farther from the box than the optimizer's activation distance,
  and the gripper mesh must collide while the wrist meshes do not:

  ```bash
  PYTHONPATH=. cpu_verify/.venv/bin/python cpu_verify/verify_mesh_collision_scene.py
  ```

- `verify_cuda_graph_steps.py`: runs the update steps of
  `curobo/tests/_src/collision/test_robot_mesh_constraint_cuda_graph.py` eagerly and reports
  the steps that do not change the result (a stale CUDA graph can only be caught at a step
  that changes it):

  ```bash
  PYTHONPATH=. cpu_verify/.venv/bin/python cpu_verify/verify_cuda_graph_steps.py
  ```

- `verify_open_mesh_sign.py`: classifies query points as inside or outside closed and open
  obstacle meshes (box without a top, flipped face, single- and double-sided sheets) with the
  mesh SDF, and checks that the containment pass and the sphere collision check agree with it:

  ```bash
  PYTHONPATH=. cpu_verify/.venv/bin/python cpu_verify/verify_open_mesh_sign.py
  ```
