# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
"""Check ``triangles_within`` against a reference triangle-triangle distance on the CPU.

Branch-local verification tool. Do not merge into ``dev/main``.

The reference minimises ``|sum(alpha_i a_i) - sum(beta_j b_j)|^2`` over the two barycentric
simplices with SciPy, independently of the vertex-face / edge-edge decomposition used by the
Warp function. For every random non-intersecting pair the Warp test must report "within"
for a threshold slightly above the reference distance and "not within" slightly below it.

Usage (from the repository root)::

    PYTHONPATH=. python cpu_verify/verify_triangle_distance.py [--pairs 2000] [--seed 0]
"""

# Standard Library
import argparse

# Third Party
import numpy as np
import warp as wp
from scipy.optimize import minimize

# CuRobo
from curobo._src.geom.collision.wp_robot_mesh_kernel import triangles_within


@wp.kernel(enable_backward=False)
def _within_kernel(
    tris_a: wp.array(dtype=wp.vec3),
    tris_b: wp.array(dtype=wp.vec3),
    dist: wp.array(dtype=wp.float32),
    intersect: wp.array(dtype=wp.int32),
    within: wp.array(dtype=wp.int32),
):
    i = wp.tid()
    a0 = tris_a[i * 3 + 0]
    a1 = tris_a[i * 3 + 1]
    a2 = tris_a[i * 3 + 2]
    b0 = tris_b[i * 3 + 0]
    b1 = tris_b[i * 3 + 1]
    b2 = tris_b[i * 3 + 2]
    intersect[i] = wp.intersect_tri_tri(a0, a1, a2, b0, b1, b2)
    d = dist[i]
    within[i] = wp.where(triangles_within(a0, a1, a2, b0, b1, b2, d * d), 1, 0)


def reference_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Distance between triangles ``a`` and ``b`` (each ``[3, 3]``) by convex minimisation."""

    def objective(x):
        diff = x[:3] @ a - x[3:] @ b
        return diff @ diff

    def gradient(x):
        diff = x[:3] @ a - x[3:] @ b
        return np.concatenate([2.0 * a @ diff, -2.0 * b @ diff])

    best = np.inf
    # Start from every vertex pair so the local solver cannot stall away from the optimum.
    for i in range(3):
        for j in range(3):
            x0 = np.zeros(6)
            x0[i] = 1.0
            x0[3 + j] = 1.0
            res = minimize(
                objective,
                x0,
                jac=gradient,
                method="SLSQP",
                bounds=[(0.0, 1.0)] * 6,
                constraints=[
                    {"type": "eq", "fun": lambda x: x[:3].sum() - 1.0},
                    {"type": "eq", "fun": lambda x: x[3:].sum() - 1.0},
                ],
                options={"ftol": 1e-14, "maxiter": 200},
            )
            best = min(best, float(res.fun))
    return float(np.sqrt(max(best, 0.0)))


def random_pairs(rng: np.random.Generator, n: int) -> tuple:
    """Random triangle pairs, including slivers, near-parallel and degenerate triangles."""
    a = rng.uniform(-0.1, 0.1, size=(n, 3, 3))
    b = rng.uniform(-0.1, 0.1, size=(n, 3, 3)) + rng.uniform(-0.15, 0.15, size=(n, 1, 3))
    kinds = rng.integers(0, 4, size=n)
    for i in range(n):
        if kinds[i] == 1:
            # Near-parallel: b is a slightly tilted, offset copy of a.
            b[i] = a[i] + rng.normal(0.0, 0.002, size=(3, 3)) + rng.uniform(-0.05, 0.05, 3)
        elif kinds[i] == 2:
            # Sliver: one vertex almost on the opposite edge.
            a[i, 2] = 0.5 * (a[i, 0] + a[i, 1]) + rng.normal(0.0, 1e-4, 3)
        elif kinds[i] == 3:
            # Degenerate: a collapses to a segment.
            a[i, 2] = a[i, 0] + 0.3 * (a[i, 1] - a[i, 0])
    return a.astype(np.float32), b.astype(np.float32)


def main() -> int:
    """Run the comparison and return a non-zero exit code on any mismatch."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pairs", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--rel-tol", type=float, default=1e-3)
    parser.add_argument("--abs-tol", type=float, default=1e-5)
    args = parser.parse_args()

    wp.init()
    rng = np.random.default_rng(args.seed)
    a, b = random_pairs(rng, args.pairs)
    ref = np.array([reference_distance(a[i], b[i]) for i in range(args.pairs)], np.float32)

    def run(threshold: np.ndarray) -> tuple:
        intersect = wp.zeros(args.pairs, dtype=wp.int32, device="cpu")
        within = wp.zeros(args.pairs, dtype=wp.int32, device="cpu")
        wp.launch(
            _within_kernel,
            dim=args.pairs,
            inputs=[
                wp.array(a.reshape(-1, 3), dtype=wp.vec3, device="cpu"),
                wp.array(b.reshape(-1, 3), dtype=wp.vec3, device="cpu"),
                wp.array(threshold.astype(np.float32), dtype=wp.float32, device="cpu"),
                intersect,
                within,
            ],
            device="cpu",
        )
        return intersect.numpy(), within.numpy()

    tol = np.maximum(ref * args.rel_tol, args.abs_tol).astype(np.float32)
    intersect, within_above = run(ref + tol)
    _, within_below = run(np.maximum(ref - tol, 0.0))

    # Pairs that touch within tolerance are left to wp.intersect_tri_tri.
    disjoint = (intersect == 0) & (ref > 2.0 * tol)
    false_neg = np.flatnonzero(disjoint & (within_above == 0))
    false_pos = np.flatnonzero(disjoint & (within_below == 1))
    print(f"pairs: {args.pairs}, disjoint: {int(disjoint.sum())}")
    print(f"false negatives (missed within ref + tol): {len(false_neg)}")
    print(f"false positives (flagged within ref - tol): {len(false_pos)}")
    for i in list(false_neg[:5]) + list(false_pos[:5]):
        print(f"  pair {i}: ref={ref[i]:.6g}\n    a={a[i].tolist()}\n    b={b[i].tolist()}")
    return 0 if len(false_neg) == 0 and len(false_pos) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
