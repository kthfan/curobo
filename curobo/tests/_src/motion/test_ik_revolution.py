# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the full-turn variants of IK solutions."""

from __future__ import annotations

import math

import torch

from curobo._src.motion.ik_revolution import dedup_joint_revolutions, expand_joint_revolutions
from curobo._src.motion.ik_chain import select_ik_chains

TURN = 2.0 * math.pi
DEG = math.pi / 180.0
# Joint 0 spans almost two turns, joint 1 less than one, joint 2 is prismatic.
LOWER = torch.tensor([-TURN + 0.1, -2.0, -10.0])
UPPER = torch.tensor([TURN - 0.1, 2.0, 10.0])
REVOLUTE = torch.tensor([True, True, False])


def test_expand_adds_only_variants_within_the_limits():
    solutions = torch.tensor([[1.0, 0.5, 3.0], [0.0, -0.5, -3.0]])
    out = expand_joint_revolutions(solutions, LOWER, UPPER, REVOLUTE)
    # 1.0 has the variant 1.0 - 2 pi; 0.0 has none, as 2 pi is beyond the limit.
    assert out.shape == (3, 3)
    assert torch.equal(out[:2], solutions)
    assert torch.allclose(out[2], torch.tensor([1.0 - TURN, 0.5, 3.0]))


def test_expand_leaves_narrow_and_prismatic_joints_alone():
    solutions = torch.tensor([[0.0, 1.0, 9.0]])
    out = expand_joint_revolutions(solutions, LOWER, UPPER, REVOLUTE)
    assert torch.equal(out, solutions)
    wide_prismatic = expand_joint_revolutions(
        solutions, LOWER, UPPER, torch.tensor([False, False, False])
    )
    assert torch.equal(wide_prismatic, solutions)


def test_expand_caps_the_number_of_solutions():
    lower, upper = torch.full((4,), -TURN), torch.full((4,), TURN)
    solutions = torch.full((2, 4), 1.0)
    revolute = torch.ones(4, dtype=torch.bool)
    assert expand_joint_revolutions(solutions, lower, upper, revolute).shape[0] == 32
    capped = expand_joint_revolutions(solutions, lower, upper, revolute, max_solutions=6)
    assert capped.shape[0] == 6
    # The variants kept first shift one joint only.
    assert bool((((capped[2:] - 1.0).abs() > 1e-3).sum(dim=-1) == 1).all())


def test_dedup_merges_solutions_a_full_turn_apart():
    solutions = torch.tensor(
        [[1.0, 0.5, 3.0], [1.0 - TURN, 0.5, 3.0], [1.01, 0.5, 3.0], [1.0, 0.5, 3.5]]
    )
    out = dedup_joint_revolutions(solutions, 0.05, LOWER, UPPER, REVOLUTE)
    assert out.shape == (2, 3)
    # Every kept solution is in the turn that starts at the lower limit.
    assert bool((out[:, 0] >= LOWER[0]).all()) and bool((out[:, 0] < LOWER[0] + TURN).all())
    assert torch.allclose(torch.remainder(out[:, 0] - 1.0, TURN), torch.zeros(2), atol=1e-5)


def test_dedup_measures_distance_around_the_circle():
    cut = float(LOWER[0])
    solutions = torch.tensor([[cut + 0.01, 0.0, 0.0], [cut + TURN - 0.01, 0.0, 0.0]])
    assert dedup_joint_revolutions(solutions, 0.05, LOWER, UPPER, REVOLUTE).shape[0] == 1


def test_chain_through_variants_avoids_unwinding_at_a_limit():
    """A joint at 0 turns to 90, 260 and 360 deg, but 360 is beyond its limit.

    The nearest-solution chain turns on to 260 and has to unwind to 0 (520 deg in total); the
    shortest chain turns back to -100 at the second waypoint and reaches 0 from there (380).
    """
    lower, upper, revolute = LOWER[:1], UPPER[:1], REVOLUTE[:1]
    start = torch.zeros(1)
    layers = [
        expand_joint_revolutions(
            dedup_joint_revolutions(torch.tensor([[angle * DEG]]), 0.05, lower, upper, revolute),
            lower,
            upper,
            revolute,
        )
        for angle in (90.0, 260.0, 0.0)
    ]
    chains, cost, _ = select_ik_chains(start, layers, 1)
    assert torch.allclose(chains[0, :, 0], torch.tensor([90.0, -100.0, 0.0]) * DEG, atol=1e-4)
    assert abs(float(cost[0]) - 380.0 * DEG) < 1e-4

    nearest, current = [], start
    for layer in layers:
        current = layer[(layer - current).abs().sum(dim=-1).argmin()]
        nearest.append(float(current))
    turned = sum(abs(b - a) for a, b in zip([0.0] + nearest[:-1], nearest))
    assert abs(turned - 520.0 * DEG) < 1e-3
