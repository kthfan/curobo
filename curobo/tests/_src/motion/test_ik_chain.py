# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shortest IK chains through waypoint layers."""

from __future__ import annotations

import itertools

import pytest
import torch

from curobo._src.motion.ik_chain import (
    ChainMetric,
    chain_step_lengths,
    pairwise_joint_distance,
    prune_ik_layers,
    select_ik_chains,
)

VELOCITY = torch.tensor([1.0, 2.0, 4.0])


def _step(delta: torch.Tensor, metric: ChainMetric) -> float:
    """Reference length of one step, written out per metric."""
    delta = delta.abs()
    if metric == ChainMetric.EUCLIDEAN:
        return float(delta.pow(2).sum().sqrt())
    if metric == ChainMetric.TOTAL_ROTATION:
        return float(delta.sum())
    if metric == ChainMetric.TIME:
        return float((delta / VELOCITY).max())
    return float(delta.max())


def _chain_length(start, chain, metric=ChainMetric.EUCLIDEAN) -> float:
    points = torch.cat((start.view(1, -1), chain))
    return sum(_step(b - a, metric) for a, b in zip(points[:-1], points[1:]))


def _brute_force(start, layers, metric=ChainMetric.EUCLIDEAN):
    """Lengths of all chains through the layers, shortest first."""
    lengths = []
    for combo in itertools.product(*[range(layer.shape[0]) for layer in layers]):
        chain = torch.stack([layer[i] for layer, i in zip(layers, combo)])
        lengths.append(_chain_length(start, chain, metric))
    return sorted(lengths)


def _random_layers(generator, dof=3):
    count = int(torch.randint(1, 5, (1,), generator=generator))
    return [
        torch.randn(int(torch.randint(1, 6, (1,), generator=generator)), dof, generator=generator)
        for _ in range(count)
    ]


def test_shortest_chain_to_every_last_solution():
    start = torch.zeros(1)
    layers = [torch.tensor([[1.0], [-1.0]]), torch.tensor([[1.5], [-3.0], [0.9]])]
    chains, cost, distinct = select_ik_chains(start, layers, num_chains=4)
    # Ends sorted by path length: 0.9 (via 1.0, 1.1), 1.5 (1.5), -3.0 (via -1.0, 3.0).
    expected = torch.tensor([[1.0, 0.9], [1.0, 1.5], [-1.0, -3.0], [1.0, 0.9]])
    torch.testing.assert_close(chains[:, :, 0], expected)
    torch.testing.assert_close(cost, torch.tensor([1.1, 1.5, 3.0, 1.1]))
    assert distinct == 3


def test_a_nearer_solution_now_can_cost_a_long_way_back_later():
    """The nearest solution of the first waypoint leads away from the only solution of the
    second one; the shortest chain takes the farther solution first."""
    start = torch.zeros(1)
    layers = [torch.tensor([[1.0], [-1.2]]), torch.tensor([[-3.0]])]
    chains, cost, _ = select_ik_chains(start, layers, num_chains=1, length_slack=0.0)
    torch.testing.assert_close(chains[0, :, 0], torch.tensor([-1.2, -3.0]))
    torch.testing.assert_close(cost, torch.tensor([3.0]))


def test_chains_within_the_slack_are_the_shortest_overall():
    generator = torch.Generator().manual_seed(0)
    for _ in range(30):
        layers = _random_layers(generator)
        start = torch.randn(3, generator=generator)
        lengths = _brute_force(start, layers)
        chains, cost, distinct = select_ik_chains(start, layers, 4, length_slack=0.25)
        assert abs(float(cost[0]) - lengths[0]) < 1e-4
        for chain, length in zip(chains, cost):
            assert abs(_chain_length(start, chain) - float(length)) < 1e-4
        assert bool((cost <= lengths[0] * 1.25 + 1e-4).all())
        assert torch.unique(chains[:distinct], dim=0).shape[0] == distinct
        # Chains repeat in order when fewer than requested are within the slack.
        torch.testing.assert_close(chains[distinct:], chains[: 4 - distinct])


def test_without_slack_the_first_chain_is_the_shortest():
    generator = torch.Generator().manual_seed(1)
    for _ in range(30):
        layers = _random_layers(generator)
        start = torch.randn(3, generator=generator)
        _, cost, _ = select_ik_chains(start, layers, 3)
        assert abs(float(cost[0]) - _brute_force(start, layers)[0]) < 1e-4


def test_pruned_layers_keep_the_shortest_chain():
    generator = torch.Generator().manual_seed(2)
    for _ in range(30):
        layers = _random_layers(generator)
        start = torch.randn(3, generator=generator)
        pruned = prune_ik_layers(start, layers, max_nodes=2)
        assert all(p.shape[0] == min(2, layer.shape[0]) for p, layer in zip(pruned, layers))
        _, cost, _ = select_ik_chains(start, pruned, 1, length_slack=0.0)
        assert abs(float(cost[0]) - _brute_force(start, layers)[0]) < 1e-4
        # The first kept solution of every waypoint is on the shortest chain.
        first = torch.stack([p[0] for p in pruned])
        assert abs(_chain_length(start, first) - float(cost[0])) < 1e-4


@pytest.mark.parametrize("metric", list(ChainMetric))
def test_step_lengths_follow_the_metric(metric):
    generator = torch.Generator().manual_seed(3)
    a, b = torch.randn(4, 3, generator=generator), torch.randn(5, 3, generator=generator)
    pairwise = pairwise_joint_distance(a, b, metric, VELOCITY)
    expected = torch.tensor([[_step(y - x, metric) for y in b] for x in a])
    torch.testing.assert_close(pairwise, expected)
    steps = chain_step_lengths(a, metric.value, VELOCITY)
    torch.testing.assert_close(steps, torch.tensor([_step(y - x, metric) for x, y in zip(a, a[1:])]))


@pytest.mark.parametrize("metric", list(ChainMetric))
def test_chains_are_the_shortest_in_every_metric(metric):
    generator = torch.Generator().manual_seed(4)
    for _ in range(20):
        layers = _random_layers(generator)
        start = torch.randn(3, generator=generator)
        lengths = _brute_force(start, layers, metric)
        chains, cost, _ = select_ik_chains(
            start, layers, 3, length_slack=0.2, metric=metric, velocity_limit=VELOCITY
        )
        assert abs(float(cost[0]) - lengths[0]) < 1e-4
        for chain, length in zip(chains, cost):
            assert abs(_chain_length(start, chain, metric) - float(length)) < 1e-4
        pruned = prune_ik_layers(start, layers, 1, metric, VELOCITY)
        assert abs(_chain_length(start, torch.stack([p[0] for p in pruned]), metric) - lengths[0]) < 1e-4


def test_metrics_choose_different_chains():
    """Reaching (1, 1, 1): moving every joint by 1 (A) or the first joint by 1.5 (B)."""
    start = torch.zeros(3)
    layers = [torch.tensor([[1.0, 1.0, 1.0], [1.5, 0.0, 0.0]])]
    chosen = {
        metric: select_ik_chains(start, layers, 1, 0.0, metric, VELOCITY)[0][0, 0, 0].item()
        for metric in ChainMetric
    }
    # A: L2 1.73, L1 3, time 1 (joint 0 at 1 rad/s), max 1. B: L2 1.5, L1 1.5, time 1.5, max 1.5.
    assert chosen == {
        ChainMetric.EUCLIDEAN: 1.5,
        ChainMetric.TOTAL_ROTATION: 1.5,
        ChainMetric.TIME: 1.0,
        ChainMetric.MAX_NORM: 1.0,
    }


def test_time_needs_velocity_limits():
    with pytest.raises(Exception):
        pairwise_joint_distance(torch.zeros(1, 3), torch.ones(1, 3), "time")
    with pytest.raises(ValueError):
        pairwise_joint_distance(torch.zeros(1, 3), torch.ones(1, 3), "manhattan")
