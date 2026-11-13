# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for the shortest IK chains through waypoint layers."""

from __future__ import annotations

import itertools

import torch

from curobo._src.motion.ik_chain import prune_ik_layers, select_ik_chains


def _chain_length(start: torch.Tensor, chain: torch.Tensor) -> float:
    points = torch.cat((start.view(1, -1), chain))
    return float((points[1:] - points[:-1]).norm(dim=-1).sum())


def _brute_force(start, layers):
    """Lengths of all chains through the layers, shortest first."""
    lengths = []
    for combo in itertools.product(*[range(layer.shape[0]) for layer in layers]):
        chain = torch.stack([layer[i] for layer, i in zip(layers, combo)])
        lengths.append(_chain_length(start, chain))
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
