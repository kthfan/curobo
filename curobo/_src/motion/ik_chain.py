# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shortest chains of IK solutions through the waypoints of a path.

A waypoint planner has a *layer* of IK solutions per waypoint. An *IK chain* takes one solution
from every layer; its length is the sum of the joint-space distances from the start
configuration along the chain. Moving to the nearest solution of the next waypoint every time
does not give the shortest chain: a solution that is slightly farther now can save a long way
back later. The functions here find the globally shortest chains by dynamic programming over
the layers.

How long a chain is depends on the :class:`ChainMetric` that measures every step between two
configurations.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence, Tuple, Union

import torch

from curobo._src.util.logging import log_and_raise


class ChainMetric(str, Enum):
    """Length of a joint-space step ``Δq`` between two configurations of an IK chain.

    A chain's length is the sum of its steps. Distances are in radians for revolute joints and
    meters for prismatic ones.
    """

    #: Euclidean distance ``‖Δq‖₂``. Several joints moving a little count as shorter than one
    #: joint moving a lot.
    EUCLIDEAN = "euclidean"

    #: Total rotation (and travel) of all joints, ``Σ|Δq_j|``.
    TOTAL_ROTATION = "total_rotation"

    #: Time (s) of the step with every joint at its velocity limit, ``max_j |Δq_j| / v_j``: the
    #: slowest joint sets the time. Acceleration is not counted, as the trajectory does not
    #: stop at intermediate waypoints.
    TIME = "time"

    #: Largest motion of one joint, ``max_j |Δq_j|``.
    MAX_NORM = "max_norm"


ChainMetricLike = Union[ChainMetric, str]


def _resolve(
    metric: ChainMetricLike, velocity_limit: Optional[torch.Tensor]
) -> Tuple[ChainMetric, Optional[torch.Tensor]]:
    """Return the metric and the per-joint scale its distances use."""
    metric = ChainMetric(metric)
    if metric != ChainMetric.TIME:
        return metric, None
    if velocity_limit is None:
        log_and_raise("ChainMetric.TIME needs the joint velocity limits")
    return metric, 1.0 / velocity_limit


def pairwise_joint_distance(
    a: torch.Tensor,
    b: torch.Tensor,
    metric: ChainMetricLike = ChainMetric.EUCLIDEAN,
    velocity_limit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Length of the step between every pair of configurations.

    Args:
        a: Configurations ``[A, dof]``.
        b: Configurations ``[B, dof]``.
        metric: How a step is measured.
        velocity_limit: Joint velocity limits ``[dof]``, needed for :attr:`ChainMetric.TIME`.

    Returns:
        Step lengths ``[A, B]``.
    """
    metric, scale = _resolve(metric, velocity_limit)
    if scale is not None:
        a, b = a * scale, b * scale
    p = {
        ChainMetric.EUCLIDEAN: 2.0,
        ChainMetric.TOTAL_ROTATION: 1.0,
        ChainMetric.TIME: float("inf"),
        ChainMetric.MAX_NORM: float("inf"),
    }[metric]
    return torch.cdist(a, b, p=p)


def chain_step_lengths(
    configs: torch.Tensor,
    metric: ChainMetricLike = ChainMetric.EUCLIDEAN,
    velocity_limit: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Length of every step along a sequence of configurations.

    Args:
        configs: Configurations ``[..., N, dof]``.
        metric: How a step is measured.
        velocity_limit: Joint velocity limits ``[dof]``, needed for :attr:`ChainMetric.TIME`.

    Returns:
        Step lengths ``[..., N - 1]``.
    """
    metric, scale = _resolve(metric, velocity_limit)
    delta = (configs[..., 1:, :] - configs[..., :-1, :]).abs()
    if scale is not None:
        delta = delta * scale
    if metric == ChainMetric.EUCLIDEAN:
        return delta.norm(dim=-1)
    if metric == ChainMetric.TOTAL_ROTATION:
        return delta.sum(dim=-1)
    return delta.amax(dim=-1)


def _chain_tables(
    start: torch.Tensor,
    layers: Sequence[torch.Tensor],
    metric: ChainMetricLike,
    velocity_limit: Optional[torch.Tensor],
) -> Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
    """Dynamic programming tables over the layers.

    Returns:
        ``forward[j]``, the length of the shortest chain from the start to every node of layer
        ``j``; ``parents[j]``, the node of layer ``j`` that chain takes before every node of
        layer ``j + 1``; ``backward[j]``, the length of the shortest chain from every node of
        layer ``j`` to the last layer; ``children[j]``, the node of layer ``j + 1`` that chain
        takes after every node of layer ``j``.
    """
    edges = [
        pairwise_joint_distance(a, b, metric, velocity_limit)
        for a, b in zip(layers[:-1], layers[1:])
    ]
    forward = [pairwise_joint_distance(start.view(1, -1), layers[0], metric, velocity_limit)[0]]
    parents = []
    for edge in edges:
        cost, parent = (forward[-1].unsqueeze(1) + edge).min(dim=0)
        forward.append(cost)
        parents.append(parent)
    backward = [torch.zeros_like(forward[-1])]
    children = []
    for edge in reversed(edges):
        cost, child = (edge + backward[0].unsqueeze(0)).min(dim=1)
        backward.insert(0, cost)
        children.insert(0, child)
    return forward, parents, backward, children


def select_ik_chains(
    start: torch.Tensor,
    layers: Sequence[torch.Tensor],
    num_chains: int,
    length_slack: Optional[float] = None,
    metric: ChainMetricLike = ChainMetric.EUCLIDEAN,
    velocity_limit: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Pick IK chains through the waypoint layers by shortest joint-space path.

    Without ``length_slack`` every chain is the shortest one that ends on one IK solution of
    the last waypoint, for the ``num_chains`` solutions with the shortest chains: the chains
    differ widely. With ``length_slack`` the chains are the shortest ones overall (each is the
    shortest chain through some IK solution), and only chains at most ``1 + length_slack``
    times as long as the shortest are returned. Chains are repeated when fewer exist.

    Args:
        start: Start configuration ``[dof]``.
        layers: IK solutions of every waypoint, ``[M_j, dof]`` each.
        num_chains: Number of chains to return.
        length_slack: Longest accepted chain, as a fraction above the shortest one; ``None``
            returns the shortest chain to every last-waypoint solution instead.
        metric: How a step between two configurations is measured.
        velocity_limit: Joint velocity limits ``[dof]``, needed for :attr:`ChainMetric.TIME`.

    Returns:
        Chains ``[num_chains, N, dof]`` in order of length, their lengths ``[num_chains]`` in
        the metric's unit, and the number of distinct chains among them.
    """
    forward, parents, backward, children = _chain_tables(start, layers, metric, velocity_limit)
    n = len(layers)
    device = start.device
    if length_slack is None:
        order = torch.argsort(forward[-1])
        distinct = min(num_chains, order.numel())
        picks = order[torch.arange(num_chains, device=device) % distinct]
        nodes = [picks]
        for parent in reversed(parents):
            nodes.append(parent[nodes[-1]])
        nodes.reverse()
        chains = torch.stack([layer[index] for layer, index in zip(layers, nodes)], dim=1)
        return chains, forward[-1][picks], distinct

    # The shortest chain through every node; a chain appears once per node it passes, so the
    # num_chains * n shortest entries hold at least num_chains distinct chains.
    through = torch.cat([f + b for f, b in zip(forward, backward)])
    layer_of = torch.cat([torch.full_like(f, j, dtype=torch.long) for j, f in enumerate(forward)])
    node_of = torch.cat([torch.arange(f.numel(), device=device) for f in forward])
    count = min(through.numel(), num_chains * n)
    cost, top = torch.topk(through, count, largest=False, sorted=True)
    layer_of, node_of = layer_of[top], node_of[top]
    rows = torch.arange(count, device=device)
    index = torch.zeros(count, n, dtype=torch.long, device=device)
    index[rows, layer_of] = node_of
    for j in range(n - 1, 0, -1):
        before = layer_of >= j
        index[before, j - 1] = parents[j - 1][index[before, j]]
    for j in range(n - 1):
        after = layer_of <= j
        index[after, j + 1] = children[j][index[after, j]]
    # Distinct chains, keeping the order of length.
    _, inverse = torch.unique(index, dim=0, return_inverse=True)
    first = torch.full((int(inverse.max()) + 1,), count, dtype=torch.long, device=device)
    first = first.scatter_reduce(0, inverse, rows, reduce="amin")
    keep = torch.sort(first).values
    index, cost = index[keep], cost[keep]
    eligible = cost <= cost[0] * (1.0 + length_slack) + 1e-6
    distinct = min(num_chains, int(eligible.sum()))
    picks = torch.arange(num_chains, device=device) % distinct
    chains = torch.stack([layers[j][index[picks, j]] for j in range(n)], dim=1)
    return chains, cost[picks], distinct


def prune_ik_layers(
    start: torch.Tensor,
    layers: Sequence[torch.Tensor],
    max_nodes: int,
    metric: ChainMetricLike = ChainMetric.EUCLIDEAN,
    velocity_limit: Optional[torch.Tensor] = None,
) -> List[torch.Tensor]:
    """Keep the IK solutions of every waypoint that lie on the shortest chains.

    Every solution is ranked by the length of the shortest chain through it, from the start
    configuration to the last waypoint. The globally shortest chain is always kept.

    Args:
        start: Start configuration ``[dof]``.
        layers: IK solutions of every waypoint, ``[M_j, dof]`` each.
        max_nodes: Most solutions kept per waypoint.
        metric: How a step between two configurations is measured.
        velocity_limit: Joint velocity limits ``[dof]``, needed for :attr:`ChainMetric.TIME`.

    Returns:
        The kept solutions of every waypoint, ``[min(M_j, max_nodes), dof]`` each, shortest
        chain first.
    """
    forward, _, backward, _ = _chain_tables(start, layers, metric, velocity_limit)
    pruned = []
    for layer, f, b in zip(layers, forward, backward):
        order = torch.argsort(f + b, stable=True)[:max_nodes]
        pruned.append(layer[order])
    return pruned
