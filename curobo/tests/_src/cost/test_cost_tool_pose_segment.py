# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Tests for per-segment tool-pose criteria and per-segment tool pose bounds."""

import pytest
import torch

from curobo._src.cost.cost_eef_bound import EEFBoundCost
from curobo._src.cost.cost_eef_bound_cfg import EEFBoundCostCfg
from curobo._src.cost.cost_eef_bound_eval import tool_pose_bounds_violation
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.cost_tool_pose_segment import ToolPoseSegmentTable, native_sample_slots
from curobo._src.cost.tool_pose_bound import ToolPoseBound
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.warp import init_warp

FRAMES = ["a", "b"]


@pytest.fixture(scope="module", autouse=True)
def setup_warp():
    init_warp(quiet=True)


@pytest.fixture(params=["cpu", "cuda:0"])
def device_cfg(request):
    if request.param.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(request.param), dtype=torch.float32)


def _criteria(device_cfg):
    return {
        "a": ToolPoseCriteria.linear_motion("z", device_cfg=device_cfg),
        "b": ToolPoseCriteria.track_pose(non_terminal_scale=0.3, device_cfg=device_cfg),
    }


def _cost(device_cfg, criteria):
    return ToolPoseCost(
        ToolPoseCostCfg(
            weight=[2.0, 3.0],
            tool_frames=FRAMES,
            tool_pose_criteria=criteria,
            device_cfg=device_cfg,
        )
    )


def _random_problem(device_cfg, batch, horizon, generator):
    def unit(shape):
        q = torch.randn(shape, generator=generator)
        return (q / q.norm(dim=-1, keepdim=True)).to(device_cfg.device)

    current = ToolPose(
        FRAMES,
        torch.randn((batch, horizon, 2, 3), generator=generator).to(device_cfg.device),
        unit((batch, horizon, 2, 4)),
    )
    goal = GoalToolPose(
        FRAMES,
        torch.randn((batch, horizon, 2, 1, 3), generator=generator).to(device_cfg.device),
        unit((batch, horizon, 2, 1, 4)),
    )
    return current, goal


def _evaluate(cost, current, goal, routes):
    position = current.position.clone().requires_grad_(True)
    quaternion = current.quaternion.clone().requires_grad_(True)
    out = cost.forward(ToolPose(FRAMES, position, quaternion), goal, routes)
    out[0].sum().backward()
    return [t.detach().clone() for t in out] + [position.grad, quaternion.grad]


def test_native_sample_slots_assign_segments_waypoints_and_rest():
    # Two waypoints at samples 3 and 6 of 9, capacity 4: reach slots start at 4.
    assert native_sample_slots([3, 6], 9, 4) == [0, 0, 0, 4, 1, 1, 5, 5, 5]
    with pytest.raises(ValueError):
        native_sample_slots([3, 3], 9, 4)
    with pytest.raises(ValueError):
        native_sample_slots([3, 6, 7], 9, 2)


def test_one_segment_table_matches_the_terminal_path_kernel(device_cfg):
    """A single segment ending at the last sample reproduces the upstream kernel exactly."""
    horizon = 9
    criteria = _criteria(device_cfg)
    reference, segmented = _cost(device_cfg, criteria), _cost(device_cfg, criteria)
    for cost in (reference, segmented):
        cost.setup_batch_tensors(2, horizon)
    table = ToolPoseSegmentTable(FRAMES, 4, horizon, 1, device_cfg)
    table.set_plan([criteria], [criteria], [horizon - 1])
    segmented.attach_segment_table(table, dense=False)
    current, goal = _random_problem(device_cfg, 2, horizon, torch.Generator().manual_seed(0))
    routes = torch.tensor([[0], [1]], device=device_cfg.device, dtype=torch.int32)
    for expected, actual in zip(
        _evaluate(reference, current, goal, routes), _evaluate(segmented, current, goal, routes)
    ):
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)


def test_segments_use_their_own_criteria(device_cfg):
    """Free segments cost nothing; tracked samples use their segment's criteria."""
    horizon = 9
    tracked = _criteria(device_cfg)
    free = {frame: ToolPoseCriteria(device_cfg=device_cfg) for frame in FRAMES}
    cost = _cost(device_cfg, tracked)
    cost.setup_batch_tensors(1, horizon)
    table = ToolPoseSegmentTable(FRAMES, 4, horizon, 1, device_cfg)
    # Segment 0 (samples 0-2) is free, segment 1 (samples 4-5) tracks; waypoints at 3 and 6.
    table.set_plan([free, tracked], [free, tracked], [3, 6])
    cost.attach_segment_table(table, dense=False)
    current, goal = _random_problem(device_cfg, 1, horizon, torch.Generator().manual_seed(1))
    routes = torch.zeros((1, 1), device=device_cfg.device, dtype=torch.int32)
    distance = cost.forward(current, goal, routes)[0]
    assert torch.all(distance[0, :3] == 0)
    assert torch.all(distance[0, 3:] > 0)
    feasible = cost.get_path_feasibility(*cost.forward(current, goal, routes)[1:3], current)
    # Every reach sample enforces its tolerance; the random poses violate all of them.
    assert not feasible[0, 3] and not feasible[0, 6] and not feasible[0, 8]
    # Path samples do not enforce a tolerance unless their criteria ask for it.
    assert feasible[0, :3].all() and feasible[0, 4:6].all()


def test_dense_samples_follow_their_reference_progress(device_cfg):
    horizon = 11
    table = ToolPoseSegmentTable(FRAMES, 4, horizon, 2, device_cfg)
    criteria = _criteria(device_cfg)
    # Active native samples 2..8; waypoints at 4 and 8.
    table.set_plan([criteria, criteria], [criteria, criteria], [4, 8])
    progress = torch.tensor([[0.0, 0.2, 0.4, 0.6, 0.9, 1.0]], device=device_cfg.device)
    counts = torch.tensor([5], device=device_cfg.device, dtype=torch.int32)
    slots = torch.zeros((1, 6), device=device_cfg.device, dtype=torch.int32)
    table.write_sample_slots(slots, counts, progress)
    # Native coordinates 2, 3.2, 4.4, 5.6, 7.4; the last valid sample (index 4) reaches the goal.
    assert slots[0, :4].tolist() == [0, 0, 1, 1]
    assert slots[0, 4].item() == 4 + 1
    assert table.segment_of_slot(slots)[0, :5].tolist() == [0, 0, 1, 1, 1]


@pytest.mark.parametrize(
    "bound",
    [
        ToolPoseBound(anchor="goal", reference_rotation=[0.8, 0.2, 0.4, 0.4], x=(-0.02, 0.03)),
        ToolPoseBound(anchor="start", reference_axes="base", y=(None, 0.01), z=(-0.3, None)),
        ToolPoseBound(
            anchor="goal",
            tool_rotation=[0.9, 0.3, -0.2, 0.25],
            tilt=(0.5, 1.4),
            yaw=(-0.3, 0.4),
            roll=(2.8, 3.6),
        ),
    ],
)
def test_anchored_at_base_keeps_the_limits(bound):
    generator = torch.Generator().manual_seed(2)
    n = 256
    anchor_position = torch.tensor([0.4, -0.2, 0.3])
    anchor_quaternion = torch.tensor([0.9, 0.1, -0.3, 0.2])
    anchor_quaternion = anchor_quaternion / anchor_quaternion.norm()
    position = anchor_position + 0.05 * torch.randn(n, 3, generator=generator)
    quaternion = torch.randn(n, 4, generator=generator)
    quaternion = quaternion / quaternion.norm(dim=-1, keepdim=True)
    anchors = (anchor_position.expand(n, 3), anchor_quaternion.expand(n, 4))
    expected = tool_pose_bounds_violation(
        [bound],
        position,
        quaternion,
        *anchors,
        anchors[0].unsqueeze(1),
        anchors[1].unsqueeze(1),
    )
    rebased = bound.anchored_at_base(anchor_position.tolist(), anchor_quaternion.tolist())
    assert rebased.anchor.name == "BASE" and rebased.reference_axes == "base"
    actual = tool_pose_bounds_violation([rebased], position, quaternion, None, None, None, None)
    assert expected.amax() > 0 and (expected == 0).any()
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-4)


def test_eef_bound_applies_each_segment_bounds(device_cfg):
    horizon = 9
    cost = EEFBoundCost(
        EEFBoundCostCfg(weight=1.0, device_cfg=device_cfg, tool_frames=["a"], metric=True)
    )
    table = ToolPoseSegmentTable(["a"], 4, horizon, 1, device_cfg)
    free = {"a": ToolPoseCriteria(device_cfg=device_cfg)}
    table.set_plan([free, free], [free, free], [3, 6])
    cost.attach_segment_table(table, dense=False)
    floor = ToolPoseBound.plane((0.0, 0.0, 1.0), max_distance=None, anchor="base")
    ceiling = ToolPoseBound(anchor="base", z=(None, 0.5))
    # Segment 0 (samples 0-3) needs z >= 0, segment 1 (samples 4-6, then the rest) z <= 0.5.
    cost.update_segment_tool_pose_bounds([{"a": floor}, {"a": ceiling}])
    z = torch.tensor([-1.0, 0.2, 0.2, -1.0, 1.0, 0.2, 1.0, 1.0, 0.2], device=device_cfg.device)
    position = torch.zeros((1, horizon, 1, 3), device=device_cfg.device)
    position[..., 2] = z.view(1, horizon, 1)
    quaternion = torch.zeros((1, horizon, 1, 4), device=device_cfg.device)
    quaternion[..., 0] = 1.0
    out = cost.forward(ToolPose(["a"], position, quaternion))[0, :, 0]
    assert (out > 0).tolist() == [True, False, False, True, True, False, True, True, False]
    assert len(cost.tool_pose_bounds["a"]) == 2
    with pytest.raises(ValueError):
        cost.update_tool_pose_bounds({"a": floor})

