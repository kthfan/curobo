    
# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

"""Unit tests for ToolPoseCost and ToolPoseCostCfg."""

# Third Party
import pytest
import torch

# CuRobo
from curobo._src.cost.cost_tool_pose import ToolPoseCost
from curobo._src.cost.cost_tool_pose_cfg import ToolPoseCostCfg
from curobo._src.cost.tool_pose_criteria import ToolPoseCriteria
from curobo._src.types.device_cfg import DeviceCfg
from curobo._src.types.tool_pose import GoalToolPose, ToolPose
from curobo._src.util.warp import init_warp


@pytest.fixture(scope="module", autouse=True)
def setup_warp():
    """Initialize warp before running any tests in this module."""
    init_warp(quiet=True)


@pytest.fixture(params=["cuda:0"])
def device_cfg(request):
    """Create tensor configuration for GPU only (warp requires CUDA)."""
    device = request.param
    if device.startswith("cuda") and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return DeviceCfg(device=torch.device(device))


class TestToolPoseCostCfg:
    """Test ToolPoseCostCfg class."""

    def test_default_init(self, device_cfg):
        """Test default initialization."""
        cfg = ToolPoseCostCfg(weight=[1.0, 1.0], device_cfg=device_cfg)
        assert cfg.class_type == ToolPoseCost
        assert cfg.tool_frames is None
        assert cfg.use_lie_group is False

    def test_init_with_tool_frames(self, device_cfg):
        """Test initialization with tool_frames."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        assert cfg.tool_frames == tool_frames
        assert cfg.num_links == 2
        assert "ee_link" in cfg.tool_pose_criteria
        assert "gripper_link" in cfg.tool_pose_criteria

    def test_set_tool_frames(self, device_cfg):
        """Test set_tool_frames method."""
        cfg = ToolPoseCostCfg(weight=[1.0, 1.0], device_cfg=device_cfg)
        tool_frames = ["link1", "link2", "link3"]
        cfg.set_tool_frames(tool_frames)
        assert cfg.tool_frames == tool_frames
        assert cfg.num_links == 3

    def test_clone(self, device_cfg):
        """Test clone method."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            use_lie_group=True,
            device_cfg=device_cfg,
        )
        cloned = cfg.clone()
        assert cloned is not cfg
        assert cloned.tool_frames == cfg.tool_frames
        assert cloned.tool_frames is not cfg.tool_frames
        assert cloned.use_lie_group is True
        assert torch.allclose(cloned.weight, cfg.weight)

    def test_rotation_method_property(self, device_cfg):
        """Test rotation_method property."""
        cfg = ToolPoseCostCfg(weight=[1.0, 1.0], use_lie_group=False, device_cfg=device_cfg)
        assert cfg.rotation_method == 0

        cfg2 = ToolPoseCostCfg(weight=[1.0, 1.0], use_lie_group=True, device_cfg=device_cfg)
        assert cfg2.rotation_method == 1


class TestToolPoseCost:
    """Test ToolPoseCost class."""

    def test_init(self, device_cfg):
        """Test ToolPoseCost initialization."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        assert cost is not None
        assert cost.config == cfg
        assert cost.num_links == 1
        assert cost.tool_frames == tool_frames

    def test_setup_batch_tensors(self, device_cfg):
        """Test setup_batch_tensors."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        cost.setup_batch_tensors(batch_size=4, horizon=10)

        num_links = len(tool_frames)
        assert cost._batch_size == 4
        assert cost._horizon == 10
        assert cost._out_distance.shape == (4, 10, 2 * num_links)
        assert cost._out_position_distance.shape == (4, 10, num_links)
        assert cost._out_rotation_distance.shape == (4, 10, num_links)
        assert cost._out_position_gradient.shape == (4, 10, num_links, 3)
        assert cost._out_rotation_gradient.shape == (4, 10, num_links, 4)

    def test_forward_single_link(self, device_cfg):
        """Test forward with a single link."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        batch_size = 4
        horizon = 10
        num_links = 1
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses (shape: batch, horizon, num_links, 3/4)
        current_position = torch.zeros(
            (batch_size, horizon, num_links, 3), **device_cfg.as_torch_dict()
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4), **device_cfg.as_torch_dict()
        )
        current_quaternion[..., 3] = 1.0
        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses (shape: B, H, num_links, num_goalset, 3/4)
        goal_position = torch.zeros((1, 1, num_links, 1, 3), **device_cfg.as_torch_dict())
        goal_quaternion = torch.zeros((1, 1, num_links, 1, 4), **device_cfg.as_torch_dict())
        goal_quaternion[..., 3] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros((batch_size, 1), device=device_cfg.device, dtype=torch.int32)

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        assert cost_val.shape == (batch_size, horizon, 2 * num_links)
        assert linear_dist.shape == (batch_size, horizon, num_links)
        assert angular_dist.shape == (batch_size, horizon, num_links)

    def test_forward_multiple_links(self, device_cfg):
        """Test forward with multiple links."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        batch_size = 4
        horizon = 10
        num_links = 2
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses (shape: batch, horizon, num_links, 3/4)
        current_position = torch.zeros(
            (batch_size, horizon, num_links, 3), **device_cfg.as_torch_dict()
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4), **device_cfg.as_torch_dict()
        )
        current_quaternion[..., 3] = 1.0
        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses (shape: B, H, num_links, num_goalset, 3/4)
        goal_position = torch.zeros((1, 1, num_links, 1, 3), **device_cfg.as_torch_dict())
        goal_quaternion = torch.zeros((1, 1, num_links, 1, 4), **device_cfg.as_torch_dict())
        goal_quaternion[..., 3] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros((batch_size, 1), device=device_cfg.device, dtype=torch.int32)

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        assert cost_val.shape == (batch_size, horizon, 2 * num_links)
        assert linear_dist.shape == (batch_size, horizon, num_links)
        assert angular_dist.shape == (batch_size, horizon, num_links)

    def test_forward_missing_poses_raises_error(self, device_cfg):
        """Test that missing poses raises error."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)
        cost.setup_batch_tensors(batch_size=4, horizon=10)

        with pytest.raises(Exception):
            cost.forward(
                current_tool_poses=None, goal_tool_poses=None, idxs_goal=None
            )

    def test_update_tool_pose_criteria(self, device_cfg):
        """Test update_tool_pose_criteria method."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0], tool_frames=tool_frames, device_cfg=device_cfg
        )
        cost = ToolPoseCost(cfg)

        # Create new criteria
        new_criteria = {
            "ee_link": ToolPoseCriteria(
                device_cfg=device_cfg,
                terminal_pose_axes_weight_factor=[2.0, 2.0, 2.0, 1.0, 1.0, 1.0],
            ),
            "gripper_link": ToolPoseCriteria(
                device_cfg=device_cfg,
                terminal_pose_axes_weight_factor=[1.0, 1.0, 1.0, 0.5, 0.5, 0.5],
            ),
        }

        # Update should not raise
        cost.update_tool_pose_criteria(new_criteria)

    def test_use_lie_group(self, device_cfg):
        """Test with use_lie_group=True."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            use_lie_group=True,
            device_cfg=device_cfg,
        )
        cost = ToolPoseCost(cfg)
        batch_size = 4
        horizon = 10
        num_links = 1
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create poses with offset (shape: batch, horizon, num_links, 3/4)
        current_position = torch.ones(
            (batch_size, horizon, num_links, 3), **device_cfg.as_torch_dict()
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4), **device_cfg.as_torch_dict()
        )
        current_quaternion[..., 3] = 1.0
        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Goal poses (shape: B, H, num_links, num_goalset, 3/4)
        goal_position = torch.zeros((1, 1, num_links, 1, 3), **device_cfg.as_torch_dict())
        goal_quaternion = torch.zeros((1, 1, num_links, 1, 4), **device_cfg.as_torch_dict())
        goal_quaternion[..., 3] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros((batch_size, 1), device=device_cfg.device, dtype=torch.int32)

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        assert cost_val.shape == (batch_size, horizon, 2 * num_links)
        # Terminal cost should be non-zero due to position offset
        assert torch.any(cost_val[:, -1, :] > 0)


class TestToolPoseCostGradients:
    """Test gradient computation for ToolPoseCost."""

    @pytest.fixture
    def gradient_device_cfg(self):
        """Create tensor configuration for gradient tests (requires CUDA)."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return DeviceCfg(device=torch.device("cuda:0"))

    def test_tool_pose_cost_gradient_position(self, gradient_device_cfg):
        """Test that gradients flow correctly through position inputs."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            use_grad_input=True,
            device_cfg=gradient_device_cfg,
        )
        cost = ToolPoseCost(cfg)
        batch_size = 2
        horizon = 3
        num_links = 1
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses with requires_grad
        current_position = torch.randn(
            (batch_size, horizon, num_links, 3),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
            requires_grad=True,
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4), dtype=torch.float32, device=gradient_device_cfg.device
        )
        current_quaternion[..., 3] = 1.0
        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses (fixed, 5D: B, H, L, G, 3/4)
        goal_position = torch.zeros(
            (1, 1, num_links, 1, 3), dtype=torch.float32, device=gradient_device_cfg.device
        )
        goal_quaternion = torch.zeros(
            (1, 1, num_links, 1, 4), dtype=torch.float32, device=gradient_device_cfg.device
        )
        goal_quaternion[..., 3] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros((batch_size, 1), device=gradient_device_cfg.device, dtype=torch.int32)

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        # Sum the cost and backpropagate
        loss = cost_val.sum()
        loss.backward()

        # Check that gradients exist
        assert current_position.grad is not None
        assert current_position.grad.shape == current_position.shape

    def test_tool_pose_cost_gradient_quaternion(self, gradient_device_cfg):
        """Test that gradients flow correctly through quaternion inputs."""
        tool_frames = ["ee_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            use_grad_input=True,
            device_cfg=gradient_device_cfg,
        )
        cost = ToolPoseCost(cfg)
        batch_size = 2
        horizon = 3
        num_links = 1
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses with requires_grad on quaternion
        current_position = torch.randn(
            (batch_size, horizon, num_links, 3),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
            requires_grad=True,
        )
        # Manually set w=1 for identity quaternion (detach and modify)
        with torch.no_grad():
            current_quaternion.data[..., 3] = 1.0

        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses with rotation offset (5D: B, H, L, G, 3/4)
        goal_position = torch.zeros(
            (1, 1, num_links, 1, 3), dtype=torch.float32, device=gradient_device_cfg.device
        )
        # Small rotation around z-axis
        goal_quaternion = torch.tensor(
            [[[[[0.0, 0.0, 0.1, 0.995]]]]], dtype=torch.float32, device=gradient_device_cfg.device
        )
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros((batch_size, 1), device=gradient_device_cfg.device, dtype=torch.int32)

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        # Sum the cost and backpropagate
        loss = cost_val.sum()
        loss.backward()

        # Check that gradients exist for quaternion
        assert current_quaternion.grad is not None
        assert current_quaternion.grad.shape == current_quaternion.shape

    def test_tool_pose_cost_multiple_links_gradient(self, gradient_device_cfg):
        """Test gradients with multiple tool frames."""
        tool_frames = ["ee_link", "gripper_link"]
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=tool_frames,
            use_grad_input=True,
            device_cfg=gradient_device_cfg,
        )
        cost = ToolPoseCost(cfg)
        batch_size = 2
        horizon = 3
        num_links = 2
        cost.setup_batch_tensors(batch_size=batch_size, horizon=horizon)

        # Create current poses with requires_grad
        current_position = torch.randn(
            (batch_size, horizon, num_links, 3),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
            requires_grad=True,
        )
        current_quaternion = torch.zeros(
            (batch_size, horizon, num_links, 4),
            dtype=torch.float32,
            device=gradient_device_cfg.device,
            requires_grad=True,
        )
        with torch.no_grad():
            current_quaternion.data[..., 3] = 1.0

        current_poses = ToolPose(
            tool_frames=tool_frames,
            position=current_position,
            quaternion=current_quaternion,
        )

        # Create goal poses (fixed, 5D: B, H, L, G, 3/4)
        goal_position = torch.zeros(
            (1, 1, num_links, 1, 3), dtype=torch.float32, device=gradient_device_cfg.device
        )
        goal_quaternion = torch.zeros(
            (1, 1, num_links, 1, 4), dtype=torch.float32, device=gradient_device_cfg.device
        )
        goal_quaternion[..., 3] = 1.0
        goal_poses = GoalToolPose(
            tool_frames=tool_frames,
            position=goal_position,
            quaternion=goal_quaternion,
        )

        idxs_goal = torch.zeros((batch_size, 1), device=gradient_device_cfg.device, dtype=torch.int32)

        cost_val, linear_dist, angular_dist, goalset_idx = cost.forward(
            current_tool_poses=current_poses,
            goal_tool_poses=goal_poses,
            idxs_goal=idxs_goal,
        )

        # Sum the cost and backpropagate
        loss = cost_val.sum()
        loss.backward()

        # Check that gradients exist for both position and quaternion
        assert current_position.grad is not None
        assert current_quaternion.grad is not None
        assert current_position.grad.shape == current_position.shape
        assert current_quaternion.grad.shape == current_quaternion.shape


def _quat_mul_wxyz(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    aw, ax, ay, az = a.unbind(-1)
    bw, bx, by, bz = b.unbind(-1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def _quat_exp_wxyz(rotation_vector: torch.Tensor) -> torch.Tensor:
    angle = rotation_vector.norm()
    axis = rotation_vector / angle
    return torch.cat([torch.cos(angle / 2).view(1), torch.sin(angle / 2) * axis])


class TestToolPoseCostRotationGradientScale:
    """The angular gradient kinematics receives must match finite differences of the cost."""

    @pytest.fixture(params=["cpu", "cuda:0"])
    def scale_device_cfg(self, request):
        if request.param.startswith("cuda") and not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        return DeviceCfg(device=torch.device(request.param))

    @staticmethod
    def _evaluate(cost, quaternion, device_cfg):
        """Return the cost and the quaternion gradient ``(w, x, y, z)`` for one tool pose."""
        kwargs = device_cfg.as_torch_dict()
        current = ToolPose(
            tool_frames=["ee_link"],
            position=torch.zeros((1, 1, 1, 3), **kwargs),
            quaternion=quaternion.view(1, 1, 1, 4).to(**kwargs),
        )
        goal_quaternion = torch.zeros((1, 1, 1, 1, 4), **kwargs)
        goal_quaternion[..., 0] = 1.0
        goal = GoalToolPose(
            tool_frames=["ee_link"],
            position=torch.zeros((1, 1, 1, 1, 3), **kwargs),
            quaternion=goal_quaternion,
        )
        idxs_goal = torch.zeros((1, 1), device=device_cfg.device, dtype=torch.int32)
        cost_val, _, _, _ = cost.forward(current, goal, idxs_goal)
        return cost_val.sum().item(), cost._out_rotation_gradient.view(4).cpu().clone()

    @pytest.mark.parametrize("use_lie_group", [False, True])
    @pytest.mark.parametrize(
        "rotation_vector", [[0.3, 0.0, 0.0], [0.5, -0.4, 0.2], [0.0, 2.0, 0.5]]
    )
    def test_received_gradient_matches_finite_difference(
        self, scale_device_cfg, use_lie_group, rotation_vector
    ):
        cfg = ToolPoseCostCfg(
            weight=[1.0, 1.0],
            tool_frames=["ee_link"],
            use_lie_group=use_lie_group,
            device_cfg=scale_device_cfg,
        )
        cost = ToolPoseCost(cfg)
        cost.setup_batch_tensors(batch_size=1, horizon=1)
        quaternion = _quat_exp_wxyz(torch.tensor(rotation_vector))

        _, quaternion_gradient = self._evaluate(cost, quaternion, scale_device_cfg)
        # Kinematics backward reads the angular gradient as 0.5 * vec(q^-1 (x) dq).
        conjugate = quaternion * torch.tensor([1.0, -1.0, -1.0, -1.0])
        received = 0.5 * _quat_mul_wxyz(conjugate, quaternion_gradient)[1:]

        eps = 1.0e-3
        finite_difference = torch.zeros(3)
        for i in range(3):
            step = torch.zeros(3)
            step[i] = eps
            plus, _ = self._evaluate(
                cost, _quat_mul_wxyz(_quat_exp_wxyz(step), quaternion), scale_device_cfg
            )
            minus, _ = self._evaluate(
                cost, _quat_mul_wxyz(_quat_exp_wxyz(-step), quaternion), scale_device_cfg
            )
            finite_difference[i] = (plus - minus) / (2.0 * eps)

        assert torch.allclose(received, finite_difference, atol=2.0e-3, rtol=1.0e-2)

