# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for the end-effector bound cost."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Type

from curobo._src.cost.cost_base_cfg import BaseCostCfg


@dataclass
class EEFBoundCostCfg(BaseCostCfg):
    """Configuration for :class:`EEFBoundCost`.

    Penalises trajectories that deviate outside an axis-aligned corridor
    (in the goal frame) between the start and goal end-effector poses.

    Args:
        disable_margin: When True the margin offsets are not applied.
        pos_margin_offset: Extra dead-zone margin (metres) added to the
            position bounds on both the start and goal sides.
        rot_margin_offset: Extra dead-zone margin (radians) added to the
            rotation bounds on both sides.
    """

    class_type: Type = None  # filled in __post_init__

    disable_margin: bool = False
    pos_margin_offset: float = 0.0
    rot_margin_offset: float = 0.0

    def __post_init__(self):
        super().__post_init__()
        from curobo._src.cost.cost_eef_bound import EEFBoundCost

        self.class_type = EEFBoundCost