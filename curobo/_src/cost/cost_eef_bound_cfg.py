# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Configuration for :class:`~curobo._src.cost.cost_eef_bound.EEFBoundCost`."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Type

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.cost.cost_eef_bound import EEFBoundCost
from curobo._src.util.logging import log_and_raise


@dataclass
class EEFBoundCostCfg(BaseCostCfg):
    """Configuration for the tool-frame bound cost.

    The bounds themselves are runtime data: set them per link through
    :attr:`ToolPoseCriteria.bound <curobo._src.cost.tool_pose_criteria.ToolPoseCriteria.bound>`.
    ``weight`` is a scalar or one value per tool frame.
    """

    class_type: Type[EEFBoundCost] = EEFBoundCost

    #: Tool frames the cost applies to. Left unset, the cost manager fills it with the robot's
    #: tool frames.
    tool_frames: Optional[List[str]] = None

    #: ``False`` (optimization cost): each interval side is shrunk inward by the bound margin so
    #: paths stay inside the user's bounds. ``True`` (metrics constraint deciding feasibility):
    #: the user's bounds, with the tolerance only on sides the cost could not shrink.
    metric: bool = False

    def set_tool_frames(self, tool_frames: List[str]) -> None:
        """Set the tool frames the cost applies to.

        Args:
            tool_frames: Non-empty list of unique link names.
        """
        if not tool_frames:
            log_and_raise("EEFBoundCostCfg: tool_frames must be a non-empty list")
        if len(set(tool_frames)) != len(tool_frames):
            log_and_raise(f"EEFBoundCostCfg: tool_frames must be unique, got {tool_frames}")
        self.tool_frames = list(tool_frames)
