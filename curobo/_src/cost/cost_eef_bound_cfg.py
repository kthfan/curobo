# SPDX-FileCopyrightText: Copyright (c) 2023-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Configuration dataclass for :class:`EEFBoundCost`."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Type

import torch

from curobo._src.cost.cost_base_cfg import BaseCostCfg
from curobo._src.types.device_cfg import DeviceCfg

# Import the cost class for the default class_type.  This is done at module
# level to match the pattern used by other cuRobo cost configs.
try:
    from curobo._src.cost.cost_eef_bound import EEFBoundCost
except ImportError:  # pragma: no cover - can happen during partial installs
    EEFBoundCost = None  # type: ignore[misc, assignment]


@dataclass
class EEFBoundCostCfg(BaseCostCfg):
    """Configuration for the EEF bound / corridor cost.

    Inherits :attr:`weight`, :attr:`device_cfg`, :attr:`convert_to_binary`, and
    :attr:`use_grad_input` from :class:`BaseCostCfg`.

    ``tool_frames`` must be a non-empty list of link names by the time the cost is
    constructed. The robot cost manager fills it from the robot's tool frames when it is
    left unset, mirroring :class:`ToolPoseCostCfg`.
    """

    class_type: Type = EEFBoundCost  # type: ignore[assignment]

    #: If ``True``, the configured margins are ignored and tight zero margins
    #: are used for every enabled axis (i.e. the trajectory must stay strictly
    #: inside the corridor).  This is useful for debugging / hard constraints.
    disable_margin: bool = False

    #: Offset added to enabled position margins (m).
    pos_margin_offset: float = 0.0

    #: Offset added to enabled rotation margins (rad).
    rot_margin_offset: float = 0.0

    #: List of link names for which to compute pose costs. Left empty, it is filled from the
    #: robot's tool frames by the cost manager.
    tool_frames: Optional[List[str]] = field(default_factory=list)

    def __post_init__(self):
        super().__post_init__()
        if self.tool_frames:
            self.set_tool_frames(self.tool_frames)

    def set_tool_frames(self, tool_frames: List[str]) -> "EEFBoundCostCfg":
        """Update the list of tool frames.  Returns ``self`` for chaining.

        Args:
            tool_frames: List of tool frames to set.

        Raises:
            ValueError: If ``tool_frames`` is empty or ``None``.
        """
        if tool_frames is None or len(tool_frames) == 0:
            raise ValueError("tool_frames must be a non-empty list of link names")
        self.tool_frames = list(tool_frames)
        # Re-run validation (re-dedupes etc.).
        seen = set()
        deduped: List[str] = []
        for name in self.tool_frames:
            if name not in seen:
                seen.add(name)
                deduped.append(name)
        self.tool_frames = deduped
        return self

    @property
    def num_links(self) -> int:
        """Number of links controlled by this cost."""
        return len(self.tool_frames) if self.tool_frames else 0
