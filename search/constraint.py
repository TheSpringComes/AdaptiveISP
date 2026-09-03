"""ConstraintResult: what SearchSpace tells the Controller."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ConstraintResult:
    """What the SearchSpace tells the Controller.

    op_mask: [B, N_ops] bool. True where the op is allowed for that sample.
    stop_allowed: [B] bool. Whether the stop action is permitted.
    param_bounds: list of (low, high) per op (physical range). A prior that
        narrows a range would override the entries.
    """
    op_mask: torch.Tensor
    stop_allowed: torch.Tensor
    param_bounds: list[tuple[float, float]]
