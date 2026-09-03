"""ISPAction: what the Controller emits at one pipeline step."""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class ISPAction:
    """Batched action from the Controller for ONE pipeline step.

    op_indices: [B] int64. Index into PipelineExecutor.canonical_order; -1 = stop.
    params: [B, max_dim] float. Physical parameters for the chosen op, padded
        with zeros beyond the op's actual `spec.dim`. Padding is ignored by
        the operator.
    is_stop: [B] bool. Convenience; equivalent to `op_indices == -1`.
    """
    op_indices: torch.Tensor
    params: torch.Tensor
    is_stop: torch.Tensor

    def __post_init__(self) -> None:
        assert self.op_indices.dim() == 1
        assert self.params.dim() == 2 and self.params.shape[0] == self.op_indices.shape[0]
        assert self.is_stop.shape == self.op_indices.shape
