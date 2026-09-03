"""Controller: emits ISPAction from PipelineState observations.

The Controller does NOT know about ISP internals: it consumes a
PipelineState and a SearchSpace's ConstraintResult, and emits an
`ISPAction` + log_prob + value + entropy for the RL loop. Concrete
subclasses live in `controller/<name>/` — each may pick its own family
of algorithm (RL, CMA-ES, Bayesian Optimization, ...).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn

from pipeline.action import ISPAction
from pipeline.state import PipelineState
from search.constraint import ConstraintResult


@dataclass
class ControllerOutput:
    """Everything a Controller emits at one rollout step.

    action:    ISPAction — consumed by PipelineExecutor.step
    logits:    [B, N_ops] raw selection logits (for policy gradient)
    log_prob:  [B, 1] log pdf of the chosen op (surrogate objective)
    value:     [B, 1] critic estimate
    entropy:   [B, 1] entropy of selection pdf (for exploration penalty)
    pdf:       [B, N_ops] post-softmax + exploration + mask, useful for logging
    """
    action: ISPAction
    logits: torch.Tensor
    log_prob: torch.Tensor
    value: torch.Tensor
    entropy: torch.Tensor
    pdf: torch.Tensor


class Controller(nn.Module, ABC):
    """Base class for anything that emits an ISPAction from a PipelineState."""

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def act(
        self,
        state: PipelineState,
        constraint: ConstraintResult,
        *,
        noise: Optional[torch.Tensor] = None,
    ) -> ControllerOutput:
        """Produce a batched ISPAction for the current state.

        noise: optional [B, 1] uniform noise in [0, 1) for training sampling.
            When None during training, uniform noise is drawn internally.
            When None during eval, argmax selection is used.
        """


__all__ = ["Controller", "ControllerOutput"]
