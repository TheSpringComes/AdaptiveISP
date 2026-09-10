"""SearchSpace: encodes which actions are valid given the current PipelineState.

The base search space is identity (all ops always valid). Additional priors
— repeat / order / group / etc — are provided at construction and composed
in `valid_actions`: each prior returns a `[B, N_ops]` bool mask, and all
masks are ANDed with the base.

An empty prior list keeps identity behavior — exactly the pre-A2 (V2)
semantics — so wiring priors is fully opt-in from config.
"""
from __future__ import annotations

from typing import Callable, Mapping, Optional, Sequence

import torch

from isp.base import ISPOperator
from pipeline.state import PipelineState
from search.constraint import ConstraintResult


# A prior is any callable  (state, canonical_order) -> [B, N_ops] bool mask.
Prior = Callable[[PipelineState, Sequence[str]], torch.Tensor]


class SearchSpace:
    def __init__(
        self,
        operators: Mapping[str, ISPOperator],
        canonical_order: Sequence[str],
        priors: Optional[Sequence[Prior]] = None,
    ) -> None:
        self.operators = operators
        self.canonical_order = list(canonical_order)
        self.n_ops = len(self.canonical_order)
        self.priors: list[Prior] = list(priors) if priors else []

    def valid_actions(self, state: PipelineState) -> ConstraintResult:
        """Compose identity base mask with each prior's sub-mask (AND).

        `param_bounds` still comes straight from each operator's
        `ParameterSpec.low/high` — priors here only touch op selection,
        not parameter ranges. A future prior that narrows a parameter
        range would live in its own hook.
        """
        b = state.batch_size
        device = state.image.device
        op_mask = torch.ones((b, self.n_ops), dtype=torch.bool, device=device)
        for prior in self.priors:
            op_mask = op_mask & prior(state, self.canonical_order)
        stop_allowed = torch.ones(b, dtype=torch.bool, device=device)
        param_bounds = [
            (float(self.operators[n].spec.low) if isinstance(self.operators[n].spec.low, (int, float))
             else self.operators[n].spec.low,
             float(self.operators[n].spec.high) if isinstance(self.operators[n].spec.high, (int, float))
             else self.operators[n].spec.high)
            for n in self.canonical_order
        ]
        return ConstraintResult(
            op_mask=op_mask,
            stop_allowed=stop_allowed,
            param_bounds=param_bounds,
        )
