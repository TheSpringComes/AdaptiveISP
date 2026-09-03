"""SearchSpace: encodes which actions are valid given the current PipelineState.

V1 uses an Identity Prior — the search space does not constrain anything.
Later versions can plug in constraints (repeat, order, group) through
`search.priors` subclasses.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

from isp.base import ISPOperator
from pipeline.state import PipelineState
from search.constraint import ConstraintResult


class SearchSpace:
    def __init__(self, operators: Mapping[str, ISPOperator], canonical_order: Sequence[str]) -> None:
        self.operators = operators
        self.canonical_order = list(canonical_order)
        self.n_ops = len(self.canonical_order)

    def valid_actions(self, state: PipelineState) -> ConstraintResult:
        """V1 identity Prior: every op valid, stop always allowed, param bounds
        from each operator's ParameterSpec."""
        b = state.batch_size
        device = state.image.device
        op_mask = torch.ones((b, self.n_ops), dtype=torch.bool, device=device)
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
