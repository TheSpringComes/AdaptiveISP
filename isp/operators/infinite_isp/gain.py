"""Infinite-ISP digital gain: constant multiplicative gain in [0.5, 2.0].

Reference: https://github.com/10x-Engineers/Infinite-ISP  (module: digital_gain)

Different from classical `exposure` (which is ±3.5 stops via 2^ev): here
the parameter maps directly to a bounded linear gain, matching what
Infinite-ISP calls "digital gain" — a simple sensor-side multiplier.
"""
from __future__ import annotations

import torch

from isp.base import ISPOperator, ParameterSpec, tanh_range
from isp.registry import register

_LO, _HI = 0.5, 2.0


@register("inf_digital_gain")
class InfDigitalGain(ISPOperator):
    short_name = "iDG"
    spec = ParameterSpec(
        dim=1, low=_LO, high=_HI,
        regressor=tanh_range(_LO, _HI, initial=1.0),
        description="linear digital gain (0.5..2.0, initialized at 1.0)",
    )
    runtime_cost = 1.0

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return (img * params.view(-1, 1, 1, 1)).clamp(0.0, 1.0)
