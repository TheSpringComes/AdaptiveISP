"""Gamma operator: `img ^ param`. Ports GammaFilter (short 'G')."""
from __future__ import annotations

import numpy as np
import torch

from isp.base import ISPOperator, ParameterSpec, tanh_range
from isp.registry import register

_GAMMA_RANGE = 3.0  # config.py cfg.gamma_range
_LOG_GAMMA = np.log(_GAMMA_RANGE)


def _gamma_regressor(features: torch.Tensor) -> torch.Tensor:
    return torch.exp(tanh_range(-_LOG_GAMMA, _LOG_GAMMA)(features))


@register("gamma")
class GammaOperator(ISPOperator):
    short_name = "G"
    spec = ParameterSpec(
        dim=1,
        low=1.0 / _GAMMA_RANGE,
        high=_GAMMA_RANGE,
        regressor=_gamma_regressor,
        description="gamma exponent",
    )
    runtime_cost = 2.0

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return torch.pow(torch.clip(img, 0.001), params[:, :, None, None])
