"""Exposure operator: `img * 2^param`. Ports ExposureFilter (short 'E')."""
from __future__ import annotations

import numpy as np
import torch

from isp.base import ISPOperator, ParameterSpec, tanh_range
from isp.registry import register

_EXPOSURE_RANGE = 3.5  # config.py cfg.exposure_range


@register("exposure")
class ExposureOperator(ISPOperator):
    short_name = "E"
    spec = ParameterSpec(
        dim=1,
        low=-_EXPOSURE_RANGE,
        high=_EXPOSURE_RANGE,
        regressor=tanh_range(-_EXPOSURE_RANGE, _EXPOSURE_RANGE, initial=0),
        description="EV shift, stops",
    )
    runtime_cost = 1.7

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return img * torch.exp(params[:, :, None, None] * np.log(2))
