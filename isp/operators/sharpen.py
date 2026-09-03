"""Sharpen operator (adjust_sharpness). Ports SharpenFilter (short 'Shr')."""
from __future__ import annotations

import torch

from isp.base import ISPOperator, ParameterSpec, tanh_range
from isp.registry import register
from isp.sharpen import adjust_sharpness

_SHARP_LOW, _SHARP_HIGH = 0.0, 10.0  # config.py cfg.sharpen_range


@register("sharpen")
class SharpenOperator(ISPOperator):
    short_name = "Shr"
    spec = ParameterSpec(
        dim=1,
        low=_SHARP_LOW,
        high=_SHARP_HIGH,
        regressor=tanh_range(_SHARP_LOW, _SHARP_HIGH),
        description="sharpen amount",
    )
    runtime_cost = 6.3

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return adjust_sharpness(img, params[:, :, None, None])
