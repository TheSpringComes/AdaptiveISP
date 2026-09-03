"""Contrast operator: luminance-based blend. Ports ContrastFilter (short 'Ct')."""
from __future__ import annotations

import math

import torch

from isp.base import ISPOperator, ParameterSpec, rgb2lum, lerp
from isp.registry import register


@register("contrast")
class ContrastOperator(ISPOperator):
    short_name = "Ct"
    spec = ParameterSpec(
        dim=1,
        low=-1.0,
        high=1.0,
        regressor=torch.tanh,
        description="contrast blend factor (-1..1)",
    )
    runtime_cost = 2.1

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        luminance = torch.clip(rgb2lum(img), 0.0, 1.0)
        contrast_lum = -torch.cos(math.pi * luminance) * 0.5 + 0.5
        contrast_image = img / (luminance + 1e-6) * contrast_lum
        return lerp(img, contrast_image, params[:, :, None, None])
