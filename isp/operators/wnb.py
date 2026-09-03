"""White-and-Black operator: blend towards luminance. Ports WNBFilter (short 'BW')."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from isp.base import ISPOperator, ParameterSpec, rgb2lum, lerp
from isp.registry import register


@register("wnb")
class WNBOperator(ISPOperator):
    short_name = "BW"
    spec = ParameterSpec(
        dim=1,
        low=0.0,
        high=1.0,
        regressor=F.sigmoid,
        description="RGB-to-BW blend (0..1)",
    )
    runtime_cost = 1.9

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        luminance = rgb2lum(img)
        return lerp(img, luminance, params[:, :, None, None])
