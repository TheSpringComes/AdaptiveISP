"""Saturation-plus operator: HSV boost then blend. Ports SaturationPlusFilter (short 'S+')."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from isp.base import ISPOperator, ParameterSpec, rgb2hsv, hsv2rgb
from isp.registry import register


@register("saturation")
class SaturationOperator(ISPOperator):
    short_name = "S+"
    spec = ParameterSpec(
        dim=1,
        low=0.0,
        high=1.0,
        regressor=F.sigmoid,
        description="saturation-plus blend (0..1)",
    )
    runtime_cost = 2.0

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        img = torch.clip(img, min=0.0, max=1.0)
        hsv = rgb2hsv(img)
        s = hsv[:, 1:2, :, :]
        v = hsv[:, 2:3, :, :]
        enhanced_s = s + (1 - s) * (0.5 - torch.abs(0.5 - v)) * 0.8
        hsv1 = torch.cat([hsv[:, 0:1, :, :], enhanced_s, hsv[:, 2:, :, :]], dim=1)
        full_color = hsv2rgb(hsv1)

        p = params[:, :, None, None]
        return img * (1.0 - p) + full_color * p
