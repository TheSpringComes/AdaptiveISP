"""Infinite-ISP saturation — multiplicative HSV saturation gain.

Reference: https://github.com/10x-Engineers/Infinite-ISP  (module: color_correction / saturation)

Distinct from classical `saturation` (which uses a value-dependent HSV
boost blended with the original). Here we simply multiply the HSV
saturation channel by a bounded scalar gain:

    hsv[:, 1] *= gain     with gain ∈ [0.5, 2.0]

Simple, monotonic, and different in character from the S+ boost above.
"""
from __future__ import annotations

import torch

from isp.base import ISPOperator, ParameterSpec, tanh_range, rgb2hsv, hsv2rgb
from isp.registry import register

_LO, _HI = 0.5, 2.0


@register("inf_saturation")
class InfSaturation(ISPOperator):
    short_name = "iSat"
    spec = ParameterSpec(
        dim=1, low=_LO, high=_HI,
        regressor=tanh_range(_LO, _HI, initial=1.0),
        description="HSV saturation multiplicative gain (0.5..2.0)",
    )
    runtime_cost = 1.7

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        img = img.clamp(0.0, 1.0)
        hsv = rgb2hsv(img)
        gain = params.view(-1, 1, 1, 1)
        s = (hsv[:, 1:2] * gain).clamp(0.0, 1.0)
        hsv1 = torch.cat([hsv[:, 0:1], s, hsv[:, 2:3]], dim=1)
        return hsv2rgb(hsv1).clamp(0.0, 1.0)
