"""Infinite-ISP unsharp masking — Gaussian low-pass + amplify high-freq.

Reference: https://github.com/10x-Engineers/Infinite-ISP  (module: sharpen)

Distinct from classical `sharpen` (which uses a 3x3 box kernel via
`adjust_sharpness`): this uses a 5x5 Gaussian blur at sigma=1.5, matching
Infinite-ISP's spec, so the two operators live at different frequency
scales in the search space.

    blurred = gaussian(img, sigma=1.5, k=5)
    Output  = clamp(img + amount * (img - blurred), 0, 1)
"""
from __future__ import annotations

import torch

from isp.base import ISPOperator, ParameterSpec, tanh_range
from isp.registry import register
from isp.sharpen import unsharp_mask

_AMOUNT_LO, _AMOUNT_HI = 0.0, 3.0
_SIGMA = 1.5


@register("inf_unsharp")
class InfUnsharp(ISPOperator):
    short_name = "iUSM"
    spec = ParameterSpec(
        dim=1, low=_AMOUNT_LO, high=_AMOUNT_HI,
        regressor=tanh_range(_AMOUNT_LO, _AMOUNT_HI),
        description="unsharp amount (fixed sigma=1.5, 5x5 Gaussian)",
    )
    runtime_cost = 5.0

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        b = img.shape[0]
        sigma = torch.full((b, 1), _SIGMA, device=img.device, dtype=img.dtype)
        amount = params.view(b, 1, 1, 1)
        return unsharp_mask(img, sigma=sigma, amount=amount, kernel_size=(5, 5), clip=True)
