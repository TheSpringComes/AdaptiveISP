"""Infinite-ISP LDCI — Local Dynamic Contrast Enhancement.

Reference: https://github.com/10x-Engineers/Infinite-ISP  (module: ldci)

Infinite-ISP's LDCI is a CLAHE variant. A faithful differentiable CLAHE is
expensive; here we use a coarse-scale local-mean subtraction as its
functional approximation:

    local_mean = box_filter(img, 33)                       # coarse tile-scale
    detail     = img - local_mean
    enhanced   = local_mean + detail * (1 + 2*alpha)       # amp detail up to 3x
    Output     = clamp(enhanced, 0, 1)

This is analogous to CLAHE's "boost the middle-frequency component per tile"
effect at a fraction of the compute — coarser scale than `sharpen` (which
uses a 3x3 kernel) so the two are visually distinct in the operator bank.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from isp.base import ISPOperator, ParameterSpec
from isp.registry import register

_KERNEL = 33     # coarse local-mean scale (~1/16 of a 512px side)


def _local_mean(img: torch.Tensor) -> torch.Tensor:
    pad = _KERNEL // 2
    padded = F.pad(img, [pad, pad, pad, pad], mode="reflect")
    return F.avg_pool2d(padded, kernel_size=_KERNEL, stride=1)


@register("inf_ldci")
class InfLDCI(ISPOperator):
    short_name = "iLDCI"
    spec = ParameterSpec(
        dim=1, low=0.0, high=1.0, regressor=torch.sigmoid,
        description="local contrast enhancement strength (CLAHE-like)",
    )
    runtime_cost = 3.5

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        img = img.clamp(0.0, 1.0)
        mean = _local_mean(img)
        detail = img - mean
        amp = 1.0 + 2.0 * params.view(-1, 1, 1, 1).clamp(0.0, 1.0)
        return (mean + detail * amp).clamp(0.0, 1.0)
