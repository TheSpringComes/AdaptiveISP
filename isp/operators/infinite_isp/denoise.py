"""Infinite-ISP denoise operators: NLM (RGB variant) + EBF (bilateral).

Reference: https://github.com/10x-Engineers/Infinite-ISP  (module: 2d_noise_reduction)

Two distinct denoisers:

- inf_nlm : Non-Local-Means with **RGB** distance metric (`NonLocalMeans`)
  — different from classical `denoise` which uses the luminance-only
  `NonLocalMeansGray`. Same window/patch (11/5).

- inf_ebf : Edge-preserving Bilateral Filter. Small 5x5 spatial window
  (sigma_s=1.5) with a range weight controlled by the RL parameter
  (sigma_r ∈ [0.05, 0.5]). Edge-aware smoothing distinct from NLM's
  patch-based averaging.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from isp.base import ISPOperator, ParameterSpec
from isp.registry import register
from isp.denoise import NonLocalMeans


_EBF_RADIUS = 2                # 5x5 spatial window
_EBF_SIGMA_S = 1.5
_EBF_SIGMA_R_LO, _EBF_SIGMA_R_HI = 0.05, 0.5


@register("inf_nlm")
class InfNLM(ISPOperator):
    short_name = "iNLM"
    spec = ParameterSpec(
        dim=1, low=0.0, high=1.0, regressor=torch.sigmoid,
        description="NLM (RGB metric) strength — distinct from classical NLM (gray)",
    )
    runtime_cost = 12.0

    def __init__(self) -> None:
        super().__init__()
        self.denoise = NonLocalMeans(search_window_size=11, patch_size=5)

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        img = img.clamp(0.0, 1.0)
        return self.denoise(img, params.view(-1, 1, 1, 1))


@register("inf_ebf")
class InfEBF(ISPOperator):
    short_name = "iEBF"
    spec = ParameterSpec(
        dim=1, low=0.0, high=1.0, regressor=torch.sigmoid,
        description="edge-preserving bilateral filter strength (range sigma)",
    )
    runtime_cost = 8.0

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        img = img.clamp(0.0, 1.0)
        b = img.shape[0]
        alpha = params.view(b, 1, 1, 1).clamp(0.0, 1.0)
        sigma_r = (_EBF_SIGMA_R_LO + (_EBF_SIGMA_R_HI - _EBF_SIGMA_R_LO) * alpha)

        # Luminance guide for range weights (edge preservation on brightness).
        y = 0.299 * img[:, 0:1] + 0.587 * img[:, 1:2] + 0.114 * img[:, 2:3]

        accum = torch.zeros_like(img)
        wsum = torch.zeros_like(y)
        for dy in range(-_EBF_RADIUS, _EBF_RADIUS + 1):
            for dx in range(-_EBF_RADIUS, _EBF_RADIUS + 1):
                spatial_w = math.exp(-(dx * dx + dy * dy) / (2.0 * _EBF_SIGMA_S ** 2))
                shifted = torch.roll(img, shifts=(dy, dx), dims=(2, 3))
                shifted_y = torch.roll(y, shifts=(dy, dx), dims=(2, 3))
                range_w = torch.exp(-((y - shifted_y) ** 2) / (2.0 * sigma_r ** 2 + 1e-8))
                w = spatial_w * range_w
                accum = accum + w * shifted
                wsum = wsum + w
        return (accum / (wsum + 1e-8)).clamp(0.0, 1.0)
