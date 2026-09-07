"""Infinite-ISP AWB operators — 3 illuminant estimators.

Reference: https://github.com/10x-Engineers/Infinite-ISP  (module: auto_white_balance)

Each op estimates a green-anchored per-channel gain from an RGB image
in [0,1] and blends toward the balanced result by alpha:

    Output = img * ( (1-alpha) * 1 + alpha * gain )

alpha ∈ [0,1] is the RL-controlled strength; alpha=0 is passthrough,
alpha=1 applies the full estimator.

Three estimators:
- Grey World         : gain = mean(G) / mean(C)         (all pixels)
- Norm-2 (Shades of Grey p=2) : gain = ||G||_2 / ||C||_2  (all pixels)
- PCA-based (Cheng et al. 2014, simplified) : gain from the mean of the top
  3.5% brightest pixels — captures the dominant illuminant direction.

All three are pure Torch, differentiable in alpha, and run on GPU.
"""
from __future__ import annotations

import torch

from isp.base import ISPOperator, ParameterSpec
from isp.registry import register


def _apply_gain(img: torch.Tensor, gain: torch.Tensor, alpha: torch.Tensor) -> torch.Tensor:
    """img (B,3,H,W) * ((1-alpha) + alpha*gain).  gain (B,3), alpha (B,1)."""
    a = alpha.view(-1, 1, 1, 1).clamp(0.0, 1.0)
    g = gain.view(-1, 3, 1, 1)
    return (img * ((1.0 - a) + a * g)).clamp(0.0, 1.0)


def _grayworld_gain(img: torch.Tensor) -> torch.Tensor:
    # mean per channel across H,W; then anchor to green.
    mu = img.mean(dim=(2, 3))                       # (B, 3)
    return mu[:, 1:2] / (mu + 1e-6)                 # (B, 3)


def _norm2_gain(img: torch.Tensor) -> torch.Tensor:
    # Minkowski p=2 (Shades of Grey with p=2). Robust to a few bright pixels.
    n2 = torch.sqrt((img ** 2).mean(dim=(2, 3)) + 1e-12)   # (B, 3)
    return n2[:, 1:2] / (n2 + 1e-6)


def _pca_gain(img: torch.Tensor, top_frac: float = 0.035) -> torch.Tensor:
    # Simplified Cheng-2014: take the top-3.5% brightest pixels and use their
    # mean color as the illuminant estimate. Averaging on the bright tail
    # picks out the illuminant far better than the whole-image mean.
    b, _, h, w = img.shape
    lum = img.mean(dim=1).reshape(b, -1)                    # (B, N)
    n_top = max(1, int(lum.shape[1] * top_frac))
    _, idx = torch.topk(lum, k=n_top, dim=1)                # (B, K)
    flat = img.reshape(b, 3, -1)                            # (B, 3, N)
    idx_e = idx.unsqueeze(1).expand(-1, 3, -1)              # (B, 3, K)
    top = torch.gather(flat, dim=2, index=idx_e)            # (B, 3, K)
    illum = top.mean(dim=2)                                 # (B, 3)
    return illum[:, 1:2] / (illum + 1e-6)


@register("inf_awb_grayworld")
class InfAwbGrayworld(ISPOperator):
    short_name = "iGW"
    spec = ParameterSpec(
        dim=1, low=0.0, high=1.0, regressor=torch.sigmoid,
        description="grey-world AWB strength",
    )
    runtime_cost = 1.5

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return _apply_gain(img, _grayworld_gain(img), params)


@register("inf_awb_norm2")
class InfAwbNorm2(ISPOperator):
    short_name = "iN2"
    spec = ParameterSpec(
        dim=1, low=0.0, high=1.0, regressor=torch.sigmoid,
        description="shades-of-grey (p=2) AWB strength",
    )
    runtime_cost = 1.5

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return _apply_gain(img, _norm2_gain(img), params)


@register("inf_awb_pca")
class InfAwbPca(ISPOperator):
    short_name = "iPCA"
    spec = ParameterSpec(
        dim=1, low=0.0, high=1.0, regressor=torch.sigmoid,
        description="PCA-based AWB (bright-pixel mean) strength",
    )
    runtime_cost = 3.0

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return _apply_gain(img, _pca_gain(img), params)
