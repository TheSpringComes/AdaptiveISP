"""SSIM + LPIPS batch metrics for the Human Quality task.

Kept intentionally small: two batch helpers + one composite `quality_score`.
Both metrics are wrapped so we can call them once per rollout terminal step
with a `(B, 3, H, W)` prediction and Expert C reference in [0, 1].

- SSIM (structural similarity): higher is better, in [-1, 1].
- LPIPS (learned perceptual): lower is better, ≥ 0. AlexNet backbone by
  default (industry standard, ~6M params, faster than VGG).

Both are lazily instantiated and cached on the target device so a second
call on the same device incurs no reload.
"""
from __future__ import annotations

from typing import Optional

import torch

from pytorch_msssim import ssim as _ssim_fn
import lpips as _lpips_module


_LPIPS_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def _get_lpips(net: str, device: torch.device) -> torch.nn.Module:
    key = (net, str(device))
    if key not in _LPIPS_CACHE:
        model = _lpips_module.LPIPS(net=net, verbose=False).to(device)
        model.eval()
        for p in model.parameters():
            p.requires_grad_(False)
        _LPIPS_CACHE[key] = model
    return _LPIPS_CACHE[key]


def ssim_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """SSIM per-sample. Inputs `(B, 3, H, W)` in `[0, 1]`; returns `(B, 1)`."""
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    per_sample = _ssim_fn(pred, target, data_range=1.0, size_average=False)
    return per_sample.view(-1, 1)


def lpips_batch(pred: torch.Tensor, target: torch.Tensor, net: str = "alex") -> torch.Tensor:
    """LPIPS per-sample. Inputs `(B, 3, H, W)` in `[0, 1]`; returns `(B, 1)`.

    LPIPS internally expects `[-1, 1]` — the wrapper below shifts. We do
    the call under `no_grad` because the network is frozen and we only
    need the scalar distance for the reward.
    """
    model = _get_lpips(net, pred.device)
    pred = pred.clamp(0.0, 1.0) * 2.0 - 1.0
    target = target.clamp(0.0, 1.0) * 2.0 - 1.0
    with torch.no_grad():
        d = model(pred, target)   # (B, 1, 1, 1)
    return d.view(-1, 1)


def quality_score(
    pred: torch.Tensor,
    target: torch.Tensor,
    lambda_ssim: float = 1.0,
    lambda_lpips: float = 1.0,
    lpips_net: str = "alex",
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Composite Q(pred) = λ_ssim * SSIM - λ_lpips * LPIPS. Returns `(B, 1)`.

    The second element is a dict of the raw component tensors, for logging.
    Both raw values live in the same [0, 1]-ish neighborhood (LPIPS-alex on
    natural images is typically 0.05 – 0.5), so the default lambda_ssim =
    lambda_lpips = 1.0 gives them comparable pull. Tune from the config.
    """
    s = ssim_batch(pred, target)
    lp = lpips_batch(pred, target, net=lpips_net)
    q = lambda_ssim * s - lambda_lpips * lp
    return q, {"ssim": s, "lpips": lp, "quality": q}


__all__ = ["ssim_batch", "lpips_batch", "quality_score"]
