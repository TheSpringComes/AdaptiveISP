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

# pytorch_msssim / lpips 惰性导入：PSNR 与 ΔE 不依赖它们，环境缺包时
# 模块仍可导入（例如 smoke 环境），只在真正调用 SSIM/LPIPS 时报错。
_ssim_fn = None
_lpips_module = None


def _ensure_backends() -> None:
    global _ssim_fn, _lpips_module
    if _ssim_fn is None:
        from pytorch_msssim import ssim as _ssim
        import lpips as _lp
        _ssim_fn, _lpips_module = _ssim, _lp


_LPIPS_CACHE: dict[tuple[str, str], torch.nn.Module] = {}


def _get_lpips(net: str, device: torch.device) -> torch.nn.Module:
    _ensure_backends()
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
    _ensure_backends()
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    per_sample = _ssim_fn(pred, target, data_range=1.0, size_average=False)
    return per_sample.view(-1, 1)


def lpips_batch(pred: torch.Tensor, target: torch.Tensor, net: str = "alex",
                grad: bool = False) -> torch.Tensor:
    """LPIPS per-sample. Inputs `(B, 3, H, W)` in `[0, 1]`; returns `(B, 1)`.

    LPIPS internally expects `[-1, 1]` — the wrapper below shifts. By
    default the call runs under `no_grad` (frozen network, scalar reward);
    pass `grad=True` when LPIPS is a *training* term (V3.1 calibration
    pretraining) — the input graph is preserved while the LPIPS weights
    themselves stay frozen.
    """
    model = _get_lpips(net, pred.device)
    pred = pred.clamp(0.0, 1.0) * 2.0 - 1.0
    target = target.clamp(0.0, 1.0) * 2.0 - 1.0
    if grad:
        d = model(pred, target)   # (B, 1, 1, 1)
    else:
        with torch.no_grad():
            d = model(pred, target)   # (B, 1, 1, 1)
    return d.view(-1, 1)


def psnr_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """PSNR per-sample. Inputs `(B, 3, H, W)` in `[0, 1]`; returns `(B, 1)` dB."""
    pred = pred.clamp(0.0, 1.0)
    target = target.clamp(0.0, 1.0)
    mse = ((pred - target) ** 2).flatten(1).mean(dim=1)
    psnr = 10.0 * torch.log10(1.0 / mse.clamp(min=1e-12))
    return psnr.view(-1, 1)


# sRGB → CIELAB 的标准矩阵/白点（D65）。ΔE76 = Lab 欧氏距离，作为
# V3.1 消融表的颜色保真指标（够用且无需外部依赖）。
_SRGB_TO_XYZ = torch.tensor([
    [0.4124564, 0.3575761, 0.1804375],
    [0.2126729, 0.7151522, 0.0721750],
    [0.0193339, 0.1191920, 0.9503041],
])
_XYZ_WHITE = torch.tensor([0.95047, 1.0, 1.08883])
_DELTA = 6.0 / 29.0


def _srgb_to_lab(x: torch.Tensor) -> torch.Tensor:
    """(B, 3, H, W) sRGB in [0,1] → CIELAB。线性化 + XYZ + f(t) 投影。"""
    lin = torch.where(x <= 0.04045, x / 12.92, ((x.clamp(min=0.0) + 0.055) / 1.055) ** 2.4)
    m = _SRGB_TO_XYZ.to(x.device, x.dtype)
    flat = lin.permute(0, 2, 3, 1) @ m.T
    xyz = flat / _XYZ_WHITE.to(x.device, x.dtype)
    f = torch.where(xyz > _DELTA ** 3, xyz.clamp(min=0.0) ** (1.0 / 3.0),
                    xyz / (3 * _DELTA ** 2) + 4.0 / 29.0)
    return torch.cat([116 * f[..., 1:2] - 16,
                      500 * (f[..., 0:1] - f[..., 1:2]),
                      200 * (f[..., 1:2] - f[..., 2:3])], dim=-1)


def delta_e_batch(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """ΔE76（CIELAB 欧氏距离）per-sample，全图像素均值。返回 `(B, 1)`。"""
    d = _srgb_to_lab(pred.clamp(0.0, 1.0)) - _srgb_to_lab(target.clamp(0.0, 1.0))
    return d.norm(dim=-1).flatten(1).mean(dim=1, keepdim=True)


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


__all__ = ["ssim_batch", "lpips_batch", "psnr_batch", "delta_e_batch", "quality_score"]
