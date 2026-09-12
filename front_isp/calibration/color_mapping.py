"""可微色彩变换 — Calibration 前向的每一步（V3.1 §1）。

变换顺序（Demosaic 固定在数据层，见 module.py 的说明）：

    WB / Channel Gain  →  CCM + Bias  →  Base Tone (Gamma)

全部支持梯度训练：
  - WB 增益以 `log_gain` 参数化（`gain = exp(log_gain) > 0`，初始 0 → 增益 1）
  - CCM 是自由 3×3 矩阵 + 自由 3 维 bias
  - Tone 以 `log_gamma` 参数化（`gamma = exp(log_gamma)`，初始 0 → gamma 1）

输入统一为 `(B, 3, H, W)`；参数按 batch 元素给出（camera-specific 行），
广播到像素维。
"""
from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class CameraParams:
    """单次前向所用的一组标定参数（batch 维收集后）。

    wb_log:   (B, 3)      白平衡增益的 log
    ccm:      (B, 3, 3)   色彩校正矩阵
    bias:     (B, 3)      CCM 后的 RGB 偏置
    log_gamma:(B, 1)       tone gamma 的 log
    """

    wb_log: torch.Tensor
    ccm: torch.Tensor
    bias: torch.Tensor
    log_gamma: torch.Tensor


def apply_wb(x: torch.Tensor, wb_log: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) × exp(wb_log (B,3))。增益恒正。"""
    gain = torch.exp(wb_log)
    return x * gain[:, :, None, None]


def apply_ccm_bias(x: torch.Tensor, ccm: torch.Tensor, bias: torch.Tensor) -> torch.Tensor:
    """(B,3,H,W) 逐样本做 `out_c = Σ_k M[c,k]·x_k + b[c]`。"""
    flat = x.permute(0, 2, 3, 1)                          # (B, H, W, 3)
    out = torch.einsum('bhwc,bkc->bhwk', flat, ccm) + bias[:, None, None, :]
    return out.permute(0, 3, 1, 2)


def apply_tone(x: torch.Tensor, log_gamma: torch.Tensor) -> torch.Tensor:
    """逐样本 `x^gamma`（gamma = exp(log_gamma)）。x 裁到 ≥0 使 pow 可微。"""
    gamma = torch.exp(log_gamma)[:, :, None, None]        # (B, 1, 1, 1)
    return x.clamp(min=0.0).pow(gamma)


def apply_calibration(x: torch.Tensor, p: CameraParams) -> torch.Tensor:
    """完整标定链：WB → CCM+Bias → Tone。输出未做 [0,1] 裁剪（由调用方决定，
    训练时保留端点外梯度更友好，推理时 clamp）。"""
    x = apply_wb(x, p.wb_log)
    x = apply_ccm_bias(x, p.ccm, p.bias)
    x = apply_tone(x, p.log_gamma)
    return x


__all__ = ["CameraParams", "apply_wb", "apply_ccm_bias", "apply_tone", "apply_calibration"]
