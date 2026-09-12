"""Canonical Front ISP — fixed, Infinite-ISP-style RAW-linear -> baseline sRGB.

原 `pipeline/backbone.py` 的 `CanonicalBackbone`（V3-A1 阶段 1）迁移而来，
注册为 `front_isp: {type: canonical}`。在 Controller 看到图像之前运行一个
*固定的* ISP，使其从 "reasonable baseline sRGB" 而非 raw-linear camera
空间开始。这是 Infinite-ISP 色彩链的 RGB-native 子集 — 无 demosaic、
无坏点校正：demosaic 由 Input Adapter（`front_isp/raw_adapter.py`，
0.5*Malvar + 0.5*Bilinear）在数据层完成，本链全程工作在非 mosaic
的 3 通道 linear RGB 上。

Stages (all fixed; no learnable params):
    1. Auto white balance      (gray-world; per-batch content-derived)
    2. Color correction matrix (linear-cam -> sRGB; fixed 3x3)
    3. Global tone mapping     (smoothstep; the forward of unprocess's
                                `inverse_smoothstep`)
    4. Gamma compression       (sRGB gamma ≈ 1/2.2)

The AWB stage reuses `_grayworld_gain` from `isp.operators.infinite_isp.awb`
so the algorithm is single-sourced with the RL-selectable
`inf_awb_grayworld` op (that op's alpha=1 case is exactly this stage).

The CCM matrix is `isp.unprocess_np.get_calibrated_cam2rgb()` — the same
calibrated linear-camera -> sRGB matrix that the RAW simulation
(`unprocess_wo_mosaic`) inverts. Semantically matched.

Reference (algorithm specs only, not runtime code):
    https://github.com/10x-Engineers/Infinite-ISP
"""
from __future__ import annotations

import torch
import torch.nn as nn

from front_isp.base import FrontISPBase
from front_isp.registry import register_front_isp
from isp.operators.infinite_isp.awb import _grayworld_gain
from isp.unprocess_np import get_calibrated_cam2rgb


@register_front_isp('canonical')
class CanonicalBackbone(FrontISPBase):
    """Fixed 4-stage Front ISP: AWB (gray-world) -> CCM -> GTM (smoothstep) -> gamma.

    Input:  (B, 3, H, W) float in [0, 1] — simulated linear-camera RGB
    Output: (B, 3, H, W) float in [0, 1] — baseline sRGB

    All stages are non-parametric. 保留类名 `CanonicalBackbone` 以兼容旧的
    `pipeline.CanonicalBackbone` 导入（`pipeline/backbone.py` 现为 re-export）。
    """

    def __init__(self, config=None) -> None:
        super().__init__(config)
        self.gamma = float(self.config.get('gamma', 1.0 / 2.2))
        cam2rgb = torch.as_tensor(get_calibrated_cam2rgb(), dtype=torch.float32)  # (3, 3)
        self.register_buffer("cam2rgb", cam2rgb, persistent=False)

    @staticmethod
    def _awb(img: torch.Tensor) -> torch.Tensor:
        gain = _grayworld_gain(img).view(-1, 3, 1, 1)   # (B, 3, 1, 1)
        return (img * gain).clamp(0.0, 1.0)

    def _ccm(self, img: torch.Tensor) -> torch.Tensor:
        # img @ cam2rgb^T along the channel axis, then clamp back to [0,1].
        b, c, h, w = img.shape
        flat = img.permute(0, 2, 3, 1).reshape(-1, 3)                # (B*H*W, 3)
        out = flat @ self.cam2rgb.T                                   # (B*H*W, 3)
        return out.reshape(b, h, w, 3).permute(0, 3, 1, 2).clamp(0.0, 1.0)

    @staticmethod
    def _smoothstep(img: torch.Tensor) -> torch.Tensor:
        # Hermite smoothstep: 3x^2 - 2x^3. Exact forward of the
        # `inverse_smoothstep` used in isp/unprocess_np.py.
        x = img.clamp(0.0, 1.0)
        return x * x * (3.0 - 2.0 * x)

    def _gamma(self, img: torch.Tensor) -> torch.Tensor:
        return img.clamp(0.0, 1.0).pow(self.gamma)

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        img = self._awb(image)
        img = self._ccm(img)
        img = self._smoothstep(img)
        img = self._gamma(img)
        return img


__all__ = ["CanonicalBackbone"]
