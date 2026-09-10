"""CanonicalBackbone: fixed, Infinite-ISP-style RAW-linear -> baseline sRGB.

V3-A1 阶段 1: before the Controller sees an image, run a *fixed* ISP so it
starts from a "reasonable baseline sRGB", not from raw-linear camera space.
This is the RGB-native subset of Infinite-ISP's color chain — no demosaic,
no dead-pixel correction, since the framework operates on non-mosaic
3-channel simulated RAW throughout.

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

from isp.operators.infinite_isp.awb import _grayworld_gain
from isp.unprocess_np import get_calibrated_cam2rgb


class CanonicalBackbone(nn.Module):
    """Fixed 4-stage ISP: AWB (gray-world) -> CCM -> GTM (smoothstep) -> gamma.

    Input:  (B, 3, H, W) float in [0, 1] — simulated linear-camera RGB
    Output: (B, 3, H, W) float in [0, 1] — baseline sRGB

    All stages are non-parametric. `nn.Module` only for `.to(device)` and
    for buffer registration of the CCM matrix.
    """

    def __init__(self, gamma: float = 1.0 / 2.2) -> None:
        super().__init__()
        cam2rgb = torch.as_tensor(get_calibrated_cam2rgb(), dtype=torch.float32)  # (3, 3)
        self.register_buffer("cam2rgb", cam2rgb, persistent=False)
        self.gamma = float(gamma)

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

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        img = self._awb(img)
        img = self._ccm(img)
        img = self._smoothstep(img)
        img = self._gamma(img)
        return img


__all__ = ["CanonicalBackbone"]
