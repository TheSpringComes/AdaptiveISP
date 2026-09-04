"""Neural chroma-mapping wrapper (Samsung PhotofinishingModule._lut_net).

Converts RGB -> YCbCr, samples the predicted 2D CbCr LUT, and converts
back. Y is passed through untouched by the LUT step; only CbCr are
remapped.
"""
from __future__ import annotations

import torch

from isp.learned.base import NeuralISPOperator
from isp.learned.samsung_modular.backend import DEFAULT_PS_MODEL, get_photofinishing
from isp.registry import register


@register("n_chroma")
class NeuralChroma(NeuralISPOperator):
    short_name = "nCH"
    input_domain = "linear_srgb"
    output_domain = "linear_srgb"
    checkpoint = str(DEFAULT_PS_MODEL)
    runtime_cost = 4.0

    def _forward_neural(self, img: torch.Tensor) -> torch.Tensor:
        ps = get_photofinishing(device=img.device)
        ycbcr = ps.rgb_to_ycbcr(img)
        lut = ps._lut_net(ycbcr)
        cbcr_out = ps._apply_2d_lut_on_cbcr(ycbcr[:, 1:, ...], lut)
        ycbcr_out = torch.cat([ycbcr[:, :1, ...], cbcr_out], dim=1)
        return ps.ycbcr_to_rgb(ycbcr_out).clamp(0.0, 1.0)
