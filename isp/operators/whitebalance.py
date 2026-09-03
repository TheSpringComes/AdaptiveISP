"""Improved white-balance operator. Ports ImprovedWhiteBalanceFilter (short 'W').

Note the two idiosyncrasies preserved verbatim from the original:
- `log_wb_range = 0.5` is hard-coded in the regressor (cfg.wb_range = 1.1 is
  never actually read by the original filter — a known upstream quirk).
- The R-channel logit is masked to zero before exponentiation; only G/B get
  scaled, and the final vector is normalized by luminance so mean gain ≈ 1.
"""
from __future__ import annotations

import numpy as np
import torch

from isp.base import ISPOperator, ParameterSpec, tanh_range
from isp.registry import register

_LOG_WB_RANGE = 0.5


def _wb_regressor(features: torch.Tensor) -> torch.Tensor:
    # Zero the R-channel logit (matches original filter behavior).
    mask = torch.tensor(np.array((0, 1, 1), dtype=np.float32).reshape(1, 3),
                        device=features.device)
    features = features * mask
    scaling = torch.exp(tanh_range(-_LOG_WB_RANGE, _LOG_WB_RANGE)(features))
    lum = 0.27 * scaling[:, 0] + 0.67 * scaling[:, 1] + 0.06 * scaling[:, 2]
    scaling = scaling * (1.0 / (1e-5 + lum)[:, None])
    return scaling


@register("whitebalance")
class WhiteBalanceOperator(ISPOperator):
    short_name = "W"
    spec = ParameterSpec(
        dim=3,
        # Per-channel gain after luminance normalization; the range varies
        # per-channel because of the row-normalization step. These are loose
        # empirical envelopes for SearchSpace prior use — not enforced here.
        low=0.3,
        high=3.0,
        regressor=_wb_regressor,
        description="per-channel gain (R fixed at 1, luminance-normalized)",
    )
    runtime_cost = 1.7

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return img * params[:, :, None, None]
