"""Tone-curve operator (8 knots). Ports ToneFilter (short 'T')."""
from __future__ import annotations

import torch

from isp.base import ISPOperator, ParameterSpec, tanh_range
from isp.registry import register

_CURVE_STEPS = 8                              # config.py cfg.curve_steps
_TONE_LOW, _TONE_HIGH = 0.5, 2.0              # config.py cfg.tone_curve_range


def _tone_regressor(features: torch.Tensor) -> torch.Tensor:
    """features: [B, 8] -> [B, 8] squashed to [0.5, 2]. Apply reshapes internally."""
    return tanh_range(_TONE_LOW, _TONE_HIGH)(features)


@register("tone")
class ToneOperator(ISPOperator):
    short_name = "T"
    spec = ParameterSpec(
        dim=_CURVE_STEPS,
        low=_TONE_LOW,
        high=_TONE_HIGH,
        regressor=_tone_regressor,
        description="tone-curve knots (8)",
    )
    runtime_cost = 2.7

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # Accept either flat [B, 8] or canonical [B, 8, 1, 1, 1].
        if params.dim() == 2:
            params = params[:, :, None, None, None]
        tone_curve = params
        tone_curve_sum = torch.sum(tone_curve, dim=1) + 1e-30
        total = img * 0
        for i in range(_CURVE_STEPS):
            total = total + torch.clip(img - 1.0 * i / _CURVE_STEPS, 0, 1.0 / _CURVE_STEPS) \
                            * params[:, i, :, :, :]
        total = total * (_CURVE_STEPS / tone_curve_sum)
        return total
