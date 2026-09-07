"""Neural digital-gain wrapper (Samsung PhotofinishingModule._gain_net)."""
from __future__ import annotations

import torch

from isp.learned.base import NeuralISPOperator
from isp.learned.samsung_modular.backend import DEFAULT_PS_MODEL, get_photofinishing
from isp.registry import register


@register("n_gain")
class NeuralGain(NeuralISPOperator):
    short_name = "nGN"
    input_domain = "linear_srgb"
    output_domain = "linear_srgb"
    checkpoint = str(DEFAULT_PS_MODEL)
    runtime_cost = 3.0

    def _forward_neural(self, img: torch.Tensor) -> torch.Tensor:
        ps = get_photofinishing(device=img.device)
        gain = ps._gain_net(img)
        return ps._apply_gain(img, gain).clamp(0.0, 1.0)
