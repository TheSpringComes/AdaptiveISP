"""Neural gamma wrapper (Samsung PhotofinishingModule._gamma_net)."""
from __future__ import annotations

import torch

from isp.learned.base import NeuralISPOperator
from isp.learned.samsung_modular.backend import DEFAULT_PS_MODEL, get_photofinishing
from isp.registry import register


@register("n_gamma")
class NeuralGamma(NeuralISPOperator):
    short_name = "nG"
    input_domain = "linear_srgb"
    output_domain = "srgb"
    checkpoint = str(DEFAULT_PS_MODEL)
    runtime_cost = 3.0

    def _forward_neural(self, img: torch.Tensor) -> torch.Tensor:
        ps = get_photofinishing(device=img.device)
        gamma = ps._gamma_net(img)
        return ps._apply_gamma(img.clamp(1e-6, 1.0), gamma).clamp(0.0, 1.0)
