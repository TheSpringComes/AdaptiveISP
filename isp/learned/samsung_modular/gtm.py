"""Neural global tone-mapping wrapper (Samsung PhotofinishingModule._gtm_net)."""
from __future__ import annotations

import torch

from isp.learned.base import NeuralISPOperator
from isp.learned.samsung_modular.backend import DEFAULT_PS_MODEL, get_photofinishing
from isp.registry import register


@register("n_gtm")
class NeuralGTM(NeuralISPOperator):
    short_name = "nTM"
    input_domain = "linear_srgb"
    output_domain = "linear_srgb"
    checkpoint = str(DEFAULT_PS_MODEL)
    runtime_cost = 4.0

    def _forward_neural(self, img: torch.Tensor) -> torch.Tensor:
        ps = get_photofinishing(device=img.device)
        params = ps._gtm_net(img)
        return ps._apply_gtm(img, params).clamp(0.0, 1.0)
