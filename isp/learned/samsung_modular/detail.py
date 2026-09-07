"""Neural detail-enhancement wrapper (Samsung NAFNet, enhancement-style-0)."""
from __future__ import annotations

import torch

from isp.learned.base import NeuralISPOperator
from isp.learned.samsung_modular.backend import DEFAULT_ENHANCE_MODEL, get_detail
from isp.registry import register


@register("n_detail")
class NeuralDetail(NeuralISPOperator):
    short_name = "nDT"
    input_domain = "srgb"
    output_domain = "srgb"
    checkpoint = str(DEFAULT_ENHANCE_MODEL)
    runtime_cost = 20.0

    def _forward_neural(self, img: torch.Tensor) -> torch.Tensor:
        net = get_detail(device=img.device)
        return net(img).clamp(0.0, 1.0)
