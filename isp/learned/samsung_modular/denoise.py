"""Neural RAW-domain denoise wrapper (Samsung NAFNet, generic_lite)."""
from __future__ import annotations

import torch

from isp.learned.base import NeuralISPOperator
from isp.learned.samsung_modular.backend import DEFAULT_DENOISE_MODEL, get_denoiser
from isp.registry import register


@register("n_denoise")
class NeuralDenoise(NeuralISPOperator):
    short_name = "nDN"
    input_domain = "raw_linear"
    output_domain = "raw_linear"
    checkpoint = str(DEFAULT_DENOISE_MODEL)
    runtime_cost = 20.0

    def _forward_neural(self, img: torch.Tensor) -> torch.Tensor:
        net = get_denoiser(device=img.device)
        return net(img).clamp(0.0, 1.0)
