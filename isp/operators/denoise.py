"""Non-local-means denoise operator. Ports DenoiseFilter (short 'NLM')."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from isp.base import ISPOperator, ParameterSpec
from isp.registry import register
from isp.denoise import NonLocalMeansGray


@register("denoise")
class DenoiseOperator(ISPOperator):
    short_name = "NLM"
    spec = ParameterSpec(
        dim=1,
        low=0.0,
        high=1.0,
        regressor=F.sigmoid,
        description="NLM strength (sigmoid-bounded)",
    )
    runtime_cost = 10.0

    def __init__(self) -> None:
        super().__init__()
        # Original: NonLocalMeansGray(search_window_size=11, patch_size=5).
        # Gray mode is ~3x faster than the RGB alternative and matches baseline.
        self.denoise = NonLocalMeansGray(search_window_size=11, patch_size=5)

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        img = torch.clip(img, min=0.0, max=1.0)
        return self.denoise(img, params[:, :, None, None])
