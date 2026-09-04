"""Base class + shared regressor for neural ISP operators.

Every neural operator follows the same contract:

    Output = x + alpha * (F(x) - x)      with alpha in [0, 1]

Subclasses implement `_forward_neural(img) -> F(img)` and inherit
`apply(img, params)` from `NeuralISPOperator`. The backend network is
frozen (`eval()` + `requires_grad_(False)`) and inference runs inside
`torch.no_grad()` — gradients w.r.t. alpha still flow through the blend.
"""
from __future__ import annotations

from typing import ClassVar

import torch
import torch.nn.functional as F

from isp.base import ISPOperator, ParameterSpec


ALPHA_SPEC = ParameterSpec(
    dim=1,
    low=0.0,
    high=1.0,
    regressor=torch.sigmoid,
    description="neural op strength alpha",
)


class NeuralISPOperator(ISPOperator):
    """Base for neural operators. Subclasses set `spec = ALPHA_SPEC` (or
    an override) and implement `_forward_neural`. Domain metadata is
    advisory — SearchSpace may filter on it but `apply` does not enforce.
    """

    input_domain: ClassVar[str] = "rgb01"
    output_domain: ClassVar[str] = "rgb01"
    implementation: ClassVar[str] = "neural"
    checkpoint: ClassVar[str] = ""
    spec: ClassVar[ParameterSpec] = ALPHA_SPEC

    def _forward_neural(self, img: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        # params: (B, 1) with alpha in [0,1]. Broadcast to (B,1,1,1).
        alpha = params.view(-1, 1, 1, 1).clamp(0.0, 1.0)
        img_c = img.clamp(0.0, 1.0)
        with torch.no_grad():
            fx = self._forward_neural(img_c)
        # F(x) is frozen; blending on alpha keeps gradient w.r.t. alpha alive.
        return img_c + alpha * (fx - img_c)


__all__ = ["NeuralISPOperator", "ALPHA_SPEC"]
