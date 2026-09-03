"""CCM operator: 3x3 color-correction matrix. Ports CCMFilter (short 'CCM')."""
from __future__ import annotations

import torch

from isp.base import ISPOperator, ParameterSpec, tanh_range
from isp.registry import register

_CCM_LOW, _CCM_HIGH = -2.0, 2.0  # config.py cfg.ccm_range


def _color_correction_matrix(image: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
    """image: NCHW; ccm: N,3,3; return: NCHW."""
    image = torch.permute(image, (0, 2, 3, 1))          # NHWC
    image = image[:, :, :, None, :]                     # N,H,W,1,C
    ccm = ccm[:, None, None, :, :]                      # N,1,1,3,3
    out = torch.sum(image * ccm, dim=-1)                # N,H,W,3
    return torch.permute(out, (0, 3, 1, 2))             # NCHW


@register("ccm")
class CCMOperator(ISPOperator):
    short_name = "CCM"
    spec = ParameterSpec(
        dim=9,
        low=_CCM_LOW,
        high=_CCM_HIGH,
        regressor=tanh_range(_CCM_LOW, _CCM_HIGH),
        description="3x3 color-correction matrix (row-normalized)",
    )
    runtime_cost = 1.9

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        m = torch.reshape(params, shape=(-1, 3, 3))
        m = m / torch.sum(m, dim=-1, keepdim=True)      # row-normalize (original behavior)
        return _color_correction_matrix(img, m)
