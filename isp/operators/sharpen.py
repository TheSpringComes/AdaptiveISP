"""Sharpen operator (adjust_sharpness). Ports SharpenFilter (short 'Shr').

Identity-centered residual parameterization (2026-09-15 redesign):

    factor = 1 + tanh(z)            （raw=0 → factor=1 → 精确 identity）

`adjust_sharpness` 的语义是 factor = 原图混合比：
    output = image·factor + blurred·(1−factor)
factor=1 = 原图（identity）、0 = 纯模糊、2 = 强锐化。

旧参数化 `tanh_range(0, 2)`（无 initial）的问题：
1) raw=0 映射到 1.0 纯属巧合（tanh01(0)=0.5 → 中点），不是设计；
2) 参数域负方向是"模糊"——参数头输出偏负时"锐化算子"在模糊图像；
3) curriculum 的 linear 模式围绕 neutral=1.0 收缩碰巧正确，但映射
   本身没有 identity 锚定。

新参数化下 raw=0 精确恒等；负方向=模糊、正方向=锐化，方向语义清晰。
Progressive curriculum（isp/curriculum.py 的 "sharpen" 模式）：
    factor' = 1 + s·(factor − 1) = 1 + s·tanh(z)
早期 s=0.2 → factor ∈ [0.8, 1.2]，只允许轻微模糊/锐化。
"""
from __future__ import annotations

import torch

from isp.base import ISPOperator, ParameterSpec
from isp.registry import register
from isp.sharpen import adjust_sharpness

_SHARP_LOW, _SHARP_HIGH = 0.0, 2.0  # factor 域（1=identity）


def _sharpen_regressor(features: torch.Tensor) -> torch.Tensor:
    """[B, 1] raw → [B, 1] factor ∈ (0, 2)。零输出 → 1.0 = identity。"""
    return 1.0 + torch.tanh(features)


@register("sharpen")
class SharpenOperator(ISPOperator):
    short_name = "Shr"
    spec = ParameterSpec(
        dim=1,
        low=_SHARP_LOW,
        high=_SHARP_HIGH,
        regressor=_sharpen_regressor,
        description="sharpen blend factor (1=identity, 0=blur, 2=strong sharpen)",
    )
    runtime_cost = 6.3

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return adjust_sharpness(img, params[:, :, None, None])
