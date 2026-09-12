"""Identity Front ISP — passthrough.

对应 `front_isp: {enabled: false}` 或 `{type: none}`：不做任何前置处理，
Adaptive Tail 直接从数据集输出的图像开始（V2 / E0 parity 行为）。
"""
import torch

from front_isp.base import FrontISPBase
from front_isp.registry import register_front_isp


@register_front_isp('none')
class IdentityFrontISP(FrontISPBase):
    """Passthrough：process() 原样返回输入。"""

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        return image

    def __repr__(self) -> str:
        return "IdentityFrontISP()"


__all__ = ["IdentityFrontISP"]
