"""Front ISP 基类定义

Front ISP 负责将 RAW/simulated-RAW 转换为稳定的 baseline RGB，
为后续的 Adaptive Tail 提供可靠的起点。

所有 Front ISP 实现必须继承 FrontISPBase 并实现 process() 方法。
"""
from abc import ABC, abstractmethod
from typing import Any, Dict, Optional
import torch
import torch.nn as nn


class FrontISPBase(nn.Module, ABC):
    """Front ISP 基类

    所有 Front ISP 实现必须：
    1. 继承此类
    2. 实现 process() 方法
    3. 接受 (image, metadata) 输入
    4. 返回 baseline RGB (B, 3, H, W)
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Args:
            config: Front ISP 的配置字典，不同实现可以有不同的配置项
        """
        super().__init__()
        self.config = config or {}

    @abstractmethod
    def process(
        self,
        image: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """处理输入图像，生成 baseline RGB

        Args:
            image: 输入图像 (B, C, H, W)
                   - 对于 RAW 输入: C=1 (Bayer) 或 C=3 (demosaicked linear)
                   - 对于 simulated-RAW: C=3 (linear RGB with noise)
            metadata: 可选的元数据，包含相机参数、噪声级别等

        Returns:
            baseline_rgb: 稳定的 baseline RGB (B, 3, H, W)，范围 [0, 1]
        """
        pass

    def forward(
        self,
        image: torch.Tensor,
        metadata: Optional[Dict[str, Any]] = None
    ) -> torch.Tensor:
        """PyTorch 标准 forward，调用 process()"""
        return self.process(image, metadata)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(config={self.config})"
