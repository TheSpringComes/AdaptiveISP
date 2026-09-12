"""Infinite-ISP (10x-Engineers) Front ISP wrapper 包。

import 安全：真正的第三方依赖延迟到 `InfiniteISPFront._load_backend()`，
本包被 `front_isp` 顶层 import 时只完成注册。
"""
from front_isp.infinite_isp.wrapper import InfiniteISPFront

__all__ = ["InfiniteISPFront"]
