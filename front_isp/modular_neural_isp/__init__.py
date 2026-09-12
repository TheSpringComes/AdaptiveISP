"""Samsung Modular Neural ISP Front ISP wrapper 包。

import 安全：真正的第三方依赖延迟到 `ModularNeuralISPFront._load_backend()`，
本包被 `front_isp` 顶层 import 时只完成注册。
"""
from front_isp.modular_neural_isp.wrapper import ModularNeuralISPFront

__all__ = ["ModularNeuralISPFront"]
