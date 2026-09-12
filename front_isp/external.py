"""External Front ISP — 第三方 ISP 统一接入（V3.1 模式之四）。

对应配置：

    front_isp:
      enabled: true
      type: external
      external:
        backend: infinite_isp | samsung_isp   # 指定具体实现
        ...                                    # 其余键原样传给该 backend 的
                                               # wrapper（repo_path 等）

设计（见 V3.1 规划）：
  - Infinite-ISP 与 Samsung Modular Neural ISP 通过各自的 wrapper 接入
    （`front_isp/infinite_isp/`、`front_isp/modular_neural_isp/`），
    统一输入输出格式；
  - 本类只做 backend 分发——AdaptiveISP 主体与训练代码只认
    `type: external`，不感知具体用了哪一套外部代码；
  - 新增第三方 ISP = 写一个 FrontISPBase wrapper + 在 `_BACKENDS` 注册，
    不需要改动本类之外的任何代码。
"""
from __future__ import annotations

import importlib
from typing import Any, Dict, Optional

from front_isp.base import FrontISPBase
from front_isp.registry import register_front_isp

# backend 名 → (wrapper 模块, 类名)。延迟 import，避免模块加载期
# 依赖第三方仓库的存在（wrapper 自身的第三方 import 也是延迟的）。
_BACKENDS = {
    'infinite_isp': ('front_isp.infinite_isp.wrapper', 'InfiniteISPFront'),
    'samsung_isp': ('front_isp.modular_neural_isp.wrapper',
                    'ModularNeuralISPFront'),
}


def list_external_backends() -> list:
    return sorted(_BACKENDS.keys())


@register_front_isp('external')
class ExternalFrontISP(FrontISPBase):
    """第三方 ISP 的统一分发壳：按 `external.backend` 委托给具体 wrapper。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        backend = str(self.config.get('backend', ''))
        if backend not in _BACKENDS:
            raise ValueError(
                f"未知 external backend: {backend!r}。"
                f"可用: {list_external_backends()}")
        module_name, cls_name = _BACKENDS[backend]
        cls = getattr(importlib.import_module(module_name), cls_name)
        # 除 backend 外的全部键原样传给 wrapper（repo_path / config_path /
        # weights_dir / stages 等，由各 wrapper 自行解释）。
        sub = {k: v for k, v in self.config.items() if k != 'backend'}
        self._impl = cls(config=sub)

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        return self._impl.process(image, metadata)

    def __repr__(self) -> str:
        return f"ExternalFrontISP({self._impl!r})"


__all__ = ["ExternalFrontISP", "list_external_backends"]
