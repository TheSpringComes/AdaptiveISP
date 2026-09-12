"""Fixed Front ISP — 人工配置的固定 ISP 链（V3.1 模式之二）。

对应配置：

    front_isp:
      enabled: true
      type: fixed
      fixed:
        modules:                # 有序列表，按顺序执行
          - {name: wb, auto: grayworld}          # 内容自适应 gray-world AWB
          - {name: ccm, matrix: cam2rgb}          # 'cam2rgb' = CycleISP 校准矩阵
          - {name: gamma, gamma: 0.4545}          # 编码 gamma (1/2.2)

设计（见 V3.1 规划）：
  - 模块顺序与参数全部由配置文件指定，训练时不更新任何参数；
  - 第一版提供 WB / CCM / Bias / Gamma / Exposure / Smoothstep 等简单模块；
  - 模块注册表 `register_fixed_module` 对外开放——后续要加 denoise、
    sharpen、contrast 等模块时，写一个 `(B,3,H,W), cfg -> (B,3,H,W)` 的
    函数并装饰注册即可，配置里 `{name: <模块名>, ...}` 直接可用，
    不需要改动 FixedFrontISP 或 AdaptiveISP 主体。

输出约定：链末 clamp 到 [0, 1]（Front ISP 的统一输出契约）。
"""
from __future__ import annotations

from typing import Any, Callable, Dict, Optional

import torch

from front_isp.base import FrontISPBase
from front_isp.registry import register_front_isp

# ---------------------------------------------------------------------------
# 固定模块注册表：fn(image (B,3,H,W), cfg dict) -> (B,3,H,W)
# 扩展方式：@register_fixed_module('denoise') 加一个函数即可。
# ---------------------------------------------------------------------------
_FIXED_MODULE_REGISTRY: Dict[str, Callable[[torch.Tensor, Dict[str, Any]],
                                           torch.Tensor]] = {}


def register_fixed_module(name: str):
    """注册一个 fixed-ISP 模块（(image, cfg) -> image 的纯函数）。"""
    def decorator(fn):
        if name in _FIXED_MODULE_REGISTRY:
            raise ValueError(
                f"fixed 模块 '{name}' 已注册为 "
                f"{_FIXED_MODULE_REGISTRY[name].__name__}，不能重复注册")
        _FIXED_MODULE_REGISTRY[name] = fn
        return fn
    return decorator


def list_fixed_modules() -> list:
    return sorted(_FIXED_MODULE_REGISTRY.keys())


# ------------------------------ 内置模块 ------------------------------------

@register_fixed_module('wb')
def _wb(image: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """白平衡。两种方式（二选一）：
    - `auto: grayworld`：按图像内容算 gray-world 增益（确定性、无参数）；
    - `gains: [r, g, b]`：固定通道增益。
    """
    auto = str(cfg.get('auto', '')).lower()
    if auto == 'grayworld':
        from isp.operators.infinite_isp.awb import _grayworld_gain
        gain = _grayworld_gain(image).view(-1, 3, 1, 1)
        return image * gain
    gains = cfg.get('gains')
    if gains is None:
        raise ValueError("fixed 模块 wb 需要 `auto: grayworld` 或 `gains: [r,g,b]`")
    g = torch.as_tensor(gains, dtype=image.dtype, device=image.device)
    return image * g.view(1, 3, 1, 1)


@register_fixed_module('ccm')
def _ccm(image: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """色彩校正矩阵：`matrix: 3x3`（或字符串 'cam2rgb' 用 CycleISP 校准矩阵），
    可选 `bias: [r,g,b]`（缺省 0）。"""
    spec = cfg.get('matrix')
    if spec is None:
        raise ValueError("fixed 模块 ccm 需要 `matrix`（3x3 列表或 'cam2rgb'）")
    if isinstance(spec, str) and spec.lower() == 'cam2rgb':
        from isp.unprocess_np import get_calibrated_cam2rgb
        spec = get_calibrated_cam2rgb()
    m = torch.as_tensor(spec, dtype=image.dtype, device=image.device)
    if tuple(m.shape) != (3, 3):
        raise ValueError(f"ccm matrix 必须是 3x3，得到 {tuple(m.shape)}")
    b = cfg.get('bias', [0.0, 0.0, 0.0])
    bias = torch.as_tensor(b, dtype=image.dtype, device=image.device)
    b_, _, h, w = image.shape
    flat = image.permute(0, 2, 3, 1).reshape(-1, 3)
    out = flat @ m.T + bias
    return out.reshape(b_, h, w, 3).permute(0, 3, 1, 2)


@register_fixed_module('bias')
def _bias(image: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """加偏置 `offset: [r, g, b]`。"""
    off = torch.as_tensor(cfg.get('offset', [0.0, 0.0, 0.0]),
                           dtype=image.dtype, device=image.device)
    return image + off.view(1, 3, 1, 1)


@register_fixed_module('gamma')
def _gamma(image: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """幂律 tone：`gamma: g` → x^g。"""
    g = float(cfg.get('gamma', 1.0))
    return image.clamp(min=0.0).pow(g)


@register_fixed_module('exposure')
def _exposure(image: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """曝光：`ev: v` → x * 2^v（单位 EV）。"""
    ev = float(cfg.get('ev', 0.0))
    return image * (2.0 ** ev)


@register_fixed_module('smoothstep')
def _smoothstep(image: torch.Tensor, cfg: Dict[str, Any]) -> torch.Tensor:
    """Hermite 全局色调映射：3x² − 2x³（canonical GTM 的同款）。"""
    x = image.clamp(0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


# ------------------------------ Front ISP ----------------------------------

@register_front_isp('fixed')
class FixedFrontISP(FrontISPBase):
    """人工配置的固定 ISP：按 `fixed.modules` 顺序依次应用，无任何可训练参数。"""

    def __init__(self, config: Optional[Dict[str, Any]] = None) -> None:
        super().__init__(config)
        modules = self.config.get('modules', None)
        if not isinstance(modules, list) or not modules:
            raise ValueError(
                "FixedFrontISP 需要 fixed.modules 非空列表，"
                "例如: [{name: wb, auto: grayworld}, {name: gamma, gamma: 0.4545}]")
        # 构建期校验：所有模块名必须在注册表里（快速失败，而不是第一次
        # forward 时才报错），并把配置固化下来。
        self._chain = []
        for i, mod in enumerate(modules):
            name = str(mod.get('name', ''))
            if name not in _FIXED_MODULE_REGISTRY:
                raise ValueError(
                    f"fixed.modules[{i}]: 未知模块 {name!r}。"
                    f"可用模块: {list_fixed_modules()}")
            self._chain.append((name, dict(mod)))

    def process(self, image: torch.Tensor, metadata=None) -> torch.Tensor:
        x = image
        for _name, cfg in self._chain:
            x = _FIXED_MODULE_REGISTRY[_name](x, cfg)
        return x.clamp(0.0, 1.0)

    def __repr__(self) -> str:
        chain = " → ".join(n for n, _ in self._chain)
        return f"FixedFrontISP({chain})"


__all__ = ["FixedFrontISP", "register_fixed_module", "list_fixed_modules"]
