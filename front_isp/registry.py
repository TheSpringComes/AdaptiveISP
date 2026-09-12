"""Front ISP 注册机制

通过 @register_front_isp 装饰器注册 Front ISP 实现，
通过 build_front_isp() 工厂函数根据 config 构建实例。
"""
from typing import Any, Dict, Optional, Type
from .base import FrontISPBase


# 全局注册表
_FRONT_ISP_REGISTRY: Dict[str, Type[FrontISPBase]] = {}

# 各类型在主配置里的子键候选（按优先级）。V3.1 统一四种模式：
#     identity | fixed | learnable | external
_SUBKEY_CANDIDATES: Dict[str, list] = {}


def register_front_isp(name: str):
    """注册 Front ISP 实现的装饰器

    Usage:
        @register_front_isp('infinite_isp')
        class InfiniteISPFront(FrontISPBase):
            ...
    """
    def decorator(cls: Type[FrontISPBase]) -> Type[FrontISPBase]:
        if name in _FRONT_ISP_REGISTRY:
            raise ValueError(
                f"Front ISP '{name}' 已注册为 {_FRONT_ISP_REGISTRY[name].__name__}，"
                f"不能重复注册为 {cls.__name__}"
            )
        if not issubclass(cls, FrontISPBase):
            raise TypeError(
                f"注册的类 {cls.__name__} 必须继承 FrontISPBase"
            )
        _FRONT_ISP_REGISTRY[name] = cls
        return cls
    return decorator


def build_front_isp(config: Dict[str, Any]) -> FrontISPBase:
    """根据 config 构建 Front ISP 实例

    Args:
        config: 包含 'type' 字段的配置字典，例如:
            {'type': 'infinite_isp', 'infinite_isp': {'config_path': ...}}
        `enabled: false`（或缺省且 type 为 'none'）返回 IdentityFrontISP，
        即 Adaptive Tail 直接从输入图像开始（E0 parity）。

    Returns:
        FrontISPBase 实例（disabled 时为 IdentityFrontISP）

    Raises:
        ValueError: 如果 type 未注册
    """
    from front_isp.identity import IdentityFrontISP

    if not bool(config.get('enabled', True)):
        return IdentityFrontISP()

    front_type = config.get('type', 'none')

    if front_type not in _FRONT_ISP_REGISTRY:
        available = list(_FRONT_ISP_REGISTRY.keys())
        raise ValueError(
            f"未知的 Front ISP 类型: '{front_type}'。"
            f"V3.1 统一四种模式: identity | fixed | learnable | external。"
            f"已注册类型: {available}"
        )

    cls = _FRONT_ISP_REGISTRY[front_type]
    # 传递对应类型的子配置。每个类型的主子键与其类型同名
    # （fixed / learnable / external）。
    sub_config = None
    for key in _SUBKEY_CANDIDATES.get(front_type, [front_type]):
        if config.get(key) is not None:
            sub_config = config.get(key)
            break
    if sub_config is None:
        sub_config = {}
    return cls(config=sub_config)


def list_front_isps() -> list:
    """列出所有已注册的 Front ISP 类型"""
    return list(_FRONT_ISP_REGISTRY.keys())
