"""front_isp: Configurable Front ISP（可插拔前置 ISP）。

整体 Pipeline（见 Pipeline 扩展方案）：

    RAW → front_isp.process(raw, metadata) → Baseline RGB
        → AdaptiveISP (Adaptive Tail) → Final RGB → Task

Front ISP 只负责产生稳定、正常的基础成像结果；Adaptive Tail 只负责
根据场景和任务做增量式优化（pipeline-level refinement，而非显式的
I_out = I_base + ΔI）。

对外只暴露一个入口：

    front_isp = build_front_isp_from_cfg(cfg)      # cfg 是整个运行配置
    baseline_rgb = front_isp(raw, metadata)       # 或 .process(raw, metadata)

主 Pipeline（trainer / evaluator / visualizer）不 import 任何具体
Infinite-ISP / Samsung 实现，第三方项目通过 wrapper 接入且不修改。

注册的实现：
    none               — IdentityFrontISP（passthrough，E0 parity）
    canonical          — CanonicalBackbone（V3-A1 固定 4 阶段 ISP）
    calibrated         — CalibratedFrontISP（V3.1 可学习相机标定：WB/CCM/Bias/Tone）
    infinite_isp       — InfiniteISPFront（10x-Engineers Infinite-ISP wrapper）
    modular_neural_isp — ModularNeuralISPFront（SamsungLabs wrapper）
"""
from front_isp.base import FrontISPBase
from front_isp.registry import (
    build_front_isp,
    list_front_isps,
    register_front_isp,
)

# 顶层 import 触发注册。所有子模块的 import 都是安全的（第三方依赖
# 延迟到实例构建时才加载），因此这里不会因为缺少第三方仓库而失败。
from front_isp.identity import IdentityFrontISP                    # noqa: F401
from front_isp.canonical import CanonicalBackbone                 # noqa: F401
from front_isp.calibration import CalibratedFrontISP               # noqa: F401
from front_isp.infinite_isp import InfiniteISPFront                # noqa: F401
from front_isp.modular_neural_isp import ModularNeuralISPFront     # noqa: F401


def build_front_isp_from_cfg(cfg) -> FrontISPBase:
    """从主运行配置构建 Front ISP。

    优先读 `cfg.front_isp`（新接口，见 Pipeline 扩展方案 2.2）：

        front_isp:
          enabled: true
          type: canonical          # none | canonical | infinite_isp | modular_neural_isp
          canonical: {}            # 各类型自己的内部配置
          infinite_isp:
            config_path: configs/front_isp/infinite.yaml

    向后兼容：`front_isp` 段缺席时回落到 V3-A1 的 legacy 键
    `canonical_backbone.enabled`（true → canonical，false → identity），
    旧的 v3 实验 config 无需改动即可运行。
    """
    front_cfg = cfg.get('front_isp', None) if hasattr(cfg, 'get') else None
    if front_cfg:
        return build_front_isp(front_cfg)
    legacy = cfg.get('canonical_backbone', {}) or {}
    if bool(legacy.get('enabled', False)):
        return build_front_isp({'type': 'canonical'})
    return build_front_isp({'type': 'none'})


__all__ = [
    "FrontISPBase",
    "IdentityFrontISP",
    "CanonicalBackbone",
    "CalibratedFrontISP",
    "InfiniteISPFront",
    "ModularNeuralISPFront",
    "build_front_isp",
    "build_front_isp_from_cfg",
    "register_front_isp",
    "list_front_isps",
]
