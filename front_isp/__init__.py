"""front_isp: 可插拔前置 ISP（V3.1 重构，四种统一模式）。

整体 Pipeline（见 V3.1 规划）：

    RAW → front_isp.process(raw, metadata) → Baseline RGB
        → AdaptiveISP (Adaptive Tail) → Final RGB → Task

Front ISP 只负责产生稳定、正常的基础成像结果（WB / CCM / Gamma 这类
基础处理不必交给 RL 逐步搜索）；AdaptiveISP 保持原有逻辑，只负责 Front
ISP 之后的算子选择与参数优化。

统一四种模式（`front_isp.type`）：

    identity  — 不使用 Front ISP，输入直接进入 AdaptiveISP（对照组）。
    fixed     — 人工配置 ISP：WB/CCM/Bias/Gamma 等简单模块，顺序与参数
                全部由配置指定，训练时不更新。模块注册表开放扩展
                （加 denoise/sharpen/contrast = 注册一个函数即可）。
    learnable — 固定结构 + 部分参数可训练（WB gain / CCM / Bias / Gamma）。
                两阶段训练：Stage 1 只训 Front ISP（LearnableTrainer，
                FiveK Input → Expert C），保存并冻结；Stage 2 运行
                AdaptiveISP 训练（HumanTrainer），Front ISP 输出作为 RL
                初始图像。不与 AdaptiveISP 联合训练。
    external  — 接入现有开源 ISP（backend: infinite_isp | samsung_isp），
                wrapper 统一输入输出，AdaptiveISP 不感知具体实现。

legacy 类型名（继续可用）：none=identity、canonical（V3-A1 固定链）、
infinite_isp / modular_neural_isp（直连 wrapper）。

对外入口不变：

    front_isp = build_front_isp_from_cfg(cfg)      # cfg 是整个运行配置
    baseline_rgb = front_isp(raw, metadata)       # 或 .process(raw, metadata)
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
from front_isp.fixed import FixedFrontISP                         # noqa: F401
from front_isp.canonical import CanonicalBackbone                 # noqa: F401
from front_isp.learnable import LearnableFrontISP                    # noqa: F401
from front_isp.external import ExternalFrontISP                   # noqa: F401
from front_isp.infinite_isp import InfiniteISPFront               # noqa: F401
from front_isp.modular_neural_isp import ModularNeuralISPFront     # noqa: F401


def build_front_isp_from_cfg(cfg) -> FrontISPBase:
    """从主运行配置构建 Front ISP。

    优先读 `cfg.front_isp`（V3.1 统一四种模式）：

        front_isp:
          enabled: true
          type: learnable          # identity | fixed | learnable | external
          learnable: {}            # 各类型自己的内部配置
          external:
            backend: infinite_isp  # infinite_isp | samsung_isp

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
    "FixedFrontISP",
    "CanonicalBackbone",
    "LearnableFrontISP",
    "ExternalFrontISP",
    "InfiniteISPFront",
    "ModularNeuralISPFront",
    "build_front_isp",
    "build_front_isp_from_cfg",
    "register_front_isp",
    "list_front_isps",
]
