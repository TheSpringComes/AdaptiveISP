"""Neutral-Distance Parameter Regularization（参数正则化，2026-09-15 重设计）。

取代旧 Progressive Parameter Bounds（range_scale 动态压缩参数范围）。
新机制：**参数始终使用完整合法范围**，但每步对实际执行参数相对 neutral
的归一化距离收取二次惩罚，惩罚系数随训练进度 cosine 衰减到 0——
早期强烈偏好 neutral 附近的小幅调整，后期完全放开（零惩罚）。

    d_t = |p_t − p_neutral| / D_max ∈ [0, 1]     （逐参数归一化距离）
    P_param,t = λ_p(ρ) · d_t²                    （加入每步 reward）
    r_t = ΔQ_t − P_param,t − P_other

三阶段 cosine schedule（默认 start=10%、end=30%）：

    ρ < start        λ = λ₀                       （保持）
    start ≤ ρ < end  λ = λ₀/2 · [1 + cos(π·(ρ−start)/(end−start))]  （衰减）
    ρ ≥ end          λ = 0                        （完全放开）

d_t 的计算按参数类型（与旧 curriculum 的分类一致，neutral 语义不变）：

  linear  （exposure/contrast/tone）: d = |p − p₀| / max(|hi−p₀|, |p₀−lo|)
  log     （gamma/WB/inf 增益）    : d = |log(p/p₀)| / log(p_far/p₀)
  sharpen （factor=1 恒等）         : d = |factor − 1| / 1
  ccm     （identity 残差）         : d = ‖M − I‖_max / δ_max（元素最大偏差）
  blend   （0=identity 强度）       : d = α / high

schedule 数据流：
    trainer (progress) → lambda_param(ρ) → reward_fn.compute → P_param
    （Controller 不再持有 range_scale——参数映射恒为完整范围）
"""
from __future__ import annotations

import math

import torch

from isp.base import ParameterSpec


# ------------------------------ schedule ------------------------------

def lambda_param_for_progress(
    progress: float,
    *,
    enabled: bool = False,
    lambda0: float = 0.001,
    start_ratio: float = 0.10,
    end_ratio: float = 0.30,
    schedule: str = "cosine",
) -> float:
    """训练进度 → 当前参数惩罚系数 λ_p(ρ)。

    三阶段：ρ<start 保持 λ₀；start≤ρ<end cosine 衰减；ρ≥end 恒 0。
    disabled / 未知 schedule → 0（无惩罚，与旧行为一致）。
    """
    if not enabled or schedule != "cosine":
        return 0.0
    if progress < start_ratio:
        return float(lambda0)
    if progress >= end_ratio:
        return 0.0
    span = max(end_ratio - start_ratio, 1e-8)
    w = (progress - start_ratio) / span               # ∈ [0, 1)
    return float(lambda0) * 0.5 * (1.0 + math.cos(math.pi * w))


def parse_param_reg_cfg(cfg: dict) -> dict:
    """读 config 的 param_regularization 块（缺省 = 关闭）。"""
    c = (cfg.get('param_regularization', {}) or {}) if cfg else {}
    return {
        'enabled': bool(c.get('enable', c.get('enabled', False))),
        'lambda0': float(c.get('lambda0', 0.001)),
        'start_ratio': float(c.get('start_ratio', 0.10)),
        'end_ratio': float(c.get('end_ratio', 0.30)),
        'schedule': str(c.get('schedule', 'cosine')),
    }


# ------------------------------ neutral & distance ------------------------------

# 逐算子的距离度量类型（与旧 curriculum 分类一致；neutral 语义见各类注释）。
_LINEAR = ("exposure", "contrast", "tone")
_SHARPEN = ("sharpen",)                     # factor = 1 + tanh(z)，neutral=1
_CCM = ("ccm",)                             # M = I + δ·tanh(z)，neutral=I
_LOG = ("gamma", "whitebalance", "inf_digital_gain", "inf_saturation")
_BLEND = ("denoise", "wnb", "saturation",
          "n_denoise", "n_awb", "n_gain", "n_gtm", "n_chroma",
          "n_gamma", "n_detail",
          "inf_awb_grayworld", "inf_awb_norm2", "inf_awb_pca",
          "inf_ldci", "inf_unsharp", "inf_nlm", "inf_ebf")

_MODES = {**{n: "linear" for n in _LINEAR},
          **{n: "sharpen" for n in _SHARPEN},
          **{n: "ccm" for n in _CCM},
          **{n: "log" for n in _LOG},
          **{n: "blend" for n in _BLEND}}


def neutral_params(name: str, spec: ParameterSpec,
                   raw: torch.Tensor) -> torch.Tensor:
    """该算子的 neutral 物理参数（identity 点），形状同 raw 的输出。

    与零输出 regressor 一致（blend 的 neutral=0 由定义给出）。
    """
    mode = _MODES.get(name, "linear")
    if mode == "blend":
        return torch.zeros_like(raw)
    with torch.no_grad():
        return spec.regressor(torch.zeros_like(raw))


def param_distance(name: str, spec: ParameterSpec,
                   physical: torch.Tensor) -> torch.Tensor:
    """实际参数 p_t 相对 neutral 的归一化距离 d_t ∈ [0, 1]（逐元素）。

    physical: [B, dim]（Controller 参数映射后的物理参数）。
    返回 [B, dim] 的距离（调用方自行聚合为每样本标量）。
    """
    mode = _MODES.get(name, "linear")
    lo = float(spec.low) if not isinstance(spec.low, tuple) else float(spec.low[0])
    hi = float(spec.high) if not isinstance(spec.high, tuple) else float(spec.high[0])

    if mode == "blend":
        # α ∈ [0, high]，neutral=0：d = α / high。
        return (physical / max(hi, 1e-8)).clamp(0.0, 1.0)

    if mode == "sharpen":
        # factor ∈ [0, 2]，neutral=1，两侧最大偏移各 1：d = |factor−1|。
        return (physical - 1.0).abs().clamp(0.0, 1.0)

    if mode == "ccm":
        # M = I + δ·tanh(z)（9 元素 flat）。残差逐元素 |M−I|，归一化 δ_max。
        eye9 = torch.eye(3, device=physical.device,
                         dtype=physical.dtype).flatten()
        # δ_max 由 ccm.py 定义为 0.3；从残差上界推断（对角 |1−1±δ|≤δ）。
        delta_max = 0.3
        return ((physical - eye9).abs() / delta_max).clamp(0.0, 1.0)

    # neutral 数值锚
    with torch.no_grad():
        p0 = spec.regressor(torch.zeros_like(physical))
        p0 = p0.expand_as(physical) if p0.shape != physical.shape else p0

    if mode == "log":
        # 对数空间：d = |log(p/p₀)| / log(p_far/p₀)，p_far = 离 p₀ 更远的边界。
        p_safe = physical.clamp(min=1e-8)
        p0_safe = p0.clamp(min=1e-8)
        log_d = (p_safe / p0_safe).log().abs()
        far = torch.where(hi / p0_safe.mean() > p0_safe.mean() / max(lo, 1e-8),
                          torch.tensor(hi, device=physical.device),
                          torch.tensor(max(lo, 1e-8), device=physical.device))
        d_max = (far.clamp(min=1e-8) / p0_safe.mean()).log().abs().clamp(min=1e-8)
        return (log_d / d_max).clamp(0.0, 1.0)

    # linear：d = |p − p₀| / max(|hi−p₀|, |p₀−lo|)。
    span = max(abs(hi - p0.mean().item()), abs(p0.mean().item() - lo), 1e-8)
    return ((physical - p0).abs() / span).clamp(0.0, 1.0)


__all__ = ["lambda_param_for_progress", "parse_param_reg_cfg",
           "neutral_params", "param_distance"]
