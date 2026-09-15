"""Progressive Parameter Bounds（参数课程）。

训练前期限制 ISP 算子的参数调整幅度，随进度逐渐放宽到完整范围。
只作用于参数映射阶段——PPO / Reward / 算子实现 / ParameterSpec
接口均不改动；`range_scale == 1` 时与原行为逐位一致。

数据流：
    trainer (iter/total → progress) ──range_scale──▶ Controller ──▶ 本模块
    Controller 参数头 raw ──▶ scale_params(name, spec, raw, s) ──▶ 物理参数

schedule（默认 quadratic，先慢后快）：
    rc = min(progress / end_progress, 1)
    s  = start_scale + (1 - start_scale) · rc²        (progress ≥ end 后恒 1)

三类参数的缩放空间（中性点 p₀ = spec.regressor(0)，即零输出的物理值）：

  linear（加性参数：exposure/contrast/tone/ccm）
      p' = p₀ + s·(p − p₀)                     线性空间围绕中性点
  log（乘法参数：gamma/whitebalance/inf 增益类）
      p' = p₀ · (p/p₀)^s                       对数空间围绕中性点——
      gamma: p' = exp(s·log3·tanh(z)) 即 [3^-s, 3^s]，正是规划公式
  blend（[0,1] 混合强度，0 = identity：sharpen/denoise/wnb/n_*/inf_* 强度类）
      p' = s · p                               最大强度从 s 渐开到 1

不变量：三类映射在 s=1 时精确还原原 regressor 输出；s<1 时
progressive 范围始终包含中性点（linear/log 的插值端点、blend 的 0），
且不越出 spec.low/high（凸组合 / 正域对数凸组合 / 纯收缩）。
"""
from __future__ import annotations

import torch

from isp.base import ParameterSpec


# ------------------------------ schedule ------------------------------

def range_scale_for_progress(
    progress: float,
    *,
    enabled: bool = False,
    start_scale: float = 0.2,
    end_progress: float = 0.25,
    schedule: str = "quadratic",
) -> float:
    """训练进度 → 当前开放的参数范围比例 s ∈ [start_scale, 1]。

    disabled / progress ≥ end_progress / 未知 schedule 一律返回 1.0
    （完整范围，与旧行为一致）。
    """
    if not enabled or progress >= end_progress:
        return 1.0
    rc = min(max(progress / max(end_progress, 1e-8), 0.0), 1.0)
    if schedule == "quadratic":
        return start_scale + (1.0 - start_scale) * rc * rc
    if schedule == "linear":
        return start_scale + (1.0 - start_scale) * rc
    return 1.0  # 未知 schedule：保守回退全范围


def parse_curriculum_cfg(cfg: dict) -> dict:
    """读 config 的 parameter_curriculum 块（缺省 = 关闭）。"""
    c = (cfg.get('parameter_curriculum', {}) or {}) if cfg else {}
    return {
        'enabled': bool(c.get('enabled', False)),
        'start_scale': float(c.get('start_scale', 0.2)),
        'end_progress': float(c.get('end_progress', 0.25)),
        'schedule': str(c.get('schedule', 'quadratic')),
    }


# ------------------------------ mode table ------------------------------

# 逐算子的缩放空间。默认 "linear"（未知算子的安全回退：围绕中性点
# 线性收缩，s=1 精确还原）。
# ccm：identity-centered residual——regressor 已输出 M = I + δ·tanh(z)，
# curriculum 在矩阵域做 residual 缩放 M' = I + s·(M − I)，发生在 apply 的
# 行归一化之前（归一化对均匀 residual 缩放不敏感，必须先缩放再归一化）。
# 无 identity 点、暂用 linear 收缩待人工重定义的算子：
#   sharpen  — factor 是 blur↔sharpen 混合（1=锐化 0=模糊，轴上无 identity）
#   inf_ebf  — 参数是 bilateral 的 range-sigma 位置（α=0 → σ=0.05，仍是有效滤波）
_LINEAR = ("exposure", "contrast", "tone", "sharpen", "inf_ebf")
_CCM = ("ccm",)
_LOG = ("gamma", "whitebalance", "inf_digital_gain", "inf_saturation")
_BLEND = ("denoise", "wnb", "saturation",
          "n_denoise", "n_awb", "n_gain", "n_gtm", "n_chroma",
          "n_gamma", "n_detail",
          "inf_awb_grayworld", "inf_awb_norm2", "inf_awb_pca",
          "inf_ldci", "inf_unsharp", "inf_nlm")

_MODES = {**{n: "linear" for n in _LINEAR},
          **{n: "ccm" for n in _CCM},
          **{n: "log" for n in _LOG},
          **{n: "blend" for n in _BLEND}}


# ------------------------------ param mapping ------------------------------

def scale_params(name: str, spec: ParameterSpec, raw: torch.Tensor,
                 range_scale: float) -> torch.Tensor:
    """带 range_scale 的参数映射：raw（参数头无界输出）→ 物理域。

    blend-strength 算子**无条件**走 tanh² 映射（α = s·high·tanh²(z+b₀)，
    与 s 无关——这是它们的参数定义，不是课程；raw=0 → α≈0 恒 identity）。
    其余类型 `range_scale >= 1` 时走原路径 `spec.regressor(raw)`——与
    旧行为逐位一致（不开课程的训练/eval 默认即此分支）。
    """
    mode = _MODES.get(name, "linear")
    s = float(range_scale)
    if mode == "blend":
        # Blend-strength 参数化（2026-09-15 重设计）：
        #     α = s · high · tanh²(z + b₀),  b₀ = 0.001
        # raw=0 → α ≈ s·high·b₀² ≈ 0（identity，而非旧 σ(z) 的半强度 0.5s）；
        # ∂α/∂z|₀ = 2·s·high·b₀ ≠ 0（tanh² 在零初始化处梯度不为零，b₀ 的作用）；
        # α ∈ [0, s·high]，s=1 时全幅度，s<1 时从 identity 侧渐开。
        # 直接计算——不再经过各算子自己的 sigmoid/tanh_range regressor
        #（那些的零输出是 0.5/中点，正是 audit 发现的 neutral mismatch）。
        b0 = 0.001
        high = float(spec.high) if not isinstance(spec.high, tuple) else float(spec.high[0])
        return (s * high
                * torch.tanh(raw + b0).pow(2))

    p = spec.regressor(raw)
    if s >= 1.0:
        return p

    if mode == "ccm":
        # Identity-centered residual：regressor 输出 M = I + δ·tanh(z)
        # （9 元素，reshape 前后同一布局）。curriculum 在矩阵域缩放残差：
        #     M' = I + s·(M − I)
        # 发生在 apply 的行归一化之前——归一化对均匀残差缩放不敏感，
        # 先缩放才能让课程真正生效。s=1 → M'=M（逐位还原）。
        eye9 = torch.eye(3, device=p.device, dtype=p.dtype).flatten()
        return eye9 + s * (p - eye9)

    # 中性点 = 零输出的物理值（tanh_range 的 initial / blend 0.5 中点等
    # 都编码在内）。no_grad：中性点是常数锚，不参与参数头梯度。
    with torch.no_grad():
        neutral = spec.regressor(torch.zeros_like(raw))

    if mode == "log":
        # 对数空间围绕中性点：p' = p₀·(p/p₀)^s。gamma ∈ [3^−s, 3^s]。
        ratio = torch.clamp(p / neutral.clamp(min=1e-8), min=1e-8)
        return neutral * torch.pow(ratio, s)
    # linear：p' = p₀ + s·(p − p₀)。
    return neutral + s * (p - neutral)


__all__ = ["range_scale_for_progress", "parse_curriculum_cfg", "scale_params"]
