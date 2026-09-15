"""CCM operator: 3x3 color-correction matrix. Ports CCMFilter (short 'CCM').

Identity-centered residual parameterization (2026-09-15 redesign):

    M = I + δ_max · tanh(z)          （z = 参数头零输出时 ΔM = 0 → M = I 严格恒等）

旧参数化（9 元素直接 tanh_range(-2,2) 再 reshape）的 neutral 是全零矩阵：
    1) 全零行经 row-normalize 0/0 = NaN（V1 遗留地雷，曾污染整条 rollout）；
    2) 即使数值上救回，"从零矩阵学一个完整 CCM" 也不符合 residual
       AdaptiveISP 的设计——中性点应是恒等（不动颜色）。

`δ_max`（residual 幅度上限）= 0.3：M 的对角 ∈ [0.7, 1.3]、非对角 ∈
[−0.3, 0.3]，每行和 ≥ 1 − 3δ = 0.1 > 0 恒成立，row-normalize 天然安全，
不再需要 NaN 兜底。

Progressive curriculum（isp/curriculum.py 的 "ccm" 模式）：
    M_t = I + s(ρ)·ΔM
训练初期 s=0.2 → M ≈ I ± 0.2δ；后期 s=1 → 全幅度。注意行归一化对
均匀缩放不敏感（s·ΔM 各元素同比缩放后归一化不变），因此 curriculum
作用在**归一化前**的 residual 上：neutral 通道保持 I 的行结构。
"""
from __future__ import annotations

import torch

from isp.base import ISPOperator, ParameterSpec
from isp.registry import register

# δ_max 的选择由行和恒正约束决定：每行和 ≥ (1−δ) + 2·(−δ) = 1−3δ，
# 要 > 0 需 δ < 1/3。取规划值 0.3：行和 ≥ 0.1 > 0 恒成立，row-normalize
# 永不除零/翻符号；对角 ∈ [0.7, 1.3]、非对角 ∈ [−0.3, 0.3]——训练初期
# s=0.2 时 M ≈ I ± 0.06，只允许很小的颜色调整（正是渐进设计的意图）。
_DELTA_MAX = 0.3
_CCM_LOW, _CCM_HIGH = -2.0, 2.0   # 完整物理范围的宽松包络（SearchSpace 信息性用）


def _color_correction_matrix(image: torch.Tensor, ccm: torch.Tensor) -> torch.Tensor:
    """image: NCHW; ccm: N,3,3; return: NCHW."""
    image = torch.permute(image, (0, 2, 3, 1))          # NHWC
    image = image[:, :, :, None, :]                     # N,H,W,1,C
    ccm = ccm[:, None, None, :, :]                      # N,1,1,3,3
    out = torch.sum(image * ccm, dim=-1)                # N,H,W,3
    return torch.permute(out, (0, 3, 1, 2))             # NCHW


def _ccm_regressor(features: torch.Tensor) -> torch.Tensor:
    """[B, 9] raw → [B, 9] 的 M 元素（identity + residual）。

    reshape 成 3×3 后即 M = I + δ_max·tanh(z)。零输出 = 严格恒等。
    """
    residual = _DELTA_MAX * torch.tanh(features)
    eye9 = torch.eye(3, device=features.device, dtype=features.dtype).flatten()
    return residual + eye9                                  # broadcast [B, 9]


@register("ccm")
class CCMOperator(ISPOperator):
    short_name = "CCM"
    spec = ParameterSpec(
        dim=9,
        low=_CCM_LOW,
        high=_CCM_HIGH,
        regressor=_ccm_regressor,
        description="3x3 CCM as identity + residual (zero output = exact identity)",
    )
    runtime_cost = 1.9

    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        m = torch.reshape(params, shape=(-1, 3, 3))
        # 行归一化（原版行为）。identity 主对角保证每行和 ≥ 1 > 0，
        # 0/0 NaN 在此参数化下不可能发生。
        m = m / torch.sum(m, dim=-1, keepdim=True)
        return _color_correction_matrix(img, m)
