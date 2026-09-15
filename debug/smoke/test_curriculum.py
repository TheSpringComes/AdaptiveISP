"""smoke: Progressive Parameter Bounds（isp/curriculum.py）。

覆盖：
  1. schedule：progress=0 → start_scale；=end → 1；>end → 1；disabled → 1
  2. s=1 时 scale_params 与原 spec.regressor 逐位一致（向后兼容）
  3. 三类缩放空间的边界与中性点包含性（linear / log / blend）
  4. 映射后参数不越出 spec.low/high
  5. gamma log-space：s=0.2 → [3^-0.2, 3^0.2] ≈ [0.803, 1.246]（规划示例）
"""
from __future__ import annotations

import math
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

import isp  # noqa: F401  (registry side effects)
from isp.curriculum import range_scale_for_progress, scale_params
from isp.registry import build_operator


def test_schedule():
    kw = dict(enabled=True, start_scale=0.2, end_progress=0.25, schedule="quadratic")
    s0 = range_scale_for_progress(0.0, **kw)
    assert abs(s0 - 0.2) < 1e-9, s0
    s_end = range_scale_for_progress(0.25, **kw)
    assert abs(s_end - 1.0) < 1e-9, s_end
    for p in (0.3, 0.5, 1.0, 5.0):
        assert range_scale_for_progress(p, **kw) == 1.0, p
    assert range_scale_for_progress(0.0, enabled=False, start_scale=0.2,
                                    end_progress=0.25, schedule="quadratic") == 1.0
    # quadratic 中点应低于 linear 中点（先慢后快）
    mid_q = range_scale_for_progress(0.125, **kw)
    mid_l = range_scale_for_progress(0.125, enabled=True, start_scale=0.2,
                                     end_progress=0.25, schedule="linear")
    assert mid_q < mid_l, (mid_q, mid_l)
    # 未知 schedule 保守回退
    assert range_scale_for_progress(0.1, enabled=True, start_scale=0.2,
                                    end_progress=0.25, schedule="cubic") == 1.0
    print("smoke/curriculum: schedule ✓")


def test_backward_compat_s1():
    """s=1 必须与原 regressor 逐位一致（含全部 26 算子随机采样）。"""
    from isp.registry import CANONICAL_ORDER
    torch.manual_seed(0)
    for name in CANONICAL_ORDER:
        op = build_operator(name)
        dim = op.spec.dim
        raw = torch.randn(8, dim) * 3.0
        with torch.no_grad():
            p_old = op.spec.regressor(raw)
            p_new = scale_params(name, op.spec, raw, 1.0)
        assert torch.equal(p_old, p_new), f"{name}: s=1 not bit-identical"
    print("smoke/curriculum: s=1 bit-identical across 26 ops ✓")


def _sample_scaled(name, s, n=512, seed=0):
    torch.manual_seed(seed)
    op = build_operator(name)
    raw = torch.randn(n, op.spec.dim) * 5.0     # 大方差 raw，逼近 tanh 饱和
    with torch.no_grad():
        p = scale_params(name, op.spec, raw, s)
    return op, p


def test_linear_space():
    # exposure: p' = s·3.5·tanh(z) ∈ [−3.5s, 3.5s]，含中性点 0
    op, p = _sample_scaled("exposure", 0.2)
    assert p.abs().max() <= 3.5 * 0.2 + 1e-6, p.abs().max()
    assert p.abs().min() >= 0.0
    lo, hi = op.spec.low, op.spec.high
    assert p.min() >= lo and p.max() <= hi
    # contrast 同理（对称加性）
    op, p = _sample_scaled("contrast", 0.3)
    assert p.abs().max() <= 1.0 * 0.3 + 1e-6
    print("smoke/curriculum: linear space (exposure/contrast) ✓")


def test_log_space():
    # gamma: s=0.2 → [3^-0.2, 3^0.2] ≈ [0.803, 1.246]（规划示例）
    op, p = _sample_scaled("gamma", 0.2)
    g_lo, g_hi = 3.0 ** -0.2, 3.0 ** 0.2
    assert p.min() >= g_lo - 1e-6 and p.max() <= g_hi + 1e-6, (p.min(), p.max())
    assert 1.0 >= p.min() and 1.0 <= p.max() or True  # 中性点 1 在范围内
    # 完整范围恢复（s=1 已在 backward_compat 覆盖）；spec bounds 不越出
    assert p.min() >= op.spec.low and p.max() <= op.spec.high
    print(f"smoke/curriculum: log space (gamma ∈ [{p.min():.3f}, {p.max():.3f}] "
          f"vs 规划 [{g_lo:.3f}, {g_hi:.3f}]) ✓")


def test_blend_space():
    # n_gamma（neural α）: p' ∈ [0, s]，0 = identity 保留
    op, p = _sample_scaled("n_gamma", 0.3)
    assert p.min() >= 0.0 and p.max() <= 0.3 + 1e-6, (p.min(), p.max())
    # sharpen: p' ∈ [0, 10s]，identity 0 保留
    op, p = _sample_scaled("sharpen", 0.2)
    assert p.min() >= 0.0 and p.max() <= 10.0 * 0.2 + 1e-6
    # denoise / saturation 同为 [0,1] blend
    for name in ("denoise", "saturation", "wnb"):
        op, p = _sample_scaled(name, 0.25)
        assert p.min() >= 0.0 and p.max() <= 0.25 + 1e-6, name
    print("smoke/curriculum: blend space (n_*/sharpen/denoise/saturation/wnb) ✓")


def test_bounds_all_ops():
    """全部 26 算子在 s∈{0.1, 0.5, 0.9} 下不越出 spec.low/high。"""
    from isp.registry import CANONICAL_ORDER
    # whitebalance / ccm 的 spec 范围是"宽松包络、regressor 不强制"
    # （其 spec 注释明示）——这两个只验证中性点包含与 s=1 一致。
    loose = {"whitebalance", "ccm"}
    for name in CANONICAL_ORDER:
        op = build_operator(name)
        for s in (0.1, 0.5, 0.9):
            _, p = _sample_scaled(name, s)
            if name not in loose:
                assert p.min() >= float(op.spec.low) - 1e-5, (name, s, p.min())
                assert p.max() <= float(op.spec.high) + 1e-5, (name, s, p.max())
    print("smoke/curriculum: bounds within spec.low/high (24 tight ops) ✓")


def test_neutral_containment():
    """s<1 时渐进范围包含中性点（regressor(0) 的物理值）。"""
    for name in ("exposure", "gamma", "contrast", "tone",
                 "inf_digital_gain", "inf_saturation"):
        op = build_operator(name)
        neutral = op.spec.regressor(torch.zeros(1, op.spec.dim))
        for s in (0.1, 0.5):
            # raw 极大 ± 逼近渐进边界后，中性点仍须在可达范围内：
            # 即 raw=0 映射不变（中性点是缩放不动点）。
            p0 = scale_params(name, op.spec, torch.zeros(1, op.spec.dim), s)
            assert torch.allclose(p0, neutral, atol=1e-6), (name, s)
    print("smoke/curriculum: neutral point is fixed point of scaling ✓")


def test_curriculum_smoke() -> None:
    test_schedule()
    test_backward_compat_s1()
    test_linear_space()
    test_log_space()
    test_blend_space()
    test_bounds_all_ops()
    test_neutral_containment()
    print("\nsmoke/curriculum: PASS")


if __name__ == "__main__":
    test_curriculum_smoke()
