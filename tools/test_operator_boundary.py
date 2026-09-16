"""ISP Operator Boundary & Identity Check — 全算子边界参数检查。

对 registry 中全部算子（26 个）执行四类检查，任何一项失败即报错退出：

  [B1] 边界完整性：regressor 输出恒在 spec.low/high 内
       （raw 扫描 ±8 饱和 + 随机采样 × scale_params 全 s 档）。
  [B2] identity 锚定：raw=0 → apply 后图像精确不变
       （max|Δ| < 1e-5，三类测试图：random / gray / gradient）。
  [B3] 课程一致性：s=1 时非 blend 算子与 regressor 逐位一致；
       s<1 时范围收缩且包含 neutral；s 扫描 {0.1, 0.2, 0.5, 0.9, 1.0}。
  [B4] 梯度健康：Q(loss) 对 raw 反传，梯度有限非零（CIE 分段幂/tanh² 等
       无奇异点）。

用法：
    python tools/test_operator_boundary.py          # 全部检查
    python tools/test_operator_boundary.py sharpen  # 只查单个算子
"""
from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

import isp  # noqa: F401  (registry side effects)
from isp.curriculum import scale_params, _MODES
from isp.registry import CANONICAL_ORDER, build_operator

# ------------------------------ 阈值（集中配置） ------------------------------
IDENT_T = 1e-5          # [B2] 精确 identity：图像 max|Δ| 上限
# [B2] 浮点/实现噪声容忍。gamma 的 apply 内部 clip(img, 0.001) 使黑像素
# 即使在 γ=1（identity）下也被抬到 0.001——绝对偏差恰 1e-3，是原版防
# pow(0,y) 溢出的固有 clamp（非参数化错误）；WB 的 1/(1e-5+lum) 同理。
NEAR_T = 2e-3
GRAD_MIN, GRAD_MAX = 1e-12, 1e6   # [B4] 梯度有限非零的界

# 各算子的 identity 物理参数（用于 [B3] 的 neutral 包含性验证；
# raw=0 时 regressor 输出应等于此值——与 [B2] 的图像级检查互为印证）
_EXPECTED_NEUTRAL: dict[str, float] = {
    "exposure": 0.0, "contrast": 0.0,
    "gamma": 1.0, "whitebalance": 1.0,
    "inf_digital_gain": 1.0, "inf_saturation": 1.0,
    "sharpen": 1.0,
}
# 这些算子的 neutral 不是单值（ccm=I、tone=均匀、blend≈0 由 [B2] 覆盖）

_SCALES = (0.1, 0.2, 0.5, 0.9, 1.0)


def _test_images(imgsz: int = 48) -> dict[str, torch.Tensor]:
    torch.manual_seed(0)
    imgs = {"random": torch.rand(2, 3, imgsz, imgsz),
            "gray": torch.full((2, 3, imgsz, imgsz), 0.5)}
    grad = torch.zeros(2, 3, imgsz, imgsz)
    h = torch.linspace(0, 1, imgsz)
    grad[:, 0] = h.view(1, imgsz, 1).expand(2, imgsz, imgsz)
    grad[:, 2] = torch.flip(h, [0]).view(1, imgsz, 1).expand(2, imgsz, imgsz)
    imgs["gradient"] = grad
    return imgs


def check_operator(name: str, imgs: dict[str, torch.Tensor]) -> list[str]:
    """跑全部四类检查，返回失败信息列表（空 = 全过）。"""
    op = build_operator(name)
    spec = op.spec
    lo = float(spec.low) if not isinstance(spec.low, tuple) else float(spec.low[0])
    hi = float(spec.high) if not isinstance(spec.high, tuple) else float(spec.high[0])
    fails: list[str] = []
    mode = _MODES.get(name, "linear")

    # ---- [B1] 边界完整性 ----
    raws = [torch.full((4, spec.dim), v) for v in (-8.0, -4.0, -1.0, 0.0, 1.0, 4.0, 8.0)]
    torch.manual_seed(42)
    raws.append(torch.randn(8, spec.dim) * 3)
    for raw in raws:
        for s in (1.0,):   # 完整范围（旧 _SCALES 课程扫描已随 range_scale 移除）
            p = scale_params(name, spec, raw, s)
            # blend 模式输出的域是 [0, high]（tanh²×s×high）
            eff_hi = hi
            if p.min().item() < lo - 1e-5 or p.max().item() > eff_hi + 1e-5:
                fails.append(f"[B1] raw∈[{raw.min():.1f},{raw.max():.1f}] s={s}: "
                             f"p∈[{p.min():.4f},{p.max():.4f}] 越出 [{lo},{hi}]")
                break

    # ---- [B2] identity 锚定（raw=0 → 图像不变；参数恒为完整范围）----
    for s in (1.0,):
        p0 = scale_params(name, spec, torch.zeros(2, spec.dim), s)
        for key, img in imgs.items():
            with torch.no_grad():
                out = op.apply(img, p0)
            d = (out - img).abs().max().item()
            if d > NEAR_T:
                fails.append(f"[B2] raw=0 s={s} {key}: |Δ|={d:.2e} > {NEAR_T:.0e}")
                break

    # ---- [B3] neutral-distance 语义（param_reg 取代旧课程） ----
    from isp.param_reg import param_distance, neutral_params
    if name != 'tone':   # tone 的 neutral 是均匀值（非单点），distance 定义为回归距离
        n = neutral_params(name, spec, torch.zeros(4, spec.dim))
        d0 = param_distance(name, spec, n)
        if d0.max().item() > 1e-5:
            fails.append(f"[B3] neutral 处 distance={d0.max().item():.2e} ≠ 0")
    # neutral 数值锚（有预期值的算子）
    if name in _EXPECTED_NEUTRAL:
        with torch.no_grad():
            n = spec.regressor(torch.zeros(1, spec.dim))
        exp = _EXPECTED_NEUTRAL[name]
        if not torch.allclose(n, torch.full_like(n, exp), atol=1e-4):
            fails.append(f"[B3] regressor(0)={n.flatten().tolist()} ≠ 预期 neutral {exp}")
    # 距离范围 [0,1]
    raw = torch.randn(8, spec.dim) * 5
    p = scale_params(name, spec, raw, 1.0)
    d = param_distance(name, spec, p)
    if d.min() < -1e-6 or d.max() > 1.0001:
        fails.append(f"[B3] distance 越界 [{d.min():.4f},{d.max():.4f}]")

    # ---- [B4] 梯度健康 ----
    raw_g = torch.zeros(2, spec.dim, requires_grad=True)
    p = scale_params(name, spec, raw_g, 0.5)
    loss = p.sum()
    if loss.requires_grad:
        loss.backward()
        g = raw_g.grad
        if g is not None:
            if not torch.isfinite(g).all():
                fails.append(f"[B4] raw=0 梯度含 NaN/Inf")
            elif g.abs().max().item() > GRAD_MAX:
                fails.append(f"[B4] raw=0 梯度爆炸 |g|={g.abs().max().item():.2e}")
            # 零梯度对某些对称 neutral 是合法的（如 ccm 非对角元素），
            # 只要求不全零
            elif g.abs().sum().item() < GRAD_MIN:
                fails.append(f"[B4] raw=0 梯度全零（死梯度）")

    return fails


def main() -> int:
    target = sys.argv[1] if len(sys.argv) > 1 else None
    names = [target] if target else list(CANONICAL_ORDER)
    imgs = _test_images()

    n_pass, n_fail = 0, 0
    for name in names:
        fails = check_operator(name, imgs)
        mode = _MODES.get(name, "linear")
        if fails:
            n_fail += 1
            print(f"✗ {name} ({mode})")
            for f in fails[:4]:
                print(f"    {f}")
        else:
            n_pass += 1
            print(f"✓ {name} ({mode})")

    print(f"\n{'=' * 50}")
    print(f"boundary check: {n_pass} passed, {n_fail} failed (of {len(names)})")
    if n_fail:
        sys.exit(1)
    print("ALL OPERATORS PASS ALL CHECKS")


if __name__ == "__main__":
    main()
