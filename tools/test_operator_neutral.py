"""ISP Operator Neutral Audit — raw=0 时每个算子是否为 identity。

对 registry 中全部算子（classical / n_* / inf_*）构造 `raw = zeros(dim)`，
经 `ParameterSpec.regressor` 映射后在三类测试图（随机 RGB / 中灰 /
渐变）上计算 apply 前后差异，输出统一审计表格与分类总结。

只审计，不修改任何算子、regressor 或训练逻辑。

阈值（集中配置）：
    MAE < 1e-6            PASS          （identity）
    1e-6 ≤ MAE < 1e-3     NEAR_IDENTITY
    MAE ≥ 1e-3            FAIL          （neutral mismatch，需要处理）

用法：
    python tools/test_operator_neutral.py            # 全部算子
    python tools/test_operator_neutral.py --max-err  # 只显示 FAIL/CHECK 明细
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import torch

import isp  # noqa: F401  (registry side effects)
from isp.registry import CANONICAL_ORDER, build_operator

# ------------------------------ 阈值（集中配置） ------------------------------
PASS_T = 1e-6
NEAR_T = 1e-3

# 每个算子的"按 apply() 公式推断的真正 neutral"与建议的 progressive 映射类。
# 这是人工分析表（audit 的第 9 项要求），数据来自逐算子读 apply() 源码：
_NEUTRAL_NOTES: dict[str, tuple[str, str]] = {
    # name: (推断 neutral, 建议 progressive 类)
    "exposure":        ("0（EV shift 线性空间）", "additive"),
    "gamma":           ("1（log 空间乘性）", "log-space"),
    "ccm":             ("3×3 identity（残差参数化）", "identity-matrix"),
    "sharpen":         ("1.0（factor=1=原图；0=模糊，2=强锐化）", "sharpen-residual"),
    "denoise":         ("0（strength=0 不去噪）", "blend-strength"),
    "tone":            ("均匀曲线（apply 内 sum 归一化 → 任意等值曲线=identity）", "identity-curve"),
    "contrast":        ("0", "additive"),
    "saturation":      ("0（blend=0 原图）", "blend-strength"),
    "wnb":             ("0（lerp=0 原图）", "blend-strength"),
    "whitebalance":    ("(1,1,1) 增益恒等", "log-space"),
    "n_denoise":       ("α=0（x + α(F(x)−x)）", "blend-strength"),
    "n_awb":           ("α=0", "blend-strength"),
    "n_gain":          ("α=0", "blend-strength"),
    "n_gtm":           ("α=0", "blend-strength"),
    "n_chroma":        ("α=0", "blend-strength"),
    "n_gamma":         ("α=0", "blend-strength"),
    "n_detail":        ("α=0", "blend-strength"),
    "inf_awb_grayworld": ("α=0", "blend-strength"),
    "inf_awb_norm2":     ("α=0", "blend-strength"),
    "inf_awb_pca":       ("α=0", "blend-strength"),
    "inf_digital_gain":  ("1.0（增益恒等）", "log-space"),
    "inf_ldci":          ("0（strength=0）", "blend-strength"),
    "inf_unsharp":       ("0（amount=0）", "blend-strength"),
    "inf_nlm":           ("0（strength=0）", "blend-strength"),
    "inf_ebf":           ("0（identity-blend：α=0 直通，α 控制 σ 与混合）", "blend-strength"),
    "inf_saturation":    ("1.0（乘性增益恒等）", "log-space"),
}


def _status(mae: float) -> str:
    if mae < PASS_T:
        return "PASS"
    if mae < NEAR_T:
        return "NEAR_IDENTITY"
    return "FAIL"


def audit(imgsz: int = 64, seed: int = 0) -> list[dict]:
    """跑全部算子审计，返回逐算子记录。"""
    dev = torch.device("cpu")
    torch.manual_seed(seed)

    imgs = {
        "random": torch.rand(2, 3, imgsz, imgsz),
        "gray": torch.full((2, 3, imgsz, imgsz), 0.5),
    }
    grad = torch.zeros(2, 3, imgsz, imgsz)
    h = torch.linspace(0, 1, imgsz)
    grad[:, 0] = h.view(1, imgsz, 1).expand(2, imgsz, imgsz)
    grad[:, 2] = torch.flip(h, [0]).view(1, imgsz, 1).expand(2, imgsz, imgsz)
    imgs["gradient"] = grad

    records = []
    for name in CANONICAL_ORDER:
        op = build_operator(name).to(dev)
        spec = op.spec
        raw = torch.zeros(2, spec.dim)
        with torch.no_grad():
            p = spec.regressor(raw)

        per_img = {}
        overall_mae = 0.0
        overall_max = 0.0
        for key, im in imgs.items():
            with torch.no_grad():
                out = op.apply(im, p)
            diff = (out - im).abs()
            mae = diff.mean().item()
            overall_mae = max(overall_mae, mae)
            overall_max = max(overall_max, diff.max().item())
            per_img[key] = mae

        st = _status(overall_mae)
        note, cls = _NEUTRAL_NOTES.get(name, ("?", "?"))
        records.append({
            "name": name, "dim": spec.dim, "p0": p[0].tolist(),
            "mae": overall_mae, "max": overall_max, "per_img": per_img,
            "status": st, "neutral_note": note, "class": cls,
            "low": spec.low, "high": spec.high,
        })
    return records


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-err", action="store_true",
                    help="只打印 FAIL/CHECK 算子的详细诊断")
    ap.add_argument("--imgsz", type=int, default=64)
    args = ap.parse_args()

    records = audit(args.imgsz)

    print("ISP Operator Neutral Audit — f_op(I, regressor(0)) ≈ I ?\n")
    header = (f"{'operator':<18} {'dim':>3}  {'physical@raw0':<24} "
              f"{'MAE':>8}  {'max_err':>8}  status")
    print(header)
    print("-" * len(header))
    for r in records:
        pv = r["p0"]
        p_disp = (f"[{pv[0]:.3f}...×{r['dim']}]"
                  if len(pv) > 4 else str([round(v, 3) for v in pv]))
        print(f"{r['name']:<18} {r['dim']:>3}  {p_disp:<24} "
              f"{r['mae']:>8.4f}  {r['max']:>8.4f}  {r['status']}")

    if args.max_err:
        print("\n---- FAIL/CHECK 明细 ----")
        for r in records:
            if r["status"] == "PASS":
                continue
            print(f"\n{r['name']}  [{r['status']}]  MAE={r['mae']:.4f}")
            print(f"  spec.low/high      : {r['low']} / {r['high']}")
            print(f"  raw=0 physical     : {[round(v, 3) for v in r['p0']]}")
            print(f"  推断 neutral        : {r['neutral_note']}")
            print(f"  建议 progressive 类 : {r['class']}")
            print(f"  分图 MAE            : "
                  + " / ".join(f"{k}={v:.4f}" for k, v in r["per_img"].items()))

    # ------------------------------ 总结 ------------------------------
    passed = [r for r in records if r["status"] == "PASS"]
    near = [r for r in records if r["status"] == "NEAR_IDENTITY"]
    failed = [r for r in records if r["status"] == "FAIL"]

    print("\n================ Summary ================")
    print(f"identity 初始化已满足（PASS）  : "
          f"{', '.join(r['name'] for r in passed) or '—'}")
    print(f"near-identity（观察即可）      : "
          f"{', '.join(r['name'] for r in near) or '—'}")
    print(f"neutral mismatch（FAIL）       : "
          f"{', '.join(r['name'] for r in failed) or '—'}")

    blend = [r["name"] for r in failed
             if r["class"] == "blend-strength"]
    print("\nprogressive mapping 分类处理建议：")
    if blend:
        print(f"  · blend-strength 类 {len(blend)} 个 {blend}：neutral=0，"
              "regressor 的零输出 0.5 → raw=0 时半强度生效。需要"
              " initial/偏置把 neutral 移到 0，或在 curriculum 用"
              " blend 模式 α=s·σ(z)（identity 侧渐开）")
    log_ops = [r["name"] for r in failed if r["class"] == "log-space"]
    if log_ops:
        print(f"  · log-space 类：neutral=1，需检查乘性 neutral")
    curve = [r["name"] for r in failed if r["class"] == "identity-curve"]
    if curve:
        print(f"  · identity-curve 类 {curve}：需按 apply 公式确认均匀曲线中性")
    print("  · 已 PASS 的算子维持现状（regressor 零输出=identity）")
    print("  注意：本审计检查 regressor 层（neutral mismatch 是 regressor 的属性）。"
          "isp/curriculum.py 的 blend 模式已在映射层用 α=s·high·tanh²(z+b₀) 覆盖修复——"
          "即 FAIL 算子在训练/eval 时已表现为 identity 初始化，但 regressor 本身未变。")
    print("==========================================\n")


if __name__ == "__main__":
    main()
