"""V3.1 五组消融（A–E）结果汇总表。

收集：
  1. Front ISP 输出质量   experiments/front_isp_eval/summary.json
  2. AdaptiveISP 最终指标  experiments/<group>/final_val.json（训练结束自动落盘）
  3. 算子选择频率          final_val.json 里的 op_pick_cum

输出：experiments/ablation_summary.md + 终端表格
"""
from __future__ import annotations

import json
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GROUPS = [
    ("A", "v31_ablation_A_identity",  "identity"),
    ("B", "v31_ablation_B_fixed",     "fixed (FittedISP)"),
    ("C", "v31_ablation_C_stage2",    "learnable (2-stage)"),
    ("D", "v31_ablation_D_infinite",  "external-infinite"),
    ("E", "v31_ablation_E_samsung",   "external-samsung"),
]
FRONT_EVAL = os.path.join(_ROOT, "experiments", "front_isp_eval", "summary.json")
OUT_MD = os.path.join(_ROOT, "experiments", "ablation_summary.md")


def load_json(path):
    try:
        with open(path) as fh:
            return json.load(fh)
    except Exception:
        return {}


def main():
    front = load_json(FRONT_EVAL)
    runs = {g: load_json(os.path.join(_ROOT, "experiments", d, "final_val.json"))
            for g, d, _ in GROUPS}
    # identity 组的 Front ISP 输出 = 输入本身
    front.setdefault("A_identity", dict(ssim=front.get("A_identity", {}).get("ssim"),
                                        lpips=front.get("A_identity", {}).get("lpips"),
                                        psnr=front.get("A_identity", {}).get("psnr")))

    lines = []
    lines.append("# V3.1 Front ISP 五组消融汇总\n")
    lines.append("统一预算：Stage 2 各 1000 iters（batch 4, imgsz 512, T=8 rollout, "
                 "val=100 张 Expert-C）。\n")

    # ---- 表 1：Front ISP 输出质量 ----
    lines.append("## 表 1 — Front ISP 输出质量（AdaptiveISP 之前）\n")
    lines.append("| 组 | Front ISP | SSIM↑ | LPIPS↓ | PSNR↑ |")
    lines.append("|---|---|---|---|---|")
    fe_keys = {"A": "A_identity", "B": "B_fixed", "C": "C_learnable",
               "D": "D_infinite", "E": "E_samsung"}
    for g, _, name in GROUPS:
        r = front.get(fe_keys[g], {})
        if r.get("ssim") is not None:
            lines.append(f"| {g} | {name} | {r['ssim']:.4f} | {r['lpips']:.4f} "
                         f"| {r['psnr']:.2f} |")
        else:
            lines.append(f"| {g} | {name} | — | — | — |")
    lines.append("")

    # ---- 表 2：AdaptiveISP 最终指标 ----
    lines.append("## 表 2 — AdaptiveISP 最终 val 指标（RL 之后）\n")
    lines.append("| 组 | Front ISP | SSIM↑ | LPIPS↓ | PSNR↑ | ΔE76↓ | Q↑ | mean len |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for g, d, name in GROUPS:
        r = runs.get(g) or {}
        if r.get("val/ssim") is not None:
            lines.append(
                f"| {g} | {name} | {r['val/ssim']:.4f} | {r['val/lpips']:.4f} "
                f"| {r['val/psnr']:.2f} | {r['val/delta_e']:.2f} "
                f"| {r['val/quality']:+.4f} | {r['val/mean_length']:.2f} |")
        else:
            lines.append(f"| {g} | {name} | — | — | — | — | — | — |")
    lines.append("")

    # ---- 表 3：基础算子选择频率 ----
    lines.append("## 表 3 — 基础算子选择频率（全程累计，%）\n")
    base_ops = ["exposure", "gamma", "ccm", "whitebalance", "tone", "contrast",
                "saturation", "sharpen", "denoise", "wnb"]
    header = "| 组 | " + " | ".join(base_ops) + " | neural% | total picks |"
    lines.append(header)
    lines.append("|" + "---|" * (len(base_ops) + 3))
    for g, d, _ in GROUPS:
        r = runs.get(g) or {}
        picks = r.get("op_pick_cum") or {}
        total = sum(picks.values())
        if not total:
            lines.append(f"| {g} | " + " | ".join(["—"] * len(base_ops)) + " | — | — |")
            continue
        neural = sum(v for k, v in picks.items() if k.startswith("n_"))
        cells = [f"{100 * picks.get(op, 0) / total:.1f}" for op in base_ops]
        lines.append(f"| {g} | " + " | ".join(cells)
                     + f" | {100 * neural / total:.1f} | {total} |")
    lines.append("")

    text = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nsaved: {OUT_MD}")


if __name__ == "__main__":
    main()
