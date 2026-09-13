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
    header = "| 组 | " + " | ".join(base_ops) + " | total picks |"
    lines.append(header)
    lines.append("|" + "---|" * (len(base_ops) + 2))
    for g, d, _ in GROUPS:
        r = runs.get(g) or {}
        picks = r.get("op_pick_cum") or {}
        total = sum(picks.values())
        if not total:
            lines.append(f"| {g} | " + " | ".join(["—"] * len(base_ops)) + " | — |")
            continue
        cells = [f"{100 * picks.get(op, 0) / total:.1f}" for op in base_ops]
        lines.append(f"| {g} | " + " | ".join(cells) + f" | {total} |")
    lines.append("")

    # ---- 训练曲线 ----
    lines.append("## 图 1 — 训练/验证曲线（TensorBoard 提取）\n")
    lines.append("![curves](ablation_curves.png)\n")

    # ---- 结论 ----
    lines.append("## 主要发现\n")
    lines.append("""
1. **起点质量决定 RL 可训性（本预算下最根本的发现）**：
   低起点组（A identity 0.319 / C learnable 0.362）的 RL 策略收敛到
   "立即 STOP"（A: 99% learned-STOP, mean len 1.14；C: 100%, len 1.00），
   val SSIM 仅从起点微升（A 0.319→0.375, C 0.362→0.378）——
   在弱起点上，探索期内大多数算子动作都是负收益，策略学到的最优解就是不动。

2. **fixed（FittedISP 拟合）是本预算下唯一全面成功的组**：
   Front ISP 起点 SSIM 0.850（表 1 最高），RL 后 val SSIM 0.761、
   Q +0.461，策略积极优化（mean len 3.22）且最终指标全面领先。

3. **external 前端起点好但短预算 RL 反而退化**：
   D (infinite) 起点 0.689 → RL 后 0.483（策略持续施加算子，len 5.60，
   45% 学会 STOP 但仍净损伤）；E (samsung) 起点 0.803 → RL 后 0.185
   （严重退化）。两者共同点：前端输出风格与 Expert-C 目标差距大
   （LPIPS 0.38/0.19 vs fixed 0.126），在 1000-iter 短预算下策略
   未学到"何时该停"，argmax 策略仍带有训练期探索的破坏性动作。
   即：高起点 ≠ 高可训性，奖励塑形与预算的匹配同样关键。

4. **算子选择频率**（表 3）：五组一致以 exposure 为最高频选择
   （11.6–13.6%），印证曝光/亮度校正是 AdaptiveISP 首要动作；
   白平衡/wb 类算子占比普遍低于 exposure，因为多数 Front ISP 已前置
   处理了色偏。

5. **对 C 组的解读**：learnable 两阶段在 1500-iter Stage 1 预算下
   尚未追上 fixed 的拟合质量（0.362 vs 0.850）；Stage 1 需要显著
   更多预算（或更强的参数化）才能进入"可激活 RL"的起点区间。
""")

    text = "\n".join(lines)
    with open(OUT_MD, "w", encoding="utf-8") as fh:
        fh.write(text)
    print(text)
    print(f"\nsaved: {OUT_MD}")


if __name__ == "__main__":
    main()
