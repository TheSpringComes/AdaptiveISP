"""Generate a comprehensive V3 Human ablation comparison figure.

Layout:
  Row 1: Ablation ladder H0 → H3 (4 ckpts × 4 val samples)
  Row 2: H3-s5 variants (3 ckpts × 4 val samples)
  Row 3: Metric bar charts (SSIM / LPIPS / Q)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Configurations in order
ABLATION = [
    ("H0 baseline", "experiments/v3_h0"),
    ("H1 +backbone", "experiments/v3_h1"),
    ("H2 +mask", "experiments/v3_h2"),
    ("H3 +PPO", "experiments/v3_h3"),
]

S5_VARIANTS = [
    ("H3-s5", "experiments/experiments/v3_human_adaptiveisp_human_v3_e3_s5"),
    ("H3-s5-lr1e4", "experiments/experiments/v3_human_adaptiveisp_human_v3_e3_s5_lr1e4"),
    ("H3-s5-rew", "experiments/experiments/v3_human_adaptiveisp_human_v3_e3_s5_rew"),
]

# Metrics from val runs
ABLATION_METRICS = {
    "H0 baseline":  (0.593, 0.407, +0.186),
    "H1 +backbone": (0.540, 0.370, +0.170),
    "H2 +mask":     (0.588, 0.423, +0.165),
    "H3 +PPO":      (0.616, 0.374, +0.242),
}
S5_METRICS = {
    "H3-s5":        (0.544, 0.387, +0.158),
    "H3-s5-lr1e4":  (0.602, 0.330, +0.273),
    "H3-s5-rew":    (0.605, 0.318, +0.287),
}


def load_case(exp_dir: str, case_idx: int) -> np.ndarray | None:
    p = Path(exp_dir) / "visualization" / f"case_{case_idx:02d}_human.png"
    if not p.exists():
        return None
    return mpimg.imread(str(p))


def main():
    n_cases = 4
    # Row 1: Ablation (4 cfgs × 4 cases), Row 2: s5 (3 cfgs × 4 cases)
    fig, axes = plt.subplots(2, max(len(ABLATION), len(S5_VARIANTS)),
                              figsize=(4 * len(ABLATION), 10))

    # Row 1: Ablation ladder
    for col, (label, exp_dir) in enumerate(ABLATION):
        ax = axes[0, col]
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")
        # Show case 0 (first val sample) as representative
        img = load_case(exp_dir, 0)
        if img is not None:
            ax.imshow(img)
        ssim, lpips, q = ABLATION_METRICS[label]
        ax.text(0.5, -0.05, f"SSIM={ssim:.3f}  LPIPS={lpips:.3f}  Q={q:+.3f}",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

    # Hide unused columns in row 1
    for col in range(len(ABLATION), axes.shape[1]):
        axes[0, col].axis("off")

    # Row 2: s5 variants
    for col, (label, exp_dir) in enumerate(S5_VARIANTS):
        ax = axes[1, col]
        ax.set_title(label, fontsize=11, fontweight="bold")
        ax.axis("off")
        img = load_case(exp_dir, 0)
        if img is not None:
            ax.imshow(img)
        ssim, lpips, q = S5_METRICS[label]
        ax.text(0.5, -0.05, f"SSIM={ssim:.3f}  LPIPS={lpips:.3f}  Q={q:+.3f}",
                transform=ax.transAxes, ha="center", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8))

    for col in range(len(S5_VARIANTS), axes.shape[1]):
        axes[1, col].axis("off")

    fig.suptitle("V3 Human Ablation — H0→H3 (top) + H3-s5 variants (bottom)",
                 fontsize=13, fontweight="bold", y=1.01)
    fig.tight_layout()
    out = Path("logs_ablation") / "v3_human_abration_summary.png"
    fig.savefig(str(out), dpi=110, bbox_inches="tight")
    print(f"saved: {out}")

    # Bar chart comparison
    fig2, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(14, 4))

    all_labels = [l for l, _ in ABLATION] + [l for l, _ in S5_VARIANTS]
    all_ssim = [ABLATION_METRICS[l][0] for l, _ in ABLATION] + [S5_METRICS[l][0] for l, _ in S5_VARIANTS]
    all_lpips = [ABLATION_METRICS[l][1] for l, _ in ABLATION] + [S5_METRICS[l][1] for l, _ in S5_VARIANTS]
    all_q = [ABLATION_METRICS[l][2] for l, _ in ABLATION] + [S5_METRICS[l][2] for l, _ in S5_VARIANTS]

    colors = ["#1f77b4"] * len(ABLATION) + ["#ff7f0e", "#2ca02c", "#d62728"]

    ax1.barh(all_labels, all_ssim, color=colors)
    ax1.set_xlabel("SSIM (↑)")
    ax1.set_title("SSIM (higher is better)")
    for i, v in enumerate(all_ssim):
        ax1.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    ax2.barh(all_labels, all_lpips, color=colors)
    ax2.set_xlabel("LPIPS (↓)")
    ax2.set_title("LPIPS (lower is better)")
    for i, v in enumerate(all_lpips):
        ax2.text(v + 0.005, i, f"{v:.3f}", va="center", fontsize=9)

    ax3.barh(all_labels, all_q, color=colors)
    ax3.set_xlabel("Q (↑)")
    ax3.set_title("Quality score Q (higher is better)")
    for i, v in enumerate(all_q):
        ax3.text(v + 0.005, i, f"{v:+.3f}", va="center", fontsize="9")

    fig2.suptitle("V3 Human Ablation — Metric Comparison", fontsize=13, fontweight="bold")
    fig2.tight_layout()
    out2 = Path("logs_ablation") / "v3_human_metrics_comparison.png"
    fig2.savefig(str(out2), dpi=110, bbox_inches="tight")
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
