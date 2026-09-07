"""Summarize V2-AI ablation experiment results into a comparison table.

Reads each run's tee-log and its final val output, produces a Markdown
table. No Python deps beyond stdlib + numpy for pretty formatting.

Usage:
    python scripts/summarize_ablations.py
    python scripts/summarize_ablations.py --logs logs_ablation --ckpt experiments
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path


ABLATIONS = [
    # (id,       kind,        save_path,        cfg)
    ("d-base",   "detection", "lod-v2ai_full",  "test_steps=10 (baseline, previously trained)"),
    ("d5",       "detection", "lod-v2ai_d5",    "test_steps=5"),
    ("dusepen5", "detection", "lod-v2ai_dusepen5",  "penalty base 5 (from 20)"),
    ("dusepen1", "detection", "lod-v2ai_dusepen1",  "penalty base 1"),
    ("dcritic10","detection", "lod-v2ai_dcritic10", "critic_logit_multiplier=10"),
    ("dclassical","detection","lod-v2ai_dclassical","10 classical only"),
    # Batch 3: Detection hyperparameter tuning
    ("dclip01",  "detection", "lod-v2ai_dclip01",   "grad_clip 1e-5→0.1 (1e4× looser)"),
    ("dlr1e-4",  "detection", "lod-v2ai_dlr1e-4",   "lr 3e-5→1e-4"),
    ("dlr1e-5",  "detection", "lod-v2ai_dlr1e-5",   "lr 3e-5→1e-5"),
    ("dbatch16", "detection", "lod-v2ai_dbatch16",  "batch_size 8→16"),
    ("dexp05",   "detection", "lod-v2ai_dexp05",    "exploration 0.2→0.5"),
    ("dexp01",   "detection", "lod-v2ai_dexp01",    "exploration 0.2→0.1"),
    ("hbase",    "human",     "v2ai_hbase",     "T=10, stop=on, repeat=on (base 5)"),
    ("hnostop",  "human",     "v2ai_hnostop",   "T=10, stop=OFF, repeat=on"),
    ("hnorepeat","human",     "v2ai_hnorepeat", "T=10, stop=on, repeat=OFF"),
    ("ht5",      "human",     "v2ai_ht5",       "T=5, stop=on, repeat=on"),
    ("husepen2", "human",     "v2ai_husepen2",  "repeat base 2 (softer)"),
    ("husepen1", "human",     "v2ai_husepen1",  "repeat base 1 (softest)"),
    ("hlambdassim","human",   "v2ai_hlambdassim","λ_ssim=2, λ_lpips=1"),
    ("hclassical","human",    "v2ai_hclassical","10 classical only, no neural"),
]


def find_final_map(log_path: Path) -> dict | None:
    """Detection: grep the last  'all  ...  mAP50  ...' line printed by val.py."""
    if not log_path.exists():
        return None
    metrics: dict = {}
    with open(log_path) as fh:
        for line in fh:
            m = re.search(
                r'^\s*all\s+\d+\s+\d+\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)',
                line,
            )
            if m:
                metrics = {
                    "P": float(m.group(1)), "R": float(m.group(2)),
                    "mAP50": float(m.group(3)), "mAP75": float(m.group(4)),
                    "mAP50-95": float(m.group(5)),
                }
    return metrics or None


def find_human_val(log_path: Path) -> dict | None:
    """Human: grep the ===== VAL ===== block at end of training log."""
    if not log_path.exists():
        return None
    with open(log_path) as fh:
        text = fh.read()
    if "===== VAL" not in text:
        return None
    # Look at everything after the LAST "===== VAL" marker; grep each metric
    # line directly (no need to bracket the block since the metrics are all
    # on their own labeled lines).
    tail = text.rsplit("===== VAL", 1)[1]
    out: dict = {}
    for k, pat in [
        ("SSIM",   r"SSIM:\s*([-0-9.]+)"),
        ("LPIPS",  r"LPIPS:\s*([-0-9.]+)"),
        ("Q",      r"Q:\s*([+\-0-9.]+)"),
        ("length", r"mean rollout length:\s*([0-9.]+)/"),
        ("pct_stop", r"pct learned-STOP.*?:\s*([-0-9.]+)"),
    ]:
        m = re.search(pat, tail)
        if m:
            out[k] = float(m.group(1))
    return out or None


def find_last_train_map(exp_dir: Path) -> dict | None:
    """Detection baseline: no ablation log, look at the training log inside experiments/."""
    log = exp_dir / "logs" / "log.txt"
    if not log.exists():
        return None
    with open(log) as fh:
        text = fh.read()
    # look for last "mAP50" line in training output (from tools/val.py output if val was run)
    matches = re.findall(
        r'\s+all\s+\d+\s+\d+\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)\s+([-0-9.]+)',
        text,
    )
    if not matches:
        return None
    P, R, m50, m75, m5095 = [float(x) for x in matches[-1]]
    return {"P": P, "R": R, "mAP50": m50, "mAP75": m75, "mAP50-95": m5095}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs_ablation")
    ap.add_argument("--exp", default="experiments")
    args = ap.parse_args()

    logs_dir = Path(args.logs)
    exp_dir = Path(args.exp)

    print("# V2-AI Ablation Summary\n")

    # Detection table
    print("## Detection (LOD)")
    print()
    print("| Run      | Config                             | P     | R     | mAP50 | mAP75 | mAP50-95 |")
    print("|----------|------------------------------------|------:|------:|------:|------:|---------:|")
    for id_, kind, save, note in ABLATIONS:
        if kind != "detection":
            continue
        log = logs_dir / f"{id_}.log"
        metrics = find_final_map(log)
        if metrics is None:
            # try inside experiments/<save>/logs/log.txt (D-BASE fallback)
            for candidate in [exp_dir / save, exp_dir / f"lod-{save}"]:
                metrics = find_last_train_map(candidate)
                if metrics:
                    break
        if metrics:
            print(f"| {id_:<8s} | {note:<34s} | "
                  f"{metrics.get('P',0):.3f} | {metrics.get('R',0):.3f} | "
                  f"{metrics.get('mAP50',0):.3f} | {metrics.get('mAP75',0):.3f} | "
                  f"{metrics.get('mAP50-95',0):.3f} |")
        else:
            print(f"| {id_:<8s} | {note:<34s} | *(no log / not run yet)* |")

    print("\n")

    # Human table
    print("## Human Quality (FiveK, val split = 100 samples)")
    print()
    print("| Run       | Config                             | SSIM ↑ | LPIPS ↓ | Q ↑    | mean len | %learned STOP |")
    print("|-----------|------------------------------------|-------:|--------:|-------:|---------:|--------------:|")
    for id_, kind, save, note in ABLATIONS:
        if kind != "human":
            continue
        log = logs_dir / f"{id_}.log"
        metrics = find_human_val(log)
        if metrics:
            print(f"| {id_:<9s} | {note:<34s} | "
                  f"{metrics.get('SSIM',0):.4f} | {metrics.get('LPIPS',0):.4f} | "
                  f"{metrics.get('Q',0):+.4f} | "
                  f"{metrics.get('length',0):.2f} | "
                  f"{metrics.get('pct_stop',0):.1f}% |")
        else:
            print(f"| {id_:<9s} | {note:<34s} | *(no log / not run yet)* |")

    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
