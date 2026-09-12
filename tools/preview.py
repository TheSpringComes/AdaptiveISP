"""tools/preview.py — Human 数据集人工校验预览 (v2 cache 专用).

每次抽样 N 张 (默认 4) val/train 样本, 生成两张 PNG:

  preview_grid.png   — 每个 case 两行:
                       row1: Input RAW (完整画幅, 保比例) | Expert C target
                       row2: 校注 (camera / 原始宽高 / 各通道均值 / corr)
  preview_metrics.png — 样本级 corr 直方图 (全 split, 快速扫错配)

比例修复: 不走 FiveKDataset._resize 的"中心裁方" (会丢 33% 画幅),
本工具 imgsz=None 加载原始画幅, 整图保比例缩放显示; 每列宽度按该
case 的宽高比自适应, matplotlib imshow(aspect='equal') 保证无形变。

输出统一到 experiments/preview/<tag>/。
用法:
    python tools/preview.py --n 4                      # val 抽样
    python tools/preview.py --split train --n 3        # train 抽样
    python tools/preview.py --n 4 --seed 123           # 换一组样本
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import torch

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.human_quality import FiveKDataset
from tasks.human_quality.metrics import ssim_batch, psnr_batch


def load_batch(split: str, n: int, seed: int, cache_dir: str):
    """抽 n 个样本 (原始画幅, 不裁方), 返回 (images, targets, cams, stems, ratios)."""
    list_file = {
        "val": "/home/jing/datasets/fivek/val_expert_c.txt",
        "train": "/home/jing/datasets/fivek/train_expert_c.txt",
    }[split]
    ds = FiveKDataset(list_file, cache_dir=cache_dir, imgsz=None)
    rng = np.random.default_rng(seed)
    idxs = sorted(rng.choice(len(ds), size=min(n, len(ds)), replace=False))
    out = []
    for i in idxs:
        im, tg, cam = ds[i]
        stem = Path(ds.paths[i]).stem
        out.append((im, tg, cam, stem))
    return out


def corr_pair(im: torch.Tensor, tg: torch.Tensor) -> float:
    a = im.flatten().numpy()
    b = tg.flatten().numpy()
    return float(np.corrcoef(a, b)[0, 1])


def make_grid(samples, out_path: Path) -> None:
    """2 行 × n 列网格: 上排 [RAW | Expert C] 并排, 下排为标注行。"""
    n = len(samples)
    # 每个 case 两个子图 (raw, target), 用 gridspec 按宽高比分列宽。
    fig = plt.figure(figsize=(4.2 * n, 9.5))
    outer = fig.add_gridspec(2, n, height_ratios=[8, 1.1], hspace=0.05, wspace=0.06)

    for ci, (im, tg, cam, stem) in enumerate(samples):
        h, w = im.shape[-2:]
        im_np = im.permute(1, 2, 0).clamp(0, 1).numpy()
        tg_np = tg.permute(1, 2, 0).clamp(0, 1).numpy()

        s = ssim_batch(im.unsqueeze(0), tg.unsqueeze(0)).item()
        p = psnr_batch(im.unsqueeze(0), tg.unsqueeze(0)).item()
        c = corr_pair(im, tg)
        cam_name = ""
        if samples and hasattr(samples, "__iter__"):
            pass
        # camera name lookup (best effort)
        cam_names = getattr(make_grid, "_cam_names", [])
        if cam_names and isinstance(cam, int):
            cam_name = cam_names[cam] if cam < len(cam_names) else str(cam)

        ax1 = fig.add_subplot(outer[0, 2 * ci] if False else outer[0, ci])
        # 注意: gridspec 2列×n, 但我们要每 case 两个图 → 改用 4×n?
        # 简化: 每 case 一列, 列内上下放 raw 和 target 不可行 (标注行占位)。
        # 最终布局: 大格子里横向 concat [raw | target] 显示。
        sep = np.ones((h, 6, 3), dtype=np.float32)  # 白色分隔条
        combo = np.concatenate([im_np, sep, tg_np], axis=1)
        ax1.imshow(combo)
        ax1.set_xticks([])
        ax1.set_yticks([])
        ax1.set_title(stem, fontsize=9, fontfamily="monospace")

        # 标注行
        ax2 = fig.add_subplot(outer[1, ci])
        ax2.axis("off")
        ax2.text(0.5, 0.95,
                 f"{w}×{h} (W/H {w / h:.2f})\n"
                 f"cam: {cam_name or cam}\n"
                 f"left=RAW  right=Expert C\n"
                 f"SSIM {s:.3f}  PSNR {p:.1f}dB\n"
                 f"corr {c:.3f}",
                 ha="center", va="top", fontsize=8,
                 transform=ax2.transAxes)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def make_hist(split: str, cache_dir: str, out_path: Path) -> None:
    """全 split corr 直方图 (快速看错配/坏样本过滤效果)。"""
    import json
    al_path = os.path.join(cache_dir, "_alignment.json")
    list_file = {
        "val": "/home/jing/datasets/fivek/val_expert_c.txt",
        "train": "/home/jing/datasets/fivek/train_expert_c.txt",
    }[split]
    stems = [Path(l.strip()).stem for l in open(list_file) if l.strip()]
    if not os.path.exists(al_path):
        print(f"(no _alignment.json, skip hist)")
        return
    al = json.load(open(al_path))
    vals = [al.get(s, 1.0) for s in stems]
    fig, ax = plt.subplots(figsize=(7, 3))
    ax.hist(vals, bins=40, range=(0, 1), color="#3b6db5", edgecolor="white")
    ax.axvline(0.5, color="#b03a3a", ls="--", lw=1.2, label="filter threshold 0.5")
    ax.set_xlabel("corr(RAW, Expert C target)")
    ax.set_ylabel("count")
    ax.set_title(f"{split}: n={len(vals)} mean={np.mean(vals):.3f} "
                 f"<0.5: {sum(1 for v in vals if v < 0.5)}")
    ax.legend(fontsize=8)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", choices=["val", "train"], default="val")
    ap.add_argument("--n", type=int, default=4, help="样本数 (抽样, 不全跑)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cache-dir", default="/home/jing/datasets/fivek/cache_expert_c")
    ap.add_argument("--tag", default="", help="输出子目录名 (默认 preview_<split>)")
    a = ap.parse_args()

    # camera name 表 (给标注)
    cam_json = os.path.join(os.path.dirname(a.cache_dir), "camera.json")
    if os.path.exists(cam_json):
        import json
        names = sorted(set(json.load(open(cam_json)).values()))
        make_grid._cam_names = names

    tag = a.tag or f"preview_{a.split}"
    out_dir = _ROOT / "experiments" / "preview" / tag
    samples = load_batch(a.split, a.n, a.seed, a.cache_dir)
    print(f"split={a.split} sampled {len(samples)} (seed={a.seed}) → {out_dir}")

    make_grid(samples, out_dir / "preview_grid.png")
    make_hist(a.split, a.cache_dir, out_dir / "preview_metrics.png")


if __name__ == "__main__":
    main()
