"""tools/vis_size_match.py — FiveK raw/target 像素级尺寸对齐可视化.

验证 raw (4,H/2,W/2) 与 target (3,H,W) 是否严格 2x 对应,以及 demosaic 后
是否与 target 像素级对齐. 每张样本一行:

  col1  Raw RGB3 (H/2 x W/2)        — 线性马赛克平面堆叠 (输入)
  col2  Raw RGB3 ↑2x nearest (H x W) — 最近邻放大到 target 分辨率
  col3  Expert-C target (H x W)      — sRGB 参考
  col4  |col2 - col3| (x4 增强)      — 差值图; 若尺寸错位 1px 会出现
                                      边缘亮线, 若严格对齐则只有噪声

输出: experiments/vis_size_match/<tag>/size_match.png
用法:
    python tools/vis_size_match.py --n 4
    python tools/vis_size_match.py --stems a0001-jmac_DSC1459 a0002-dgw_005
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tasks.human_quality import FiveKDataset


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--stems", nargs="*", default=None)
    ap.add_argument("--val-list", default="/home/jing/datasets/fivek/val_expert_c.txt")
    ap.add_argument("--cache-dir", default="/home/jing/datasets/fivek/cache_expert_c")
    a = ap.parse_args()

    out_dir = _ROOT / "experiments" / "vis_size_match" / (a.tag or f"seed{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = FiveKDataset(list_file=a.val_list, cache_dir=a.cache_dir,
                      imgsz=None, return_camera=True, alignment_threshold=0.0)

    if a.stems:
        stem2idx = {Path(p).stem: i for i, p in enumerate(ds.paths)}
        idxs = [stem2idx[s] for s in a.stems if s in stem2idx]
    else:
        rng = np.random.default_rng(a.seed)
        idxs = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False).tolist()

    # alignment scores
    align = {}
    aj = Path(a.cache_dir) / "_alignment.json"
    if aj.exists():
        align = json.load(open(aj))

    n = len(idxs)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n), squeeze=False)
    titles = ["1. Raw RGB3\n(H/2 x W/2)", "2. Raw ↑2x nearest\n(H x W)",
              "3. Target\n(H x W)", "4. |↑raw - target| x4"]

    lines = []
    for row, idx in enumerate(idxs):
        # 手动加载原始 npz 拿全尺寸 target (dataset 会 downsample 到 raw 尺寸)
        path = ds.paths[idx]
        stem = Path(path).stem
        z = np.load(path)
        raw4 = z["raw"].astype(np.float32)      # (4, h, w)
        tgt3 = z["target"].astype(np.float32)   # (3, H, W)

        # 与 dataset 相同的处理
        rgb3 = np.stack([raw4[0], 0.5 * (raw4[1] + raw4[2]), raw4[3]], axis=0)
        img = torch.from_numpy(rgb3).clamp_(0, 1)                    # (3, h, w)
        h, w = img.shape[-2], img.shape[-1]

        # 手动 upsample 到 target 分辨率 (nearest, 看像素级对齐)
        img_up = F.interpolate(img.unsqueeze(0), size=(2 * h, 2 * w),
                               mode="nearest").squeeze(0)             # (3, 2h, 2w)
        # target 全分辨率 (若 target 大 1px 则裁)
        tgt = torch.from_numpy(tgt3).clamp_(0, 1)[:, : 2 * h, : 2 * w]

        diff = (img_up - tgt).abs().mean(dim=0, keepdim=True).expand(3, -1, -1) * 4

        panels = [img, img_up, tgt, diff]
        for col, (ax, x) in enumerate(zip(axes[row], panels)):
            if col == 3:
                ax.imshow(x.permute(1, 2, 0).clamp(0, 1).cpu().numpy(),
                          cmap="hot", aspect="equal")
            else:
                ax.imshow(_to_np(x), aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax.set_title(titles[col], fontsize=12)
            if col == 0:
                ax.set_xlabel(f"{h} x {w}", fontsize=9)
            elif col == 1:
                ax.set_xlabel(f"{2 * h} x {2 * w}", fontsize=9)
            elif col == 2:
                ax.set_xlabel(f"{tgt.shape[-2]} x {tgt.shape[-1]}", fontsize=9)

        corr = align.get(stem, float("nan"))
        axes[row][0].set_ylabel(f"{stem}\ncorr={corr:.3f}", fontsize=9,
                                rotation=0, ha="right", va="center", labelpad=70)

        # 量化: 计算 img_up vs tgt 的 MAE 和 1px 错位相关性
        mae = (img_up - tgt).abs().mean().item()
        # 检查是否严格对齐: 若错位 1px, corr(shifted) 会显著更高
        up_np = img_up.permute(1, 2, 0).numpy()
        tg_np = tgt.permute(1, 2, 0).numpy()
        corr_center = np.corrcoef(up_np[1:-1, 1:-1].ravel(), tg_np[1:-1, 1:-1].ravel())[0, 1]
        corr_right = np.corrcoef(up_np[1:-1, 1:-1].ravel(), tg_np[1:-1, 2:].ravel())[0, 1] if tg_np.shape[1] > 2 else 0
        lines.append(f"{stem}: raw {h}x{w} -> {2*h}x{2*w} vs tgt {tgt.shape[-2]}x{tgt.shape[-1]}  "
                     f"MAE={mae:.4f}  corr(中心)={corr_center:.4f}  corr(右移1px)={corr_right:.4f}")

    fig.suptitle("Raw / Target 像素级尺寸对齐验证 (nearest upsample, 不做 ISP)", fontsize=14)
    fig.tight_layout(rect=(0.08, 0, 1, 0.97))
    png = out_dir / "size_match.png"
    fig.savefig(png, dpi=110)
    print(f"saved {png}\n")
    print("\n".join(lines))
    (out_dir / "report.txt").write_text("\n".join(lines) + "\n")
    print(f"\nsaved {out_dir / 'report.txt'}")


if __name__ == "__main__":
    main()
