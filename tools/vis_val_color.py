"""tools/vis_val_color.py — FiveK val 颜色链路 4 格对比可视化.

诊断 "呈现的颜色有问题" 究竟在哪一级引入。每张 val 样本一行四列:

  col1  Dataset 直出 RGB3   — Input Adapter: Bayer 重建 + demosaic
                              (0.5*Malvar + 0.5*Bilinear, 未 WB / 未 CCM,
                               线性相机空间, 全分辨率)
  col2  Canonical front-ISP — 固定 AWB+CCM+smoothstep+gamma
                              (cam2rgb 为 CycleISP 合成管线校准,
                               对真实 FiveK RAW 失配,作为"坏基线"对照)
  col3  Learnable identity — LearnableFrontISP(init=identity, 未训练)
                              (展示 Stage-1 尚未学习时的直通输出;
                               若传入 --ckpt 则加载 Stage-1 结果)
  col4  Expert-C target     — sRGB 参考

每张图角落标注逐通道均值;行标题给 stem / camera / corr(_alignment.json)。
同时打印每张 col2/col3 vs target 的 PSNR/SSIM/ΔE, 终端汇总中位数。

输出: experiments/vis_val_color/<tag>/val_color_grid.png (+ metrics.txt)

用法:
    python tools/vis_val_color.py --n 4
    python tools/vis_val_color.py --n 6 --ckpt experiments/v31_stage1/ckpt/CalibISP_iter_2500.pth
    python tools/vis_val_color.py --stems a0001-jmac_DSC1459 a0006-IMG_2787
"""
from __future__ import annotations

import argparse
import json
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
from tasks.human_quality.metrics import psnr_batch, ssim_batch, delta_e_batch
from front_isp.canonical import CanonicalBackbone
from front_isp.learnable import LearnableFrontISP as CalibratedFrontISP


def _to_np(x: torch.Tensor) -> np.ndarray:
    return x.clamp(0, 1).permute(1, 2, 0).cpu().numpy()


def _chan_means(x: torch.Tensor) -> str:
    m = x.mean(dim=(1, 2))
    return f"R{m[0]:.2f} G{m[1]:.2f} B{m[2]:.2f}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="")
    ap.add_argument("--ckpt", default=None,
                    help="Stage-1 CalibISP ckpt; 缺省则 col3 用 identity")
    ap.add_argument("--stems", nargs="*", default=None,
                    help="指定 stem 而不是随机抽样")
    ap.add_argument("--val-list",
                    default="/home/jing/datasets/fivek/val_expert_c.txt")
    ap.add_argument("--cache-dir",
                    default="/home/jing/datasets/fivek/cache_expert_c")
    a = ap.parse_args()

    out_dir = _ROOT / "experiments" / "vis_val_color" / (a.tag or f"seed{a.seed}")
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = FiveKDataset(list_file=a.val_list, cache_dir=a.cache_dir,
                      imgsz=None, return_camera=True, alignment_threshold=0.5)

    # 选样本
    if a.stems:
        stem2idx = {Path(p).stem: i for i, p in enumerate(ds.paths)}
        idxs = [stem2idx[s] for s in a.stems if s in stem2idx]
        missing = [s for s in a.stems if s not in stem2idx]
        if missing:
            print(f"warn: stems not in kept val set: {missing}")
    else:
        rng = np.random.default_rng(a.seed)
        idxs = rng.choice(len(ds), size=min(a.n, len(ds)), replace=False).tolist()

    # front ISPs
    canonical = CanonicalBackbone({}).eval()
    calib_cfg = {"calibration": {
        "camera_specific": True, "n_cameras": ds.n_cameras,
        "init": {"type": "identity"},
    }}
    if a.ckpt:
        calib_cfg["calibration"]["ckpt"] = a.ckpt
    calibrated = CalibratedFrontISP(calib_cfg).eval()  # legacy alias for LearnableFrontISP

    # alignment scores for row titles
    align = {}
    aj = Path(a.cache_dir) / "_alignment.json"
    if aj.exists():
        align = json.load(open(aj))

    cam_names = ds.camera_names or ["<unknown>"]

    n = len(idxs)
    fig, axes = plt.subplots(n, 4, figsize=(20, 5 * n), squeeze=False)
    col_titles = ["1. Dataset RGB3\n(demosaiced linear)",
                  "2. Canonical\n(fixed ISP)",
                  "3. Calibrated\n(%s)" % ("Stage-1 ckpt" if a.ckpt else "identity"),
                  "4. Expert-C target"]

    lines = []
    for row, idx in enumerate(idxs):
        img, tgt, cam_id = ds[idx]                       # (3,h,w) in [0,1]
        stem = Path(ds.paths[idx]).stem
        cam = cam_names[cam_id] if cam_id < len(cam_names) else str(cam_id)
        corr = align.get(stem, float("nan"))

        with torch.no_grad():
            b = img.unsqueeze(0)
            out_can = canonical.process(b).squeeze(0)
            out_cal = calibrated.process(
                b, {"camera_id": torch.tensor([cam_id])}).squeeze(0)

        panels = [img, out_can, out_cal, tgt]
        for col, (ax, x) in enumerate(zip(axes[row], panels)):
            ax.imshow(_to_np(x), aspect="equal")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_xlabel(_chan_means(x), fontsize=9)
            if row == 0:
                ax.set_title(col_titles[col], fontsize=12)
        axes[row][0].set_ylabel(
            f"{stem}\n{cam}\ncorr={corr:.3f}", fontsize=9, rotation=0,
            ha="right", va="center", labelpad=70)

        # metrics col2/col3 vs target (full-res, on same grid)
        with torch.no_grad():
            t = tgt.unsqueeze(0)
            for name, out in (("canonical", out_can), ("calibrated", out_cal)):
                o = out.unsqueeze(0)
                psnr = psnr_batch(o, t).item()
                ssim = ssim_batch(o, t).item()
                de = delta_e_batch(o, t).mean().item()
                lines.append(f"{stem}  {name:10s}  PSNR={psnr:6.2f}  "
                             f"SSIM={ssim:.4f}  dE76={de:6.2f}")

    fig.suptitle("FiveK val color-chain diagnosis — where does the color go wrong?",
                 fontsize=14)
    fig.tight_layout(rect=(0.10, 0, 1, 0.97))
    png = out_dir / "val_color_grid.png"
    fig.savefig(png, dpi=110)
    print(f"saved {png}")

    txt = out_dir / "metrics.txt"
    txt.write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"saved {txt}")


if __name__ == "__main__":
    main()
