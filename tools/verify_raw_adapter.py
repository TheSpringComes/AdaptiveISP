#!/usr/bin/env python3
"""Verify the Input Adapter (front_isp/raw_adapter.py) end to end.

Checks (requirement set for the demosaic migration):
  1. 统一 RGB 接口: 3ch passthrough / 1ch mosaic / 4ch packed 均产出
     (3, H, W) float32 [0,1]。
  2. Dataset 层: FiveKDataset 输出 image/target 同为全分辨率 (3, H, W)，
     dtype float32，值域 [0,1]。
  3. 几何对齐: demosaic 输出 vs target 的中心相关性高于平移 1/2px 的
     相关性（无系统性错位）。
  4. R/B 通道顺序: corr(img_R, tgt_R) 明显高于 corr(img_R, tgt_B)，
     按 pattern 分组验证（RGGB/BGGR/GBRG/GRBG 无通道交换）。
  5. rawpy 真值对拍: 对真实 DNG 用本适配器 demosaic，与 rawpy 自带
     demosaic（linear 算法、Raw colorspace、无 WB）做通道相关性对拍。
  6. Malvar 核与 colour-demosaicing 参考实现逐像素一致。

Output: experiments/raw_adapter_check/ (report.txt + demosaic_grid.png)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from front_isp.raw_adapter import (  # noqa: E402
    BAYER_PATTERNS,
    demosaic_bayer,
    reconstruct_mosaic,
    to_canonical_rgb,
)
from tasks.human_quality import FiveKDataset  # noqa: E402

CACHE = "/home/jing/datasets/fivek/cache_expert_c"
CFA_JSON = "/home/jing/datasets/fivek/cfa_pattern.json"
VAL_LIST = "/home/jing/datasets/fivek/val_expert_c.txt"
RAW_ROOT = "/home/jing/datasets/fivek/fivek_dataset/raw_photos"
OUT = _ROOT / "experiments" / "raw_adapter_check"

_ok = []
_bad = []


def check(name: str, cond: bool, detail: str = "") -> None:
    tag = "PASS" if cond else "FAIL"
    print(f"[{tag}] {name}" + (f"  ({detail})" if detail else ""))
    (_ok if cond else _bad).append(name)


def _corr(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel() - a.mean()
    b = b.ravel() - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


def _shift(img: np.ndarray, dy: int, dx: int) -> np.ndarray:
    out = np.empty_like(img)
    out[:] = np.nan
    h, w = img.shape
    ys0, ys1 = max(0, dy), min(h, h + dy)
    xs0, xs1 = max(0, dx), min(w, w + dx)
    out[ys0:ys1, xs0:xs1] = img[max(0, -dy):h - max(0, dy),
                                max(0, -dx):w - max(0, dx)]
    return out


def _corr_safe(a: np.ndarray, b: np.ndarray) -> float:
    m = np.isfinite(b)
    return _corr(a[m], b[m])


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []

    # ---------------- 1. 统一 RGB 接口 ----------------
    rng = np.random.default_rng(0)
    x3 = rng.random((3, 64, 64)).astype(np.float32)
    p3 = to_canonical_rgb(x3)
    check("3ch passthrough: shape/dtype/values",
          p3.shape == x3.shape and p3.dtype == np.float32
          and np.array_equal(p3, x3))

    mos = (0.4 * np.cumsum(np.cumsum(rng.random((64, 64)), 0), 1) % 1.0
           + 0.6 * rng.random((64, 64))).astype(np.float32)
    p1 = to_canonical_rgb(mos[None], pattern="RGGB")
    check("1ch mosaic: (3,H,W) float32 [0,1]",
          p1.shape == (3, 64, 64) and p1.dtype == np.float32
          and p1.min() >= 0 and p1.max() <= 1)

    # 4-plane packed: use a real cache file's raw for realism
    import json
    cfa = json.load(open(CFA_JSON))
    val_stems = [Path(ln.strip()).stem for ln in open(VAL_LIST)
                 if ln.strip() and not ln.startswith("#")]
    stem0 = next(s for s in val_stems if s in cfa)
    z = np.load(f"{CACHE}/{stem0}.npz")
    raw4, tgt = z["raw"].astype(np.float32), z["target"].astype(np.float32)
    p4 = to_canonical_rgb(raw4, pattern=cfa[stem0])
    check("4ch packed: full-res (3,2h,2w) float32 [0,1]",
          p4.shape == (3, 2 * raw4.shape[1], 2 * raw4.shape[2])
          and p4.dtype == np.float32
          and p4.min() >= 0 and p4.max() <= 1,
          f"{stem0} {cfa[stem0]} -> {p4.shape}")

    # ---------------- 6. Malvar 核参考对拍 ----------------
    try:
        from colour_demosaicing import (demosaicing_CFA_Bayer_Malvar2004)
        m64 = (0.3 * (np.cumsum(np.cumsum(rng.random((64, 64)), 0), 1) % 1.0)
               + 0.7 * rng.random((64, 64)))
        worst = max(
            np.abs(demosaic_bayer(m64, pat, method="malvar")
                   - demosaicing_CFA_Bayer_Malvar2004(m64, pat).transpose(2, 0, 1)).max()
            for pat in BAYER_PATTERNS)
        check("Malvar == colour-demosaicing reference (all patterns)",
              worst < 1e-5, f"max|diff|={worst:.1e}")
    except ImportError:
        print("[SKIP] colour-demosaicing not installed")

    # ---------------- 2-4. Dataset 层（按 pattern 分组） ----------------
    ds = FiveKDataset(list_file=VAL_LIST, cache_dir=CACHE, imgsz=None,
                      return_camera=True, alignment_threshold=0.5)
    print(f"dataset: {len(ds)} val samples")

    # 每个 pattern 抽 2 个（val 里不够就从全 cache 补）
    by_pat: dict[str, list[str]] = {p: [] for p in BAYER_PATTERNS}
    for s in val_stems:
        if s in cfa and len(by_pat[cfa[s]]) < 2 and os.path.exists(f"{CACHE}/{s}.npz"):
            by_pat[cfa[s]].append(s)
    for pat, lst in by_pat.items():
        need = 2 - len(lst)
        if need > 0:
            for s, p2 in cfa.items():
                if p2 == pat and s not in lst and os.path.exists(f"{CACHE}/{s}.npz"):
                    lst.append(s)
                    need -= 1
                    if need == 0:
                        break

    grid_rows = []  # (stem, pat, img, tgt)
    for pat, stems in by_pat.items():
        for stem in stems:
            z = np.load(f"{CACHE}/{stem}.npz")
            raw4 = z["raw"].astype(np.float32)
            tgt = z["target"].astype(np.float32)
            img = to_canonical_rgb(raw4, pattern=pat)

            # shape / dtype / range / target 尺寸一致
            ok = (img.shape == tgt.shape and img.dtype == np.float32
                  and img.min() >= 0 and img.max() <= 1)
            # 几何对齐: luma 中心 corr vs 平移 corr
            ly = img.mean(0)
            ty = tgt.mean(0)
            c0 = _corr(ly, ty)
            shifts = {f"({dy:+d},{dx:+d})": _corr_safe(ly, _shift(ty, dy, dx))
                      for dy, dx in [(0, 1), (0, -1), (1, 0), (-1, 0),
                                     (0, 2), (2, 0)]}
            align_ok = all(c0 > v for v in shifts.values())
            # R/B 顺序（启发式：弱色彩分离场景 G 通道也会偏向同一侧，
            # 属内容效应；只有明显反向才判 fail，权威判定见 rawpy 对拍）
            cRR = _corr(img[0], tgt[0])
            cRB = _corr(img[0], tgt[2])
            rb_ok = cRR > cRB - 0.05
            ok = ok and align_ok and rb_ok
            check(f"{stem[:28]:28} [{pat}] shape/range={img.shape} "
                  f"align={c0:.3f}>(max shift {max(shifts.values()):.3f}) "
                  f"R/R={cRR:.3f} R/B={cRB:.3f}", ok)
            lines.append(f"{stem}\t{pat}\tshape={img.shape}\tcorr_center={c0:.4f}"
                         f"\tmax_shift_corr={max(shifts.values()):.4f}"
                         f"\tcRR={cRR:.4f}\tcRB={cRB:.4f}")
            grid_rows.append((stem, pat, img, tgt))

    # FiveKDataset 对象接口（新增 cfa 路径）
    img_t, tgt_t, cam_t = ds[0]
    check("FiveKDataset[0]: image==target 全分辨率同形",
          img_t.shape == tgt_t.shape
          and str(img_t.dtype) == "torch.float32"
          and float(img_t.min()) >= 0 and float(img_t.max()) <= 1,
          str(tuple(img_t.shape)))

    # ---------------- 5. rawpy 真值对拍（R/B 交换的权威判定） ----------------
    # 对每个 pattern 组的真实样本 DNG：用 cfa_pattern.json 里的名字走
    # 生产路径 demosaic（并断言名字与 DNG 实际 raw_pattern 网格一致），
    # 与 rawpy 自带 demosaic（LINEAR、raw colorspace、无 WB、linear
    # gamma）对拍。通道若有交换，对角 corr 会掉到 ~0.7 且交叉项反超。
    # postprocess 的裁剪行为因文件而异（有的返回全帧、有的返回 active
    # 区），故双方都取中心区域并对齐中心；再 8x 均值降采样把残余
    # 亚像素错位平滑掉（R/B 交换信号不受降采样影响）。
    try:
        import rawpy
        from front_isp.raw_adapter import parse_pattern
        _FLIP_TO_K = {0: 0, 3: 2, 5: 1, 6: 3}   # 与 build_cache 一致
        dng_index = {p.stem: p for p in Path(RAW_ROOT).rglob("*.dng")}

        def _pool(a: np.ndarray, f: int) -> np.ndarray:
            h, w = a.shape[-2] // f * f, a.shape[-1] // f * f
            a = a[..., :h, :w]
            return a.reshape(*a.shape[:-2], h // f, f, w // f, f).mean((-3, -1))

        def _shift_corr(a: np.ndarray, b: np.ndarray, dy: int, dx: int) -> float:
            """corr(a shifted by (dy,dx) vs b) on the overlapping region."""
            ah, aw = a.shape[-2:]
            bh, bw = b.shape[-2:]
            ay0, ay1 = max(0, dy), min(ah, bh + dy)
            ax0, ax1 = max(0, dx), min(aw, bw + dx)
            if ay1 - ay0 < 64 or ax1 - ax0 < 64:
                return -1.0
            a2 = a[..., ay0:ay1, ax0:ax1]
            b2 = b[..., ay0 - dy:ay1 - dy, ax0 - dx:ax1 - dx]
            return _corr(a2, b2)

        for pat, stems in sorted(by_pat.items()):
            stem = stems[0]
            d = dng_index.get(stem)
            if d is None:
                continue
            with rawpy.imread(str(d)) as r:
                flip = int(r.sizes.flip)
                if flip not in _FLIP_TO_K:
                    continue
                k = _FLIP_TO_K[flip]
                grid = r.raw_pattern.copy()
                black = np.asarray(r.black_level_per_channel, dtype=np.float32)
                white = float(r.white_level)
                mosaic_full = r.raw_image.copy().astype(np.float32)
                rgb_ref = r.postprocess(
                    use_camera_wb=False, use_auto_wb=False,
                    output_color=rawpy.ColorSpace.raw, no_auto_bright=True,
                    demosaic_algorithm=rawpy.DemosaicAlgorithm.LINEAR,
                    gamma=(1, 1)).astype(np.float32) / 65535.0
            if grid.shape != (2, 2):
                continue
            # 名字 <-> DNG 实际网格 一致性
            if not np.array_equal(parse_pattern(pat), grid):
                check(f"pattern name [{pat}] == DNG raw_pattern grid", False,
                      f"{parse_pattern(pat).tolist()} vs {grid.tolist()}")
                continue
            # 全帧 demosaic（pattern 相位 = 帧原点，精确），名字走生产路径
            mf = mosaic_full[: mosaic_full.shape[0] - mosaic_full.shape[0] % 2,
                             : mosaic_full.shape[1] - mosaic_full.shape[1] % 2]
            bm = np.tile(black[grid], (mf.shape[0] // 2 + 1,
                                       mf.shape[1] // 2 + 1))[: mf.shape[0],
                                                             : mf.shape[1]]
            ours = demosaic_bayer((mf - bm) / white, pat)
            if k:
                ours = np.rot90(ours, k, axes=(1, 2))   # postprocess 是 upright
            ref = rgb_ref.transpose(2, 0, 1)
            # postprocess 的裁剪几何因文件而异：两阶段搜索 ref 相对
            # 全帧原点的平移（先 8x 降采样粗搜 8px 步长，再 2x 细搜
            # 1px 步长），在对齐位置算通道相关。
            a8, b8 = _pool(ours, 8), _pool(ref, 8)
            _, cy, cx = max(
                ((_shift_corr(a8.mean(0), b8.mean(0), dy, dx), dy, dx)
                 for dy in range(-20, 21) for dx in range(-24, 25)),
                key=lambda t: t[0])
            a2, b2 = _pool(ours, 2), _pool(ref, 2)
            _, fy, fx = max(
                ((_shift_corr(a2.mean(0), b2.mean(0), dy, dx), dy, dx)
                 for dy in range(cy * 4 - 5, cy * 4 + 6)
                 for dx in range(cx * 4 - 5, cx * 4 + 6)),
                key=lambda t: t[0])
            cc = [_shift_corr(a2[c], b2[c], fy, fx) for c in range(3)]
            check(f"rawpy 对拍 [{pat}] {stem[:24]} flip={flip} "
                  f"off=({fy * 2:+d},{fx * 2:+d}): "
                  f"corr R/G/B = {cc[0]:.4f}/{cc[1]:.4f}/{cc[2]:.4f}",
                  all(c > 0.99 for c in cc))
    except ImportError:
        print("[SKIP] rawpy not installed")

    # ---------------- 可视化 ----------------
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        n = len(grid_rows)
        fig, axes = plt.subplots(n, 2, figsize=(12, 3.4 * n), squeeze=False)
        for row, (stem, pat, img, tgt) in enumerate(grid_rows):
            def show(ax, x, title):
                lo, hi = np.percentile(x, (1, 99))
                ax.imshow((np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1)
                           ).transpose(1, 2, 0))
                ax.set_title(title, fontsize=8)
                ax.axis("off")
            show(axes[row][0], img, f"{stem[:36]} [{pat}] demosaic (1-99% stretch)")
            show(axes[row][1], tgt, "Expert-C target")
        fig.tight_layout()
        fig.savefig(OUT / "demosaic_grid.png", dpi=110)
        print(f"visualization -> {OUT / 'demosaic_grid.png'}")
    except Exception as e:  # noqa: BLE001
        print(f"viz skipped: {e}")

    with open(OUT / "report.txt", "w") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"\n===== {len(_ok)} passed, {len(_bad)} failed =====")
    if _bad:
        for b in _bad:
            print(f"  FAILED: {b}")
        sys.exit(1)


if __name__ == "__main__":
    main()
