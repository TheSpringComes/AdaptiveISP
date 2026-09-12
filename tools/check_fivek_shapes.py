"""tools/check_fivek_shapes.py — FiveK 缓存 raw/target 尺寸一致性审计.

检查点:
  1. 所有 npz: raw (4, h, w) 与 target (3, H, W) 是否严格满足 H=2h, W=2w
     (Input Adapter 的 mosaic 重建假设: demosaic 后 (3, 2h, 2w) 与
     target 全分辨率对齐).
  2. 统计不满足 2x 关系的样本 (允许 target 大 1px, 加载时裁剪).
  3. 输出尺寸分布摘要 (常见分辨率).

用法:
    python tools/check_fivek_shapes.py
    python tools/check_fivek_shapes.py --show-mismatch
"""
from __future__ import annotations

import argparse
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/jing/datasets/fivek/cache_expert_c")
    ap.add_argument("--show-mismatch", action="store_true",
                    help="打印所有不满足 2x 关系的样本名")
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.cache) if f.endswith(".npz"))
    n_ok, n_ok_crop, n_bad = 0, 0, 0
    size_counter = Counter()
    mismatches = []

    for f in files:
        try:
            z = np.load(os.path.join(a.cache, f))
            raw, tgt = z["raw"], z["target"]
            if raw.shape[0] != 4 or tgt.shape[0] != 3:
                mismatches.append((f[:-4], f"channel 数不对: raw {raw.shape} / tgt {tgt.shape}"))
                n_bad += 1
                continue
            h, w = raw.shape[1], raw.shape[2]
            th, tw = tgt.shape[1], tgt.shape[2]
            size_counter[(h, w)] += 1
            # 允许 target 恰好 2x 或比 2x 大 1px (加载时裁)
            if th == 2 * h and tw == 2 * w:
                n_ok += 1
            elif th in (2 * h, 2 * h + 1) and tw in (2 * w, 2 * w + 1):
                n_ok_crop += 1
                mismatches.append((f[:-4], f"target 大 1px (可裁剪): raw {h}x{w} / tgt {th}x{tw}"))
            else:
                n_bad += 1
                mismatches.append((f[:-4], f"尺寸关系异常: raw {h}x{w} / tgt {th}x{tw}"))
        except Exception as e:
            n_bad += 1
            mismatches.append((f[:-4], f"加载失败: {e}"))

    print(f"缓存目录: {a.cache}")
    print(f"总文件数: {len(files)}")
    print(f"  严格 2x 匹配:        {n_ok} ({n_ok / len(files) * 100:.1f}%)")
    print(f"  target 大 1px (可裁): {n_ok_crop} ({n_ok_crop / len(files) * 100:.1f}%)")
    print(f"  异常 (无法对齐):      {n_bad}")
    print()
    print(f"不同分辨率组合数: {len(size_counter)}")
    print("最常见的 10 种 (h x w):")
    for (h, w), cnt in size_counter.most_common(10):
        print(f"  {h:4d} x {w:4d}  x {cnt}")

    if a.show_mismatch and mismatches:
        print(f"\n--- 非严格匹配样本 ({len(mismatches)} 个) ---")
        for stem, msg in mismatches[:50]:
            print(f"  {stem}: {msg}")
        if len(mismatches) > 50:
            print(f"  ... 还有 {len(mismatches) - 50} 个")


if __name__ == "__main__":
    main()
