"""Scan a rebuilt FiveK cache for geometrically misaligned raw/target pairs.

Computes corr(RGB3(raw), pooled target) per file (aspect-preserving resize to
a common size, so rotation/aspect bugs show up as low correlation) and writes
`_alignment.json`: {stem: corr}. Pairs below the threshold are candidates for
exclusion at train time (their Expert-C TIFFs mismatch the DNG — a legacy
cache-generation error we cannot fix by rebuilding raw).

Usage:
    python tools/fivek_scan_cache.py --cache /home/jing/datasets/fivek/cache_expert_c \
        [--threshold 0.5] [--workers 8]
"""
from __future__ import annotations

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor


def scan_one(path):
    import numpy as np
    import torch
    import torch.nn.functional as F

    d = np.load(path)
    raw = d["raw"].astype(np.float32)
    tgt = d["target"].astype(np.float32)
    rgb3 = np.stack([raw[0], (raw[1] + raw[2]) / 2, raw[3]])
    pooled = F.avg_pool2d(torch.from_numpy(tgt).unsqueeze(0), 2, 2).squeeze(0)
    pooled = pooled[:, : rgb3.shape[1], : rgb3.shape[2]]
    if pooled.shape[1] != rgb3.shape[1] or pooled.shape[2] != rgb3.shape[2]:
        pooled = F.interpolate(
            pooled.unsqueeze(0), size=rgb3.shape[1:], mode="bilinear", align_corners=False
        ).squeeze(0)
    a = F.interpolate(torch.from_numpy(rgb3).unsqueeze(0), size=(256, 256),
                      mode="bilinear", align_corners=False)[0].numpy()
    b = pooled.numpy()
    if b.shape[1:] != (256, 256):
        b = F.interpolate(torch.from_numpy(b).unsqueeze(0), size=(256, 256),
                          mode="bilinear", align_corners=False)[0].numpy()
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default="/home/jing/datasets/fivek/cache_expert_c")
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    files = sorted(f for f in os.listdir(a.cache) if f.endswith(".npz"))
    out = {}
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        for fn, c in zip(files, ex.map(scan_one, [os.path.join(a.cache, f) for f in files])):
            out[fn[:-4]] = round(c, 4)

    with open(os.path.join(a.cache, "_alignment.json"), "w") as fh:
        json.dump(out, fh)

    vals = sorted(out.values())
    bad = [(s, c) for s, c in out.items() if c < a.threshold]
    import numpy as np
    print(f"n={len(vals)} mean={np.mean(vals):.3f} median={np.median(vals):.3f} "
          f"min={min(vals):.3f} p1={np.percentile(vals,1):.3f}")
    print(f"below {a.threshold}: {len(bad)}")
    for s, c in sorted(bad, key=lambda x: x[1])[:20]:
        print(f"  {c:+.3f}  {s}")


if __name__ == "__main__":
    main()
