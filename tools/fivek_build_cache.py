"""Rebuild the FiveK Expert-C cache with orientation-correct RAW planes.

Why: the original cache generation (script never committed) packed the DNG
mosaic into RGGB planes and resized to the target's half-size, but did NOT
apply the DNG EXIF flip to the RAW plane — while the target TIFFs were
already EXIF-corrected. For ~1/3 of images (portrait shots, flip=5/6) the
stored raw and target are rotated 90° relative to each other.
`fix_rotation.json` patched this at load time, but that patch itself breaks
`_resize`'s center-crop (crop happens in mismatched aspect frames), leaving
32/100 val images geometrically misaligned (corr < 0.5).

This script rebuilds ONLY the `raw` plane from the local DNGs:

    raw  = DNG mosaic → per-CFA black subtraction / white normalize
           → pack R,G,G,B planes (per-camera raw_pattern)
           → apply EXIF flip (empirically mapped: 0→k0, 5→k1, 6→k3, 3→k2)
           → cv2.INTER_AREA resize to the existing target's (H/2, W/2)

`target` is copied verbatim from the old cache (it was generated from
EXIF-corrected TIFFs and is correct). The new cache needs no
fix_rotation.json at all.

Usage:
    python tools/fivek_build_cache.py \
        --old-cache /home/jing/datasets/fivek/cache_expert_c \
        --raw-root  /home/jing/datasets/fivek/fivek_dataset/raw_photos \
        --out       /home/jing/datasets/fivek/cache_expert_c \
        [--limit 50] [--workers 8]

Output: <out>/<stem>.npz {raw: (4,H/2,W/2) f16, target: (3,H,W) f16},
plus <out>/_build_report.json with per-file stats.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np

# libraw flip code → numpy rot90 k on the packed planes (axes=(1,2)).
# Empirically validated against rawpy postprocess (which applies flip) on a
# random sample covering flip=0/5/6: best-k unanimous, corr≈0.92.
#  - flip 0: no rotation
#  - flip 3: 180°
#  - flip 5: 90° CCW   (rot90 k=1)
#  - flip 6: 90° CW    (rot90 k=3)
_FLIP_TO_K = {0: 0, 1: -1, 2: -1, 3: 2, 4: -1, 5: 1, 6: 3, 7: -1, 8: -1}
# -1 = mirrored variant (never seen in FiveK; file is skipped loudly)


def build_one(args):
    stem, dng_path, old_npz, out_dir = args
    import cv2
    import rawpy

    try:
        # ---- target: copy from old cache (already EXIF-correct) ----
        old = np.load(old_npz)
        target = old["target"]  # (3, H, W) f16, keep dtype/bytes verbatim

        # ---- raw: rebuild from DNG ----
        with rawpy.imread(dng_path) as r:
            flip = r.sizes.flip
            black = np.asarray(r.black_level_per_channel, dtype=np.float32)
            white = float(r.white_level)
            pat = r.raw_pattern.copy()
            mosaic_full = r.raw_image.copy().astype(np.float32)
            sizes = r.sizes

        k = _FLIP_TO_K.get(int(flip), -1)
        if k < 0:
            return (stem, "skipped", f"unsupported flip={flip} (mirrored)")

        # X-Trans (6x6 CFA, e.g. Fujifilm) cannot be packed into 4 Bayer
        # planes — skip loudly. ~1/300 of FiveK; not worth a lossy hack.
        if pat.shape != (2, 2):
            return (stem, "skipped", f"non-Bayer CFA {pat.shape}")

        # Use the active-image crop (what postprocess sees), not the raw
        # frame with optical-black margins: margins carry sensor-border
        # garbage that wrecks alignment with the target. Crop offsets are
        # parity-aligned to the CFA when even; when odd, expand by 1 px.
        cl = int(sizes.crop_left_margin)
        ct = int(sizes.crop_top_margin)
        cw = int(sizes.crop_width)
        ch = int(sizes.crop_height)
        if cl % 2: cl, cw = cl - 1, cw + 1
        if ct % 2: ct, ch = ct - 1, ch + 1
        mosaic = mosaic_full[ct:ct + ch, cl:cl + cw]

        # Normalize: per-CFA-position black level tiled to full mosaic.
        h, w = mosaic.shape
        mosaic = mosaic[: h - (h % 2), : w - (w % 2)]
        bm2 = black[pat]  # (2,2)
        black_map = np.tile(bm2, (mosaic.shape[0] // 2 + 1,
                                  mosaic.shape[1] // 2 + 1))[: mosaic.shape[0],
                                                             : mosaic.shape[1]]
        mosaic = (mosaic - black_map) / white

        # Pack planes by CFA color code (0=R, 1=G, 2=B, 3=G2) → (4, H/2, W/2)
        # in canonical R, G, G, B order (dataset averages the two greens).
        planes = {0: [], 1: [], 2: [], 3: []}
        for (i, j), c in np.ndenumerate(pat):
            planes[int(c)].append(mosaic[i::2, j::2])
        raw4 = np.stack([
            planes[0][0],
            planes[1][0],
            planes[3][0],
            planes[2][0],
        ])

        # Apply EXIF orientation so raw is upright, matching the target.
        if k:
            raw4 = np.ascontiguousarray(np.rot90(raw4, k, axes=(1, 2)))

        # Resize to the target's half spatial size (INTER_AREA, same as the
        # original cache generation — verified MAE=3e-5 on flip=0 files).
        tH, tW = int(target.shape[1]), int(target.shape[2])
        rH, rW = tH // 2, tW // 2
        raw4 = np.stack([
            cv2.resize(raw4[i], (rW, rH), interpolation=cv2.INTER_AREA)
            for i in range(4)
        ]).astype(np.float16)

        # Shape sanity: portrait targets get portrait raws and vice versa.
        if raw4.shape[1:] != (rH, rW):
            return (stem, "skipped", f"shape mismatch {raw4.shape} vs ({rH},{rW})")

        np.savez_compressed(
            os.path.join(out_dir, f"{stem}.npz"),
            raw=raw4, target=target,
        )
        return (stem, "ok", {
            "flip": int(flip), "k": k,
            "raw_hw": [int(raw4.shape[1]), int(raw4.shape[2])],
            "target_hw": [tH, tW],
            "aspect_raw": round(raw4.shape[2] / raw4.shape[1], 4),
            "aspect_tgt": round(tW / tH, 4),
        })
    except Exception:
        return (stem, "error", traceback.format_exc(limit=3))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-cache", default="/home/jing/datasets/fivek/cache_expert_c")
    ap.add_argument("--raw-root",
                    default="/home/jing/datasets/fivek/fivek_dataset/raw_photos")
    ap.add_argument("--out", default="/home/jing/datasets/fivek/cache_expert_c")
    ap.add_argument("--limit", type=int, default=0, help="only first N (smoke)")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    # rawpy/OpenMP deadlocks under fork-start-method workers — use spawn.
    mp.set_start_method("spawn", force=True)

    os.makedirs(a.out, exist_ok=True)

    # Stems come from the old cache (we need its target anyway).
    old_files = sorted(f for f in os.listdir(a.old_cache) if f.endswith(".npz"))
    stems = [f[:-4] for f in old_files]
    if a.limit:
        stems = stems[: a.limit]

    # Locate each stem's DNG (raw_photos/**/photos/<stem>.dng).
    dng_index = {}
    for p in Path(a.raw_root).rglob("*.dng"):
        dng_index.setdefault(p.stem, str(p))

    jobs, missing = [], []
    for stem in stems:
        dng = dng_index.get(stem)
        if dng is None:
            missing.append(stem)
            continue
        jobs.append((stem, dng, os.path.join(a.old_cache, f"{stem}.npz"), a.out))

    print(f"stems: {len(stems)} | dng found: {len(jobs)} | missing dng: {len(missing)}")
    if missing:
        print("  missing (first 5):", missing[:5])

    report, n_ok = {}, 0
    with ProcessPoolExecutor(max_workers=a.workers) as ex:
        futs = {ex.submit(build_one, j): j[0] for j in jobs}
        for i, fut in enumerate(as_completed(futs)):
            stem, status, info = fut.result()
            report[stem] = {"status": status, "info": info}
            if status == "ok":
                n_ok += 1
            elif status == "error":
                print(f"ERROR {stem}:\n{info}")
            if (i + 1) % 250 == 0:
                print(f"  {i + 1}/{len(jobs)} done ({n_ok} ok)")

    with open(os.path.join(a.out, "_build_report.json"), "w") as fh:
        json.dump(report, fh)

    n_err = sum(1 for v in report.values() if v["status"] == "error")
    n_skip = sum(1 for v in report.values() if v["status"] == "skipped")
    print(f"\nrebuild complete: {n_ok} ok, {n_err} error, {n_skip} skipped, "
          f"{len(missing)} missing dng")

    # Aspect agreement summary (raw vs target orientation consistency).
    aspects = [v["info"] for v in report.values()
               if v["status"] == "ok" and isinstance(v["info"], dict)]
    if aspects:
        both_portrait = sum(1 for x in aspects
                            if x["aspect_raw"] < 1 and x["aspect_tgt"] < 1)
        both_landscape = sum(1 for x in aspects
                             if x["aspect_raw"] > 1 and x["aspect_tgt"] > 1)
        disagree = len(aspects) - both_portrait - both_landscape
        print(f"orientation agreement: {both_portrait} portrait+portrait, "
              f"{both_landscape} landscape+landscape, {disagree} DISAGREE")


if __name__ == "__main__":
    main()
