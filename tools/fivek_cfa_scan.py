#!/usr/bin/env python3
"""Scan all FiveK DNGs for their CFA pattern (raw_pattern) distribution.

Answers: are all raws RGGB? (No — Canon is RGGB, Nikon is often BGGR.)
The cache packer (tools/fivek_build_cache.py) is pattern-agnostic: it packs
by color code into canonical R, G, G, B channels regardless of the sensor
CFA, so downstream sees R,G,G,B either way.

Usage:
    python tools/fivek_cfa_scan.py [--raw-root DIR] [--workers 8]
Output: experiments/fivek_cfa_scan/report.txt (+ per-camera cross-tab)
"""
import argparse
import collections
import json
import multiprocessing as mp
import os
from pathlib import Path

import rawpy

_PAT_NAME = {
    ((0, 1), (3, 2)): "RGGB",
    ((2, 3), (1, 0)): "BGGR",
    ((3, 2), (0, 1)): "GBRG",
    ((1, 0), (2, 3)): "GRBG",
}


def _scan(dng_path):
    stem = os.path.basename(dng_path)[:-4]
    try:
        with rawpy.imread(dng_path) as r:
            pat = r.raw_pattern.copy()
        if pat.shape != (2, 2):
            return (stem, f"non-bayer {pat.shape}")
        name = _PAT_NAME.get(tuple(map(tuple, pat.tolist())), str(pat.tolist()))
        return (stem, name)
    except Exception as e:  # noqa: BLE001
        return (stem, f"error: {e}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-root",
                    default="/home/jing/datasets/fivek/fivek_dataset/raw_photos")
    ap.add_argument("--out", default="experiments/fivek_cfa_scan")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()

    dngs = [str(p) for p in Path(a.raw_root).rglob("*.dng")]
    print(f"scanning {len(dngs)} DNGs with {a.workers} workers ...", flush=True)

    # rawpy deadlocks under fork — use spawn.
    mp.set_start_method("spawn", force=True)
    with mp.Pool(a.workers) as pool:
        results = pool.map(_scan, dngs, chunksize=16)

    pat_count = collections.Counter(r[1] for r in results)

    # Cross-tab with camera.json (stem -> camera model) if available.
    cam_json = "/home/jing/datasets/fivek/camera.json"
    pat_by_cam = collections.defaultdict(collections.Counter)
    cams = {}
    if os.path.exists(cam_json):
        with open(cam_json) as fh:
            cams = json.load(fh)
        for stem, pat in results:
            cam = cams.get(stem, "?")
            pat_by_cam[cam][pat] += 1

    os.makedirs(a.out, exist_ok=True)
    lines = []
    lines.append(f"total DNGs scanned: {len(results)}")
    lines.append("pattern distribution:")
    for pat, n in pat_count.most_common():
        lines.append(f"  {pat!s:20} {n}")
    if pat_by_cam:
        lines.append("pattern x camera (top 15 cameras):")
        for cam in sorted(pat_by_cam, key=lambda c: -sum(pat_by_cam[c].values()))[:15]:
            lines.append(f"  {cam[:46]:46} {dict(pat_by_cam[cam])}")
    report = "\n".join(lines)
    print(report)

    with open(os.path.join(a.out, "report.txt"), "w") as fh:
        fh.write(report + "\n")
    with open(os.path.join(a.out, "per_file.json"), "w") as fh:
        json.dump(dict(results), fh, indent=0)


if __name__ == "__main__":
    main()
