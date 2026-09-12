"""Extract camera Make/Model from every FiveK DNG, write a metadata JSON.

V3.1 camera-specific Calibration needs a `<stem> → "<Make> <Model>"` map
per sample. This tool walks the `fivek_dataset/raw_photos/` tree, reads
EXIF tags 271 (Make) / 272 (Model) from every DNG, and writes the JSON
that `tasks.human_quality.dataset.FiveKDataset` auto-discovers as
`camera.json` next to the cache_dir. The dataset turns the sorted name
list into integer camera ids.

One-time setup, same pattern as `tools/fivek_orientation_metadata.py`:

    python tools/fivek_camera_metadata.py \
        --raw-root  /home/jing/datasets/fivek/fivek_dataset/raw_photos \
        --out       /home/jing/datasets/fivek/camera.json

A DNG with missing/unreadable EXIF maps to "unknown" (all such samples
share one calibration row).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image


def _read_camera(path: Path) -> str:
    """Return "<Make> <Model>" from EXIF, or "unknown" if missing."""
    try:
        with Image.open(path) as im:
            ex = im.getexif()
            make = str(ex.get(271, "")).strip()
            model = str(ex.get(272, "")).strip()
            if make and model:
                return f"{make} {model}"
            return model or make or "unknown"
    except Exception:
        return "unknown"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--raw-root", type=Path, required=True,
                   help="root containing DNGs (e.g. .../fivek_dataset/raw_photos)")
    p.add_argument("--out", type=Path, required=True,
                   help="output JSON path (e.g. .../fivek/camera.json)")
    args = p.parse_args()

    dngs = list(args.raw_root.rglob("*.dng"))
    if not dngs:
        raise SystemExit(f"no DNGs found under {args.raw_root}")
    print(f"scanning {len(dngs)} DNGs...")

    mapping: dict[str, str] = {}
    for i, dng in enumerate(dngs):
        mapping[dng.stem] = _read_camera(dng)
        if i and i % 500 == 0:
            print(f"  {i}/{len(dngs)}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump(mapping, fh, indent=0, ensure_ascii=False)

    names = sorted(set(mapping.values()))
    print(f"wrote {args.out} ({len(mapping)} entries)")
    print(f"distinct cameras: {len(names)}")
    for n in names:
        print(f"  {sum(1 for v in mapping.values() if v == n):5d}  {n}")


if __name__ == "__main__":
    main()
