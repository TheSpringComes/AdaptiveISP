#!/usr/bin/env python3
"""
Generate COCO SynRAW train/val list files for AdaptiveISP.

Usage:
  python make_coco_synraw_lists.py --coco-root /path/to/coco2017
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable


VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def collect_images(folder: Path) -> list[Path]:
    if not folder.exists():
        raise FileNotFoundError(f"Image directory not found: {folder}")
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in VALID_EXTS]
    if not files:
        raise RuntimeError(f"No image files found in: {folder}")
    return sorted(files)


def write_rel_list(paths: Iterable[Path], root: Path, out_file: Path) -> int:
    lines = []
    for p in paths:
        rel = p.relative_to(root).as_posix()
        lines.append(f"./{rel}")
    out_file.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return len(lines)


def ensure_label_symlink(link_path: Path, target_path: Path) -> None:
    if link_path.exists() or link_path.is_symlink():
        # Keep user's existing folder/symlink untouched.
        return
    link_path.parent.mkdir(parents=True, exist_ok=True)
    link_path.symlink_to(target_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build train/val txt lists for SynRAW COCO data.")
    parser.add_argument("--coco-root", type=Path, required=True, help="COCO root directory")
    parser.add_argument("--train-raw", type=str, default="raw_images/train2017_SynRAW", help="Train SynRAW folder")
    parser.add_argument("--val-raw", type=str, default="raw_images/val2017_SynRAW", help="Val SynRAW folder")
    parser.add_argument("--train-txt", type=str, default="train2017_synraw.txt", help="Output train txt name")
    parser.add_argument("--val-txt", type=str, default="val2017_synraw.txt", help="Output val txt name")
    parser.add_argument(
        "--create-label-symlink",
        action="store_true",
        help="Create labels/train2017_SynRAW -> labels/train2017 and val symlink if missing",
    )
    args = parser.parse_args()

    coco_root = args.coco_root.resolve()
    train_raw = (coco_root / args.train_raw).resolve()
    val_raw = (coco_root / args.val_raw).resolve()
    train_txt = coco_root / args.train_txt
    val_txt = coco_root / args.val_txt

    train_files = collect_images(train_raw)
    val_files = collect_images(val_raw)

    n_train = write_rel_list(train_files, coco_root, train_txt)
    n_val = write_rel_list(val_files, coco_root, val_txt)

    if args.create_label_symlink:
        ensure_label_symlink(coco_root / "labels/train2017_SynRAW", coco_root / "labels/train2017")
        ensure_label_symlink(coco_root / "labels/val2017_SynRAW", coco_root / "labels/val2017")

    print(f"Wrote {n_train} entries -> {train_txt}")
    print(f"Wrote {n_val} entries -> {val_txt}")
    print("Done.")


if __name__ == "__main__":
    main()
