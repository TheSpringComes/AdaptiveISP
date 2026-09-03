#!/usr/bin/env python3
# YOLOv3-style GT visualization for COCO json annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))

from ultralytics.utils.plotting import Annotator, colors
from utils.general import LOGGER, increment_path, print_args


def load_coco(annotation_path):
    with open(annotation_path, "r", encoding="utf-8") as f:
        coco = json.load(f)

    images = coco.get("images", [])
    annotations = coco.get("annotations", [])
    categories = coco.get("categories", [])

    file_to_image = {}
    id_to_image = {}
    for im in images:
        file_name = Path(im["file_name"]).name
        file_to_image[file_name] = im
        id_to_image[im["id"]] = im

    anns_by_image = defaultdict(list)
    for ann in annotations:
        anns_by_image[ann["image_id"]].append(ann)

    cat_id_to_name = {c["id"]: c["name"] for c in categories}
    sorted_cat_ids = sorted(cat_id_to_name.keys())
    cat_id_to_color_idx = {cid: i for i, cid in enumerate(sorted_cat_ids)}

    return file_to_image, anns_by_image, cat_id_to_name, cat_id_to_color_idx


def draw_gt(
    image,
    anns,
    cat_id_to_name,
    cat_id_to_color_idx,
    hide_labels=False,
):
    h, w = image.shape[:2]
    fs = int((h + w) * 0.01)
    lw = max(round(fs / 10), 1)
    annotator = Annotator(image, line_width=lw, font_size=fs, example=str(list(cat_id_to_name.values())))

    for ann in anns:
        if ann.get("iscrowd", 0):
            continue

        x, y, bw, bh = ann["bbox"]  # COCO xywh, absolute pixels
        x1, y1 = x, y
        x2, y2 = x + bw, y + bh

        cls_id = ann["category_id"]
        cls_name = cat_id_to_name.get(cls_id, str(cls_id))
        color_idx = cat_id_to_color_idx.get(cls_id, 0)
        color = colors(color_idx, True)

        label = None if hide_labels else f"{cls_name}"
        annotator.box_label([x1, y1, x2, y2], label, color=color)

    return annotator.result()


def run(
    source=ROOT / "data/images",
    annotations=ROOT / "data/coco128/annotations/instances_val2017.json",
    project=ROOT / "runs/gt",
    name="exp",
    exist_ok=False,
    hide_labels=False
):
    source = Path(source)
    annotations = Path(annotations)

    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)
    save_dir.mkdir(parents=True, exist_ok=True)

    file_to_image, anns_by_image, cat_id_to_name, cat_id_to_color_idx = load_coco(annotations)

    image_files = sorted([p for p in source.glob("*.jpg")])
    if not image_files:
        raise FileNotFoundError(f"No .jpg images found in {source}")

    LOGGER.info(f"Images found: {len(image_files)}")
    LOGGER.info(f"Saving to: {save_dir}")

    seen = 0
    missing = 0

    for img_path in image_files:
        im0 = cv2.imread(str(img_path))
        if im0 is None:
            LOGGER.warning(f"Image Not Found {img_path}")
            continue

        info = file_to_image.get(img_path.name)
        if info is None:
            LOGGER.warning(f"No matching COCO image entry for {img_path.name}")
            missing += 1
            continue

        image_id = info["id"]
        anns = anns_by_image.get(image_id, [])

        im_out = draw_gt(
            im0,
            anns,
            cat_id_to_name,
            cat_id_to_color_idx,
            hide_labels=hide_labels
        )

        save_path = save_dir / img_path.name
        cv2.imwrite(str(save_path), im_out)

        seen += 1
        if seen % 100 == 0:
            LOGGER.info(f"Processed {seen}/{len(image_files)}")

    LOGGER.info(f"Done. Saved {seen} images to {save_dir}")
    if missing:
        LOGGER.info(f"Skipped {missing} images not found in COCO json")


def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source",
        type=str,
        default="/home/jing/datasets/COCO/val2017_SynRAW",
        help="directory containing val_SynRAW jpg images",
    )
    parser.add_argument(
        "--annotations",
        type=str,
        default="/home/jing/datasets/COCO/annotations/instances_val2017.json",
        help="COCO annotation json path",
    )
    parser.add_argument("--project", default=ROOT / "val_results", help="save results to project/name")
    parser.add_argument("--name", default="val_gt", help="save results to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--hide-labels", default=False, action="store_true", help="hide labels")
    opt = parser.parse_args()
    print_args(vars(opt))
    return opt


def main(opt):
    run(**vars(opt))


if __name__ == "__main__":
    opt = parse_opt()
    main(opt)