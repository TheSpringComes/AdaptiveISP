"""tools/val.py — CLI wrapper for mAP evaluation.

Delegates to the vendored val_adaptiveisp.py, which auto-detects V1
Controller ckpt schema.

Usage:
    python tools/val.py --weights pretrained/yolov3.pt \
        --isp_weights experiments/coco-<run>/ckpt/DynamicISP_iter_37000.pth \
        --data_name coco --data configs/datasets/coco_synraw.yaml \
        --imgsz 512 --batch-size 1 --steps 5 --cfg_file configs/adaptiveisp.yaml
"""
from __future__ import annotations

import os
import runpy
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_VAL_SCRIPT = os.path.join(_ROOT, "tasks", "third_party", "yolov3", "val_adaptiveisp.py")
_YOLOV3 = os.path.join(_ROOT, "tasks", "third_party", "yolov3")

for p in (_ROOT, _YOLOV3):
    if p not in sys.path:
        sys.path.insert(0, p)


if __name__ == "__main__":
    runpy.run_path(_VAL_SCRIPT, run_name="__main__")
