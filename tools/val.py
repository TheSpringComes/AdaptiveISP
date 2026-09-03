"""tools/val.py — CLI entry point for mAP evaluation.

The implementation lives in `engine.evaluator.evaluate`. This file is
just argparse + dispatch.

Example:
    python tools/val.py \\
        --weights pretrained/yolov3.pt \\
        --isp_weights experiments/lod-.../ckpt/DynamicISP_iter_30000.pth \\
        --data tasks/third_party/yolov3/data/lod.yaml \\
        --data_name lod --imgsz 512 --batch-size 1 --steps 5 \\
        --cfg configs/adaptiveisp.yaml \\
        --project val_results --name my_run --exist-ok
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument('--weights', type=str, default='pretrained/yolov3.pt')
    p.add_argument('--isp_weights', type=str, required=True,
                   help="V1 Controller checkpoint (schema: {'controller_model', ...})")
    p.add_argument('--data', type=str, required=True,
                   help='dataset yaml (e.g. tasks/third_party/yolov3/data/lod.yaml)')
    p.add_argument('--data_name', type=str, default='lod', choices=['lod', 'coco'])
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch-size', dest='batch_size', type=int, default=1)
    p.add_argument('--steps', type=int, default=5, help='ISP rollout steps')
    p.add_argument('--cfg', dest='cfg_path', type=str, default='configs/adaptiveisp.yaml',
                   help='framework config yaml (controller net dims etc.)')
    p.add_argument('--cfg_file', dest='cfg_path', type=str,
                   help='alias for --cfg')
    p.add_argument('--project', type=str, default='val_results')
    p.add_argument('--name', type=str, default='exp')
    p.add_argument('--exist-ok', dest='exist_ok', action='store_true')
    p.add_argument('--conf-thres', dest='conf_thres', type=float, default=0.001)
    p.add_argument('--iou-thres', dest='iou_thres', type=float, default=0.6)
    p.add_argument('--max-det', dest='max_det', type=int, default=300)
    p.add_argument('--seed', type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    from engine.evaluator import evaluate
    evaluate(**vars(args))


if __name__ == '__main__':
    main()
