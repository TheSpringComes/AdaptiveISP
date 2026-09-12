"""tools/val.py — unified CLI for AdaptiveISP evaluation.

Auto-routes on the ckpt's `task` field:

  - `detection` (default) → mAP loop on LOD/COCO. Needs `--data`,
    `--data_name`, `--weights` (YOLOv3 pretrained).
  - `human_quality` / `human` → SSIM/LPIPS/Q on FiveK val split. FiveK
    paths come from the ckpt's cfg (`human_quality:` block). Detection-only
    flags are accepted but ignored.

Both paths write canary PNGs under `experiments/<exp>/visualization/`
(derived from `--isp_weights`) unless `--skip_viz`. See
`engine.evaluator.evaluate` for details.

Example (detection):
    python tools/val.py \\
        --isp_weights experiments/lod-.../ckpt/DynamicISP_iter_30000.pth \\
        --data tasks/third_party/yolov3/data/lod.yaml --data_name lod \\
        --cfg configs/adaptiveisp.yaml \\
        --name lod_run --exist-ok

Example (human):
    python tools/val.py \\
        --isp_weights experiments/v2ai_human/ckpt/HumanISP_iter_28000.pth \\
        --cfg experiments/v2ai_human/adaptiveisp_human.yaml \\
        --name v2ai_human_iter28000 --exist-ok
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
    p.add_argument('--isp_weights', type=str, required=True,
                   help="Controller checkpoint (schema: {'controller_model', ...}); "
                        "the ckpt's `task` field picks the eval branch.")
    p.add_argument('--cfg', dest='cfg_path', type=str, default='configs/adaptiveisp.yaml',
                   help='framework config yaml (controller net dims etc.)')
    p.add_argument('--cfg_file', dest='cfg_path', type=str,
                   help='alias for --cfg')

    # Shared knobs.
    p.add_argument('--imgsz', type=int, default=512)
    p.add_argument('--batch-size', dest='batch_size', type=int, default=1)
    p.add_argument('--steps', type=int, default=5, help='ISP rollout steps (detection)')
    p.add_argument('--project', type=str, default='experiments/val_results')
    p.add_argument('--name', type=str, default='exp')
    p.add_argument('--exist-ok', dest='exist_ok', action='store_true')
    p.add_argument('--seed', type=int, default=0)

    # Detection-only knobs. Ignored on the human path.
    p.add_argument('--weights', type=str, default='pretrained/yolov3.pt',
                   help='YOLOv3 pretrained (detection only)')
    p.add_argument('--data', type=str, default=None,
                   help='dataset yaml (detection only; required if task=detection)')
    p.add_argument('--data_name', type=str, default='lod', choices=['lod', 'coco'],
                   help='dataset name (detection only)')
    p.add_argument('--conf-thres', dest='conf_thres', type=float, default=0.001)
    p.add_argument('--iou-thres', dest='iou_thres', type=float, default=0.6)
    p.add_argument('--max-det', dest='max_det', type=int, default=300)

    # Canary visualization.
    p.add_argument('--skip_viz', action='store_true', default=False,
                   help='skip the canary visualization step (default: run it)')
    p.add_argument('--viz_cases', type=int, default=4,
                   help='number of canary cases to visualize')
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    from engine.evaluator import evaluate
    kwargs = vars(args)
    kwargs['run_viz'] = not kwargs.pop('skip_viz')
    evaluate(**kwargs)


if __name__ == '__main__':
    main()
