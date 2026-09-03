"""tools/train.py — CLI entry point for training.

Usage:
    python tools/train.py [--seed 0] [--epochs 150] ...

The heavy lifting lives in `engine.trainer.Trainer`. This file only does
argument parsing + sys.path setup + dispatch.
"""
from __future__ import annotations

import argparse
import os
import sys

# Make project root importable regardless of where the script is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from engine.util import set_seed


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", type=str, default='train_val', choices=['train', 'train_val'],
                        help="train, or train and val (val-mode mAP eval lives in tools/val.py)")
    parser.add_argument("--batch_size", type=int, default=4, help="batch size")
    parser.add_argument("--epochs", type=int, default=800, help="epochs")
    parser.add_argument("--patience", type=int, default=20, help="early stopping patience (unused: dead code)")
    parser.add_argument("--lr", type=float, default=3e-5, help="learning rate")
    parser.add_argument("--scheduler_step_size", type=int, default=20)
    parser.add_argument("--scheduler_lr_gamma", type=float, default=0.5)
    parser.add_argument("--imgsz", type=int, default=512, help="image size")
    parser.add_argument("--workers", type=int, default=4)

    parser.add_argument('--weights', type=str, default='pretrained/yolov3.pt', help='yolov3 pretrained path')
    parser.add_argument('--yolo_cfg', type=str, default='tasks/third_party/yolov3/models/yolov3.yaml', help='model yaml path')
    parser.add_argument('--hyp', type=str, default='tasks/third_party/yolov3/data/hyps/hyp.scratch-low.yaml', help='hyperparameters path')

    parser.add_argument("--save_path", type=str, default='adaptiveisp', help="save path at experiments/save_path/")
    parser.add_argument("--data_name", type=str, default='coco', choices=['lod', 'coco'])
    parser.add_argument('--data_cfg', type=str, default='tasks/third_party/yolov3/data/coco_synraw.yaml')
    parser.add_argument("--add_noise", type=bool, default=False)
    parser.add_argument("--use_linear", action='store_true', default=False)
    parser.add_argument("--bri_range", type=float, default=None, nargs='*')
    parser.add_argument("--noise_level", type=float, default=None)

    parser.add_argument('--use_truncated', type=bool, default=True)
    parser.add_argument("--runtime_penalty", action='store_true', default=False)
    parser.add_argument("--runtime_penalty_lambda", type=float, default=0.01)
    parser.add_argument('--resume', type=str, default=None)

    parser.add_argument("--steps", type=int, default=5, help="rollout steps at eval")
    parser.add_argument("--cfg", type=str, default="configs/adaptiveisp.yaml", help="config yaml")
    parser.add_argument("--seed", type=int, default=0, help="global RNG seed")
    parser.add_argument("--nondeterministic", action='store_true', default=False)

    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    args.save_path = args.data_name + '-' + args.save_path
    if args.data_name in ("lod",):
        args.add_noise = False
        args.bri_range = None
        args.use_linear = False

    # set_seed(args.seed, deterministic=not args.nondeterministic)

    from engine.trainer import Trainer
    trainer = Trainer(args, args.task)
    trainer.train()


if __name__ == "__main__":
    main()
