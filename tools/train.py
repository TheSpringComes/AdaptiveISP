"""tools/train.py — unified CLI entry point for AdaptiveISP training.

Dispatches to the right trainer subclass by `--task`:

    detection  →  engine.trainer.Trainer         (LOD / COCO + YOLOv3)
    human      →  engine.trainer_human.HumanTrainer  (FiveK + Expert C)

Both trainers now share the same BaseTrainer scaffold (`engine/base_trainer.py`)
— dir setup, controller/executor/search_space construction, checkpoint save,
config loading, print-block helpers — so this file only owns argument
parsing + dispatch. The two loops still differ in shape (1-step + Replay vs
full T-step rollout); that is a Stage-C concern, not touched by Plan A.

Usage:
    python tools/train.py --task detection --data_name lod --cfg configs/adaptiveisp.yaml
    python tools/train.py --task human --cfg configs/adaptiveisp_human.yaml
"""
from __future__ import annotations

import argparse
import os
import sys

# Make project root importable regardless of where the script is invoked from.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)


def _add_shared_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--task", type=str, default="detection",
                   choices=["detection", "human", "learnable", "calibration"],
                   help="downstream task; picks the trainer subclass "
                        "(learnable = Stage 1 learnable Front ISP pretrain; "
                        "calibration 是旧名别名)")
    p.add_argument("--mode", type=str, default="train_val",
                   choices=["train", "train_val"],
                   help="train, or train and val (val-mode mAP eval lives in tools/val.py)")
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--imgsz", type=int, default=512, help="image size")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save_path", type=str, default="adaptiveisp",
                   help="save path at experiments/save_path/")
    p.add_argument("--cfg", type=str, default="configs/adaptiveisp.yaml",
                   help="config yaml")
    p.add_argument("--max_iters", type=int, default=0,
                   help="cap training iterations (small-scale / smoke runs); "
                        "0 = use config-derived value")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--runtime_penalty", action="store_true", default=False)
    p.add_argument("--runtime_penalty_lambda", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0, help="global RNG seed")
    p.add_argument("--nondeterministic", action="store_true", default=False)


def _add_detection_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--patience", type=int, default=20,
                   help="early stopping patience (unused: dead code)")
    p.add_argument("--scheduler_step_size", type=int, default=20)
    p.add_argument("--scheduler_lr_gamma", type=float, default=0.5)
    p.add_argument("--weights", type=str, default="pretrained/yolov3.pt",
                   help="yolov3 pretrained path")
    p.add_argument("--yolo_cfg", type=str,
                   default="tasks/third_party/yolov3/models/yolov3.yaml",
                   help="model yaml path")
    p.add_argument("--hyp", type=str,
                   default="tasks/third_party/yolov3/data/hyps/hyp.scratch-low.yaml",
                   help="hyperparameters path")
    p.add_argument("--data_name", type=str, default="coco",
                   choices=["lod", "coco"])
    p.add_argument("--data_cfg", type=str,
                   default="tasks/third_party/yolov3/data/coco_synraw.yaml")
    p.add_argument("--add_noise", type=bool, default=False)
    p.add_argument("--use_linear", action="store_true", default=False)
    p.add_argument("--bri_range", type=float, default=None, nargs="*")
    p.add_argument("--noise_level", type=float, default=None)
    p.add_argument("--use_truncated", type=bool, default=True)
    p.add_argument("--steps", type=int, default=5, help="rollout steps at eval")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    _add_shared_args(parser)
    _add_detection_args(parser)   # detection args always parseable; ignored by human path
    return parser.parse_args()


def _run_detection(args) -> None:
    args.save_path = args.data_name + "-" + args.save_path
    if args.data_name in ("lod",):
        args.add_noise = False
        args.bri_range = None
        args.use_linear = False

    from engine.trainer import Trainer
    trainer = Trainer(args, task=args.mode)
    trainer.train()


def _run_learnable(args) -> None:
    """V3.1 Stage 1: train only the learnable Front ISP params."""
    import isp  # noqa: F401

    from engine.trainer_learnable import LearnableTrainer
    trainer = LearnableTrainer(args, task="train")
    trainer.train()


def _run_human(args) -> None:
    # Human path uses the raw --save_path (no data_name prefix; FiveK/Expert C
    # is implicit). Neural op registry loads via import side effect.
    import isp  # noqa: F401

    from engine.trainer_human import HumanTrainer
    trainer = HumanTrainer(args, task="train")
    trainer.train()


def main() -> None:
    args = _parse_args()
    if args.task == "detection":
        _run_detection(args)
    elif args.task == "human":
        _run_human(args)
    elif args.task == "learnable":
        _run_learnable(args)
    elif args.task == "calibration":      # legacy 别名
        _run_learnable(args)
    else:
        raise ValueError(f"unknown --task: {args.task!r}")


if __name__ == "__main__":
    main()
