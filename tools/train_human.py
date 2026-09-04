"""tools/train_human.py — CLI entry for the Human Quality (FiveK) task.

Mirror of tools/train.py but wires up `engine.trainer_human.HumanTrainer`
and defaults to `configs/adaptiveisp_human.yaml`.

Example:
    python tools/train_human.py --epochs 30 --batch_size 4 --imgsz 512 \\
        --save_path v2ai_human --cfg configs/adaptiveisp_human.yaml
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
    p.add_argument("--task", type=str, default="train", choices=["train"])
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--imgsz", type=int, default=512)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--save_path", type=str, default="v2ai_human",
                   help="save path at experiments/save_path/")
    p.add_argument("--cfg", type=str, default="configs/adaptiveisp_human.yaml")
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--runtime_penalty", action="store_true", default=False)
    p.add_argument("--runtime_penalty_lambda", type=float, default=0.01)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    # Ensure isp package + neural op registry are populated before the
    # HumanTrainer instantiates operators. (Import-time side effects.)
    import isp  # noqa: F401

    from engine.trainer_human import HumanTrainer
    trainer = HumanTrainer(args, task=args.task)
    trainer.train()


if __name__ == "__main__":
    main()
