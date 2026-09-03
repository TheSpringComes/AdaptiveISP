"""Runner: high-level dispatch (train / val / test).

V1 provides a thin dispatcher. Users typically invoke tools/train.py or
tools/val.py directly; this file exists for programmatic composition of
multiple experiment stages.
"""
from __future__ import annotations

from engine.trainer import Trainer


def run(args, task: str = "train_val") -> None:
    if task in ("train", "train_val"):
        Trainer(args, task).train()
    elif task == "val":
        raise NotImplementedError(
            "--task val is served by tools/val.py, which wraps "
            "yolov3/val_adaptiveisp.py for mAP computation."
        )
    else:
        raise ValueError(f"unknown task: {task}")


__all__ = ["run"]
