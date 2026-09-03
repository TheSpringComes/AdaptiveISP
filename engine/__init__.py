"""engine: experiment orchestration.

Answers "how does the whole experiment run?" — assembles the four subsystems
(isp / search / controller / pipeline / tasks) into a training or eval loop.

Files:
    runner.py     — high-level dispatch (train / val / test)
    trainer.py    — Trainer class (was DynamicISP)
    evaluator.py  — Evaluator (defers to tools/val.py for mAP)
"""
from engine.trainer import Trainer

__all__ = ["Trainer"]
