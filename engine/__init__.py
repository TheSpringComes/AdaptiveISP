"""engine: experiment orchestration.

Answers "how does the whole experiment run?" — assembles the four subsystems
(isp / search / controller / pipeline / tasks) into a training or eval loop.

Files:
    trainer.py    Trainer class (was DynamicISP)
    runner.py     high-level dispatch (train / val / test)
    util.py       small shared utilities (set_seed, Tee, Dict, AsyncTaskManager, ...)

NOTE: This __init__ deliberately does NOT eagerly import Trainer. Doing so
creates a circular import when tasks/detection/dataset.py imports
engine.util.AsyncTaskManager — Python starts loading engine/__init__.py,
which reaches into Trainer, which imports tasks.detection.replay, which
imports tasks.detection.dataset (still in the middle of its own load).

Callers should import specifically: `from engine.trainer import Trainer`
or `from engine.util import set_seed`.
"""
