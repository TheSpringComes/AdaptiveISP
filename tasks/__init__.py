"""tasks: downstream task interface + concrete implementations.

Layout:
    tasks/base.py           — Task ABC + TaskMetrics
    tasks/detection/        — detection-family (v1: YOLOv3)
        task.py             — family orchestrator
        detector.py         — Detector interface (V2 target)
        metrics.py          — metric functions (V2 target)
        implementations/    — concrete detector implementations
            yolov3.py       — YOLOv3Detection
    tasks/third_party/      — vendored source, called only by implementations/
        yolov3/             — vendored yolov3 repo
"""
import os
import sys

# Make the vendored yolov3 subpackage importable via both:
#   `from yolov3.utils.general import ...` (package form, used by our code)
#   `from utils.general import ...`       (bare form, used inside vendored code)
_HERE = os.path.dirname(os.path.abspath(__file__))
_THIRD_PARTY = os.path.join(_HERE, "third_party")
_YOLOV3 = os.path.join(_THIRD_PARTY, "yolov3")
for _p in (_THIRD_PARTY, _YOLOV3):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

from tasks.base import Task, TaskMetrics

__all__ = ["Task", "TaskMetrics"]
