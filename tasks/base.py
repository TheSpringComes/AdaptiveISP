"""Task interface: base class + shared TaskMetrics container.

A `Task` is the downstream evaluator that consumes the ISP-processed image
and produces `TaskMetrics`. Concrete implementations live under
`tasks/<family>/implementations/`.

Family layers (`tasks/detection/`, `tasks/segmentation/`, `tasks/depth/`)
group related tasks so V2 multi-task extension is a config change rather
than a code rewrite.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import torch


@dataclass
class TaskMetrics:
    """Metrics produced by a Task after evaluating the pipeline output.

    values: dict of scalar or per-sample tensors. Standard keys per family:
        detection: {"detect_loss" [B,1], "detect_retouch_loss" (alias),
                    "box_loss", "obj_loss", "cls_loss"; val only: mAP*}
        segmentation: {"seg_loss", "mIoU"}
        depth: {"depth_loss", "AbsRel"}
    Reward.compute consumes this by key; each Task documents its key set.
    """
    values: dict[str, torch.Tensor] = field(default_factory=dict)
    extras: dict[str, Any] = field(default_factory=dict)

    def __getitem__(self, k: str) -> torch.Tensor:
        return self.values[k]

    def get(self, k: str, default: Any = None) -> Any:
        return self.values.get(k, default)


class Task(ABC):
    """Base class for downstream task models.

    Subclasses ARE the task model (not wrappers around one); they own
    weights, preprocessing, and metric computation, and MUST document which
    keys they populate in `TaskMetrics.values`.
    """
    name: str

    @abstractmethod
    def compute_metrics(
        self,
        images: torch.Tensor,
        targets: Any,
        **kwargs,
    ) -> TaskMetrics:
        """Run the task on `images` and return metrics."""

    def to(self, device):
        return self


__all__ = ["Task", "TaskMetrics"]
