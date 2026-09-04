"""HumanQualityTask: SSIM + LPIPS composite reward against Expert C.

Standard-form `Task` per `tasks.base.Task`. `compute_metrics(image, target)`
returns `TaskMetrics` with keys:

    - `ssim`      (B, 1)  higher is better
    - `lpips`     (B, 1)  lower is better
    - `quality`   (B, 1)  Q(I) = λ_ssim · SSIM − λ_lpips · LPIPS

Unlike YOLOv3Detection, this task has no learnable parameters — LPIPS is a
frozen pretrained AlexNet cached at first use.
"""
from __future__ import annotations

from typing import Any

import torch

from tasks.base import Task, TaskMetrics
from tasks.human_quality.metrics import quality_score


class HumanQualityTask(Task):
    """Task that scores an image against a paired Expert-C reference.

    lambda_ssim / lambda_lpips are the weights in
      Q(I) = lambda_ssim · SSIM(I, ref) - lambda_lpips · LPIPS(I, ref)

    lpips_net is the LPIPS backbone; "alex" is fastest and typical for
    training-time reward. Switch to "vgg" for eval if desired.
    """
    name = "human_quality"

    def __init__(
        self,
        lambda_ssim: float = 1.0,
        lambda_lpips: float = 1.0,
        lpips_net: str = "alex",
        device: torch.device = torch.device("cuda"),
    ) -> None:
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_lpips = float(lambda_lpips)
        self.lpips_net = lpips_net
        self.device = device

    def compute_metrics(
        self,
        images: torch.Tensor,
        targets: torch.Tensor,
        **kwargs,
    ) -> TaskMetrics:
        q, parts = quality_score(
            images, targets,
            lambda_ssim=self.lambda_ssim,
            lambda_lpips=self.lambda_lpips,
            lpips_net=self.lpips_net,
        )
        return TaskMetrics(values={
            "ssim": parts["ssim"],
            "lpips": parts["lpips"],
            "quality": parts["quality"],
        })

    def train(self) -> None:
        pass  # no learnable params; kept for interface parity with YOLO Task

    def eval(self) -> None:
        pass

    def to(self, device):
        self.device = device
        return self


__all__ = ["HumanQualityTask"]
