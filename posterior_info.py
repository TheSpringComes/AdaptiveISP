"""Posterior-information rewards for task-driven ISP training.

This module computes the class-posterior entropy, objectness entropy and the
normalized posterior-information reward described in the low-light RAW ISP
experiment design.  It operates directly on Ultralytics YOLO-style raw detection
heads, so it can be used during training without non-maximum suppression.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Mapping

import torch


@dataclass(frozen=True)
class PosteriorInfoStats:
    """Per-image posterior uncertainty statistics."""

    h_cls: torch.Tensor
    h_obj: torch.Tensor
    h_det: torch.Tensor
    reward: torch.Tensor

    def as_log_dict(self) -> Mapping[str, torch.Tensor]:
        return {
            "posterior/h_cls": self.h_cls.mean(),
            "posterior/h_obj": self.h_obj.mean(),
            "posterior/h_det": self.h_det.mean(),
            "posterior/reward": self.reward.mean(),
        }


def _flatten_yolo_predictions(preds: torch.Tensor | Iterable[torch.Tensor]) -> torch.Tensor:
    """Return YOLO predictions as ``[batch, num_candidates, 5 + num_classes]``.

    Ultralytics YOLO models may return a tensor, a tuple whose first item is the
    inference tensor, or a list of training detection heads.  The training heads
    usually have shape ``[B, A, H, W, 5+C]``; this function flattens all anchor
    and spatial dimensions while preserving the final prediction dimension.
    """

    if isinstance(preds, torch.Tensor):
        if preds.ndim == 3:
            return preds
        if preds.ndim >= 4:
            return preds.reshape(preds.shape[0], -1, preds.shape[-1])
        raise ValueError(f"Unsupported prediction tensor shape: {tuple(preds.shape)}")

    if isinstance(preds, tuple):
        first = preds[0]
        if isinstance(first, torch.Tensor):
            return _flatten_yolo_predictions(first)
        preds = first

    flattened: List[torch.Tensor] = []
    for head in preds:
        if not isinstance(head, torch.Tensor):
            continue
        if head.ndim < 3:
            raise ValueError(f"Unsupported YOLO head shape: {tuple(head.shape)}")
        flattened.append(head.reshape(head.shape[0], -1, head.shape[-1]))
    if not flattened:
        raise ValueError("No tensor predictions were found in YOLO output")
    return torch.cat(flattened, dim=1)


def posterior_information_reward(
    preds: torch.Tensor | Iterable[torch.Tensor],
    beta: float = 1.0,
    eps: float = 1e-8,
    topk: int | None = 1000,
) -> PosteriorInfoStats:
    """Compute normalized posterior-information reward from YOLO predictions.

    The reward is ``1 - (H_cls + beta * H_obj) / (log(C) + beta * log(2))``.
    Candidate entropies are weighted by objectness so boxes that are more likely
    to contain an object dominate the aggregate.  ``topk`` keeps the computation
    bounded for dense YOLO heads while preserving the most task-relevant boxes.
    """

    flat = _flatten_yolo_predictions(preds)
    if flat.shape[-1] <= 5:
        raise ValueError("YOLO predictions must contain class logits/probabilities")

    obj = flat[..., 4].sigmoid().clamp(eps, 1.0 - eps)
    cls_prob = flat[..., 5:].softmax(dim=-1).clamp_min(eps)

    if topk is not None and 0 < topk < obj.shape[1]:
        _, idx = torch.topk(obj, k=topk, dim=1)
        gather_pred = idx.unsqueeze(-1).expand(-1, -1, cls_prob.shape[-1])
        obj = torch.gather(obj, 1, idx)
        cls_prob = torch.gather(cls_prob, 1, gather_pred)

    weights = obj / (obj.sum(dim=1, keepdim=True) + eps)
    h_cls_i = -(cls_prob * torch.log(cls_prob)).sum(dim=-1)
    h_obj_i = -obj * torch.log(obj) - (1.0 - obj) * torch.log(1.0 - obj)

    h_cls = (weights * h_cls_i).sum(dim=1, keepdim=True)
    h_obj = (weights * h_obj_i).sum(dim=1, keepdim=True)
    h_det = h_cls + float(beta) * h_obj

    num_classes = cls_prob.shape[-1]
    normalizer = torch.log(torch.tensor(float(num_classes), device=flat.device, dtype=flat.dtype))
    normalizer = normalizer + float(beta) * torch.log(torch.tensor(2.0, device=flat.device, dtype=flat.dtype))
    reward = 1.0 - h_det / (normalizer + eps)
    reward = reward.clamp(0.0, 1.0)

    return PosteriorInfoStats(h_cls=h_cls, h_obj=h_obj, h_det=h_det, reward=reward)
