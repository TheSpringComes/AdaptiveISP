"""PipelineState: batched snapshot of the ISP pipeline mid-rollout."""
from __future__ import annotations

from dataclasses import dataclass, field

import torch

from pipeline.action import ISPAction


@dataclass
class PipelineState:
    """Batched snapshot of the ISP pipeline mid-rollout.

    image:      [B, C, H, W] — current image (float in [0, 1]).
    step:       [B] int64    — number of ops applied so far.
    stopped:    [B] bool     — whether this sample emitted the stop action.
    has_reward: [B] bool     — whether the terminal reward has been recorded.
    op_usage:   [B, N_ops] int64 — how many times each op has been applied to
                this sample so far. `0` = never used; `k` = used `k` times.
                Used by AdaptiveISPReward.usage_penalty for the 2^k
                exponential repeat penalty.
    history:    per-batch list of ISPActions applied (kept CPU-side for logging).
    """
    image: torch.Tensor
    step: torch.Tensor
    stopped: torch.Tensor
    has_reward: torch.Tensor
    op_usage: torch.Tensor
    history: list[ISPAction] = field(default_factory=list)

    @property
    def batch_size(self) -> int:
        return self.image.shape[0]
