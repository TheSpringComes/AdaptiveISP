"""Reward function for the AdaptiveISP Controller.

The reward is intimately tied to the training objective, so it lives inside
the controller subpackage (`controller/adaptiveisp/`). A future controller
family (BayesOpt, CMA-ES, etc.) would have its own `<name>/reward.py` if it
needs a different formulation.

Original AdaptiveISP formula:
    reward = scale * (detect_before - detect_after) * critic_logit_multiplier
             - overflow_penalty - entropy_penalty - usage_penalty
             - early_stop_penalty - runtime_penalty
See agent.py:234-277 + train.py:289-293 in the pre-refactor tree.
"""
from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import torch

from pipeline.action import ISPAction
from pipeline.state import PipelineState
from tasks.base import TaskMetrics


@dataclass
class RewardBreakdown:
    """Component-wise reward for logging/debugging."""
    task_delta: torch.Tensor
    overflow_penalty: torch.Tensor
    entropy_penalty: torch.Tensor
    usage_penalty: torch.Tensor
    early_stop_penalty: torch.Tensor
    runtime_penalty: torch.Tensor
    total: torch.Tensor
    # Optional: stop-time bonus for Detection reward (learned STOP).
    stop_bonus: Optional[torch.Tensor] = None


class Reward(ABC):
    """Base class for reward functions."""

    @abstractmethod
    def compute(self, *args, **kwargs) -> tuple[torch.Tensor, RewardBreakdown]:
        """Return (per-batch total reward [B, 1], component breakdown)."""


class AdaptiveISPReward(Reward):
    """Original AdaptiveISP reward. V1 keeps constants matching config values."""

    def __init__(
        self,
        *,
        n_ops: int,
        max_steps: int,
        critic_logit_multiplier: float = 100.0,
        all_reward: float = 1.0,
        filter_usage_penalty: float = 1.0,
        exploration_penalty: float = 0.05,
        early_stop_penalty: float = 1.0,
        runtime_penalty_enabled: bool = False,
        runtime_penalty_lambda: float = 0.01,
        runtime_costs: Optional[list[float]] = None,
        detect_loss_key: str = "detect_loss",
        use_penalty: bool = True,
        stop_bonus_scale: float = 0.0,
    ) -> None:
        self.n_ops = int(n_ops)
        self.max_steps = int(max_steps)
        self.critic_logit_multiplier = float(critic_logit_multiplier)
        self.all_reward = float(all_reward)
        self.filter_usage_penalty = float(filter_usage_penalty)
        self.exploration_penalty = float(exploration_penalty)
        self.early_stop_penalty = float(early_stop_penalty)
        self.runtime_penalty_enabled = bool(runtime_penalty_enabled)
        self.runtime_penalty_lambda = float(runtime_penalty_lambda)
        self.runtime_costs = list(runtime_costs or [])
        self.detect_loss_key = detect_loss_key
        self.use_penalty = bool(use_penalty)
        self.stop_bonus_scale = float(stop_bonus_scale)

        if self.runtime_penalty_enabled and len(self.runtime_costs) != self.n_ops:
            raise ValueError(
                f"AdaptiveISPReward: runtime_costs has {len(self.runtime_costs)} "
                f"entries but n_ops = {self.n_ops}"
            )

    def compute(
        self,
        metrics_before: TaskMetrics,
        metrics_after: TaskMetrics,
        state_before: PipelineState,
        action: ISPAction,
        state_after: PipelineState,
        *,
        entropy: Optional[torch.Tensor] = None,
        progress: float = 0.0,
    ) -> tuple[torch.Tensor, RewardBreakdown]:
        B = state_after.batch_size
        device = state_after.image.device

        # 1. Task-delta (main reward signal)
        detect_before = metrics_before[self.detect_loss_key]
        detect_after = metrics_after[self.detect_loss_key]
        stopped_after = state_after.stopped.float().unsqueeze(-1)
        scale = self.all_reward + (1.0 - self.all_reward) * stopped_after
        task_delta = scale * (detect_before.detach() - detect_after) * self.critic_logit_multiplier

        # 2. Overflow penalty (post-op image values above 1)
        image_after = state_after.image
        overflow_penalty = torch.mean(
            torch.clip(image_after - 1.0, min=0.0) ** 2, dim=(1, 2, 3),
        ).unsqueeze(-1)

        # 3. Entropy penalty (encourages high entropy early in training)
        if entropy is None:
            entropy_penalty = torch.zeros((B, 1), device=device)
        else:
            entropy = entropy if entropy.dim() == 2 else entropy.unsqueeze(-1)
            entropy_penalty = (
                (1.0 - float(progress))
                * self.exploration_penalty
                * (math.log(self.n_ops) - entropy)
            )

        # 4. Usage penalty — exponential in the number of prior uses of the
        # chosen op. Prevents policy from collapsing into a "keep picking the
        # same op" loop. Formula:
        #     first pick of op         : penalty = 0
        #     k-th pick (k >= 2)       : penalty = base * 2^(k-1)
        # where `k-1 = prev_count` is `state_before.op_usage[b, chosen]`.
        # base_penalty=5.0 with default config → 5 for 2nd, 10 for 3rd,
        # 20 for 4th, 40 for 5th, … so k = 5 accumulated repeats already
        # cost 40× the per-step task_delta and is essentially blocked.
        safe_op = torch.where(
            action.is_stop, torch.zeros_like(action.op_indices), action.op_indices,
        )
        batch_idx = torch.arange(B, device=device)
        prev_count = state_before.op_usage[batch_idx, safe_op].float().unsqueeze(-1)
        is_repeat = (prev_count > 0).float()
        scale = torch.pow(torch.tensor(2.0, device=device), prev_count)
        usage_penalty = (
            is_repeat * scale * self.filter_usage_penalty
            * (~action.is_stop).float().unsqueeze(-1)
        )

        # 5. Early-stop penalty (submitted before final step)
        is_last_step = (state_after.step == self.max_steps).float().unsqueeze(-1)
        submitted = action.is_stop.float().unsqueeze(-1)
        early_stop_penalty = (1.0 - is_last_step) * submitted * self.early_stop_penalty

        # 6. Runtime penalty (per-op wall-clock cost)
        if self.runtime_penalty_enabled:
            costs = torch.tensor(self.runtime_costs, device=device, dtype=torch.float32)
            chosen_cost = costs[safe_op].unsqueeze(-1)
            chosen_cost = chosen_cost * (~action.is_stop).float().unsqueeze(-1)
            runtime_penalty = chosen_cost * self.runtime_penalty_lambda
        else:
            runtime_penalty = torch.zeros((B, 1), device=device)

        # 7. Stop bonus — rewards the Controller for stopping when the current
        # image already scores well. Encourages learned STOP action.
        # Formula:
        #     stop_bonus = submitted × max(0, 1 - detect_after) × stop_bonus_scale × critic_logit_multiplier
        # detect_after ∈ [0, ~1]: closer to 0 (better) → bigger positive bonus.
        # `max(0, ...)` clamps so pathologically-bad frames still get 0, not
        # negative. Off-by-default (stop_bonus_scale=0) preserves legacy.
        if self.stop_bonus_scale != 0.0:
            quality = torch.clamp(1.0 - detect_after.detach(), min=0.0)
            stop_bonus = (
                submitted
                * quality
                * self.stop_bonus_scale
                * self.critic_logit_multiplier
            )
        else:
            stop_bonus = torch.zeros((B, 1), device=device)

        if self.use_penalty:
            penalty = (
                overflow_penalty + entropy_penalty + usage_penalty
                + early_stop_penalty + runtime_penalty
            )
            reward = task_delta - penalty + stop_bonus
        else:
            reward = task_delta + stop_bonus

        return reward, RewardBreakdown(
            task_delta=task_delta,
            overflow_penalty=overflow_penalty,
            entropy_penalty=entropy_penalty,
            usage_penalty=usage_penalty,
            early_stop_penalty=early_stop_penalty,
            runtime_penalty=runtime_penalty,
            total=reward,
            stop_bonus=stop_bonus,
        )


__all__ = ["Reward", "RewardBreakdown", "AdaptiveISPReward"]
