"""HumanReward: Actor-Critic terminal reward for the Human Quality task.

Reward semantics — MATCHES the design brief:
  - Intermediate steps: `task_delta = 0`. Only the auxiliary penalties
    (entropy / usage / early-stop / runtime) fire.
  - Terminal step:      `task_delta = Q(I_T) - Q(I_0)` where
      Q(I) = λ_ssim · SSIM(I, Expert C) − λ_lpips · LPIPS(I, Expert C)

The Critic (value net) does credit assignment via TD, propagating the
sparse terminal reward back to earlier actions. This mirrors the standard
AdaptiveISP TD loop; only `task_delta` differs from `AdaptiveISPReward`.

The auxiliary penalties are re-used verbatim from `reward.py` — no change
in shape or scale — so a run switching Detection ↔ Human sees comparable
gradient magnitudes from those terms.
"""
from __future__ import annotations

import math
from typing import Optional

import torch

from controller.adaptiveisp.reward import Reward, RewardBreakdown
from pipeline.action import ISPAction
from pipeline.state import PipelineState
from tasks.human_quality.metrics import quality_score


class HumanReward(Reward):
    """Terminal-only quality reward.

    Signature is compatible with `AdaptiveISPReward.compute` but takes two
    extra tensors — `image_initial` (I_0) and `target` (Expert C reference)
    — since Q(·) needs both. The trainer supplies these once per rollout.
    """

    def __init__(
        self,
        *,
        n_ops: int,
        max_steps: int,
        lambda_ssim: float = 1.0,
        lambda_lpips: float = 1.0,
        lpips_net: str = "alex",
        critic_logit_multiplier: float = 100.0,
        all_reward: float = 1.0,
        filter_usage_penalty: float = 1.0,
        exploration_penalty: float = 0.05,
        early_stop_penalty: float = 1.0,
        runtime_penalty_enabled: bool = False,
        runtime_penalty_lambda: float = 0.01,
        runtime_costs: Optional[list[float]] = None,
        use_penalty: bool = True,
    ) -> None:
        self.n_ops = int(n_ops)
        self.max_steps = int(max_steps)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_lpips = float(lambda_lpips)
        self.lpips_net = lpips_net
        self.critic_logit_multiplier = float(critic_logit_multiplier)
        self.all_reward = float(all_reward)
        self.filter_usage_penalty = float(filter_usage_penalty)
        self.exploration_penalty = float(exploration_penalty)
        self.early_stop_penalty = float(early_stop_penalty)
        self.runtime_penalty_enabled = bool(runtime_penalty_enabled)
        self.runtime_penalty_lambda = float(runtime_penalty_lambda)
        self.runtime_costs = list(runtime_costs or [])
        self.use_penalty = bool(use_penalty)

        if self.runtime_penalty_enabled and len(self.runtime_costs) != self.n_ops:
            raise ValueError(
                f"HumanReward: runtime_costs has {len(self.runtime_costs)} "
                f"entries but n_ops = {self.n_ops}"
            )

    def compute(
        self,
        image_initial: torch.Tensor,
        target: torch.Tensor,
        state_before: PipelineState,
        action: ISPAction,
        state_after: PipelineState,
        *,
        entropy: Optional[torch.Tensor] = None,
        progress: float = 0.0,
        q_initial: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, RewardBreakdown, dict[str, torch.Tensor]]:
        """Compute per-batch reward + breakdown + terminal quality parts.

        Third return element `q_parts` is populated only on terminal steps
        (keys `ssim`, `lpips`, `quality`, each `(B, 1)`); an empty dict on
        intermediate steps. The trainer reuses this for logging so we
        don't run SSIM+LPIPS twice per iter.

        `q_initial` is optional — passing it in avoids recomputing Q(I_0)
        every step of a rollout. When None, it is recomputed inline from
        `image_initial`.
        """
        B = state_after.batch_size
        device = state_after.image.device

        # 1. Task-delta: only fires on the terminal step of each sample.
        # In the current HumanTrainer, all samples share the same rollout
        # length so `is_terminal` is either all-True or all-False; skip the
        # expensive SSIM/LPIPS forward entirely on intermediate steps.
        is_terminal_mask = (
            (state_after.step >= self.max_steps) | state_after.stopped
        )
        is_terminal = is_terminal_mask.float().unsqueeze(-1)

        q_parts: dict[str, torch.Tensor] = {}
        if is_terminal_mask.any():
            if q_initial is None:
                q_initial, _ = quality_score(
                    image_initial, target,
                    lambda_ssim=self.lambda_ssim,
                    lambda_lpips=self.lambda_lpips,
                    lpips_net=self.lpips_net,
                )
            q_after, q_parts = quality_score(
                state_after.image, target,
                lambda_ssim=self.lambda_ssim,
                lambda_lpips=self.lambda_lpips,
                lpips_net=self.lpips_net,
            )
            r_terminal = (q_after - q_initial) * self.critic_logit_multiplier
            task_delta = is_terminal * r_terminal
        else:
            task_delta = torch.zeros((B, 1), device=device)

        # 2. Overflow penalty (values pushed above 1)
        image_after = state_after.image
        overflow_penalty = torch.mean(
            torch.clip(image_after - 1.0, min=0.0) ** 2, dim=(1, 2, 3),
        ).unsqueeze(-1)

        # 3. Entropy penalty (unchanged from Detection reward)
        if entropy is None:
            entropy_penalty = torch.zeros((B, 1), device=device)
        else:
            entropy = entropy if entropy.dim() == 2 else entropy.unsqueeze(-1)
            entropy_penalty = (
                (1.0 - float(progress))
                * self.exploration_penalty
                * (math.log(self.n_ops) - entropy)
            )

        # 4. Usage penalty — exponential in prior op-count (2^prev_count × base).
        # Matches AdaptiveISPReward semantics; requires state.op_usage to store
        # counts (int64), not bool. First pick pays 0; each repeat doubles.
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

        # 6. Runtime penalty (optional)
        if self.runtime_penalty_enabled:
            costs = torch.tensor(self.runtime_costs, device=device, dtype=torch.float32)
            chosen_cost = costs[safe_op].unsqueeze(-1)
            chosen_cost = chosen_cost * (~action.is_stop).float().unsqueeze(-1)
            runtime_penalty = chosen_cost * self.runtime_penalty_lambda
        else:
            runtime_penalty = torch.zeros((B, 1), device=device)

        if self.use_penalty:
            penalty = (
                overflow_penalty + entropy_penalty + usage_penalty
                + early_stop_penalty + runtime_penalty
            )
            reward = task_delta - penalty
        else:
            reward = task_delta

        return reward, RewardBreakdown(
            task_delta=task_delta,
            overflow_penalty=overflow_penalty,
            entropy_penalty=entropy_penalty,
            usage_penalty=usage_penalty,
            early_stop_penalty=early_stop_penalty,
            runtime_penalty=runtime_penalty,
            total=reward,
        ), q_parts


__all__ = ["HumanReward"]
