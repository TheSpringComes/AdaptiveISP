"""AdaptiveISPController: RL policy + value net for the AdaptiveISP task.

V1 keeps the original AdaptiveISP paper's algorithm intact, only relocated:
- Two independent FeatureExtractor CNNs (param regression / action selection)
- Per-op parameter heads keyed by op name
- Softmax selection + exploration mix + search-space mask
- Time-limit stop (no learned stop head)
- Value net is a sub-module of the Controller

Simplification vs the original (Option B in REFACTOR_V1.md §7): forward
ONLY the selected op's param head, not all N ops. V1 experiment-level
parity permits this — the gradient path differs but training dynamics
converge to a comparable mAP.
"""
from __future__ import annotations

from typing import Mapping, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from controller.base import Controller, ControllerOutput
from controller.adaptiveisp.network import AdaptiveISPValueNet, FeatureExtractor, pdf_sample
from isp.base import ISPOperator
from pipeline.action import ISPAction
from pipeline.state import PipelineState
from search.constraint import ConstraintResult


class AdaptiveISPController(Controller):
    """Concrete Controller for V1: RL policy + per-op parameter heads + critic."""
    def __init__(
        self,
        operators: Mapping[str, ISPOperator],
        canonical_order: Sequence[str],
        *,
        obs_hw: int = 64,
        mid_channels: int = 32,
        fc1_size: int = 128,
        feature_dim: int = 4096,
        dropout_keep_prob: float = 0.5,
        exploration: float = 0.05,
        max_steps: int = 5,
    ) -> None:
        super().__init__()
        missing = [n for n in canonical_order if n not in operators]
        if missing:
            raise KeyError(f"Controller: operators missing: {missing}")
        self.operators = nn.ModuleDict({n: operators[n] for n in canonical_order})
        self.canonical_order = list(canonical_order)
        self.n_ops = len(self.canonical_order)

        n_state = 3 + self.n_ops
        obs_channels = 3 + n_state
        dropout = 1.0 - dropout_keep_prob

        self.obs_hw = obs_hw
        self.down_sample = nn.AdaptiveAvgPool2d((obs_hw, obs_hw))
        self.param_features = FeatureExtractor(
            in_channels=obs_channels, input_hw=obs_hw,
            mid_channels=mid_channels, output_dim=feature_dim, dropout_prob=dropout,
        )
        self.select_features = FeatureExtractor(
            in_channels=obs_channels, input_hw=obs_hw,
            mid_channels=mid_channels, output_dim=feature_dim, dropout_prob=dropout,
        )
        self.param_heads = nn.ModuleDict()
        for name in self.canonical_order:
            op = operators[name]
            self.param_heads[name] = nn.Sequential(
                nn.Linear(feature_dim, fc1_size),
                nn.LeakyReLU(negative_slope=0.2),
                nn.Linear(fc1_size, op.spec.dim),
            )
        self.select_head = nn.Sequential(
            nn.Linear(feature_dim, fc1_size),
            nn.LeakyReLU(negative_slope=0.2),
            # n_ops op logits + 1 STOP logit (learned; last column)
            nn.Linear(fc1_size, self.n_ops + 1),
        )
        self.value_net = AdaptiveISPValueNet(
            n_ops=self.n_ops, obs_hw=obs_hw,
            mid_channels=mid_channels, fc1_size=fc1_size, feature_dim=feature_dim,
        )
        self.exploration = float(exploration)
        self.max_steps = int(max_steps)

    def _build_obs(self, state: PipelineState) -> torch.Tensor:
        img = self.down_sample(state.image)
        hw = img.shape[-1]
        B = state.batch_size
        state_ch = torch.cat([
            state.step.float().view(B, 1, 1, 1).expand(B, 1, hw, hw),
            state.stopped.float().view(B, 1, 1, 1).expand(B, 1, hw, hw),
            state.has_reward.float().view(B, 1, 1, 1).expand(B, 1, hw, hw),
            state.op_usage.float().view(B, self.n_ops, 1, 1).expand(B, self.n_ops, hw, hw),
        ], dim=1)
        return torch.cat([img, state_ch], dim=1)

    def act(
        self,
        state: PipelineState,
        constraint: ConstraintResult,
        *,
        noise: Optional[torch.Tensor] = None,
    ) -> ControllerOutput:
        obs = self._build_obs(state)
        pf = self.param_features(obs)
        sf = self.select_features(obs)

        # Action space: n_ops op choices + 1 STOP action (index = n_ops).
        n_actions = self.n_ops + 1
        logits = self.select_head(sf)                             # [B, n_ops + 1]
        pdf = F.softmax(logits, dim=1) + 1e-37
        pdf = pdf * (1.0 - self.exploration) + self.exploration / n_actions

        # Extend the op-mask with a STOP column. STOP is forbidden at step 0
        # (policy must apply at least one op before it can stop) — otherwise
        # the trivial all-STOP policy is a shallow local optimum.
        stop_col_ok = (state.step > 0).float().unsqueeze(-1)      # (B, 1)
        extended_mask = torch.cat(
            [constraint.op_mask.float(), stop_col_ok], dim=1,
        )                                                          # [B, n_ops + 1]
        pdf = pdf * extended_mask
        pdf = pdf / (pdf.sum(dim=1, keepdim=True) + 1e-30)
        entropy = (-pdf * torch.log(pdf + 1e-10)).sum(dim=1, keepdim=True)

        B = state.batch_size
        device = obs.device

        if self.training:
            if noise is None:
                noise = torch.rand(B, 1, device=device)
            action_indices = pdf_sample(pdf, noise).to(torch.int64)
        else:
            action_indices = pdf.argmax(dim=1).to(torch.int64)

        log_prob = torch.log(pdf.gather(1, action_indices.unsqueeze(-1)) + 1e-10)

        # Split STOP (action == n_ops) from op picks. STOP samples get a dummy
        # op_index=0; PipelineExecutor filters them out via `~is_stop`.
        stop_action = action_indices == self.n_ops
        op_indices = torch.where(
            stop_action, torch.zeros_like(action_indices), action_indices,
        )

        max_dim = max(self.operators[n].spec.dim for n in self.canonical_order)
        params_flat = torch.zeros(B, max_dim, device=device)
        for idx, name in enumerate(self.canonical_order):
            mask = (op_indices == idx) & (~stop_action)
            if not mask.any():
                continue
            op = self.operators[name]
            raw = self.param_heads[name](pf[mask])
            physical = op.spec.regressor(raw)
            if physical.dim() > 2:
                physical = physical.reshape(physical.shape[0], -1)
            params_flat[mask, :op.spec.dim] = physical

        # Final is_stop: learned STOP OR time-limit reached OR already stopped.
        is_stop_time = state.step >= (self.max_steps - 1)
        is_stop = stop_action | is_stop_time | state.stopped

        value = self.value_net(state)

        return ControllerOutput(
            action=ISPAction(op_indices=op_indices, params=params_flat, is_stop=is_stop),
            logits=logits,
            log_prob=log_prob,
            value=value,
            entropy=entropy,
            pdf=pdf,
        )

    def evaluate(
        self,
        state: PipelineState,
        constraint: ConstraintResult,
        op_indices: torch.Tensor,
        is_stop: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Recompute (log_prob, value, entropy) for a FIXED action under the
        current policy — the PPO update path.

        Same pdf construction as `.act` (exploration mix + extended STOP
        mask); the only difference is that instead of sampling / argmaxing
        we `gather` log-prob at the pre-recorded action index. `op_indices`
        + `is_stop` encode the same categorical choice as the rollout-time
        sample: `sampled_idx = n_ops if is_stop else op_indices`.

        Params are deterministic (no separate parameter distribution) so
        this method does NOT re-project params — PPO clips only the
        categorical op-choice distribution.
        """
        obs = self._build_obs(state)
        sf = self.select_features(obs)

        n_actions = self.n_ops + 1
        logits = self.select_head(sf)                              # [B, n_ops + 1]
        pdf = F.softmax(logits, dim=1) + 1e-37
        pdf = pdf * (1.0 - self.exploration) + self.exploration / n_actions

        stop_col_ok = (state.step > 0).float().unsqueeze(-1)
        extended_mask = torch.cat(
            [constraint.op_mask.float(), stop_col_ok], dim=1,
        )
        pdf = pdf * extended_mask
        pdf = pdf / (pdf.sum(dim=1, keepdim=True) + 1e-30)
        entropy = (-pdf * torch.log(pdf + 1e-10)).sum(dim=1, keepdim=True)

        # Rebuild the sampled action index (op or STOP).
        sampled_idx = torch.where(
            is_stop, torch.full_like(op_indices, self.n_ops), op_indices,
        ).to(torch.int64).unsqueeze(-1)
        log_prob = torch.log(pdf.gather(1, sampled_idx) + 1e-10).squeeze(-1)

        value = self.value_net(state).reshape(-1)
        entropy = entropy.reshape(-1)
        return log_prob, value, entropy


__all__ = ["AdaptiveISPController"]
