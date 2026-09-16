"""Human task rewards: terminal-only (legacy) and stepwise (dense).

HumanReward (legacy, terminal-only):
  - Intermediate steps: `task_delta = 0`. Only the auxiliary penalties
    (entropy / usage / early-stop / runtime) fire.
  - Terminal step:      `task_delta = Q(I_T) - Q(I_0)` where
      Q(I) = λ_ssim · SSIM(I, Expert C) − λ_lpips · LPIPS(I, Expert C)
  Credit assignment relies on the value net (TD/GAE).

StepwiseHumanReward (dense, 2026-09-13 redesign):
  Every step gets its own quality delta — the policy sees immediately
  whether THIS op helped:

      r_t = α·[Q(I_{t+1}) − Q(I_t)]                      (per-step quality)
            − P_usage − P_runtime − P_invalid            (small guard terms)
            + 1[STOP]·β·max(Q(I_t) − Q(I_0), 0)          (progress-gated stop)

  Design notes (vs legacy):
  - No big terminal task reward — ΔQ is credited where it happens, so the
    total already telescopes to α·[Q(I_T) − Q(I_0)]; adding a terminal term
    would double-count and re-inflate the value scale.
  - STOP bonus is disabled by default (β=0); opt-in ablation only.
    When enabled, STOP is not a flat bonus/penalty: stopping pays in proportion to
    the improvement over the Front-ISP baseline. Stop when nothing has
    improved → no payoff; stop after real gains → positive payoff. This is
    the "见好就收" shaping that keeps STOP from dominating at step 1.
  - Guard penalties are sized to stay BELOW a typical positive per-step
    ΔQ (~0.02–0.08 measured on fixed-front runs) so they can only filter
    out meaningless moves, never out-compete real quality gains:
      usage   = 2^prev_count × base      (base ~0.002, exponential — the
                                          2nd use of an op already costs 4×)
      runtime = cost[op] × λ             (λ ~0.0005)
      invalid = overflow + NaN           (strong — genuinely abnormal only)
  - Entropy / early-stop penalties are DROPPED from this variant: entropy
    shaping conflicts with PPO's own entropy bonus; the flat early-stop fee
    is superseded by the progress-gated stop reward.
"""
from __future__ import annotations

from typing import Optional

import torch

from controller.adaptiveisp.reward import Reward, RewardBreakdown
from pipeline.action import ISPAction
from pipeline.state import PipelineState
from tasks.human_quality.metrics import quality_score


class HumanReward(Reward):
    """Terminal-only quality reward (legacy — kept for ablation parity).

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
        lambda_lab_ab: float = 0.0,
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
        self.lambda_lab_ab = float(lambda_lab_ab)
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
        entropy_max: Optional[torch.Tensor] = None,
        progress: float = 0.0,
        q_initial: Optional[torch.Tensor] = None,
        q_before: Optional[torch.Tensor] = None,
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
                    lambda_lab_ab=self.lambda_lab_ab,
                )
            q_after, q_parts = quality_score(
                state_after.image, target,
                lambda_ssim=self.lambda_ssim,
                lambda_lpips=self.lambda_lpips,
                lpips_net=self.lpips_net,
                lambda_lab_ab=self.lambda_lab_ab,
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

        # 3. Entropy gap over the actual masked categorical support.
        if entropy is None:
            entropy_penalty = torch.zeros((B, 1), device=device)
        else:
            entropy = entropy if entropy.dim() == 2 else entropy.unsqueeze(-1)
            if entropy_max is None:
                raise ValueError("HumanReward requires masked entropy_max when entropy is supplied")
            entropy_max = entropy_max.reshape(B, 1)
            entropy_penalty = (
                (1.0 - float(progress))
                * self.exploration_penalty
                * (entropy_max - entropy).clamp_min(0.0)
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


class StepwiseHumanReward(Reward):
    """Dense per-step quality reward (2026-09-13 redesign — see module doc).

    r_t = α·ΔQ_t − P_usage − P_runtime − P_invalid
          + 1[STOP]·β·max(Q(I_t) − Q(I_0), 0)

    The trainer must pass `q_before` (Q of the pre-step image, computed on
    the PREVIOUS step's after-image or the Front-ISP baseline at t=0) so
    each step needs exactly one new quality_score call.
    """

    def __init__(
        self,
        *,
        n_ops: int,
        max_steps: int,
        lambda_ssim: float = 1.0,
        lambda_lpips: float = 1.0,
        lpips_net: str = "alex",
        lambda_lab_ab: float = 0.0,
        quality_scale: float = 1.0,            # α
        stop_bonus_beta: float = 0.0,          # β
        usage_penalty: float = 0.002,          # base × 2^k guard
        runtime_penalty_enabled: bool = False,
        runtime_penalty_lambda: float = 0.0005,
        runtime_costs: Optional[list[float]] = None,
        invalid_penalty: float = 0.1,          # overflow scale; NaN pays flat 1.0
        lambda_param: float = 0.0,             # neutral-distance param penalty coef
    ) -> None:
        self.n_ops = int(n_ops)
        self.max_steps = int(max_steps)
        self.lambda_ssim = float(lambda_ssim)
        self.lambda_lpips = float(lambda_lpips)
        self.lambda_lab_ab = float(lambda_lab_ab)
        self.lpips_net = lpips_net
        # param_penalty 查表用（trainer 注入：op 名列表 + spec 表）
        self._op_names: list = []
        self._op_specs: dict = {}
        self.alpha = float(quality_scale)
        self.beta = float(stop_bonus_beta)
        self.usage_base = float(usage_penalty)
        self.runtime_penalty_enabled = bool(runtime_penalty_enabled)
        self.runtime_penalty_lambda = float(runtime_penalty_lambda)
        self.runtime_costs = list(runtime_costs or [])
        self.invalid_penalty = float(invalid_penalty)
        self.lambda_param = float(lambda_param)

        if self.runtime_penalty_enabled and len(self.runtime_costs) != self.n_ops:
            raise ValueError(
                f"StepwiseHumanReward: runtime_costs has {len(self.runtime_costs)} "
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
        entropy: Optional[torch.Tensor] = None,   # unused (kept for signature parity)
        progress: float = 0.0,                     # unused
        q_initial: Optional[torch.Tensor] = None,
        q_before: Optional[torch.Tensor] = None,
        physical_params: Optional[torch.Tensor] = None,   # [B, max_dim] 实际执行参数
        param_op_name: Optional[str] = None,              # 本步（sample 0 参考）算子
        param_op_indices: Optional[torch.Tensor] = None,  # [B] 每样本选中算子 idx
    ) -> tuple[torch.Tensor, RewardBreakdown, dict[str, torch.Tensor]]:
        """One new quality_score call per step; returns (reward, breakdown, q_parts).

        `q_parts` is ALWAYS populated (dense variant — every step has a
        fresh Q) so the trainer's logging uses the last one as Q_T without
        a special terminal branch. `q_initial` / `q_before` must be passed
        by the trainer (see HumanTrainer rollout loop).
        """
        B = state_after.batch_size
        device = state_after.image.device
        if q_initial is None:
            q_initial, _ = quality_score(
                image_initial, target,
                lambda_ssim=self.lambda_ssim,
                lambda_lpips=self.lambda_lpips,
                lpips_net=self.lpips_net,
                lambda_lab_ab=self.lambda_lab_ab,
            )
        if q_before is None:
            # Fallback (single-step semantics): before-image == baseline.
            q_before = q_initial

        # ---- per-step quality parts (one fresh SSIM+LPIPS forward) ----
        q_after, q_parts = quality_score(
            state_after.image, target,
            lambda_ssim=self.lambda_ssim,
            lambda_lpips=self.lambda_lpips,
            lpips_net=self.lpips_net,
            lambda_lab_ab=self.lambda_lab_ab,
        )
        # Already-stopped samples: image is frozen by the executor, so
        # ΔQ_t = 0 naturally — but the executor may emit clamped/unchanged
        # tensors; force exact zero to avoid float noise accumulating.
        # NaN 隔离：NaN 帧的 quality_score 是 nan，先把 delta 归零 —— 该帧
        # 的信号完全由 invalid_penalty（恒罚 1.0）承载，绝不让 nan 进 PPO。
        was_alive = (~state_before.stopped).float().unsqueeze(-1)
        delta_q = torch.nan_to_num(q_after - q_before, nan=0.0,
                                   posinf=0.0, neginf=0.0) * was_alive

        # 1. Per-step task reward: α · ΔQ_t
        task_delta = self.alpha * delta_q

        # 2. Progress-gated STOP reward: β · max(Q(I_t) − Q(I_0), 0).
        #    Fires only on the step where the policy submits STOP (learned
        #    stop; time-limit stop gets nothing — no bonus for being forced).
        learned_stop = (action.is_stop & ~state_before.stopped
                        & (state_before.step < self.max_steps - 1)).float().unsqueeze(-1)
        global_gain = torch.clamp(torch.nan_to_num(
            q_after - q_initial, nan=0.0, posinf=0.0, neginf=0.0), min=0.0)
        stop_reward = self.beta * learned_stop * global_gain

        # 3. Usage penalty — small exponential guard (base × 2^k).
        safe_op = torch.where(
            action.is_stop, torch.zeros_like(action.op_indices), action.op_indices,
        )
        batch_idx = torch.arange(B, device=device)
        prev_count = state_before.op_usage[batch_idx, safe_op].float().unsqueeze(-1)
        is_repeat = (prev_count > 0).float()
        usage_penalty = (
            is_repeat * torch.pow(torch.tensor(2.0, device=device), prev_count)
            * self.usage_base * (~action.is_stop).float().unsqueeze(-1)
        )

        # 4. Runtime penalty — tiny, optional.
        if self.runtime_penalty_enabled:
            costs = torch.tensor(self.runtime_costs, device=device, dtype=torch.float32)
            chosen_cost = costs[safe_op].unsqueeze(-1)
            runtime_penalty = (
                chosen_cost * (~action.is_stop).float().unsqueeze(-1)
                * self.runtime_penalty_lambda
            )
        else:
            runtime_penalty = torch.zeros((B, 1), device=device)

        # 5. Invalid penalty — STRONG guard, but only for GENUINELY abnormal
        #    frames. Untrained CCM params already push ~0.5-0.6 mean-square
        #    overflow (values to 4×, no clamp in the executor), and that is
        #    a NORMAL exploration state, not an anomaly — so scale linearly
        #    with a small coefficient (default 0.1): typical untrained
        #    overflow costs ~0.06 (< typical ΔQ 0.02–0.08), while a truly
        #    broken frame (overflow ≥ 1 or NaN) still pays ≥ 0.1–1.1.
        #    NaN 先 nan_to_num 归零再进 mean —— 否则 overflow 也是 nan，
        #    会污染整条 reward（PPO 对 nan 的 advantage 直接失效）。
        image_after = state_after.image
        is_nan = (torch.isnan(image_after).flatten(1).any(dim=1)
                  | torch.isinf(image_after).flatten(1).any(dim=1)
                  ).float().unsqueeze(-1)
        clean_after = torch.nan_to_num(image_after, nan=0.0,
                                       posinf=0.0, neginf=0.0)
        overflow = torch.mean(
            torch.clip(clean_after - 1.0, min=0.0) ** 2, dim=(1, 2, 3),
        ).unsqueeze(-1)
        # NaN/Inf 是硬失效：单独支付全额 1.0（invalid_penalty 只缩放 overflow）。
        invalid_penalty = overflow * self.invalid_penalty + is_nan * 1.0

        # 6. Neutral-distance parameter regularization:
        #    P_param = λ_p · d_t²，d_t = 实际参数相对 neutral 的归一化距离。
        #    λ_p 由 trainer 按 cosine schedule 设置（progress <30% 衰减到 0）；
        #    只对实际执行的算子参数计罚（STOP / 未执行步不罚）。
        from isp.param_reg import param_distance
        param_penalty = torch.zeros((B, 1), device=device)
        if self.lambda_param > 0.0 and physical_params is not None \
                and param_op_indices is not None:
            # 逐样本：其选中算子的参数距离。physical_params 是 [B, max_dim]，
            # 每个样本只有自己算子的前 spec.dim 维有效。
            for b in range(B):
                if action.is_stop[b] or state_before.stopped[b]:
                    continue
                op_idx = int(param_op_indices[b].item())
                op_name = self._op_names[op_idx]
                spec = self._op_specs[op_name]
                d = param_distance(op_name, spec,
                                   physical_params[b:b+1, :spec.dim])
                param_penalty[b, 0] = (d.mean() ** 2).item()

        # Total: r_t = αΔQ_t − guards + stop_reward − P_param
        reward = (task_delta - usage_penalty - runtime_penalty
                  - invalid_penalty - self.lambda_param * param_penalty
                  + stop_reward)

        # Breakdown maps onto the existing logging fields: `task_delta`
        # holds the dense αΔQ_t; `stop_bonus` (unused in legacy human) now
        # carries the progress-gated stop reward; entropy/early-stop are
        # hard zeros in this variant.
        return reward, RewardBreakdown(
            task_delta=task_delta,
            overflow_penalty=invalid_penalty,
            entropy_penalty=torch.zeros((B, 1), device=device),
            usage_penalty=usage_penalty,
            early_stop_penalty=torch.zeros((B, 1), device=device),
            runtime_penalty=runtime_penalty,
            total=reward,
            stop_bonus=stop_reward,
            param_penalty=param_penalty if self.lambda_param > 0.0 else None,
        ), q_parts


__all__ = ["HumanReward", "StepwiseHumanReward"]
