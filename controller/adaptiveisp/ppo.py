"""PPO update — clipped policy + MSE value + entropy bonus, K epochs.

Runs on a flat batch produced by `pipeline.TrajectoryBuffer.flat_alive_batch()`,
using the Controller's own `.evaluate(state, constraint, op_indices, is_stop)`
to recompute the new log-prob / value / entropy at every minibatch.

Scope (V3-E3 phase 1):
- Only the categorical op-choice distribution is clipped. Params (the
  continuous outputs of `param_heads`) are deterministic under the
  current Controller and are trained separately during rollout via
  reward-graph backprop through the executor — this file does not
  touch `param_features` / `param_heads`.
- The `optimizer` passed in is expected to cover the "policy + value"
  parameter group only (select_features + select_head + value_net).
  The trainer wires the two-optimizer split.
- Gradient clipping norm applies to that optimizer's own param set.

Hyperparams from cfg.rl_algo.ppo (all optional):
    epochs           — K, number of passes over the batch (default 4)
    clip_range       — ε in the clipped surrogate    (default 0.2)
    value_coef       — weight on the value MSE       (default 0.5)
    entropy_coef     — weight on the entropy bonus   (default 0.01)
    advantage_norm   — normalize advantages per minibatch (default True)
    minibatch_size   — SGD minibatch size            (default 64)
    grad_clip_norm   — grad-norm clip for policy opt (default 0.5)
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn.functional as F

from pipeline.state import PipelineState
from pipeline.trajectory import TrajectoryBuffer


def _rebuild_state(mb: dict) -> PipelineState:
    """Reassemble a PipelineState from a flat-batch minibatch dict."""
    return PipelineState(
        image=mb["images"],
        step=mb["step"],
        stopped=mb["stopped"],
        has_reward=mb["has_reward"],
        op_usage=mb["op_usage"],
    )


class PPOUpdater:
    def __init__(self, ppo_cfg: dict | None) -> None:
        cfg = ppo_cfg or {}
        self.epochs = int(cfg.get("epochs", 4))
        self.clip_range = float(cfg.get("clip_range", 0.2))
        self.value_coef = float(cfg.get("value_coef", 0.5))
        self.entropy_coef = float(cfg.get("entropy_coef", 0.01))
        self.advantage_norm = bool(cfg.get("advantage_norm", True))
        self.minibatch_size = int(cfg.get("minibatch_size", 64))
        self.grad_clip_norm = float(cfg.get("grad_clip_norm", 0.5))

    def update(
        self,
        controller,
        search_space,
        optimizer: torch.optim.Optimizer,
        flat_batch: dict,
    ) -> dict:
        """Run K-epoch PPO on `flat_batch`. Returns diagnostic averages."""
        n = flat_batch["op_indices"].shape[0]
        if n == 0:
            return {
                "policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                "approx_kl": 0.0, "clip_frac": 0.0, "n_mbs": 0,
            }

        totals = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0,
                  "approx_kl": 0.0, "clip_frac": 0.0, "n_mbs": 0}

        # Params under this optimizer — for grad clipping. Flatten once.
        opt_params: list[torch.nn.Parameter] = [
            p for g in optimizer.param_groups for p in g["params"]
        ]

        for mb in TrajectoryBuffer.iter_minibatches(
            flat_batch, self.minibatch_size, self.epochs, shuffle=True,
        ):
            state_mb = _rebuild_state(mb)
            c_mb = search_space.valid_actions(state_mb)
            new_lp, new_v, new_ent = controller.evaluate(
                state_mb, c_mb, mb["op_indices"], mb["is_stop"],
            )

            adv = mb["advantages"]
            if self.advantage_norm and adv.numel() > 1:
                adv = (adv - adv.mean()) / (adv.std() + 1e-8)

            ratio = torch.exp(new_lp - mb["old_log_probs"])
            surr1 = ratio * adv
            surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * adv
            policy_loss = -torch.min(surr1, surr2).mean()

            value_loss = 0.5 * F.mse_loss(new_v, mb["returns"])
            entropy = new_ent.mean()

            loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(opt_params, self.grad_clip_norm)
            optimizer.step()

            with torch.no_grad():
                totals["policy_loss"] += float(policy_loss.item())
                totals["value_loss"] += float(value_loss.item())
                totals["entropy"] += float(entropy.item())
                # approx_kl = E[old - new]; sign convention matches CleanRL.
                totals["approx_kl"] += float(
                    (mb["old_log_probs"] - new_lp).mean().item()
                )
                totals["clip_frac"] += float(
                    ((ratio - 1.0).abs() > self.clip_range).float().mean().item()
                )
                totals["n_mbs"] += 1

        n_mbs = max(totals["n_mbs"], 1)
        return {
            "policy_loss": totals["policy_loss"] / n_mbs,
            "value_loss":  totals["value_loss"]  / n_mbs,
            "entropy":     totals["entropy"]     / n_mbs,
            "approx_kl":   totals["approx_kl"]   / n_mbs,
            "clip_frac":   totals["clip_frac"]   / n_mbs,
            "n_mbs":       totals["n_mbs"],
        }


__all__ = ["PPOUpdater"]
