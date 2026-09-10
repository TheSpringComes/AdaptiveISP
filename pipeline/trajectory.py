"""T-step trajectory buffer + GAE for on-policy PPO training.

Collected during a single rollout of `T = cfg.test_steps` steps on a
batch of `B` samples. The buffer stores per-step, per-sample data;
after the rollout ends it computes advantages via GAE and yields
minibatches suitable for a PPO K-epoch update.

Layout convention: every stored tensor has leading shape `[T, B, ...]`.
Fields:
    images       [T, B, 3, H, W]   state.image at the START of step t
    op_usage     [T, B, N_ops]     state.op_usage at the START of step t
    step_idx     [T, B]            state.step at the START of step t
    stopped      [T, B]            state.stopped at the START of step t
    has_reward   [T, B]            state.has_reward at the START of step t
    op_indices   [T, B]            action.op_indices sampled at step t
    is_stop      [T, B]            action.is_stop at step t
    log_probs    [T, B]            log π_old(a_t | s_t) at rollout time
    values       [T, B]            V_old(s_t) at rollout time
    entropies    [T, B]            H[π_old(· | s_t)] at rollout time
    rewards      [T, B]            r_t (per-step reward)
    alive        [T, B]            True iff sample was NOT already stopped
                                    at the START of step t; loss ignores
                                    dead entries.
    dones        [T, B]            True iff the rollout terminates at t
                                    (is_stop OR truncation OR t == T-1).

Not stored: params_flat, backbone-processed inputs (see Detection
trainer for backbone application), z-noise. Params are deterministic
under the current Controller so PPO clips only the categorical op-choice
distribution.

After rollout the trainer calls:
    buf.compute_gae(gamma, lam, bootstrap_value)
where `bootstrap_value` = V(s_T) for still-alive samples (or zeros for
samples that terminated).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterator

import torch

from pipeline.action import ISPAction
from pipeline.state import PipelineState


@dataclass
class TrajectoryBuffer:
    """One (T, B) rollout of on-policy data + derived GAE advantages/returns.

    Populate by calling `push(...)` T times, then `compute_gae(...)`.
    """
    max_steps: int
    batch_size: int
    # -------- per-step lists; each entry is a length-B tensor / dataclass --------
    _states: list[PipelineState] = field(default_factory=list)
    _op_indices: list[torch.Tensor] = field(default_factory=list)
    _is_stop: list[torch.Tensor] = field(default_factory=list)
    _log_probs: list[torch.Tensor] = field(default_factory=list)
    _values: list[torch.Tensor] = field(default_factory=list)
    _entropies: list[torch.Tensor] = field(default_factory=list)
    _rewards: list[torch.Tensor] = field(default_factory=list)
    _alive: list[torch.Tensor] = field(default_factory=list)

    # -------- computed by compute_gae --------
    advantages: torch.Tensor | None = None    # [T, B]
    returns: torch.Tensor | None = None       # [T, B]

    def push(
        self,
        *,
        state: PipelineState,
        action: ISPAction,
        log_prob: torch.Tensor,
        value: torch.Tensor,
        entropy: torch.Tensor,
        reward: torch.Tensor,
    ) -> None:
        """Record one step. All tensors are batched over B (dim 0).

        `state` is snapshotted with `image.detach()` so PPO's later
        forward passes build fresh graphs; leaving the rollout graph
        attached would blow up on the second minibatch backward.
        """
        # `alive` is derived from state.stopped BEFORE the step is applied.
        # (state.stopped[b] means sample b had already emitted STOP earlier;
        # its per-step contribution to the loss must be zeroed.)
        alive = ~state.stopped
        detached_state = PipelineState(
            image=state.image.detach().clone(),
            step=state.step.clone(),
            stopped=state.stopped.clone(),
            has_reward=state.has_reward.clone(),
            op_usage=state.op_usage.clone(),
        )
        self._states.append(detached_state)
        self._op_indices.append(action.op_indices.clone())
        self._is_stop.append(action.is_stop.clone())
        # Flatten (B,1) -> (B,) for the scalar fields to keep [T, B] tidy.
        self._log_probs.append(log_prob.detach().reshape(-1))
        self._values.append(value.detach().reshape(-1))
        self._entropies.append(entropy.detach().reshape(-1))
        self._rewards.append(reward.detach().reshape(-1))
        self._alive.append(alive.clone())

    def __len__(self) -> int:
        return len(self._states)

    def compute_gae(
        self,
        gamma: float,
        lam: float,
        bootstrap_value: torch.Tensor,
    ) -> None:
        """Populate `self.advantages` and `self.returns` via GAE(λ).

        `bootstrap_value` is V(s_T) for samples that reach step T without
        stopping; passing zeros gives Monte-Carlo returns on truncation
        (same shape as a per-step value: [B]). For samples that emitted
        STOP mid-rollout, the bootstrap is naturally masked out below.
        """
        T = len(self)
        assert T > 0, "compute_gae called on an empty buffer"
        B = self.batch_size
        device = self._values[0].device

        values = torch.stack(self._values, dim=0)                         # [T, B]
        rewards = torch.stack(self._rewards, dim=0)                       # [T, B]
        alive = torch.stack(self._alive, dim=0).float()                   # [T, B]
        is_stop = torch.stack(self._is_stop, dim=0).float()               # [T, B]

        # `done_t`: rollout terminates FOR THIS SAMPLE at step t iff
        # (a) sample emits STOP at t, OR (b) t is the last step. Both
        # zero-out the bootstrap term of the Bellman equation.
        dones = is_stop.clone()
        dones[-1] = 1.0                                                    # truncation at t = T-1

        # Backward GAE recursion.
        advantages = torch.zeros((T, B), device=device)
        last_adv = torch.zeros(B, device=device)
        for t in reversed(range(T)):
            if t == T - 1:
                next_value = bootstrap_value.reshape(-1)
            else:
                next_value = values[t + 1]
            delta = rewards[t] + gamma * next_value * (1.0 - dones[t]) - values[t]
            last_adv = delta + gamma * lam * (1.0 - dones[t]) * last_adv
            advantages[t] = last_adv
        # Dead entries (alive == 0) contribute nothing.
        advantages = advantages * alive
        returns = advantages + values

        self.advantages = advantages
        self.returns = returns

    # ------------------------------ flat batches ------------------------------

    def flat_alive_batch(self) -> dict[str, torch.Tensor]:
        """Concatenate T×B along a single axis and keep ONLY alive entries.

        Returns dict with keys:
            images, op_usage, step, stopped, has_reward   — for state reconstruction
            op_indices, is_stop                            — sampled action
            old_log_probs, old_values                      — behavior policy stats
            advantages, returns                            — GAE outputs

        PPO minibatch shuffling operates on this flattened view.
        """
        assert self.advantages is not None and self.returns is not None, \
            "call compute_gae before flat_alive_batch"

        images     = torch.cat([s.image      for s in self._states], dim=0)   # [T*B, 3, H, W]
        op_usage   = torch.cat([s.op_usage   for s in self._states], dim=0)   # [T*B, N_ops]
        step       = torch.cat([s.step       for s in self._states], dim=0)   # [T*B]
        stopped    = torch.cat([s.stopped    for s in self._states], dim=0)   # [T*B]
        has_reward = torch.cat([s.has_reward for s in self._states], dim=0)   # [T*B]

        op_indices = torch.cat(self._op_indices, dim=0)
        is_stop    = torch.cat(self._is_stop, dim=0)
        old_lp     = torch.cat(self._log_probs, dim=0)
        old_v      = torch.cat(self._values, dim=0)
        alive      = torch.cat(self._alive, dim=0)

        # advantages / returns are [T, B] — reshape to [T*B] in the same order.
        adv = self.advantages.reshape(-1)
        ret = self.returns.reshape(-1)

        keep = alive
        return {
            "images":       images[keep],
            "op_usage":     op_usage[keep],
            "step":         step[keep],
            "stopped":      stopped[keep],
            "has_reward":   has_reward[keep],
            "op_indices":   op_indices[keep],
            "is_stop":      is_stop[keep],
            "old_log_probs": old_lp[keep],
            "old_values":    old_v[keep],
            "advantages":    adv[keep],
            "returns":       ret[keep],
        }

    @staticmethod
    def iter_minibatches(
        batch: dict[str, torch.Tensor],
        minibatch_size: int,
        num_epochs: int,
        shuffle: bool = True,
    ) -> Iterator[dict[str, torch.Tensor]]:
        """Yield `num_epochs` passes over `batch`, chunked into `minibatch_size` rows.

        Each yielded minibatch is a dict with the same keys, each value
        sliced along the leading axis.
        """
        # All fields share the same leading dimension after flat_alive_batch.
        n = next(iter(batch.values())).shape[0]
        device = next(iter(batch.values())).device
        for _ in range(num_epochs):
            idx = torch.randperm(n, device=device) if shuffle else torch.arange(n, device=device)
            for start in range(0, n, minibatch_size):
                sel = idx[start:start + minibatch_size]
                if sel.numel() == 0:
                    continue
                yield {k: v[sel] for k, v in batch.items()}


__all__ = ["TrajectoryBuffer"]
