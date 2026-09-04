"""Executor (PipelineExecutor): applies ISPActions to PipelineStates.

Design
------
- PipelineExecutor.step(state, action) -> state: dispatches to the operator selected
  by action.op_indices, leaves stopped samples untouched, updates step and
  op_usage bookkeeping.
- PipelineExecutor does NOT know about Controllers, YOLO, or Reward — it only knows
  operators + state.

Also provides replay-format adapters that bridge PipelineState to/from the
flat [B, 3+N_ops] tensor that ReplayMemory continues to store.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import torch

from isp.base import ISPOperator
from pipeline.action import ISPAction
from pipeline.state import PipelineState


class PipelineExecutor:
    """Applies ISPActions to PipelineStates.

    canonical_order fixes the mapping between integer op_index and named
    operator. This mapping must be stable within a run (e.g., for
    checkpoint compatibility).
    """

    def __init__(self, operators: Mapping[str, ISPOperator], canonical_order: Sequence[str]) -> None:
        missing = [n for n in canonical_order if n not in operators]
        if missing:
            raise KeyError(f"PipelineExecutor: operators missing from registry: {missing}")
        self.operators = operators
        self.canonical_order = list(canonical_order)
        self.n_ops = len(self.canonical_order)

    def initial_state(self, image: torch.Tensor) -> PipelineState:
        """Fresh state for a new rollout batch."""
        b = image.shape[0]
        device = image.device
        return PipelineState(
            image=image,
            step=torch.zeros(b, dtype=torch.long, device=device),
            stopped=torch.zeros(b, dtype=torch.bool, device=device),
            has_reward=torch.zeros(b, dtype=torch.bool, device=device),
            op_usage=torch.zeros((b, self.n_ops), dtype=torch.long, device=device),
            history=[],
        )

    def step(self, state: PipelineState, action: ISPAction) -> PipelineState:
        """Apply one action to each batch element.

        - Elements already stopped are left untouched.
        - Elements that stop this step transition stopped=True but image is
          left as-is.
        - Otherwise: dispatch to `operators[canonical_order[op_idx]].apply`.
        """
        b = state.batch_size
        assert action.op_indices.shape[0] == b, (action.op_indices.shape, b)

        new_image = state.image.clone()
        new_op_usage = state.op_usage.clone()

        for op_idx in range(self.n_ops):
            mask = (action.op_indices == op_idx) & (~state.stopped) & (~action.is_stop)
            if not mask.any():
                continue
            op = self.operators[self.canonical_order[op_idx]]
            dim = op.spec.dim
            imgs = state.image[mask]
            params = action.params[mask, :dim]
            new_image[mask] = op.apply(imgs, params)
            new_op_usage[mask, op_idx] += 1

        new_stopped = state.stopped | action.is_stop
        new_step = state.step + (~state.stopped).long()

        return PipelineState(
            image=new_image,
            step=new_step,
            stopped=new_stopped,
            has_reward=state.has_reward,
            op_usage=new_op_usage,
            history=state.history + [action],
        )


# ------------------------- Replay-format adapters --------------------------
# ReplayMemory continues to store states as the legacy [B, 3+N_ops] float
# tensor: [has_reward, stopped, step, op_usage(0..N-1)]. These adapters
# translate between the tensor form and the PipelineState dataclass.

def pipeline_state_from_replay(
    images: torch.Tensor,
    states_tensor: torch.Tensor,
    n_ops: int,
) -> PipelineState:
    """Build a PipelineState from the flat [B, 3+n_ops] replay-memory tensor."""
    assert states_tensor.shape[1] == 3 + n_ops, \
        f"expected [B, {3 + n_ops}], got {tuple(states_tensor.shape)}"
    return PipelineState(
        image=images,
        step=states_tensor[:, 2].long(),
        stopped=states_tensor[:, 1].bool(),
        has_reward=states_tensor[:, 0].bool(),
        op_usage=states_tensor[:, 3:3 + n_ops].long(),
    )


def pipeline_state_to_replay(state: PipelineState) -> torch.Tensor:
    """Serialize a PipelineState to the flat [B, 3+n_ops] replay tensor."""
    return torch.cat([
        state.has_reward.float().unsqueeze(-1),
        state.stopped.float().unsqueeze(-1),
        state.step.float().unsqueeze(-1),
        state.op_usage.float(),
    ], dim=1)
