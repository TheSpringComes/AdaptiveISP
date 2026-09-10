"""Action-mask priors for the Adaptive Tail (V3-A2 阶段 1).

Each prior is a callable that produces a `[B, N_ops]` bool mask given the
current per-sample op_usage counts. `False` means "this op is forbidden for
this sample right now"; `SearchSpace` ANDs every prior's mask into the base
(all-True) op_mask.

Three concrete priors, all opt-in via config:

1. `NoRepeatMask`
   Forbid picking an op a second time. Any op with `op_usage[b, i] > 0`
   is masked out.

2. `OrderMask`
   Hard order rules. Each rule = ({after}, {before}): once ANY op in
   `after` has been used, every op in `before` is masked out. Example
   from the V3 plan: `sharpen` used → `denoise` forbidden afterwards.

3. `GroupBudgetMask`
   Group-level cap on total picks. Each group = (ops, max_select). Once
   the sample has used ops in the group a total of `max_select` times,
   every op in the group is masked. Used both for hard same-class mutex
   (max_select=1, e.g. two denoise implementations, three AWB flavors)
   and softer tonal-group caps (max_select=2 on
   exposure/gamma/tone/contrast).

All three read only `state.op_usage` — no image content, no history walk —
so the mask computation is O(B × N_ops) and stays on the same device as
the state. Priors are pure; construction fixes their rule/group tables
from the config, and `__call__` is stateless.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch

from pipeline.state import PipelineState


def _indices_for(names: Sequence[str], canonical_order: Sequence[str]) -> list[int]:
    """Resolve a list of op names into canonical-order indices.

    Silently drops names that aren't in `canonical_order` — this keeps
    config groups portable across experiments that use different op subsets
    (an rule that mentions `n_denoise` still works even when the config
    omits `n_denoise` from its `operators` list).
    """
    idx_map = {n: i for i, n in enumerate(canonical_order)}
    return [idx_map[n] for n in names if n in idx_map]


class NoRepeatMask:
    """Every op is single-use per rollout, unless whitelisted in `exempt`.

    `exempt` names an op set that MAY be applied more than once (e.g. a
    small residual-refinement neural op that legitimately benefits from
    iteration). Names not present in `canonical_order` are silently
    dropped, so the whitelist stays portable across op subsets.
    """

    def __init__(
        self, exempt: Sequence[str] | None = None,
        canonical_order: Sequence[str] | None = None,
    ) -> None:
        # Pre-resolve exempt indices to a set for O(1) skip.
        if exempt and canonical_order is not None:
            self._exempt_idx = set(_indices_for(list(exempt), canonical_order))
        else:
            self._exempt_idx = set()

    def __call__(self, state: PipelineState, canonical_order: Sequence[str]) -> torch.Tensor:
        # Base rule: `op_usage[b, i] > 0`  →  forbid op i for sample b.
        mask = state.op_usage == 0
        if not self._exempt_idx:
            return mask
        # Whitelisted ops: force to True regardless of usage.
        N = len(canonical_order)
        device = mask.device
        exempt_col = torch.zeros(N, dtype=torch.bool, device=device)
        exempt_col[torch.as_tensor(sorted(self._exempt_idx), device=device)] = True
        return mask | exempt_col.unsqueeze(0)


class OrderMask:
    """Hard sequencing rules: once an `after` op is used, its `before` ops are forbidden.

    `rules` is a list of dicts (as loaded from yaml):
        {"after": ["sharpen", "inf_unsharp"],
         "before": ["denoise", "inf_nlm", "inf_ebf"]}
    """

    def __init__(self, rules: list[dict], canonical_order: Sequence[str]) -> None:
        # Pre-resolve to (after_idx: list[int], before_idx: list[int]) — cheap
        # at __call__ time. Skip rules whose `after` side is empty after the
        # canonical-order filter (nothing can trigger them).
        self._rules: list[tuple[list[int], list[int]]] = []
        for r in rules:
            a_names = list(r.get("after", []) or [])
            b_names = list(r.get("before", []) or [])
            a_idx = _indices_for(a_names, canonical_order)
            b_idx = _indices_for(b_names, canonical_order)
            if a_idx and b_idx:
                self._rules.append((a_idx, b_idx))

    def __call__(self, state: PipelineState, canonical_order: Sequence[str]) -> torch.Tensor:
        B = state.batch_size
        N = len(canonical_order)
        device = state.image.device
        mask = torch.ones((B, N), dtype=torch.bool, device=device)
        if not self._rules:
            return mask
        usage = state.op_usage                                            # (B, N) int64
        for a_idx, b_idx in self._rules:
            triggered = usage[:, a_idx].sum(dim=1) > 0                    # (B,) bool
            if not triggered.any():
                continue
            # Outer product: rows = triggered samples, cols = forbidden ops.
            b_col = torch.zeros(N, dtype=torch.bool, device=device)
            b_col[torch.as_tensor(b_idx, device=device)] = True           # (N,)
            forbid = triggered.unsqueeze(1) & b_col.unsqueeze(0)          # (B, N)
            mask = mask & ~forbid
        return mask


class GroupBudgetMask:
    """Total-picks cap per group. Same-class mutex is just `max_select=1`.

    `groups` is a list of dicts:
        {"name": "denoise",
         "ops":  ["denoise", "inf_nlm", "inf_ebf"],
         "max_select": 1}
    """

    def __init__(self, groups: list[dict], canonical_order: Sequence[str]) -> None:
        self._groups: list[tuple[list[int], int]] = []
        for g in groups:
            ops = list(g.get("ops", []) or [])
            max_sel = int(g.get("max_select", 1))
            idx = _indices_for(ops, canonical_order)
            if idx and max_sel >= 0:
                self._groups.append((idx, max_sel))

    def __call__(self, state: PipelineState, canonical_order: Sequence[str]) -> torch.Tensor:
        B = state.batch_size
        N = len(canonical_order)
        device = state.image.device
        mask = torch.ones((B, N), dtype=torch.bool, device=device)
        if not self._groups:
            return mask
        usage = state.op_usage
        for idx, max_sel in self._groups:
            used = usage[:, idx].sum(dim=1)                               # (B,)
            over = used >= max_sel                                        # (B,) bool
            if not over.any():
                continue
            grp_col = torch.zeros(N, dtype=torch.bool, device=device)
            grp_col[torch.as_tensor(idx, device=device)] = True           # (N,)
            forbid = over.unsqueeze(1) & grp_col.unsqueeze(0)             # (B, N)
            mask = mask & ~forbid
        return mask


@dataclass
class ActionMaskPipeline:
    """Aggregates zero or more priors; `__call__` ANDs their masks together."""
    priors: list

    def __call__(self, state: PipelineState, canonical_order: Sequence[str]) -> torch.Tensor:
        B = state.batch_size
        N = len(canonical_order)
        mask = torch.ones((B, N), dtype=torch.bool, device=state.image.device)
        for p in self.priors:
            mask = mask & p(state, canonical_order)
        return mask


def build_from_config(cfg_block: dict, canonical_order: Sequence[str]) -> ActionMaskPipeline:
    """Read a config `action_mask:` block and produce the composed pipeline.

    Any missing / disabled section is skipped; an entirely absent block
    yields an empty pipeline (identity behavior — V2 parity).
    """
    priors: list = []
    if not cfg_block:
        return ActionMaskPipeline(priors=priors)
    nr = cfg_block.get("no_repeat", {}) or {}
    if nr.get("enabled", False):
        priors.append(NoRepeatMask(
            exempt=nr.get("exempt", []) or [],
            canonical_order=canonical_order,
        ))
    order = cfg_block.get("order", {}) or {}
    if order.get("enabled", False):
        priors.append(OrderMask(order.get("rules", []) or [], canonical_order))
    gb = cfg_block.get("group_budget", {}) or {}
    if gb.get("enabled", False):
        priors.append(GroupBudgetMask(gb.get("groups", []) or [], canonical_order))
    return ActionMaskPipeline(priors=priors)


__all__ = [
    "NoRepeatMask", "OrderMask", "GroupBudgetMask",
    "ActionMaskPipeline", "build_from_config",
]
