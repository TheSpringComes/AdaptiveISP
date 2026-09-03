"""smoke: PipelineExecutor state transitions on a dummy batch.

Checks: initial_state fresh, step advances image + step + op_usage,
stopped samples are frozen, replay adapters round-trip.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

import tasks  # noqa: F401
from isp.registry import CANONICAL_ORDER, build_operator
from pipeline import (
    ISPAction, PipelineExecutor, PipelineState,
    pipeline_state_from_replay, pipeline_state_to_replay,
)


def test_pipeline_step() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ops = {n: build_operator(n).to(device) for n in CANONICAL_ORDER}
    exe = PipelineExecutor(ops, CANONICAL_ORDER)
    n_ops = exe.n_ops

    B = 4
    img = torch.rand(B, 3, 32, 32, device=device) * 0.5 + 0.25
    state0 = exe.initial_state(img)
    assert state0.step.tolist() == [0, 0, 0, 0]
    assert not state0.stopped.any()
    assert not state0.op_usage.any()

    # Sample 3 stops; the rest pick a distinct op each.
    op_idx = torch.tensor([0, 1, 2, -1], device=device)
    max_dim = max(op.spec.dim for op in ops.values())
    params = torch.zeros(B, max_dim, device=device)
    params[0, 0] = 0.5
    params[1, 0] = 1.2
    params[2, :9] = torch.tensor([1, 0, 0, 0, 1, 0, 0, 0, 1], dtype=torch.float32, device=device)
    action = ISPAction(op_indices=op_idx, params=params, is_stop=(op_idx == -1))

    state1 = exe.step(state0, action)
    # Sample 3 (stopped) image unchanged; step still bumped once by executor.step.
    assert torch.equal(state1.image[3], state0.image[3]), "stopped sample image was modified"
    assert state1.stopped[3].item(), "stop action did not flip stopped=True"
    for i in range(3):
        assert state1.op_usage[i, i].item(), f"op {i} not marked as used"

    # Second step for already-stopped sample 3 should not advance step.
    action2 = ISPAction(
        op_indices=torch.tensor([5, 5, 5, 5], device=device),
        params=torch.zeros(B, max_dim, device=device),
        is_stop=torch.zeros(B, dtype=torch.bool, device=device),
    )
    state2 = exe.step(state1, action2)
    assert state2.step[3].item() == 1, f"stopped sample step advanced: {state2.step[3]}"

    # Replay-format round-trip.
    flat = pipeline_state_to_replay(state2)
    state2b = pipeline_state_from_replay(state2.image, flat, n_ops=n_ops)
    assert torch.equal(state2b.step, state2.step)
    assert torch.equal(state2b.stopped, state2.stopped)
    assert torch.equal(state2b.op_usage, state2.op_usage)


if __name__ == "__main__":
    test_pipeline_step()
    print("smoke/test_pipeline: PASS")
