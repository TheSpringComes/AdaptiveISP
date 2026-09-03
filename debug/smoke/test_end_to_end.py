"""smoke: full rollout on dummy data — Controller → Executor → Reward.

Skips the YOLO forward (task metrics are faked with plausible tensors)
so this runs in a few seconds without loading pretrained weights or
real data. Confirms the coupling between the five subsystems.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

import tasks  # noqa: F401
from controller.adaptiveisp import AdaptiveISPController, AdaptiveISPReward
from isp.registry import CANONICAL_ORDER, build_operator
from pipeline import PipelineExecutor
from search import SearchSpace
from tasks.base import TaskMetrics


def test_full_rollout() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ops = {n: build_operator(n).to(device) for n in CANONICAL_ORDER}
    exe = PipelineExecutor(ops, CANONICAL_ORDER)
    ss = SearchSpace(ops, CANONICAL_ORDER)
    ctrl = AdaptiveISPController(
        ops, CANONICAL_ORDER, obs_hw=64, mid_channels=32, fc1_size=128,
        feature_dim=4096, dropout_keep_prob=0.5, exploration=0.05, max_steps=5,
    ).to(device)
    reward_fn = AdaptiveISPReward(
        n_ops=len(CANONICAL_ORDER), max_steps=5,
        critic_logit_multiplier=100.0, all_reward=1.0,
        filter_usage_penalty=1.0, exploration_penalty=0.05,
        early_stop_penalty=1.0, runtime_penalty_enabled=False,
    )

    B = 2
    img = torch.rand(B, 3, 128, 128, device=device) * 0.5 + 0.25
    state = exe.initial_state(img)

    ctrl.train()
    for t in range(3):
        out = ctrl.act(state, ss.valid_actions(state))
        new_state = exe.step(state, out.action)
        # Fake task metrics: some plausible per-sample loss.
        mb = TaskMetrics(values={"detect_loss": torch.rand(B, 1, device=device) * 0.1 + 0.05})
        ma = TaskMetrics(values={"detect_loss": torch.rand(B, 1, device=device) * 0.1 + 0.03})
        r, bd = reward_fn.compute(
            mb, ma, state, out.action, new_state,
            entropy=out.entropy, progress=t / 5.0,
        )
        assert r.shape == (B, 1), f"reward shape {r.shape} != ({B}, 1)"
        assert torch.isfinite(r).all(), f"non-finite reward: {r}"
        assert torch.isfinite(bd.total).all()
        state = new_state
    assert state.step[0].item() == 3
    # After 3 steps, none should be stopped yet (max_steps=5).
    assert not state.stopped.any()


if __name__ == "__main__":
    test_full_rollout()
    print("smoke/test_end_to_end: PASS")
