"""smoke: Controller.act produces valid ISPAction; gradient flows.

Skips the YOLO backbone and Reward — those are exercised in
smoke/test_end_to_end.py. Here we only need a Controller.act call plus
a manual policy+value loss.backward() to confirm gradients reach the
network heads.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

import tasks  # noqa: F401
from controller.adaptiveisp import AdaptiveISPController
from isp.registry import CANONICAL_ORDER, build_operator
from pipeline import PipelineExecutor
from search import SearchSpace


def test_controller_act() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ops = {n: build_operator(n).to(device) for n in CANONICAL_ORDER}
    exe = PipelineExecutor(ops, CANONICAL_ORDER)
    ss = SearchSpace(ops, CANONICAL_ORDER)
    ctrl = AdaptiveISPController(
        ops, CANONICAL_ORDER,
        obs_hw=64, mid_channels=32, fc1_size=128, feature_dim=4096,
        dropout_keep_prob=0.5, exploration=0.05, max_steps=5,
    ).to(device)

    B = 4
    img = torch.rand(B, 3, 512, 512, device=device) * 0.5 + 0.25
    state = exe.initial_state(img)

    ctrl.train()
    out = ctrl.act(state, ss.valid_actions(state))
    # Shapes.
    assert out.action.op_indices.shape == (B,)
    assert out.action.params.shape[0] == B
    assert out.logits.shape == (B, ctrl.n_ops)
    assert out.value.shape == (B, 1)
    assert out.log_prob.shape == (B, 1)
    assert out.entropy.shape == (B, 1)
    # Entropy in [0, log(N_ops)].
    import math
    assert 0.0 <= out.entropy.min().item() <= math.log(ctrl.n_ops) + 1e-4

    # Gradient flow: fake advantage, backward, confirm select_head + value_net grads populated.
    adv = torch.randn(B, device=device)
    loss = (-out.log_prob.squeeze(-1) * adv).mean() + (out.value.squeeze(-1) ** 2).mean()
    loss.backward()
    for group in ("select_head", "select_features", "value_net"):
        grads = [p.grad for name, p in ctrl.named_parameters() if group in name and p.grad is not None]
        assert grads, f"{group}: no grad-bearing params found"
        norm = sum(g.abs().sum().item() for g in grads)
        assert norm > 0, f"{group}: total grad norm is zero"

    # Eval mode: argmax selection deterministic given fixed state.
    ctrl.eval()
    with torch.no_grad():
        out2 = ctrl.act(state, ss.valid_actions(state))
        out2b = ctrl.act(state, ss.valid_actions(state))
    assert torch.equal(out2.action.op_indices, out2b.action.op_indices), \
        "eval mode should be deterministic (argmax), got different op_indices"


if __name__ == "__main__":
    test_controller_act()
    print("smoke/test_controller: PASS")
