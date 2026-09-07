"""smoke: HumanQuality single-iter end-to-end.

Runs one iteration of `HumanTrainer.train` semantics on 2 FiveK samples,
verifies loss / reward / gradient / logged quality delta all finite.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

# side effects: registry populate + tasks import
import isp  # noqa: F401
import tasks  # noqa: F401

from controller.adaptiveisp import AdaptiveISPController
from controller.adaptiveisp.human_reward import HumanReward
from isp.registry import CANONICAL_ORDER, build_operator
from pipeline import PipelineExecutor
from search import SearchSpace
from tasks.human_quality import FiveKDataset, HumanQualityTask, collate_fivek
from tasks.human_quality.metrics import quality_score


def test_human_quality_smoke() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # dataset (val split is 100, cheap)
    ds = FiveKDataset(
        list_file="/home/jing/datasets/fivek/val_expert_c.txt",
        cache_dir="/home/jing/datasets/fivek/cache_expert_c",
        imgsz=128,   # small for speed
    )
    assert len(ds) > 0, "FiveKDataset is empty"
    img0, tgt0 = ds[0]
    print(f"sample 0  image {tuple(img0.shape)} in [{img0.min():.3f},{img0.max():.3f}]  "
          f"target {tuple(tgt0.shape)} in [{tgt0.min():.3f},{tgt0.max():.3f}]")

    batch = collate_fivek([ds[i] for i in range(2)])
    imgs, targets = batch
    imgs = imgs.to(device)
    targets = targets.to(device)

    # metrics
    q0, parts0 = quality_score(imgs, targets, lambda_ssim=1.0, lambda_lpips=1.0)
    assert torch.isfinite(q0).all()
    print(f"Q0: {q0.squeeze().tolist()}   SSIM: {parts0['ssim'].squeeze().tolist()}   "
          f"LPIPS: {parts0['lpips'].squeeze().tolist()}")

    # subsystems
    ops = {n: build_operator(n).to(device) for n in CANONICAL_ORDER}
    n_ops = len(CANONICAL_ORDER)
    exe = PipelineExecutor(ops, CANONICAL_ORDER)
    ss = SearchSpace(ops, CANONICAL_ORDER)
    T = 5
    ctrl = AdaptiveISPController(
        ops, CANONICAL_ORDER, obs_hw=64, mid_channels=32, fc1_size=128,
        feature_dim=4096, dropout_keep_prob=0.5, exploration=0.05, max_steps=T,
    ).to(device)
    ctrl.train()

    task = HumanQualityTask(lambda_ssim=1.0, lambda_lpips=1.0)
    reward_fn = HumanReward(
        n_ops=n_ops, max_steps=T,
        lambda_ssim=1.0, lambda_lpips=1.0,
        critic_logit_multiplier=100.0, all_reward=1.0,
        filter_usage_penalty=1.0, exploration_penalty=0.05, early_stop_penalty=1.0,
        runtime_penalty_enabled=False,
    )

    optim = torch.optim.Adam(ctrl.parameters(), lr=3e-5)
    optim.zero_grad()

    state = exe.initial_state(imgs)
    step_losses = []
    op_names_sample0 = []
    r_task_deltas = []
    for t in range(T):
        out = ctrl.act(state, ss.valid_actions(state))
        new_state = exe.step(state, out.action)
        picked_0 = CANONICAL_ORDER[int(out.action.op_indices[0].item())]
        op_names_sample0.append(picked_0)

        r, bd, q_parts = reward_fn.compute(
            image_initial=imgs, target=targets,
            state_before=state, action=out.action, state_after=new_state,
            entropy=out.entropy, progress=t / T, q_initial=q0,
        )
        r_task_deltas.append(bd.task_delta.mean().item())

        old_v = out.value
        new_v = ctrl.value_net(new_state)
        stopped_after = new_state.stopped.float().unsqueeze(-1)
        new_v = new_v * (1.0 - stopped_after)
        q_target = r + (1.0 - stopped_after) * 1.0 * new_v
        advantage = q_target.detach() - old_v
        v_loss = (advantage ** 2).mean()
        a_loss = (-q_target * 1.0).mean() + (out.log_prob * (-advantage.detach())).mean()
        step_losses.append(v_loss + a_loss)

        state = new_state
        if state.stopped.all():
            break

    total_loss = torch.stack(step_losses).sum()
    total_loss.backward()
    torch.nn.utils.clip_grad_norm_(ctrl.parameters(), 1e-5)
    optim.step()

    # asserts
    q_final, parts_final = quality_score(state.image, targets)
    q_delta = (q_final - q0).mean().item()
    assert torch.isfinite(total_loss)
    assert torch.isfinite(q_final).all()
    n_grad = sum(1 for p in ctrl.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in ctrl.parameters())

    print(f"\nrollout picks (sample 0): {' → '.join(op_names_sample0)}")
    print(f"per-step task_delta.mean(): {[f'{x:+.4f}' for x in r_task_deltas]}")
    print(f"Q_final: {q_final.squeeze().tolist()}   ΔQ mean: {q_delta:+.4f}")
    print(f"total_loss: {total_loss.item():.4f}")
    print(f"params with nonzero grad: {n_grad}/{n_total}")

    # confirm task_delta zero except on terminal step
    assert abs(r_task_deltas[-1]) > 0, "terminal task_delta was zero"
    for x in r_task_deltas[:-1]:
        assert abs(x) < 1e-8, f"intermediate task_delta nonzero: {x}"

    print("\nsmoke/test_human_quality: PASS")


if __name__ == "__main__":
    test_human_quality_smoke()
