"""Wider smoke test for V2-AI neural-op integration.

Runs everything the training loop would, minus real data + YOLO backward.
Catches classes of bugs the narrow smoke (`smoke_test.py`) misses:

  1. Env sanity — every Python package Samsung code needs
     (colour_demosaicing, cv2, rawpy, ...) actually importable in the
     current interpreter.
  2. Checkpoint presence — all Samsung .pth files exist on disk.
  3. Full backend import chain — every backend loaded via
     `_samsung_import` after `tasks` (yolov3) is already imported.
  4. Every neural op fires through `PipelineExecutor.step` (not just
     `op.apply`), so the runtime dispatch path is exercised.
  5. Multi-step rollout — `Controller.act` → `Executor.step` → next
     `Controller.act` chain, with state carried forward.

Run:
    conda activate adaptiveisp
    python -m isp.learned.samsung_modular.tests.smoke_test_wide
"""
from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import torch


REQUIRED_PACKAGES = [
    "colour_demosaicing",
    "cv2",
    "rawpy",
    "torch",
    "torchvision",
    "numpy",
    "yaml",
    "tqdm",
    "tensorboard",
]


def _check_env() -> list[str]:
    print("=" * 60)
    print("env")
    print("=" * 60)
    print(f"  python:  {sys.executable}")
    print(f"  version: {sys.version.split()[0]}")
    print(f"  cwd:     {os.getcwd()}")
    print(f"  cuda:    {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  device:  {torch.cuda.get_device_name(0)}")

    missing = []
    print("\n  packages:")
    for pkg in REQUIRED_PACKAGES:
        try:
            m = importlib.import_module(pkg)
            ver = getattr(m, "__version__", "?")
            print(f"    ok    {pkg}={ver}")
        except ImportError as exc:
            print(f"    MISS  {pkg} — {exc}")
            missing.append(pkg)
    return missing


def _check_checkpoints() -> list[str]:
    print("\n" + "=" * 60)
    print("checkpoints")
    print("=" * 60)
    from isp.learned.samsung_modular import backend

    paths = {
        "denoiser": backend.DEFAULT_DENOISE_MODEL,
        "detail": backend.DEFAULT_ENHANCE_MODEL,
        "photofinishing": backend.DEFAULT_PS_MODEL,
        "awb": backend.DEFAULT_AWB_MODEL,
    }
    missing = []
    for name, p in paths.items():
        exists = Path(p).exists()
        size_mb = Path(p).stat().st_size / 1e6 if exists else 0.0
        tag = "ok  " if exists else "MISS"
        print(f"  {tag} {name}: {p} ({size_mb:.1f} MB)")
        if not exists:
            missing.append(str(p))
    return missing


def _run_stack() -> None:
    print("\n" + "=" * 60)
    print("integration: import tasks (yolov3), then Samsung backends")
    print("=" * 60)

    # 1. Simulate the trainer: import tasks FIRST so yolov3's `utils/` is
    #    registered in sys.modules before Samsung imports.
    import tasks  # noqa: F401

    import utils as yolo_utils_before
    print(f"  yolov3 utils: {yolo_utils_before.__file__}")

    # 2. Now populate isp registry (side-effect imports register neural ops)
    import isp  # noqa: F401
    from isp.registry import CANONICAL_ORDER, OPERATORS, build_operator
    from isp.learned.samsung_modular import backend

    print(f"  registered operators ({len(OPERATORS)}): {sorted(OPERATORS)}")

    # 3. Force-load every backend. This trips the full Samsung import chain
    #    (utils.constants, utils.img_utils, colour_demosaicing, ...).
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"  loading backends on {device} ...")
    _ = backend.get_denoiser(device)
    _ = backend.get_detail(device)
    _ = backend.get_photofinishing(device)
    _ = backend.get_awb(device)
    print("  all 4 backends loaded")

    # 4. yolov3 utils must still resolve after Samsung imports swapped
    #    sys.modules around.
    import utils as yolo_utils_after
    assert yolo_utils_after.__file__ == yolo_utils_before.__file__, (
        "yolov3 utils shadowed by Samsung"
    )
    print(f"  yolov3 utils after swaps: {yolo_utils_after.__file__} (unchanged)")

    # 5. Executor + SearchSpace + Controller — same objects the Trainer uses.
    from pipeline import PipelineExecutor
    from pipeline.action import ISPAction
    from search import SearchSpace
    from controller.adaptiveisp import AdaptiveISPController, AdaptiveISPReward
    from tasks.base import TaskMetrics

    ops = {n: build_operator(n).to(device) for n in CANONICAL_ORDER}
    exe = PipelineExecutor(ops, CANONICAL_ORDER)
    ss = SearchSpace(ops, CANONICAL_ORDER)
    n_ops = len(CANONICAL_ORDER)
    print(f"  n_ops={n_ops}")

    # 6. Force every op to fire via Executor.step, batch by batch, so every
    #    op index is dispatched. We DON'T need the Controller here — just
    #    hand-craft an action. Use random raw features (mimics the
    #    Controller's untrained-NN output) so pathological zero-input regimes
    #    (e.g. `ccm` regressor → all-zero matrix → NaN after luminance norm)
    #    don't fire — those are V1 numerical corners real training never hits.
    print("\n" + "=" * 60)
    print(f"forced dispatch: fire every one of {n_ops} ops through Executor.step")
    print("=" * 60)
    B = n_ops
    img = torch.rand(B, 3, 128, 128, device=device) * 0.5 + 0.25
    state = exe.initial_state(img)
    max_dim = max(ops[n].spec.dim for n in CANONICAL_ORDER)
    torch.manual_seed(0)

    for round_idx in range(2):
        op_indices = torch.tensor([(b + round_idx) % n_ops for b in range(B)],
                                  device=device, dtype=torch.long)
        params = torch.zeros(B, max_dim, device=device)
        for b in range(B):
            name = CANONICAL_ORDER[int(op_indices[b].item())]
            op = ops[name]
            raw = torch.randn(1, op.spec.dim, device=device) * 2.0
            physical = op.spec.regressor(raw)
            params[b, :op.spec.dim] = physical.view(-1)
        action = ISPAction(
            op_indices=op_indices,
            params=params,
            is_stop=torch.zeros(B, dtype=torch.bool, device=device),
        )
        state = exe.step(state, action)
        assert torch.isfinite(state.image).all(), (
            f"NaN/Inf at round {round_idx}, ops: "
            f"{[CANONICAL_ORDER[i.item()] for i in op_indices]}"
        )
        names_fired = [CANONICAL_ORDER[i.item()] for i in op_indices]
        print(f"  round {round_idx}: dispatched {names_fired}  → image finite")

    # 7. Full Controller rollout (no YOLO forward, just Controller + Executor).
    print("\n" + "=" * 60)
    print("controller rollout (5 steps, batch=4, untrained policy)")
    print("=" * 60)
    ctrl = AdaptiveISPController(
        ops, CANONICAL_ORDER, obs_hw=64, mid_channels=32, fc1_size=128,
        feature_dim=4096, dropout_keep_prob=0.5, exploration=0.05, max_steps=5,
    ).to(device)
    ctrl.train()

    img4 = torch.rand(4, 3, 256, 256, device=device) * 0.5 + 0.25
    state = exe.initial_state(img4)
    for t in range(5):
        out = ctrl.act(state, ss.valid_actions(state))
        state = exe.step(state, out.action)
        picks = [CANONICAL_ORDER[i.item()] for i in out.action.op_indices]
        assert torch.isfinite(state.image).all()
        assert torch.isfinite(out.value).all()
        assert torch.isfinite(out.log_prob).all()
        print(f"  step {t}: picks={picks}")

    # 8. Reward path.
    print("\n" + "=" * 60)
    print("reward compute (fake task metrics)")
    print("=" * 60)
    reward_fn = AdaptiveISPReward(
        n_ops=n_ops, max_steps=5, critic_logit_multiplier=100.0, all_reward=1.0,
        filter_usage_penalty=1.0, exploration_penalty=0.05, early_stop_penalty=1.0,
        runtime_penalty_enabled=False,
    )
    state_before = exe.initial_state(img4)
    out = ctrl.act(state_before, ss.valid_actions(state_before))
    state_after = exe.step(state_before, out.action)
    mb = TaskMetrics(values={"detect_loss": torch.rand(4, 1, device=device) * 0.1 + 0.05})
    ma = TaskMetrics(values={"detect_loss": torch.rand(4, 1, device=device) * 0.1 + 0.03})
    r, bd = reward_fn.compute(mb, ma, state_before, out.action, state_after,
                               entropy=out.entropy, progress=0.1)
    assert r.shape == (4, 1) and torch.isfinite(r).all()
    print(f"  reward: {r.squeeze().tolist()}")
    print(f"  breakdown finite: task_delta={bd.task_delta.mean().item():.4f} "
          f"entropy_penalty={bd.entropy_penalty.mean().item():.4f} "
          f"usage_penalty={bd.usage_penalty.mean().item():.4f}")

    # 9. Backward.
    print("\n" + "=" * 60)
    print("policy backward")
    print("=" * 60)
    q = r + 0.9 * ctrl.value_net(state_after)
    advantage = q.detach() - out.value
    loss = (advantage ** 2).mean() + (out.log_prob * (-advantage.detach())).mean()
    loss.backward()
    n_grad = sum(1 for p in ctrl.parameters() if p.grad is not None and p.grad.abs().sum() > 0)
    n_total = sum(1 for p in ctrl.parameters())
    print(f"  loss={loss.item():.4f}; params w/ nonzero grad: {n_grad}/{n_total}")


def main() -> int:
    missing_pkgs = _check_env()
    missing_ckpts = _check_checkpoints()
    if missing_pkgs or missing_ckpts:
        print("\n" + "=" * 60)
        print("BLOCKING FAILURES")
        print("=" * 60)
        if missing_pkgs:
            print(f"  missing packages: {missing_pkgs}")
            print(f"  fix:  pip install {' '.join(missing_pkgs).replace('cv2','opencv-python')}")
        if missing_ckpts:
            print(f"  missing checkpoints: {missing_ckpts}")
        return 1

    try:
        _run_stack()
    except Exception as exc:  # noqa: BLE001
        print("\n" + "=" * 60)
        print(f"WIDER SMOKE FAILED: {type(exc).__name__}: {exc}")
        print("=" * 60)
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("wider smoke: PASS")
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main())
