"""BaseTrainer: shared scaffold for Detection and Human trainers.

Owns the parts that are identical between `engine.trainer.Trainer` (Detection
+ ReplayMemory) and `engine.trainer_human.HumanTrainer` (FiveK + full-rollout):

  - experiment directories, Tee, TensorBoard writer, config copy
  - loading the config yaml (`engine.util.load_config`) + storing it on self
  - constructing the 4 pipeline subsystems that don't vary by task:
      Controller, PipelineExecutor, SearchSpace
  - optimizer + LambdaLR scheduler
  - checkpoint save / resume (with a per-task filename prefix and extra fields)
  - print-block helpers: `_fmt_elapsed`, `_fmt_step`, window accumulators, and
    the reward-breakdown / policy-diagnostics window updater

Task-specific code (data loading, task model, reward class, per-iter body)
stays in the subclass. This is deliberate — the two RL loop shapes differ
(1-step + Replay vs. full T-step rollout), and Plan A did not commit to
unifying them.
"""
from __future__ import annotations

import logging
import math
import os
import shutil
from typing import Optional

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter

from controller.adaptiveisp import AdaptiveISPController
from engine.util import Tee, load_config
from front_isp import build_front_isp_from_cfg
from isp.registry import build_operator
from pipeline import PipelineExecutor
from search import SearchSpace
from search.priors.action_mask import build_from_config as build_action_mask


logger = logging.getLogger(__name__)


class BaseTrainer:
    """Common scaffold; subclasses own task/data/reward/train-loop body."""

    #: banner shown when this trainer starts. Subclasses override.
    banner: str = "-------- BaseTrainer --------"

    #: prefix for periodic ckpt filenames, e.g. "DynamicISP" -> "DynamicISP_iter_100.pth"
    ckpt_prefix: str = "ISP"

    def _setup_experiment(self, args, save_enabled: bool) -> None:
        """Create experiment dirs, Tee, SummaryWriter, and copy the cfg.

        Only performed when save_enabled=True (train/train_val); pure-eval
        callers can skip this and directly load the config.
        """
        if not save_enabled:
            return
        self.base_dir = os.path.join('experiments', args.save_path)
        os.makedirs(self.base_dir, exist_ok=True)
        self.log_dir = os.path.join(self.base_dir, "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.ckpt_dir = os.path.join(self.base_dir, "ckpt")
        os.makedirs(self.ckpt_dir, exist_ok=True)
        self.tee = Tee(os.path.join(self.log_dir, 'log.txt'))
        self.writer = SummaryWriter(self.log_dir)
        self.image_dir = os.path.join(self.base_dir, "images")
        os.makedirs(self.image_dir, exist_ok=True)

        if os.path.exists(args.cfg):
            shutil.copy(args.cfg, os.path.join(self.base_dir, os.path.basename(args.cfg)))
        print(self.banner)

    def _load_cfg(self, args) -> "Dict":  # noqa: F821 (Dict is duck-typed)
        cfg = load_config(args.cfg)
        cfg.filter_runtime_penalty = bool(getattr(args, 'runtime_penalty', False))
        cfg.filter_runtime_penalty_lambda = float(getattr(args, 'runtime_penalty_lambda', 0.01))
        return cfg

    def _build_pipeline_subsystems(self, cfg, device) -> None:
        """Build ops / runtime / search_space / controller / front_isp.

        Sets `self.runtime`, `self.search_space`, `self.controller`, and
        `self.front_isp`. Uses `cfg.operators` for the op list; every other
        cfg field consumed here (`base_channels`, `fc1_size`,
        `feature_extractor_dims`, ...) is identical between Detection and
        Human configs.

        `self.front_isp` is the Configurable Front ISP (RAW → baseline RGB)
        built from `cfg.front_isp` via `build_front_isp_from_cfg`. It is
        ALWAYS a FrontISPBase module — identity when disabled — so callers
        can apply it unconditionally:
            imgs = self.front_isp(imgs).clamp(0.0, 1.0)
        `cfg.front_isp` falls back to legacy `canonical_backbone.enabled`
        (V3-A1 configs keep working unchanged).
        """
        ops = {name: build_operator(name).to(device) for name in cfg.operators}
        self.runtime = PipelineExecutor(ops, cfg.operators)
        # V3-A2: compose action-mask priors from cfg.action_mask (empty when
        # section absent — V2 identity behavior preserved).
        action_mask_pipeline = build_action_mask(
            cfg.get('action_mask', {}) or {}, cfg.operators,
        )
        priors = [action_mask_pipeline] if action_mask_pipeline.priors else None
        self.search_space = SearchSpace(ops, cfg.operators, priors=priors)
        self.front_isp = build_front_isp_from_cfg(cfg).to(device)
        self.controller = AdaptiveISPController(
            ops, cfg.operators,
            obs_hw=int(cfg.get('obs_hw', 64)),
            mid_channels=cfg.base_channels,
            fc1_size=cfg.fc1_size,
            feature_dim=cfg.feature_extractor_dims,
            dropout_keep_prob=cfg.dropout_keep_prob,
            exploration=cfg.exploration,
            max_steps=cfg.test_steps,
            min_rollout_length=int(cfg.get('min_rollout_length', 1)),
        ).to(device)

    def _finalize_cfg_derived_fields(self, args, cfg) -> None:
        """Populate derived fields both trainers set after data init."""
        images_per_epoch = int(cfg.get('images_per_epoch', 1000))
        cfg.max_iter_step = int(args.epochs * images_per_epoch // args.batch_size)
        if cfg.show_img_num > args.batch_size:
            cfg.show_img_num = args.batch_size

        self._grad_clip_norm = float(cfg.get('grad_clip_norm', 1e-5))
        train_cfg = cfg.get('train', {}) or {}
        self._lr_decay = float(train_cfg.get('lr_decay', 0.1))
        self._lr_segments = int(train_cfg.get('lr_segments', 3))

    # -------------------- optimizer + resume + save --------------------

    def _build_optimizer_and_scheduler(self, args, cfg):
        # V3.1 joint_finetune：把 Front ISP 的可训练标定参数加入 optimizer，
        # lr 乘 lr_calib_mul（≪ 1），避免基础颜色空间被任务奖励大幅破坏。
        stage = str((cfg.get('training', {}) or {}).get('stage', 'adaptive_train'))
        lr_calib_mul = float((cfg.get('training', {}) or {}).get('lr_calib_mul', 0.01))
        calib_params = []
        if stage == 'joint_finetune':
            calib_params = [p for p in self.front_isp.parameters() if p.requires_grad]
            if calib_params:
                print(f"joint_finetune: +{len(calib_params)} calibration tensors "
                      f"at lr={args.lr * lr_calib_mul:.2e} (mul={lr_calib_mul})")
        groups = [{'params': self.controller.parameters(), 'lr': args.lr}]
        if calib_params:
            groups.append({'params': calib_params, 'lr': args.lr * lr_calib_mul})
        optim = torch.optim.Adam(groups)
        max_iter_step = int(cfg.max_iter_step)
        lr_decay = self._lr_decay
        segments = self._lr_segments
        lr_lambda = lambda it: lr_decay ** (1.0 * it * segments / max(max_iter_step, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        print(f"init learning rate: {scheduler.get_last_lr()[0]}")
        return optim, scheduler

    def _maybe_resume(self, resume_path: Optional[str]) -> None:
        if not resume_path:
            return
        print(f"Resume from {resume_path}")
        ckpt = torch.load(resume_path, weights_only=False)
        if 'controller_model' in ckpt:
            self.controller.load_state_dict(ckpt['controller_model'])
        else:
            logger.warning(
                "Resume ckpt is legacy format; Controller has different "
                "architecture — starting fresh."
            )
        # V3.1: resume the calibration state when present.
        if 'front_isp' in ckpt:
            from front_isp.calibration import CalibratedFrontISP
            if isinstance(getattr(self, 'front_isp', None), CalibratedFrontISP):
                try:
                    self.front_isp.load_state_dict(ckpt['front_isp'])
                    print("resumed calibration state from ckpt")
                except RuntimeError as exc:
                    logger.warning(f"calibration state mismatch: {exc}")

    def _save_ckpt(self, iter_idx: int, optim, extra: Optional[dict] = None) -> None:
        self.controller.eval()
        ckpt = {
            'iter': iter_idx,
            'controller_model': self.controller.state_dict(),
            'optimizer': optim.state_dict(),
            'operators': list(self.cfg.operators),
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, os.path.join(self.ckpt_dir, f'{self.ckpt_prefix}_iter_{iter_idx}.pth'))

    # -------------------- print-block helpers --------------------

    @staticmethod
    def _fmt_elapsed(seconds: float) -> str:
        seconds = max(0.0, float(seconds))
        m, s = divmod(int(seconds + 0.5), 60)
        h, m = divmod(m, 60)
        return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    @staticmethod
    def _fmt_step(op_name: str, phys: np.ndarray) -> str:
        """Render one op-name + physical-parameter pair for the example line."""
        first = float(phys[0]) if phys.size else 0.0
        if op_name.startswith("n_"):
            return f"{op_name}(α={first:.2f})"
        if phys.size > 1:
            return f"{op_name}({first:.2f},+{phys.size - 1})"
        return f"{op_name}({first:.2f})"

    @staticmethod
    def _make_window() -> dict:
        """Windowed accumulators for reward-breakdown + policy diagnostics.

        `n_steps` counts per-step samples in-window (both trainers accumulate
        one entry per per-step reward compute — Detection's per-iter step and
        Human's per-rollout step both fit).
        """
        return {
            'n_iters': 0, 'n_steps': 0,
            'task': 0.0, 'ent_pen': 0.0, 'use': 0.0, 'estop': 0.0,
            'ovfl': 0.0, 'stop_b': 0.0, 'runt': 0.0,
            'pol_ent': 0.0, 'argmax_hits': 0, 'argmax_seen': 0,
            'n_stop': 0, 'n_stop_learned': 0, 'n_stop_timelimit': 0,
        }

    @staticmethod
    def _reset_window(win: dict) -> None:
        for k in win:
            win[k] = 0 if isinstance(win[k], int) else 0.0

    def _accumulate_window(
        self,
        win: dict,
        breakdown,
        entropy: torch.Tensor,
        ctrl_out,
        state_after,
        n_ops: int,
        max_steps: int,
    ) -> None:
        """Add one step's reward-breakdown + policy stats into `win`.

        `breakdown` is a `RewardBreakdown`; `ctrl_out` is a controller output
        with `.logits`, `.action.op_indices`, `.action.is_stop`; `state_after`
        is the post-step `PipelineState`.
        """
        win['n_steps'] += 1
        win['task'] += float(breakdown.task_delta.mean().item())
        win['ent_pen'] += float(breakdown.entropy_penalty.mean().item())
        win['use'] += float(breakdown.usage_penalty.mean().item())
        win['estop'] += float(breakdown.early_stop_penalty.mean().item())
        win['ovfl'] += float(breakdown.overflow_penalty.mean().item())
        win['runt'] += float(breakdown.runtime_penalty.mean().item())
        if getattr(breakdown, 'stop_bonus', None) is not None:
            win['stop_b'] += float(breakdown.stop_bonus.mean().item())
        win['pol_ent'] += float(entropy.mean().item())

        with torch.no_grad():
            argmax_idx = ctrl_out.logits.argmax(dim=-1)
            sampled_idx = torch.where(
                ctrl_out.action.is_stop,
                torch.full_like(ctrl_out.action.op_indices, n_ops),
                ctrl_out.action.op_indices,
            )
            win['argmax_hits'] += int((argmax_idx == sampled_idx).sum().item())
            win['argmax_seen'] += int(sampled_idx.numel())

        is_stop_np = ctrl_out.action.is_stop.detach().cpu().numpy()
        n_stop = int(is_stop_np.sum())
        is_last_step = int((state_after.step == max_steps).sum().item())
        n_stop_tl = min(n_stop, is_last_step)
        win['n_stop'] += n_stop
        win['n_stop_timelimit'] += n_stop_tl
        win['n_stop_learned'] += (n_stop - n_stop_tl)


__all__ = ["BaseTrainer"]
