"""Trainer: the AdaptiveISP training loop.

Assembles the four subsystems (isp / search / controller / pipeline / tasks)
around the replay-memory-based 1-step TD training used by the original
AdaptiveISP paper. `Trainer` corresponds to what was `DynamicISP` in the
pre-refactor tree.

CLI entry lives in `tools/train.py`; this file is pure library code.
"""
from __future__ import annotations

import datetime
import logging
import os
import shutil
from typing import Optional

import cv2
import numpy as np
import torch
import yaml
from torch.utils.tensorboard import SummaryWriter

# `tasks.detection.implementations.yolov3` is our sole boundary to the vendored
# yolov3 code; the trainer must not import yolov3.* directly.
from tasks.base import TaskMetrics                                # noqa: F401
from tasks.detection.implementations.yolov3 import YOLOv3Detection

from controller.adaptiveisp import AdaptiveISPController, AdaptiveISPReward
from isp.registry import build_operator
from pipeline import PipelineExecutor, pipeline_state_from_replay, pipeline_state_to_replay
from search import SearchSpace
from tasks.detection.replay import ReplayMemory, create_input_tensor
from engine.util import Tee


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'
WORLD_SIZE = 1


class Trainer:
    """Assembles subsystems + runs the training / periodic-save loop."""

    def __init__(self, args, task: str = "train_val") -> None:
        train = task in ("train", "train_val")
        val = task == "train_val"
        if train:
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
            print("Training begin....")
            print("------- V1 Refactor: Controller + PipelineExecutor + Task + Reward ---------")

        cfg = _load_config(args.cfg)
        cfg.filter_runtime_penalty = args.runtime_penalty
        cfg.filter_runtime_penalty_lambda = args.runtime_penalty_lambda

        self.device = torch.device('cuda')

        # Hyperparameters
        hyp = args.hyp
        if isinstance(hyp, str):
            with open(hyp, errors='ignore') as f:
                hyp = yaml.safe_load(f)
        LOGGER = logger
        LOGGER.info('hyperparameters: ' + ', '.join(f'{k}={v}' for k, v in hyp.items()))
        args.hyp = hyp.copy()
        data_dict = YOLOv3Detection.parse_data_cfg(args.data_cfg)
        nc = int(data_dict['nc'])

        # Downstream task
        self.task_model = YOLOv3Detection(
            weights=args.weights, yolo_cfg=args.yolo_cfg, hyp=hyp,
            nc=nc, imgsz=args.imgsz, device=self.device,
            detect_loss_weight=cfg.detect_loss_weight,
        )
        gs = self.task_model.gs
        args.imgsz = self.task_model.align_imgsz(args.imgsz)

        # Data loaders (ReplayMemory holds partial-trajectory (image, state) pairs)
        train_path, val_path = data_dict['train'], data_dict['val']
        if task == "test":
            val_path = data_dict['test']
        self.train_loader = ReplayMemory(cfg, train, train_path, args.imgsz, args.batch_size, gs,
                                          single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0,
                                          rect=False, image_weights=False, prefix='train: ', limit=-1,
                                          add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range,
                                          noise_level=args.noise_level, use_linear=args.use_linear)
        if val:
            self.val_loader = ReplayMemory(cfg, val, val_path, args.imgsz, args.batch_size, gs,
                                            single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0,
                                            rect=False, image_weights=False, prefix='val: ', limit=-1,
                                            add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range,
                                            noise_level=args.noise_level, use_linear=args.use_linear)
            self.val_loader = self.val_loader.get_feed_dict_and_states(8)

        self.task_model.attach_class_weights(self.train_loader.dataset.labels, nc)
        self.task_model.attach_names(data_dict['names'])
        self.data_dict = data_dict

        # Four subsystems
        ops = {name: build_operator(name).to(self.device) for name in cfg.operators}
        self.runtime = PipelineExecutor(ops, cfg.operators)
        self.search_space = SearchSpace(ops, cfg.operators)
        self.controller = AdaptiveISPController(
            ops, cfg.operators,
            obs_hw=64,
            mid_channels=cfg.base_channels,
            fc1_size=cfg.fc1_size,
            feature_dim=cfg.feature_extractor_dims,
            dropout_keep_prob=cfg.dropout_keep_prob,
            exploration=cfg.exploration,
            max_steps=cfg.test_steps,
        ).to(self.device)
        self.reward_fn = AdaptiveISPReward(
            n_ops=len(cfg.operators),
            max_steps=cfg.test_steps,
            critic_logit_multiplier=cfg.critic_logit_multiplier,
            all_reward=cfg.all_reward,
            filter_usage_penalty=cfg.filter_usage_penalty,
            exploration_penalty=cfg.exploration_penalty,
            early_stop_penalty=cfg.early_stop_penalty,
            runtime_penalty_enabled=cfg.filter_runtime_penalty,
            runtime_penalty_lambda=cfg.filter_runtime_penalty_lambda,
            runtime_costs=cfg.filters_runtime,
            use_penalty=cfg.use_penalty,
            stop_bonus_scale=float(cfg.get('stop_bonus_scale', 0.0)),
        )

        print("Controller: ", self.controller)
        n_params = sum(p.numel() for p in self.controller.parameters())
        print(f"Number of Controller parameters: {n_params / 1e6:.2f}M")

        self.args = args
        images_per_epoch = int(cfg.get('images_per_epoch', 1000))
        cfg.max_iter_step = int(self.args.epochs * images_per_epoch // args.batch_size)
        if cfg.show_img_num > args.batch_size:
            cfg.show_img_num = args.batch_size

        self.gs = gs
        self.hyp = hyp
        self.val_path = val_path
        self.filter_name = list(cfg.operators)

        print("----------------- args ------------------")
        for k, v in vars(args).items():
            print(k, ":", v)
        print("---------------- config ------------------")
        for k, v in cfg.items():
            print(k, ":", v)
        self.cfg = cfg

        train_cfg = cfg.get('train', {}) or {}
        self.max_bri = float(train_cfg.get('bright_hi', 0.9))
        self._bright_lo = float(train_cfg.get('bright_lo', 0.01))
        self._grad_clip_norm = float(cfg.get('grad_clip_norm', 1e-5))
        self._lr_decay = float(train_cfg.get('lr_decay', 0.1))
        self._lr_segments = int(train_cfg.get('lr_segments', 3))

        # Fixed canary sample: drawn once at init, reused every print. Keeping
        # the input constant across iters lets `example / canary` show how the
        # Controller's decisions evolve on the SAME image over training — the
        # cleanest RL-visualization signal.
        self._canary_img = self._grab_dataset_image()

    def _grab_dataset_image(self) -> Optional[torch.Tensor]:
        """Pull one raw sample from the dataset, return `(1, 3, H, W)` on device.

        `dataset.get_next_batch` returns either torch tensors (Normalize path)
        or numpy arrays (RAW path) depending on the loader class — normalize
        both here.
        """
        try:
            im_list, _, _, _ = self.train_loader.dataset.get_next_batch(1)
        except Exception as exc:
            print(f"dataset image capture failed: {exc!r}")
            return None
        im = im_list[0]
        if isinstance(im, np.ndarray):
            im = torch.from_numpy(im)
        return im.unsqueeze(0).to(self.device).float()

    # ------------------------- shadow rollout helper -------------------------

    def _shadow_rollout(self, single_img: torch.Tensor) -> tuple[list[str], int]:
        """Run one T-step eval-argmax rollout on a `(1,3,H,W)` image.
        Returns (pretty seq incl. optional final "STOP", op count excluding STOP).
        """
        def _fmt_step(op_name: str, phys: np.ndarray) -> str:
            first = float(phys[0]) if phys.size else 0.0
            if op_name.startswith("n_"):
                return f"{op_name}(α={first:.2f})"
            if phys.size > 1:
                return f"{op_name}({first:.2f},+{phys.size - 1})"
            return f"{op_name}({first:.2f})"

        T_max = int(self.cfg.test_steps)
        self.controller.eval()
        with torch.no_grad():
            state = self.runtime.initial_state(single_img.clone())
            seq: list[str] = []
            for _t in range(T_max):
                c = self.search_space.valid_actions(state)
                o = self.controller.act(state, c)
                if o.action.is_stop[0].item() and not state.stopped[0].item():
                    seq.append("STOP")
                    break
                idx = int(o.action.op_indices[0].item())
                name = self.cfg.operators[idx]
                dim = self.runtime.operators[name].spec.dim
                phys = o.action.params[0, :dim].detach().cpu().numpy()
                seq.append(_fmt_step(name, phys))
                state = self.runtime.step(state, o.action)
                if state.stopped[0].item():
                    break
        self.controller.train()
        path_len = sum(1 for s in seq if s != "STOP")
        return seq, path_len

    def train(self) -> None:
        if self.args.resume is not None:
            print(f"Resume from {self.args.resume}")
            ckpt = torch.load(self.args.resume, weights_only=False)
            if 'controller_model' in ckpt:
                self.controller.load_state_dict(ckpt['controller_model'])
            else:
                logger.warning(
                    "Resume ckpt is legacy format; Controller has different "
                    "architecture — starting fresh."
                )

        optim = torch.optim.Adam(self.controller.parameters(), lr=self.args.lr)
        lr_decay = self._lr_decay
        segments = self._lr_segments
        max_iter_step = self.cfg.max_iter_step
        lr_lambda = lambda it: lr_decay ** (1.0 * it * segments / max_iter_step)
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        print(f"'init learning rate: {scheduler.get_last_lr()[0]}")

        logger.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val | '
                    f'Using {self.args.workers} dataloader workers | '
                    f'Logging results to {self.args.save_path} | '
                    f'Starting training for {0} epochs...')
        mloss_agent, mloss_value = 0.0, 0.0
        mloss_detect = np.zeros(3, dtype=np.float32)

        n_ops = len(self.cfg.operators)

        # V2-AI: windowed accumulators for the expanded print block. Reset
        # every `print_freq` iters so the numbers reflect the current
        # window, not the drift-heavy from-start EMA. Order:
        #   A — reward breakdown (task/ent/use/estop/ovfl/stop_bonus/runtime)
        #   C — policy/exploration (entropy, argmax-match, STOP split)
        win = {
            'n': 0,               # iters in window
            'task': 0.0, 'ent_pen': 0.0, 'use': 0.0, 'estop': 0.0,
            'ovfl': 0.0, 'stop_b': 0.0, 'runt': 0.0,
            'pol_ent': 0.0, 'argmax_hits': 0, 'argmax_seen': 0,
            'n_stop': 0, 'n_stop_learned': 0, 'n_stop_timelimit': 0,
        }
        log_n_ops_plus1 = float(np.log(n_ops + 1))   # entropy max for select_head

        # V2-AI: per-op cumulative selection counts, plus a windowed counter
        # reset each `print_freq` block. Prints show classical vs neural share
        # so we can watch whether the Controller actually learns to pick the
        # n_* ops.
        op_pick_cum = np.zeros(n_ops, dtype=np.int64)
        op_pick_window = np.zeros(n_ops, dtype=np.int64)
        neural_mask = np.array([n.startswith("n_") for n in self.cfg.operators], dtype=bool)

        # V2-AI: elapsed + ETA tracking. `t_start` covers the whole run;
        # `t_prev_print` gives the rolling iter/s for the last print window,
        # which is a better ETA estimate than the run-average once warmup ends.
        import time as _time
        t_start = _time.perf_counter()
        t_prev_print = t_start
        iter_prev_print = 0
        max_steps_display = int(self.cfg.test_steps)

        def _fmt_elapsed(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            m, s = divmod(int(seconds + 0.5), 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        for iter in range(self.cfg.max_iter_step + 1):
            self.controller.train()
            self.task_model.train()
            progress = float(iter) / self.cfg.max_iter_step

            feed_dict = self.train_loader.get_feed_dict_and_states(self.args.batch_size)
            imgs, targets, paths, shapes, states_raw = create_input_tensor(
                (feed_dict['im'], feed_dict['label'], feed_dict['path'],
                 feed_dict['shape'], feed_dict['state']))
            imgs = imgs.to(self.device, non_blocking=True).float()
            states_raw = states_raw.to(self.device)
            targets = targets.to(self.device)
            state_before = pipeline_state_from_replay(imgs, states_raw, n_ops=n_ops)

            optim.zero_grad()

            metrics_before = self.task_model.compute_metrics(state_before.image, targets)
            constraint = self.search_space.valid_actions(state_before)
            ctrl_out = self.controller.act(state_before, constraint)
            state_after = self.runtime.step(state_before, ctrl_out.action)
            metrics_after = self.task_model.compute_metrics(state_after.image, targets)
            r, breakdown = self.reward_fn.compute(
                metrics_before, metrics_after,
                state_before, ctrl_out.action, state_after,
                entropy=ctrl_out.entropy, progress=progress,
            )

            # Per-op selection bookkeeping.
            op_idx_np = ctrl_out.action.op_indices.detach().cpu().numpy()
            counts = np.bincount(op_idx_np, minlength=n_ops)
            op_pick_cum += counts
            op_pick_window += counts

            # V2-AI (A+C): windowed reward-breakdown + policy diagnostics.
            win['n'] += 1
            win['task'] += float(breakdown.task_delta.mean().item())
            win['ent_pen'] += float(breakdown.entropy_penalty.mean().item())
            win['use'] += float(breakdown.usage_penalty.mean().item())
            win['estop'] += float(breakdown.early_stop_penalty.mean().item())
            win['ovfl'] += float(breakdown.overflow_penalty.mean().item())
            win['runt'] += float(breakdown.runtime_penalty.mean().item())
            if breakdown.stop_bonus is not None:
                win['stop_b'] += float(breakdown.stop_bonus.mean().item())
            win['pol_ent'] += float(ctrl_out.entropy.mean().item())
            # argmax-match: how often the sampled action equals the greedy
            # choice from logits. Low = exploring; high = policy locked in.
            with torch.no_grad():
                argmax_idx = ctrl_out.logits.argmax(dim=-1)
                # `logits` covers n_ops op-logits + 1 STOP logit; op_indices
                # is n_ops for STOP so this compare is well-defined.
                sampled_idx = torch.where(
                    ctrl_out.action.is_stop,
                    torch.full_like(ctrl_out.action.op_indices, n_ops),
                    ctrl_out.action.op_indices,
                )
                hits = (argmax_idx == sampled_idx).sum().item()
                seen = int(sampled_idx.numel())
            win['argmax_hits'] += int(hits)
            win['argmax_seen'] += seen
            # STOP breakdown: total-STOP vs learned-STOP (Controller chose
            # STOP before time limit) vs time-limit-STOP (max_steps hit).
            is_stop_np = ctrl_out.action.is_stop.detach().cpu().numpy()
            n_stop = int(is_stop_np.sum())
            is_last_step = int((state_after.step == self.cfg.test_steps).sum().item())
            n_stop_timelimit = min(n_stop, is_last_step)
            win['n_stop'] += n_stop
            win['n_stop_timelimit'] += n_stop_timelimit
            win['n_stop_learned'] += (n_stop - n_stop_timelimit)

            # 1-step TD
            old_value = ctrl_out.value
            new_value = self.controller.value_net(state_after)

            clear_final = (state_after.step.float() > self.cfg.maximum_trajectory_length).float().unsqueeze(-1)
            new_value = new_value * (1.0 - clear_final)

            stopped_after = state_after.stopped.float().unsqueeze(-1)
            if self.args.use_truncated:
                retouch_mean = torch.mean(state_after.image, dim=(1, 2, 3)).unsqueeze(-1)
                truncated = torch.where(self._bright_lo < retouch_mean, 1.0, 0.0)
                truncated = torch.where(retouch_mean < self.max_bri, truncated, torch.zeros_like(truncated))
                q_value = r + (1.0 - stopped_after) * self.cfg.discount_factor * new_value * (1.0 - truncated)
            else:
                q_value = r + (1.0 - stopped_after) * self.cfg.discount_factor * new_value

            advantage = q_value.detach() - old_value
            value_loss = torch.mean(advantage ** 2)

            if self.cfg.use_TD:
                routine_loss = -q_value * self.cfg.parameter_lr_mul
                policy_advantage = -advantage
            else:
                routine_loss = -r
                policy_advantage = -r
            assert routine_loss.shape == ctrl_out.log_prob.shape, (routine_loss.shape, ctrl_out.log_prob.shape)
            agent_loss = torch.mean(routine_loss + ctrl_out.log_prob * policy_advantage.detach())

            detect_retouch_loss = metrics_after['detect_loss']
            box_l = metrics_after['box_loss'].item()
            obj_l = metrics_after['obj_loss'].item()
            cls_l = metrics_after['cls_loss'].item()

            if iter % self.cfg.summary_freq == 0:
                try:
                    self.writer.add_scalar('agent_loss', agent_loss, global_step=iter)
                    self.writer.add_scalar('value_loss', value_loss, global_step=iter)
                    self.writer.add_scalar('detect_loss', detect_retouch_loss.mean(), global_step=iter)
                    self.writer.add_scalar('reward', r.mean(), global_step=iter)
                    self.writer.add_scalar('penalty_task_delta', breakdown.task_delta.mean(), global_step=iter)
                    self.writer.add_scalar('penalty_entropy', breakdown.entropy_penalty.mean(), global_step=iter)
                    self.writer.add_scalar('penalty_usage', breakdown.usage_penalty.mean(), global_step=iter)
                    if breakdown.stop_bonus is not None:
                        self.writer.add_scalar('stop_bonus', breakdown.stop_bonus.mean(), global_step=iter)
                    self.writer.add_images('input',
                        torch.clip(state_before.image[:self.cfg.show_img_num], 0.0, 1.0),
                        global_step=iter, dataformats="NCHW")

                    # V2-AI: per-op cumulative pick shares + neural/classical split.
                    total_picks = float(op_pick_cum.sum()) or 1.0
                    for i, name in enumerate(self.cfg.operators):
                        self.writer.add_scalar(f'op_pick_share/{name}',
                                               op_pick_cum[i] / total_picks, global_step=iter)
                    self.writer.add_scalar('op_pick_share/_neural_total',
                                           op_pick_cum[neural_mask].sum() / total_picks, global_step=iter)
                    self.writer.add_scalar('op_pick_share/_classical_total',
                                           op_pick_cum[~neural_mask].sum() / total_picks, global_step=iter)
                except Exception:
                    print("write log error!")
                op_idx_cpu = ctrl_out.action.op_indices.detach().cpu().tolist()
                select_names = [self.cfg.operators[i] if 0 <= i < n_ops else "STOP" for i in op_idx_cpu]
                out_image = torch.clip(state_after.image, 0.0, 1.0).detach().cpu().numpy()
                out_image_res = []
                for b_i in range(out_image.shape[0]):
                    tmp = np.transpose(out_image[b_i], (1, 2, 0)).astype(np.float32).copy()
                    tmp = cv2.putText(tmp, select_names[b_i], (50, 50),
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), thickness=2)
                    out_image_res.append(np.array(tmp))
                try:
                    self.writer.add_images('retouch',
                        np.array(out_image_res[:self.cfg.show_img_num]),
                        global_step=iter, dataformats="NHWC")
                except Exception:
                    print("write log error!")

            total_loss = value_loss + agent_loss
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self._grad_clip_norm)
            optim.step()
            scheduler.step()

            mloss_agent = (mloss_agent * iter + agent_loss.item()) / (iter + 1)
            mloss_value = (mloss_value * iter + value_loss.item()) / (iter + 1)
            new_detect = np.array([box_l, obj_l, cls_l], dtype=np.float32)
            mloss_detect = (mloss_detect * iter + new_detect) / (iter + 1)

            if iter % self.cfg.print_freq == 0:
                t_now = _time.perf_counter()
                elapsed = t_now - t_start
                dt_win = max(t_now - t_prev_print, 1e-6)
                di_win = max(iter - iter_prev_print, 1)
                it_per_s = di_win / dt_win if iter > 0 else 0.0
                remaining = self.cfg.max_iter_step - iter
                eta = remaining / it_per_s if it_per_s > 0 else 0.0
                header_time = datetime.datetime.now().strftime("%H:%M:%S")
                header = (
                    f"----- iter {iter}/{self.cfg.max_iter_step} [{header_time}] "
                    f"elapsed {_fmt_elapsed(elapsed)} | "
                    f"{it_per_s:.2f} it/s | ETA {_fmt_elapsed(eta)} -----"
                )
                print(header)
                print(
                    f"  loss     agent={mloss_agent:.4f} val={mloss_value:.4f} "
                    f"detect={detect_retouch_loss.mean().item():.4f} "
                    f"reward={r.mean().item():+.4f}"
                )

                # # V2-AI (A): windowed reward-breakdown means.
                # wn = max(1, win['n'])
                # stop_bonus_on = float(self.cfg.get('stop_bonus_scale', 0.0)) != 0.0
                # bits_A = [
                #     f"task={win['task'] / wn:+.3f}",
                #     f"ent=-{abs(win['ent_pen'] / wn):.3f}",
                #     f"use=-{abs(win['use'] / wn):.3f}",
                #     f"estop=-{abs(win['estop'] / wn):.3f}",
                #     f"ovfl=-{abs(win['ovfl'] / wn):.3f}",
                # ]
                # if stop_bonus_on:
                #     bits_A.append(f"stop+={win['stop_b'] / wn:+.3f}")
                # if self.cfg.filter_runtime_penalty:
                #     bits_A.append(f"runt=-{abs(win['runt'] / wn):.3f}")
                # print(f"  reward   {' '.join(bits_A)}")

                # # V2-AI (C): policy/exploration diagnostics.
                # pol_ent = win['pol_ent'] / wn
                # argmax_pct = (100.0 * win['argmax_hits'] / max(1, win['argmax_seen']))
                # stop_pct = (100.0 * win['n_stop'] / max(1, win['argmax_seen']))
                # learned_pct = (100.0 * win['n_stop_learned'] / max(1, win['n_stop'])
                #                if win['n_stop'] else 0.0)
                # timelimit_pct = 100.0 - learned_pct if win['n_stop'] else 0.0
                # print(
                #     f"  policy   entropy={pol_ent:.3f}/{log_n_ops_plus1:.3f}  "
                #     f"argmax={argmax_pct:.0f}%  "
                #     f"stop={stop_pct:.0f}% (learned={learned_pct:.0f}%, "
                #     f"timelimit={timelimit_pct:.0f}%)"
                # )

                # traj — batch and replay-pool trajectory-length statistics.
                # ReplayMemory stores state as a flat [has_reward, stopped, step, op_usage...]
                # np array; step is at index 2.
                T_max = int(self.cfg.test_steps)

                # example — three eval-argmax shadow rollouts to reveal what
                # the current Controller would do end-to-end on:
                #   canary : a fixed image drawn once at init  → shows policy
                #            evolution on the same input over training
                #   batch  : a random sample of the current iter's batch
                #   fresh  : a newly-drawn sample from the dataset each print
                # Together these separate "the policy changed" from "the image
                # was different" when a single rollout looks unfamiliar.
                b_idx = int(torch.randint(0, imgs.shape[0], (1,)).item())
                batch_img = imgs[b_idx:b_idx + 1]
                fresh_img = self._grab_dataset_image()

                rollouts: list[tuple[str, list[str], int]] = []
                if self._canary_img is not None:
                    seq, ln = self._shadow_rollout(self._canary_img)
                    rollouts.append(("canary", seq, ln))
                seq, ln = self._shadow_rollout(batch_img)
                rollouts.append((f"batch", seq, ln))
                if fresh_img is not None:
                    seq, ln = self._shadow_rollout(fresh_img)
                    rollouts.append(("fresh", seq, ln))

                print("  example  eval-argmax rollouts on {}:".format(
                    " / ".join(name for name, _, _ in rollouts)
                ))
                for name, seq, ln in rollouts:
                    print(f"           {name:9s} (len={ln}/{T_max}): "
                          f"{' → '.join(seq) if seq else '(none)'}")

                # ops window/cum
                w_total = int(op_pick_window.sum())
                if w_total:
                    top_idx = np.argsort(-op_pick_window)[:6]
                    top_str = " ".join(
                        f"{self.cfg.operators[i]}:{op_pick_window[i]}"
                        for i in top_idx if op_pick_window[i] > 0
                    )
                    c_total = int(op_pick_cum.sum())
                    c_neural = int(op_pick_cum[neural_mask].sum())
                    print(f"  ops      window({w_total}) top: {top_str}")
                    print(
                        f"           cum neural {c_neural}/{c_total} = "
                        f"{100 * c_neural / max(c_total, 1):.1f}%"
                    )
                op_pick_window[:] = 0
                # V2-AI: reset windowed A+C accumulators.
                for k in win:
                    win[k] = 0 if isinstance(win[k], int) else 0.0
                t_prev_print = t_now
                iter_prev_print = iter
                # self.train_loader.debug()

            if torch.isnan(state_after.image).any() or torch.isinf(state_after.image).any():
                print("retouch is nan or inf", torch.mean(state_after.image).detach().cpu().numpy())
                self.train_loader.fill_pool()
            else:
                state_after_flat = pipeline_state_to_replay(state_after)
                self.train_loader.replace_memory(
                    self.train_loader.images_and_states_to_records(
                        state_after.image.detach().cpu().numpy(),
                        feed_dict['label'], feed_dict['path'], feed_dict['shape'],
                        state_after_flat.detach().cpu().numpy(),
                    ))

            if iter % self.cfg.save_model_freq == 0:
                self.controller.eval()
                ckpt = {
                    'iter': iter,
                    'controller_model': self.controller.state_dict(),
                    'optimizer': optim.state_dict(),
                    'operators': list(self.cfg.operators),
                }
                torch.save(ckpt, os.path.join(self.ckpt_dir, f'DynamicISP_iter_{iter}.pth'))
                del ckpt

        torch.cuda.empty_cache()


def _load_config(path: str):
    """Load a config yaml (or fall back to a python module for legacy .py paths).

    Returns a util.Dict for dot-attribute access, with derived fields computed.
    """
    from engine.util import Dict

    if path.endswith(".yaml") or path.endswith(".yml"):
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        cfg = Dict(data)
    else:
        # Legacy: python module import (e.g., --cfg config)
        import importlib
        cfg = importlib.import_module(path).cfg

    # Derived fields
    if 'operators' not in cfg:
        raise ValueError(f"config missing 'operators' list: {path}")
    if 'num_state_dim' not in cfg:
        cfg.num_state_dim = 3 + len(cfg.operators)
    if 'z_dim' not in cfg:
        cfg.z_dim = 3 + len(cfg.operators) * cfg.get('z_dim_per_filter', 16)
    return cfg


__all__ = ["Trainer"]
