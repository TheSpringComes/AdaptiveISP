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
from typing import Optional

import cv2
import numpy as np
import torch
import yaml

# `tasks.detection.implementations.yolov3` is our sole boundary to the vendored
# yolov3 code; the trainer must not import yolov3.* directly.
from tasks.base import TaskMetrics                                # noqa: F401
from tasks.detection.implementations.yolov3 import YOLOv3Detection

from controller.adaptiveisp import AdaptiveISPReward
from engine.base_trainer import BaseTrainer
from pipeline import PipelineState, pipeline_state_from_replay, pipeline_state_to_replay
from tasks.detection.replay import ReplayMemory, create_input_tensor


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s %(levelname)s: %(message)s')


LOCAL_RANK = int(os.getenv('LOCAL_RANK', -1))
RANK = int(os.getenv('RANK', -1))
PIN_MEMORY = str(os.getenv('PIN_MEMORY', True)).lower() == 'true'
WORLD_SIZE = 1


class Trainer(BaseTrainer):
    """Assembles subsystems + runs the training / periodic-save loop."""

    banner = ("Training begin....\n"
              "------- V1 Refactor: Controller + PipelineExecutor + Task + Reward ---------")
    ckpt_prefix = "DynamicISP"

    def __init__(self, args, task: str = "train_val") -> None:
        train = task in ("train", "train_val")
        val = task == "train_val"
        self._setup_experiment(args, save_enabled=train)

        cfg = self._load_cfg(args)

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

        # Pipeline subsystems (Controller, Executor, SearchSpace, Backbone).
        # Built BEFORE ReplayMemory so the (optional) V3 CanonicalBackbone is
        # available at pool-fill time — the pool invariant is
        # "images stored here are already post-backbone".
        self._build_pipeline_subsystems(cfg, self.device)

        # Data loaders (ReplayMemory holds partial-trajectory (image, state) pairs)
        train_path, val_path = data_dict['train'], data_dict['val']
        if task == "test":
            val_path = data_dict['test']
        self.train_loader = ReplayMemory(cfg, train, train_path, args.imgsz, args.batch_size, gs,
                                          single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0,
                                          rect=False, image_weights=False, prefix='train: ', limit=-1,
                                          add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range,
                                          noise_level=args.noise_level, use_linear=args.use_linear,
                                          backbone=self.backbone, backbone_device=self.device)
        if val:
            self.val_loader = ReplayMemory(cfg, val, val_path, args.imgsz, args.batch_size, gs,
                                            single_cls=False, hyp=hyp, augment=False, cache=False, pad=0.0,
                                            rect=False, image_weights=False, prefix='val: ', limit=-1,
                                            add_noise=args.add_noise, data_name=args.data_name, brightness_range=args.bri_range,
                                            noise_level=args.noise_level, use_linear=args.use_linear,
                                            backbone=self.backbone, backbone_device=self.device)
            self.val_loader = self.val_loader.get_feed_dict_and_states(8)

        self.task_model.attach_class_weights(self.train_loader.dataset.labels, nc)
        self.task_model.attach_names(data_dict['names'])
        self.data_dict = data_dict

        # Reward. (Pipeline subsystems already built above.)
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
        self._finalize_cfg_derived_fields(args, cfg)

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

        V3-A1: when `self.backbone` is set, apply it once so canary rollouts
        start from the same baseline sRGB state as training samples do
        (the pool holds post-backbone images).
        """
        try:
            im_list, _, _, _ = self.train_loader.dataset.get_next_batch(1)
        except Exception as exc:
            print(f"dataset image capture failed: {exc!r}")
            return None
        im = im_list[0]
        if isinstance(im, np.ndarray):
            im = torch.from_numpy(im)
        im = im.unsqueeze(0).to(self.device).float()
        if self.backbone is not None:
            with torch.no_grad():
                im = self.backbone(im).clamp(0.0, 1.0)
        return im

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
        self._maybe_resume(self.args.resume)
        optim, scheduler = self._build_optimizer_and_scheduler(self.args, self.cfg)

        # V3-E3: opt-in PPO branch. `rl_algo.name == 'ppo'` swaps 1-step TD
        # + ReplayMemory for T-step rollout → GAE → K-epoch PPO. The pool is
        # bypassed under PPO (on-policy); backbone still applies via
        # ReplayMemory.get_next_RAW.
        rl_cfg = self.cfg.get('rl_algo', {}) or {}
        use_ppo = str(rl_cfg.get('name', 'actor_critic')).lower() == 'ppo'
        if use_ppo:
            from controller.adaptiveisp import PPOUpdater
            from pipeline import TrajectoryBuffer
            ppo_cfg = rl_cfg.get('ppo', {}) or {}
            self._ppo = PPOUpdater(ppo_cfg)
            self._TrajectoryBuffer = TrajectoryBuffer
            self._gae_gamma = float(ppo_cfg.get('gamma', self.cfg.discount_factor))
            self._gae_lambda = float(ppo_cfg.get('gae_lambda', 0.95))
            print(f"RL algo: PPO (K={self._ppo.epochs} clip={self._ppo.clip_range} "
                  f"gae_lambda={self._gae_lambda})")

        logger.info(f'Image sizes {self.args.imgsz} train, {self.args.imgsz} val | '
                    f'Using {self.args.workers} dataloader workers | '
                    f'Logging results to {self.args.save_path} | '
                    f'Starting training for {0} epochs...')
        mloss_agent, mloss_value = 0.0, 0.0
        mloss_detect = np.zeros(3, dtype=np.float32)

        n_ops = len(self.cfg.operators)

        # Windowed accumulators for reward-breakdown / policy diagnostics.
        # Detection currently suppresses the A+C print sub-blocks (see 86f512a),
        # so the accumulator runs but is not read at print time; keeping the
        # helper call in place preserves parity with HumanTrainer's format.
        win = self._make_window()
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

        for iter in range(self.cfg.max_iter_step + 1):
            self.controller.train()
            self.task_model.train()
            progress = float(iter) / self.cfg.max_iter_step

            if use_ppo:
                # -------- PPO branch: T-step rollout, no replay --------
                # Fresh samples only (backbone applied inside get_next_RAW).
                im_list, label_list, path_list, shapes_list, _states_list = \
                    self.train_loader.get_next_RAW(self.args.batch_size)
                imgs = torch.from_numpy(np.stack(im_list, 0)).to(
                    self.device, non_blocking=True).float()
                # Set per-batch image index in each label row (mirrors
                # create_input_tensor's loop). `label_list` entries are numpy
                # arrays [n_boxes, 6] where col 0 = image index.
                for i, lb in enumerate(label_list):
                    lb[:, 0] = i
                targets = torch.from_numpy(np.concatenate(label_list, 0)).to(self.device)

                state = self.runtime.initial_state(imgs)
                metrics_prev = self.task_model.compute_metrics(state.image, targets)
                buf = self._TrajectoryBuffer(
                    max_steps=int(self.cfg.test_steps),
                    batch_size=imgs.shape[0],
                )

                # For post-loop diagnostics parity with the Actor-Critic
                # branch, capture the final step's outputs into the same
                # variable names.
                state_before = state
                ctrl_out = None
                r = None
                breakdown = None
                metrics_after = metrics_prev
                state_after = state

                # `param_loss` is applied per step (backward + free graph)
                # instead of accumulated over the whole rollout, because
                # keeping T=8 YOLOv3 forwards live at once OOMs a 24GB GPU.
                # Gradients accumulate on param_features + param_heads; a
                # single optim.step() at rollout end applies them.
                optim.zero_grad()
                T_max = int(self.cfg.test_steps)
                for t in range(T_max):
                    state_before = state
                    constraint = self.search_space.valid_actions(state)
                    ctrl_out = self.controller.act(state, constraint)
                    new_state = self.runtime.step(state, ctrl_out.action)
                    metrics_after = self.task_model.compute_metrics(
                        new_state.image, targets,
                    )
                    r, breakdown = self.reward_fn.compute(
                        metrics_prev, metrics_after,
                        state, ctrl_out.action, new_state,
                        entropy=ctrl_out.entropy, progress=progress,
                    )

                    # per-op bookkeeping (parity with Actor-Critic branch)
                    op_idx_np = ctrl_out.action.op_indices.detach().cpu().numpy()
                    counts = np.bincount(op_idx_np, minlength=n_ops)
                    op_pick_cum += counts
                    op_pick_window += counts

                    self._accumulate_window(win, breakdown, ctrl_out.entropy, ctrl_out,
                                            new_state, n_ops, T_max)

                    # Op-param loss: image-differentiable pieces (task_delta
                    # − overflow_penalty). Backprops into param_features +
                    # param_heads (select_head / value_net stay grad-None).
                    # Immediate backward frees this step's task_model +
                    # op-forward graph.
                    alive_mask = (~state.stopped).float().unsqueeze(-1)
                    step_param_loss = -(
                        (breakdown.task_delta - breakdown.overflow_penalty) * alive_mask
                    ).mean()
                    if step_param_loss.requires_grad:
                        step_param_loss.backward()
                    else:
                        # Diagnostic: happens if the whole step is "dead"
                        # (all samples already stopped so alive_mask=0 kills
                        # the graph) or if action was all-STOP so no op was
                        # applied and image/params never entered the graph.
                        # Nothing to update — safe skip.
                        pass

                    # Buffer stores fully-detached snapshots (see push()).
                    buf.push(state=state, action=ctrl_out.action,
                             log_prob=ctrl_out.log_prob, value=ctrl_out.value,
                             entropy=ctrl_out.entropy, reward=r)

                    # Detach across steps so the next step's forward builds
                    # a fresh graph rather than extending this one.
                    state = PipelineState(
                        image=new_state.image.detach(),
                        step=new_state.step, stopped=new_state.stopped,
                        has_reward=new_state.has_reward,
                        op_usage=new_state.op_usage,
                    )
                    # Detach metrics for the next step's reward (only detect_loss
                    # is referenced, and reward_fn already .detach()s it inside).
                    from tasks.base import TaskMetrics
                    metrics_prev = TaskMetrics(
                        values={k: v.detach() for k, v in metrics_after.values.items()},
                        extras=metrics_after.extras,
                    )
                    state_after = new_state
                    if state.stopped.all():
                        break

                # 1. Op-param step (gradients accumulated per-step above).
                torch.nn.utils.clip_grad_norm_(
                    self.controller.parameters(), self._grad_clip_norm,
                )
                optim.step()
                scheduler.step()

                # 2. PPO K-epoch update — policy + value only.
                with torch.no_grad():
                    bootstrap = self.controller.value_net(state).reshape(-1)
                buf.compute_gae(
                    gamma=self._gae_gamma, lam=self._gae_lambda,
                    bootstrap_value=bootstrap,
                )
                flat = buf.flat_alive_batch()
                ppo_stats = self._ppo.update(
                    self.controller, self.search_space, optim, flat,
                )
                agent_loss_val = float(ppo_stats['policy_loss'])
                value_loss_val = float(ppo_stats['value_loss'])
                detect_retouch_loss = metrics_after['detect_loss']
                box_l = metrics_after['box_loss'].item()
                obj_l = metrics_after['obj_loss'].item()
                cls_l = metrics_after['cls_loss'].item()

                mloss_agent = (mloss_agent * iter + agent_loss_val) / (iter + 1)
                mloss_value = (mloss_value * iter + value_loss_val) / (iter + 1)
                new_detect = np.array([box_l, obj_l, cls_l], dtype=np.float32)
                mloss_detect = (mloss_detect * iter + new_detect) / (iter + 1)

                # Fall through to shared print/save block (below).
                _skip_legacy_body = True
            else:
                _skip_legacy_body = False

            if _skip_legacy_body:
                pass
            else:
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
                self._accumulate_window(win, breakdown, ctrl_out.entropy, ctrl_out,
                                        state_after, n_ops, int(self.cfg.test_steps))

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
                    f"elapsed {self._fmt_elapsed(elapsed)} | "
                    f"{it_per_s:.2f} it/s | ETA {self._fmt_elapsed(eta)} -----"
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
                self._reset_window(win)
                t_prev_print = t_now
                iter_prev_print = iter
                # self.train_loader.debug()

            if not use_ppo:
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
                self._save_ckpt(iter, optim)

        torch.cuda.empty_cache()


__all__ = ["Trainer"]
