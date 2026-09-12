"""HumanTrainer: single-task training loop for the Human Quality objective.

Independent of the existing Detection `Trainer` — deliberately duplicated
so the two paths can be understood in isolation. Task-alternation between
LOD (Detection) and FiveK (Human) is a Stage-B concern; this file only
ships the FiveK/Human single-task path.

Structural differences from `engine.trainer.Trainer`:
  - No `ReplayMemory`. A FiveK sample is one full `T = cfg.test_steps`
    rollout episode per iter. The whole rollout is unrolled inside a
    single call to `train()` and produces one `.backward()`.
  - Reward is `HumanReward`: intermediate `task_delta = 0`; the sparse
    `Q(I_T) - Q(I_0)` fires only on the terminal step. Auxiliary penalties
    (entropy / usage / early-stop / runtime) fire every step, same as
    Detection.
  - Print block matches the Detection trainer's format-A: one header line
    with progress / it/s / ETA, then loss / example (with the full op
    sequence for sample 0) / ops (window + cum share).
"""
from __future__ import annotations

import datetime
import logging
import os
import time
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from controller.adaptiveisp.human_reward import HumanReward
from engine.base_trainer import BaseTrainer
from tasks.human_quality import FiveKDataset, HumanQualityTask, collate_fivek


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s: %(message)s')


class HumanTrainer(BaseTrainer):
    """Assembles Controller + Executor + HumanReward + FiveK loader."""

    banner = ("HumanTrainer begin....\n"
              "------- V2-AI: FiveK + Expert C  →  SSIM/LPIPS terminal reward ---------")
    ckpt_prefix = "HumanISP"

    def __init__(self, args, task: str = "train") -> None:
        train = task in ("train", "train_val")
        self._setup_experiment(args, save_enabled=train)

        cfg = self._load_cfg(args)

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Task ---
        hq_cfg = cfg.get('human_quality', {}) or {}
        self.task_model = HumanQualityTask(
            lambda_ssim=float(hq_cfg.get('lambda_ssim', 1.0)),
            lambda_lpips=float(hq_cfg.get('lambda_lpips', 1.0)),
            lpips_net=hq_cfg.get('lpips_net', 'alex'),
            device=self.device,
        )

        # --- Data (FiveK) ---
        fivek_root = hq_cfg.get('fivek_root', '/home/jing/datasets/fivek')
        train_list = hq_cfg.get('train_list', os.path.join(fivek_root, 'train_expert_c.txt'))
        val_list = hq_cfg.get('val_list', os.path.join(fivek_root, 'val_expert_c.txt'))
        cache_dir = hq_cfg.get('cache_dir', os.path.join(fivek_root, 'cache_expert_c'))
        imgsz = int(getattr(args, 'imgsz', 512))

        self.train_dataset = FiveKDataset(train_list, cache_dir=cache_dir, imgsz=imgsz)
        self.train_loader = DataLoader(
            self.train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True,
            collate_fn=collate_fivek,
        )
        self._train_iter = iter(self.train_loader)

        # Val split — used for periodic + end-of-training quality metrics.
        # 100 samples in val_expert_c.txt; kept tiny so val is cheap.
        self.val_dataset = FiveKDataset(val_list, cache_dir=cache_dir, imgsz=imgsz)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=max(args.workers // 2, 1), pin_memory=True, drop_last=False,
            collate_fn=collate_fivek,
        )

        # Pipeline subsystems shared with detection Trainer (Controller,
        # Executor, SearchSpace) — task-agnostic construction lives in
        # BaseTrainer. Before building, give a camera-specific learnable
        # Front ISP the dataset's camera count so its parameter table is
        # sized correctly.
        fi_cfg = cfg.get('front_isp', {}) or {}
        calib_cfg = (fi_cfg.get('learnable', {}) or {})
        if fi_cfg.get('type') == 'learnable' and calib_cfg.get('camera_specific'):
            calib_cfg['n_cameras'] = max(self.train_dataset.n_cameras,
                                         self.val_dataset.n_cameras)
        self._build_pipeline_subsystems(cfg, self.device)

        # V3.1 两阶段训练：本 trainer 即 Stage 2 —— Front ISP（无论何种
        # 模式）恒为冻结，只做前向；可学习参数只在 Stage 1
        # （LearnableTrainer）训练，不与 AdaptiveISP 联合训练。
        self.front_isp.eval()
        for p in self.front_isp.parameters():
            p.requires_grad_(False)

        self.reward_fn = HumanReward(
            n_ops=len(cfg.operators),
            max_steps=cfg.test_steps,
            lambda_ssim=self.task_model.lambda_ssim,
            lambda_lpips=self.task_model.lambda_lpips,
            lpips_net=self.task_model.lpips_net,
            critic_logit_multiplier=cfg.critic_logit_multiplier,
            all_reward=cfg.all_reward,
            filter_usage_penalty=cfg.filter_usage_penalty,
            exploration_penalty=cfg.exploration_penalty,
            early_stop_penalty=cfg.early_stop_penalty,
            runtime_penalty_enabled=cfg.filter_runtime_penalty,
            runtime_penalty_lambda=cfg.filter_runtime_penalty_lambda,
            runtime_costs=cfg.filters_runtime,
            use_penalty=cfg.use_penalty,
        )

        n_params = sum(p.numel() for p in self.controller.parameters())
        print(f"HumanTrainer Controller parameters: {n_params / 1e6:.2f}M")
        print(f"FiveK train samples: {len(self.train_dataset)}   imgsz={imgsz}")

        self.args = args
        # `images_per_epoch` default = full dataset for Human (Detection uses 1000).
        cfg.setdefault('images_per_epoch', len(self.train_dataset))
        self._finalize_cfg_derived_fields(args, cfg)

        self.cfg = cfg

    # ----------------------- helpers -----------------------

    def _next_batch(self):
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    # ----------------------- train loop -----------------------

    def train(self) -> None:
        self._maybe_resume(self.args.resume)
        optim, scheduler = self._build_optimizer_and_scheduler(self.args, self.cfg)
        max_iter_step = int(self.cfg.max_iter_step)

        # V3-E3: opt-in PPO path. `rl_algo.name == 'ppo'` swaps the per-iter
        # 1-step TD loss for T-step rollout → GAE → K-epoch PPO update.
        # Everything else (dataset, front ISP, reward_fn, canary/val, print
        # block) is shared with the Actor-Critic path.
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

        logger.info(f'Image size {self.args.imgsz} | '
                    f'Using {self.args.workers} dataloader workers | '
                    f'Logging results to {self.args.save_path}')

        n_ops = len(self.cfg.operators)
        T = int(self.cfg.test_steps)
        mloss_agent, mloss_value = 0.0, 0.0
        mloss_quality_delta = 0.0

        # per-op selection tracking (window + cumulative)
        op_pick_cum = np.zeros(n_ops, dtype=np.int64)
        op_pick_window = np.zeros(n_ops, dtype=np.int64)
        neural_mask = np.array([n.startswith("n_") for n in self.cfg.operators], dtype=bool)

        # V2-AI (A+C): windowed accumulators for reward-breakdown + policy
        # diagnostics. For Human, reward components are per-step (aggregated
        # across the T-step rollout inside each iter), then summed across
        # iters in the window. `n_steps` is total per-step samples in-window.
        win = self._make_window()
        log_n_ops_plus1 = float(np.log(n_ops + 1))

        t_start = time.perf_counter()
        t_prev_print = t_start
        iter_prev_print = 0
        max_steps_display = int(self.cfg.test_steps)

        for it in range(max_iter_step + 1):
            self.controller.train()
            progress = float(it) / max(max_iter_step, 1)

            imgs, targets, cam_ids = self._next_batch()
            imgs = imgs.to(self.device, non_blocking=True).float()
            targets = targets.to(self.device, non_blocking=True).float()
            cam_ids = cam_ids.to(self.device)

            # Configurable Front ISP (RAW → baseline RGB) runs before the
            # Controller sees the image — identity when disabled (E0 parity).
            # `m0` is computed on the post-front_isp image so the reward
            # Q(final) - Q(initial) measures the Controller's (Adaptive Tail)
            # contribution alone. V3.1: Front ISP 恒冻结（两阶段训练，不
            # 联合），前向在 no_grad 下执行。
            with torch.no_grad():
                imgs = self.front_isp(imgs, {'camera_id': cam_ids}).clamp(0.0, 1.0)

            optim.zero_grad()

            # Precompute Q(I_0) once per rollout — saves T-1 LPIPS/SSIM calls.
            # Route through the Task interface (parity with Detection's
            # `task_model.compute_metrics(...)` call in engine.trainer).
            m0 = self.task_model.compute_metrics(imgs, targets)
            q_initial = m0['quality']

            state = self.runtime.initial_state(imgs)
            value_losses: list[torch.Tensor] = []
            agent_losses: list[torch.Tensor] = []
            traj_step_info: list[tuple[str, np.ndarray]] = []   # sample 0's per-step (op_name, phys)
            reward_totals: list[torch.Tensor] = []
            q_final_parts: dict[str, torch.Tensor] = {}
            # V3-E3 PPO branch state
            if use_ppo:
                buf = self._TrajectoryBuffer(max_steps=T, batch_size=imgs.shape[0])
                param_loss_terms: list[torch.Tensor] = []

            for t in range(T):
                constraint = self.search_space.valid_actions(state)
                ctrl_out = self.controller.act(state, constraint)
                new_state = self.runtime.step(state, ctrl_out.action)

                # per-op window/cum + sample-0 op + physical param
                op_idx_np = ctrl_out.action.op_indices.detach().cpu().numpy()
                counts = np.bincount(op_idx_np, minlength=n_ops)
                op_pick_cum += counts
                op_pick_window += counts
                op_idx_0 = int(op_idx_np[0])
                op_name_0 = self.cfg.operators[op_idx_0]
                spec_dim = self.runtime.operators[op_name_0].spec.dim
                phys_0 = ctrl_out.action.params[0, :spec_dim].detach().cpu().numpy()
                traj_step_info.append((op_name_0, phys_0))

                r, breakdown, q_parts = self.reward_fn.compute(
                    image_initial=imgs, target=targets,
                    state_before=state, action=ctrl_out.action, state_after=new_state,
                    entropy=ctrl_out.entropy, progress=progress, q_initial=q_initial,
                )
                reward_totals.append(r.mean().detach())
                if q_parts:
                    q_final_parts = q_parts

                # V2-AI (A+C): accumulate reward-breakdown + policy stats.
                self._accumulate_window(win, breakdown, ctrl_out.entropy, ctrl_out,
                                        new_state, n_ops, T)

                if use_ppo:
                    # PPO branch: collect trajectory for K-epoch policy/value
                    # update AND accumulate image-differentiable op-param loss
                    # (task_delta − overflow_penalty). Alive samples only —
                    # already-stopped ones' image is untouched by the executor,
                    # so their task_delta is ~0 anyway.
                    alive_mask = (~state.stopped).float().unsqueeze(-1)
                    param_loss_terms.append(
                        -((breakdown.task_delta - breakdown.overflow_penalty) * alive_mask).mean()
                    )
                    buf.push(state=state, action=ctrl_out.action,
                             log_prob=ctrl_out.log_prob, value=ctrl_out.value,
                             entropy=ctrl_out.entropy, reward=r)
                else:
                    # Actor-Critic branch (legacy): per-step 1-step TD.
                    old_value = ctrl_out.value
                    new_value = self.controller.value_net(new_state)
                    stopped_after = new_state.stopped.float().unsqueeze(-1)
                    new_value = new_value * (1.0 - stopped_after)
                    q_value = r + (1.0 - stopped_after) * self.cfg.discount_factor * new_value
                    advantage = q_value.detach() - old_value
                    value_loss = torch.mean(advantage ** 2)

                    if self.cfg.use_TD:
                        routine_loss = -q_value * self.cfg.parameter_lr_mul
                        policy_advantage = -advantage
                    else:
                        routine_loss = -r
                        policy_advantage = -r
                    agent_loss = torch.mean(
                        routine_loss + ctrl_out.log_prob * policy_advantage.detach()
                    )
                    value_losses.append(value_loss)
                    agent_losses.append(agent_loss)

                state = new_state
                if state.stopped.all():
                    break

            if use_ppo:
                # 1. Op-param update (image-path gradients only).
                if param_loss_terms:
                    total_param_loss = torch.stack(param_loss_terms).sum()
                else:
                    total_param_loss = torch.zeros((), device=self.device)
                # optim.zero_grad() already called before rollout; backward
                # here fills gradients only on param_features + param_heads
                # (verified — image path is independent of select_head/value_net).
                total_param_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.controller.parameters(), self._grad_clip_norm,
                )
                optim.step()
                scheduler.step()

                # 2. PPO K-epoch update — policy + value only. `PPOUpdater`
                # calls zero_grad + backward + step internally each minibatch,
                # so op-params (grad=None after each of its zero_grad) stay
                # frozen through the K epochs.
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
                agent_val = float(ppo_stats['policy_loss'])
                value_val = float(ppo_stats['value_loss'])
            else:
                total_loss = torch.stack(value_losses).sum() + torch.stack(agent_losses).sum()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self._grad_clip_norm)
                optim.step()
                scheduler.step()
                agent_val = float(torch.stack(agent_losses).mean().item())
                value_val = float(torch.stack(value_losses).mean().item())

            # Terminal quality — reuse `q_final_parts` from the last reward
            # call to avoid an extra SSIM+LPIPS forward. Reconstruct q_final
            # from its parts (Q = λ_ssim·SSIM − λ_lpips·LPIPS).
            if q_final_parts:
                q_final = q_final_parts["quality"]
            else:
                m_T = self.task_model.compute_metrics(state.image, targets)
                q_final = m_T['quality']
                q_final_parts = {'ssim': m_T['ssim'], 'lpips': m_T['lpips'], 'quality': m_T['quality']}
            q_delta = (q_final - q_initial).mean().item()

            # loss stats (moving averages). `agent_val` / `value_val` are set
            # above in each branch — PPO uses ppo_stats, Actor-Critic uses the
            # per-step losses.
            mloss_agent = (mloss_agent * it + agent_val) / (it + 1)
            mloss_value = (mloss_value * it + value_val) / (it + 1)
            mloss_quality_delta = (mloss_quality_delta * it + q_delta) / (it + 1)

            # summary logs
            if it % self.cfg.summary_freq == 0:
                try:
                    self.writer.add_scalar('agent_loss', agent_val, global_step=it)
                    self.writer.add_scalar('value_loss', value_val, global_step=it)
                    self.writer.add_scalar('quality/q_initial', q_initial.mean().item(), global_step=it)
                    self.writer.add_scalar('quality/q_final', q_final.mean().item(), global_step=it)
                    self.writer.add_scalar('quality/q_delta', q_delta, global_step=it)
                    self.writer.add_scalar('quality/ssim_final', q_final_parts['ssim'].mean().item(), global_step=it)
                    self.writer.add_scalar('quality/lpips_final', q_final_parts['lpips'].mean().item(), global_step=it)
                    total_picks = float(op_pick_cum.sum()) or 1.0
                    for i, name in enumerate(self.cfg.operators):
                        self.writer.add_scalar(f'op_pick_share/{name}',
                                               op_pick_cum[i] / total_picks, global_step=it)
                    self.writer.add_scalar('op_pick_share/_neural_total',
                                           op_pick_cum[neural_mask].sum() / total_picks, global_step=it)
                    self.writer.add_images('input',
                        torch.clip(imgs[:self.cfg.show_img_num], 0.0, 1.0),
                        global_step=it, dataformats="NCHW")
                    self.writer.add_images('output',
                        torch.clip(state.image[:self.cfg.show_img_num], 0.0, 1.0),
                        global_step=it, dataformats="NCHW")
                    self.writer.add_images('target',
                        torch.clip(targets[:self.cfg.show_img_num], 0.0, 1.0),
                        global_step=it, dataformats="NCHW")
                except Exception:
                    print("write log error!")

            # print block
            if it % self.cfg.print_freq == 0:
                t_now = time.perf_counter()
                elapsed = t_now - t_start
                dt_win = max(t_now - t_prev_print, 1e-6)
                di_win = max(it - iter_prev_print, 1)
                it_per_s = di_win / dt_win if it > 0 else 0.0
                remaining = max_iter_step - it
                eta = remaining / it_per_s if it_per_s > 0 else 0.0
                header_time = datetime.datetime.now().strftime("%H:%M:%S")
                print(
                    f"----- iter {it}/{max_iter_step} [{header_time}] "
                    f"elapsed {self._fmt_elapsed(elapsed)} | {it_per_s:.2f} it/s | "
                    f"ETA {self._fmt_elapsed(eta)} -----"
                )
                print(
                    f"  loss     agent={mloss_agent:.4f} val={mloss_value:.4f} "
                    f"Q_0={q_initial.mean().item():+.4f} Q_T={q_final.mean().item():+.4f} "
                    f"ΔQ={q_delta:+.4f} SSIM={q_final_parts['ssim'].mean().item():.4f} "
                    f"LPIPS={q_final_parts['lpips'].mean().item():.4f}"
                )

                # V2-AI (A): windowed reward-breakdown means (per rollout step).
                wn = max(1, win['n_steps'])
                stop_bonus_on = float(self.cfg.get('stop_bonus_scale', 0.0)) != 0.0
                bits_A = [
                    f"task={win['task'] / wn:+.3f}",
                    f"ent=-{abs(win['ent_pen'] / wn):.3f}",
                    f"use=-{abs(win['use'] / wn):.3f}",
                    f"estop=-{abs(win['estop'] / wn):.3f}",
                    f"ovfl=-{abs(win['ovfl'] / wn):.3f}",
                ]
                if stop_bonus_on:
                    bits_A.append(f"stop+={win['stop_b'] / wn:+.3f}")
                if self.cfg.filter_runtime_penalty:
                    bits_A.append(f"runt=-{abs(win['runt'] / wn):.3f}")
                print(f"  reward   {' '.join(bits_A)}")

                # V2-AI (C): policy/exploration diagnostics.
                pol_ent = win['pol_ent'] / wn
                argmax_pct = (100.0 * win['argmax_hits'] / max(1, win['argmax_seen']))
                stop_pct = (100.0 * win['n_stop'] / max(1, win['argmax_seen']))
                learned_pct = (100.0 * win['n_stop_learned'] / max(1, win['n_stop'])
                               if win['n_stop'] else 0.0)
                timelimit_pct = 100.0 - learned_pct if win['n_stop'] else 0.0
                print(
                    f"  policy   entropy={pol_ent:.3f}/{log_n_ops_plus1:.3f}  "
                    f"argmax={argmax_pct:.0f}%  "
                    f"stop={stop_pct:.0f}% (learned={learned_pct:.0f}%, "
                    f"timelimit={timelimit_pct:.0f}%)"
                )

                # example — sample 0's full op sequence with per-step α / first param
                seq_pretty = [self._fmt_step(name, phys) for (name, phys) in traj_step_info]
                q0_s0 = q_initial[0, 0].item()
                qT_s0 = q_final[0, 0].item()
                print(
                    f"  example  sample 0  Q_0={q0_s0:+.4f} → Q_T={qT_s0:+.4f} "
                    f"(ΔQ={qT_s0 - q0_s0:+.4f})"
                )
                print(f"           picks: {' → '.join(seq_pretty)}")

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
                    print(f"           cum neural {c_neural}/{c_total} = "
                          f"{100 * c_neural / max(c_total, 1):.1f}%")
                op_pick_window[:] = 0
                self._reset_window(win)
                t_prev_print = t_now
                iter_prev_print = it

            # sanity check
            if torch.isnan(state.image).any() or torch.isinf(state.image).any():
                print("output is nan or inf")

            if it % self.cfg.save_model_freq == 0:
                extra = {'task': 'human_quality'}
                # Front ISP 有可保存状态时（learnable 模式）一并存入，
                # 便于 resume / evaluator 复用 Stage-1 冻结参数。
                if len(self.front_isp.state_dict()) > 0:
                    extra['front_isp'] = self.front_isp.state_dict()
                self._save_ckpt(it, optim, extra=extra)

        # Final val on the held-out 100 images (mean SSIM/LPIPS/Q + mean length).
        self._run_val(step=max_iter_step)
        torch.cuda.empty_cache()

    def _run_val(self, step: int) -> dict:
        """Run eval-argmax rollout on the val split; log mean quality + length.

        Called at end of training. `step` is used as the global_step tag for
        TensorBoard. Returns the metrics dict for the caller's log/summary use.
        """
        self.controller.eval()
        T = int(self.cfg.test_steps)
        from tasks.human_quality import psnr_batch, delta_e_batch
        ssim_sum, lpips_sum, q_sum, len_sum, stop_count, n = 0.0, 0.0, 0.0, 0, 0, 0
        psnr_sum, de_sum = 0.0, 0.0
        with torch.no_grad():
            for imgs_v, targets_v, cam_v in self.val_loader:
                imgs_v = imgs_v.to(self.device, non_blocking=True).float()
                targets_v = targets_v.to(self.device, non_blocking=True).float()
                cam_v = cam_v.to(self.device)
                imgs_v = self.front_isp(imgs_v, {'camera_id': cam_v}).clamp(0.0, 1.0)
                state = self.runtime.initial_state(imgs_v)
                lengths = torch.zeros(imgs_v.shape[0], dtype=torch.long, device=self.device)
                stopped_learned = torch.zeros(imgs_v.shape[0], dtype=torch.bool, device=self.device)
                for t in range(T):
                    c = self.search_space.valid_actions(state)
                    o = self.controller.act(state, c)
                    # A sample "learn-STOP-ed" if action.is_stop AND not-time-limit
                    step_before = state.step
                    is_time_limit = (step_before >= (T - 1))
                    stop_learned_this_step = o.action.is_stop & ~is_time_limit & ~state.stopped
                    stopped_learned = stopped_learned | stop_learned_this_step
                    lengths = lengths + (~state.stopped).long() * (~o.action.is_stop).long()
                    state = self.runtime.step(state, o.action)
                    if state.stopped.all():
                        break
                m_v = self.task_model.compute_metrics(state.image, targets_v)
                q_final = m_v['quality']
                parts = {'ssim': m_v['ssim'], 'lpips': m_v['lpips']}
                b = imgs_v.shape[0]
                ssim_sum += parts['ssim'].sum().item()
                lpips_sum += parts['lpips'].sum().item()
                q_sum += q_final.sum().item()
                psnr_sum += psnr_batch(state.image, targets_v).sum().item()
                de_sum += delta_e_batch(state.image, targets_v).sum().item()
                len_sum += int(lengths.sum().item())
                stop_count += int(stopped_learned.sum().item())
                n += b
        self.controller.train()

        metrics = {
            'val/ssim': ssim_sum / max(n, 1),
            'val/lpips': lpips_sum / max(n, 1),
            'val/quality': q_sum / max(n, 1),
            'val/psnr': psnr_sum / max(n, 1),
            'val/delta_e': de_sum / max(n, 1),
            'val/mean_length': len_sum / max(n, 1),
            'val/pct_learned_stop': stop_count / max(n, 1),
            'val/n_samples': n,
        }
        try:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, global_step=step)
        except Exception:
            pass
        print("\n===== VAL (end of training) =====")
        print(f"  samples: {metrics['val/n_samples']}")
        print(f"  SSIM:  {metrics['val/ssim']:.4f}")
        print(f"  LPIPS: {metrics['val/lpips']:.4f}")
        print(f"  PSNR:  {metrics['val/psnr']:.2f} dB   ΔE76: {metrics['val/delta_e']:.2f}")
        print(f"  Q:     {metrics['val/quality']:+.4f}")
        print(f"  mean rollout length: {metrics['val/mean_length']:.2f}/{T}")
        print(f"  pct learned-STOP (before time-limit): {100 * metrics['val/pct_learned_stop']:.1f}%")
        print("=================================\n")
        return metrics


__all__ = ["HumanTrainer"]
