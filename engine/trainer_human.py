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
import shutil
import time
from typing import Optional

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from controller.adaptiveisp import AdaptiveISPController
from controller.adaptiveisp.human_reward import HumanReward
from engine.util import Tee
from isp.registry import build_operator
from pipeline import PipelineExecutor
from search import SearchSpace
from tasks.human_quality import FiveKDataset, HumanQualityTask, collate_fivek
from tasks.human_quality.metrics import quality_score


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s: %(message)s')


class HumanTrainer:
    """Assembles Controller + Executor + HumanReward + FiveK loader."""

    def __init__(self, args, task: str = "train") -> None:
        train = task in ("train", "train_val")
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
            print("HumanTrainer begin....")
            print("------- V2-AI: FiveK + Expert C  →  SSIM/LPIPS terminal reward ---------")

        cfg = _load_config(args.cfg)
        cfg.filter_runtime_penalty = getattr(args, 'runtime_penalty', False)
        cfg.filter_runtime_penalty_lambda = getattr(args, 'runtime_penalty_lambda', 0.01)

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

        # --- Four subsystems (same as Detection Trainer) ---
        ops = {name: build_operator(name).to(self.device) for name in cfg.operators}
        self.runtime = PipelineExecutor(ops, cfg.operators)
        self.search_space = SearchSpace(ops, cfg.operators)
        self.controller = AdaptiveISPController(
            ops, cfg.operators,
            obs_hw=int(cfg.get('obs_hw', 64)),
            mid_channels=cfg.base_channels,
            fc1_size=cfg.fc1_size,
            feature_dim=cfg.feature_extractor_dims,
            dropout_keep_prob=cfg.dropout_keep_prob,
            exploration=cfg.exploration,
            max_steps=cfg.test_steps,
        ).to(self.device)

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
        images_per_epoch = int(cfg.get('images_per_epoch', len(self.train_dataset)))
        cfg.max_iter_step = int(self.args.epochs * images_per_epoch // args.batch_size)
        if cfg.show_img_num > args.batch_size:
            cfg.show_img_num = args.batch_size

        self.cfg = cfg
        self._grad_clip_norm = float(cfg.get('grad_clip_norm', 1e-5))
        train_cfg = cfg.get('train', {}) or {}
        self._lr_decay = float(train_cfg.get('lr_decay', 0.1))
        self._lr_segments = int(train_cfg.get('lr_segments', 3))

    # ----------------------- helpers -----------------------

    def _next_batch(self):
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    # ----------------------- train loop -----------------------

    def train(self) -> None:
        if self.args.resume is not None:
            print(f"Resume from {self.args.resume}")
            ckpt = torch.load(self.args.resume, weights_only=False)
            if 'controller_model' in ckpt:
                self.controller.load_state_dict(ckpt['controller_model'])
            else:
                logger.warning("Resume ckpt is legacy; Controller has different architecture — starting fresh.")

        optim = torch.optim.Adam(self.controller.parameters(), lr=self.args.lr)
        max_iter_step = self.cfg.max_iter_step
        lr_lambda = lambda it: self._lr_decay ** (1.0 * it * self._lr_segments / max(max_iter_step, 1))
        scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda)
        print(f"init learning rate: {scheduler.get_last_lr()[0]}")

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
        win = {
            'n_iters': 0, 'n_steps': 0,
            'task': 0.0, 'ent_pen': 0.0, 'use': 0.0, 'estop': 0.0,
            'ovfl': 0.0, 'stop_b': 0.0, 'runt': 0.0,
            'pol_ent': 0.0, 'argmax_hits': 0, 'argmax_seen': 0,
            'n_stop': 0, 'n_stop_learned': 0, 'n_stop_timelimit': 0,
        }
        log_n_ops_plus1 = float(np.log(n_ops + 1))

        t_start = time.perf_counter()
        t_prev_print = t_start
        iter_prev_print = 0
        max_steps_display = int(self.cfg.test_steps)

        def _fmt_elapsed(seconds: float) -> str:
            seconds = max(0.0, float(seconds))
            m, s = divmod(int(seconds + 0.5), 60)
            h, m = divmod(m, 60)
            return f"{h:d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

        # For the per-step example line: how each op's first physical
        # parameter is displayed. Neural ops -> `alpha=`, others -> raw val.
        def _fmt_step(op_name: str, phys: np.ndarray) -> str:
            first = float(phys[0]) if phys.size else 0.0
            if op_name.startswith("n_"):
                return f"{op_name}(α={first:.2f})"
            if phys.size > 1:
                return f"{op_name}({first:.2f},+{phys.size - 1})"
            return f"{op_name}({first:.2f})"

        for it in range(max_iter_step + 1):
            self.controller.train()
            progress = float(it) / max(max_iter_step, 1)
            win['n_iters'] += 1

            imgs, targets = self._next_batch()
            imgs = imgs.to(self.device, non_blocking=True).float()
            targets = targets.to(self.device, non_blocking=True).float()

            optim.zero_grad()

            # Precompute Q(I_0) once per rollout — saves T-1 LPIPS/SSIM calls.
            q_initial, q0_parts = quality_score(
                imgs, targets,
                lambda_ssim=self.task_model.lambda_ssim,
                lambda_lpips=self.task_model.lambda_lpips,
                lpips_net=self.task_model.lpips_net,
            )

            state = self.runtime.initial_state(imgs)
            value_losses: list[torch.Tensor] = []
            agent_losses: list[torch.Tensor] = []
            traj_step_info: list[tuple[str, np.ndarray]] = []   # sample 0's per-step (op_name, phys)
            reward_totals: list[torch.Tensor] = []
            q_final_parts: dict[str, torch.Tensor] = {}

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
                win['n_steps'] += 1
                win['task'] += float(breakdown.task_delta.mean().item())
                win['ent_pen'] += float(breakdown.entropy_penalty.mean().item())
                win['use'] += float(breakdown.usage_penalty.mean().item())
                win['estop'] += float(breakdown.early_stop_penalty.mean().item())
                win['ovfl'] += float(breakdown.overflow_penalty.mean().item())
                win['runt'] += float(breakdown.runtime_penalty.mean().item())
                if breakdown.stop_bonus is not None:
                    win['stop_b'] += float(breakdown.stop_bonus.mean().item())
                win['pol_ent'] += float(ctrl_out.entropy.mean().item())
                with torch.no_grad():
                    argmax_idx = ctrl_out.logits.argmax(dim=-1)
                    sampled_idx = torch.where(
                        ctrl_out.action.is_stop,
                        torch.full_like(ctrl_out.action.op_indices, n_ops),
                        ctrl_out.action.op_indices,
                    )
                    win['argmax_hits'] += int((argmax_idx == sampled_idx).sum().item())
                    win['argmax_seen'] += int(sampled_idx.numel())
                is_stop_batch = ctrl_out.action.is_stop.detach().cpu().numpy()
                n_stop = int(is_stop_batch.sum())
                is_last = int((new_state.step == T).sum().item())
                n_stop_tl = min(n_stop, is_last)
                win['n_stop'] += n_stop
                win['n_stop_timelimit'] += n_stop_tl
                win['n_stop_learned'] += (n_stop - n_stop_tl)

                # 1-step TD
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

            total_loss = torch.stack(value_losses).sum() + torch.stack(agent_losses).sum()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.controller.parameters(), self._grad_clip_norm)
            optim.step()
            scheduler.step()

            # Terminal quality — reuse `q_final_parts` from the last reward
            # call to avoid an extra SSIM+LPIPS forward. Reconstruct q_final
            # from its parts (Q = λ_ssim·SSIM − λ_lpips·LPIPS).
            if q_final_parts:
                q_final = q_final_parts["quality"]
            else:
                q_final, q_final_parts = quality_score(
                    state.image, targets,
                    lambda_ssim=self.task_model.lambda_ssim,
                    lambda_lpips=self.task_model.lambda_lpips,
                    lpips_net=self.task_model.lpips_net,
                )
            q_delta = (q_final - q_initial).mean().item()

            # loss stats (moving averages) — agent + value tracked separately.
            agent_val = float(torch.stack(agent_losses).mean().item())
            value_val = float(torch.stack(value_losses).mean().item())
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
                    f"elapsed {_fmt_elapsed(elapsed)} | {it_per_s:.2f} it/s | "
                    f"ETA {_fmt_elapsed(eta)} -----"
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
                seq_pretty = [_fmt_step(name, phys) for (name, phys) in traj_step_info]
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
                # V2-AI: reset windowed A+C accumulators.
                for k in win:
                    win[k] = 0 if isinstance(win[k], int) else 0.0
                t_prev_print = t_now
                iter_prev_print = it

            # sanity check
            if torch.isnan(state.image).any() or torch.isinf(state.image).any():
                print("output is nan or inf")

            if it % self.cfg.save_model_freq == 0:
                self.controller.eval()
                ckpt = {
                    'iter': it,
                    'controller_model': self.controller.state_dict(),
                    'optimizer': optim.state_dict(),
                    'operators': list(self.cfg.operators),
                    'task': 'human_quality',
                }
                torch.save(ckpt, os.path.join(self.ckpt_dir, f'HumanISP_iter_{it}.pth'))
                del ckpt

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
        ssim_sum, lpips_sum, q_sum, len_sum, stop_count, n = 0.0, 0.0, 0.0, 0, 0, 0
        with torch.no_grad():
            for imgs_v, targets_v in self.val_loader:
                imgs_v = imgs_v.to(self.device, non_blocking=True).float()
                targets_v = targets_v.to(self.device, non_blocking=True).float()
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
                q_final, parts = quality_score(
                    state.image, targets_v,
                    lambda_ssim=self.task_model.lambda_ssim,
                    lambda_lpips=self.task_model.lambda_lpips,
                    lpips_net=self.task_model.lpips_net,
                )
                b = imgs_v.shape[0]
                ssim_sum += parts['ssim'].sum().item()
                lpips_sum += parts['lpips'].sum().item()
                q_sum += q_final.sum().item()
                len_sum += int(lengths.sum().item())
                stop_count += int(stopped_learned.sum().item())
                n += b
        self.controller.train()

        metrics = {
            'val/ssim': ssim_sum / max(n, 1),
            'val/lpips': lpips_sum / max(n, 1),
            'val/quality': q_sum / max(n, 1),
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
        print(f"  Q:     {metrics['val/quality']:+.4f}")
        print(f"  mean rollout length: {metrics['val/mean_length']:.2f}/{T}")
        print(f"  pct learned-STOP (before time-limit): {100 * metrics['val/pct_learned_stop']:.1f}%")
        print("=================================\n")
        return metrics


def _load_config(path: str):
    """Load a config yaml. Returns a util.Dict for dot-attribute access."""
    from engine.util import Dict

    with open(path, "r") as f:
        data = yaml.safe_load(f)
    cfg = Dict(data)

    if 'operators' not in cfg:
        raise ValueError(f"config missing 'operators' list: {path}")
    if 'num_state_dim' not in cfg:
        cfg.num_state_dim = 3 + len(cfg.operators)
    if 'z_dim' not in cfg:
        cfg.z_dim = 3 + len(cfg.operators) * cfg.get('z_dim_per_filter', 16)
    return cfg


__all__ = ["HumanTrainer"]
