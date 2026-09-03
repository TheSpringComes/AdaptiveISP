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
                    self.writer.add_images('input',
                        torch.clip(state_before.image[:self.cfg.show_img_num], 0.0, 1.0),
                        global_step=iter, dataformats="NCHW")
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
                mem = f'{torch.cuda.memory_reserved() / 1E9 if torch.cuda.is_available() else 0:.3g}G'
                n_targets = targets.shape[0]
                penalty_total = -(breakdown.total.mean().item() - breakdown.task_delta.mean().item())
                print(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                      ('%11s' + '%8s,') % (f'{iter}/{self.cfg.max_iter_step - 1}', mem),
                      f"agent loss: {mloss_agent:.4f},",
                      f"value loss: {mloss_value:.4f},",
                      f"box_loss: {mloss_detect[0]:.4f}, obj_loss: {mloss_detect[1]:.4f}, cls_loss: {mloss_detect[2]:.4f},",
                      f"detect_retouch_loss: {detect_retouch_loss.mean().item():.2f},",
                      f"instances: {n_targets:2d},",
                      f"lr: {scheduler.get_last_lr()[0]:.4e},",
                      f"penalty: {penalty_total:.4e}",
                      f"reward: {r.mean().item():.4e}",
                )
                self.train_loader.debug()

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
