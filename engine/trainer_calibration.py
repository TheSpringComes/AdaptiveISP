"""CalibrationTrainer: V3.1 Stage 1 — calibration 预训练。

关闭 AdaptiveISP（不构建 Controller/Executor），只训练 Front ISP 的
标定参数：

    RAW → Calibration(WB, CCM, Bias, Tone) → Base RGB
                                        ↘ L_calib vs Expert C

损失（V3.1 §4 Stage 1）：

    L_calib = λ1·L1 + λs·(1 - SSIM) + λp·LPIPS

由 `training:` 配置段驱动（calibration_pretrain），入口为
`tools/train.py --task calibration`。保存的 ckpt 含标定 state_dict 与
camera 名单，Stage 2 通过 `front_isp.calibration.init.params` 复用，
Stage 3 在其基础上联合微调。
"""
from __future__ import annotations

import datetime
import logging
import os
import time

import torch
from torch.utils.data import DataLoader

from engine.base_trainer import BaseTrainer
from front_isp.calibration import CalibratedFrontISP
from tasks.human_quality import (
    FiveKDataset, HumanQualityTask, collate_fivek,
    lpips_batch, psnr_batch, delta_e_batch,
)


logger = logging.getLogger(__name__)
if not logger.hasHandlers():
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(name)s %(levelname)s: %(message)s')


class CalibrationTrainer(BaseTrainer):
    """Assembles Front ISP (calibrated) + FiveK loader + L_calib loop."""

    banner = ("CalibrationTrainer begin....\n"
              "------- V3.1 Stage 1: RAW → Calibration → Base RGB vs Expert C ---------")
    ckpt_prefix = "CalibISP"

    def __init__(self, args, task: str = "train") -> None:
        train = task in ("train", "train_val")
        self._setup_experiment(args, save_enabled=train)
        cfg = self._load_cfg(args)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # --- Data (FiveK, with camera ids) ---
        hq_cfg = cfg.get('human_quality', {}) or {}
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
        self.val_dataset = FiveKDataset(val_list, cache_dir=cache_dir, imgsz=imgsz)
        self.val_loader = DataLoader(
            self.val_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=max(args.workers // 2, 1), pin_memory=True, drop_last=False,
            collate_fn=collate_fivek,
        )

        # --- Front ISP (must be learnable / legacy 'calibrated') ---
        # 注入 n_cameras：camera_specific 时按数据侧实际相机数建表。
        fi_cfg = cfg.get('front_isp', {}) or {}
        if fi_cfg.get('type') not in ('learnable', 'calibrated'):
            raise SystemExit(
                "CalibrationTrainer 需要 front_isp.type == learnable，"
                f"得到 {fi_cfg.get('type', 'none')!r}。请使用 V3.1 预训练 config。"
            )
        calib_cfg = ((fi_cfg.get('learnable', {}) or {})
                     or (fi_cfg.get('calibration', {}) or {}))
        if calib_cfg.get('camera_specific', False):
            calib_cfg['n_cameras'] = max(self.train_dataset.n_cameras,
                                         self.val_dataset.n_cameras)
        from front_isp import build_front_isp
        self.front_isp = build_front_isp(fi_cfg).to(self.device)
        if not isinstance(self.front_isp, CalibratedFrontISP):
            raise SystemExit(f"front_isp 解析为 {type(self.front_isp).__name__}，"
                             "CalibrationTrainer 只接受 CalibratedFrontISP。")
        n_learn = len(self.front_isp.trainable_parameters())
        if n_learn == 0:
            raise SystemExit("标定参数全部被冻结（learnable 全 false），无可训练项。")

        # --- LPIPS/SSIM via the task metric stack ---
        t_cfg = cfg.get('training', {}) or {}
        self.lambda_l1 = float(t_cfg.get('lambda_l1', 1.0))
        self.lambda_ssim = float(t_cfg.get('lambda_ssim', 1.0))
        self.lambda_lpips = float(t_cfg.get('lambda_lpips', 1.0))
        self.task_model = HumanQualityTask(
            lambda_ssim=self.lambda_ssim, lambda_lpips=self.lambda_lpips,
            lpips_net=hq_cfg.get('lpips_net', 'alex'), device=self.device,
        )

        self.args = args
        cfg.setdefault('images_per_epoch', 500)
        self._finalize_cfg_derived_fields(args, cfg)
        self.cfg = cfg

        print(f"CalibrationTrainer: cameras={self.front_isp.table.n_cameras} "
              f"({self.train_dataset.n_cameras} in train split), "
              f"learnable tensors={n_learn}")
        print(f"  loss = {self.lambda_l1}·L1 + {self.lambda_ssim}·(1-SSIM) "
              f"+ {self.lambda_lpips}·LPIPS")

    # ---------------- helpers ----------------

    def _save_ckpt(self, iter_idx: int, optim, extra=None) -> None:
        """Override: no Controller to save — ckpt 只含标定 state 与相机表。"""
        ckpt = {
            'iter': iter_idx,
            'optimizer': optim.state_dict(),
            'front_isp': self.front_isp.state_dict(),
            'camera_names': self.train_dataset.camera_names,
        }
        if extra:
            ckpt.update(extra)
        torch.save(ckpt, os.path.join(self.ckpt_dir, f'{self.ckpt_prefix}_iter_{iter_idx}.pth'))

    def _next_batch(self):
        try:
            return next(self._train_iter)
        except StopIteration:
            self._train_iter = iter(self.train_loader)
            return next(self._train_iter)

    def _loss(self, pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """L_calib = λ1·L1 + λs·(1-SSIM) + λp·LPIPS（全部可微）。"""
        l1 = (pred - target).abs().mean()
        ssim = self.task_model.compute_metrics(pred, target)['ssim'].mean()
        lpips = lpips_batch(pred, target, net=self.task_model.lpips_net,
                            grad=True).mean()
        loss = (self.lambda_l1 * l1
                + self.lambda_ssim * (1.0 - ssim)
                + self.lambda_lpips * lpips)
        return loss, {'l1': l1.detach(), 'ssim': ssim.detach(),
                      'lpips': lpips.detach()}

    # ---------------- train loop ----------------

    def train(self) -> None:
        from front_isp.calibration.fittedisp_loader import load_fittedisp_params  # noqa: F401
        self._maybe_resume_calibration(self.args.resume)

        params = self.front_isp.trainable_parameters()
        optim = torch.optim.Adam(params, lr=self.args.lr)
        max_iter_step = int(self.cfg.max_iter_step)
        lr_decay = self._lr_decay
        segments = self._lr_segments
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optim, lr_lambda=lambda it: lr_decay ** (1.0 * it * segments / max(max_iter_step, 1)))

        t_start = time.perf_counter()
        mloss = 0.0
        print_freq = int(self.cfg.get('print_freq', 100))

        for it in range(max_iter_step + 1):
            imgs, targets, cam_ids = self._next_batch()
            imgs = imgs.to(self.device, non_blocking=True).float()
            targets = targets.to(self.device, non_blocking=True).float()
            cam_ids = cam_ids.to(self.device)

            optim.zero_grad()
            base = self.front_isp(imgs, {'camera_id': cam_ids})
            loss, parts = self._loss(base, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            optim.step()
            scheduler.step()

            mloss = (mloss * it + float(loss.item())) / (it + 1)

            if it % print_freq == 0:
                elapsed = time.perf_counter() - t_start
                eta = elapsed / max(it, 1) * (max_iter_step - it)
                print(f"----- iter {it}/{max_iter_step} "
                      f"[{datetime.datetime.now().strftime('%H:%M:%S')}] "
                      f"loss={mloss:.4f} l1={parts['l1']:.4f} "
                      f"ssim={parts['ssim']:.4f} lpips={parts['lpips']:.4f} "
                      f"| ETA {int(eta // 60)}m{int(eta % 60):02d}s -----")
                try:
                    self.writer.add_scalar('calib/loss', float(loss.item()), it)
                    self.writer.add_scalar('calib/l1', float(parts['l1']), it)
                    self.writer.add_scalar('calib/ssim', float(parts['ssim']), it)
                    self.writer.add_scalar('calib/lpips', float(parts['lpips']), it)
                    self.writer.add_images('raw', torch.clip(imgs[:2], 0, 1), it)
                    self.writer.add_images('base_rgb', torch.clip(base[:2], 0, 1), it)
                    self.writer.add_images('target', torch.clip(targets[:2], 0, 1), it)
                except Exception:
                    print("write log error!")

            if it % self.cfg.save_model_freq == 0 or it == max_iter_step:
                self._save_ckpt(it, optim,
                                extra={'task': 'calibration_pretrain'})

        self._run_val(step=max_iter_step)
        torch.cuda.empty_cache()

    def _maybe_resume_calibration(self, resume_path) -> None:
        if not resume_path:
            return
        print(f"Resume calibration from {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu', weights_only=False)
        state = ckpt.get('front_isp', ckpt)
        self.front_isp.load_state_dict(state)

    def _run_val(self, step: int) -> dict:
        """Val: PSNR / SSIM / LPIPS / ΔE（V3.1 §7 消融指标）。"""
        self.front_isp.eval()
        psnr_s = ssim_s = lpips_s = de_s = n = 0
        with torch.no_grad():
            for imgs_v, targets_v, cam_v in self.val_loader:
                imgs_v = imgs_v.to(self.device).float()
                targets_v = targets_v.to(self.device).float()
                cam_v = cam_v.to(self.device)
                base = self.front_isp(imgs_v, {'camera_id': cam_v})
                psnr_s += psnr_batch(base, targets_v).sum().item()
                ssim_s += self.task_model.compute_metrics(base, targets_v)['ssim'].sum().item()
                lpips_s += lpips_batch(base, targets_v,
                                       net=self.task_model.lpips_net).sum().item()
                de_s += delta_e_batch(base, targets_v).sum().item()
                n += imgs_v.shape[0]
        self.front_isp.train()

        metrics = {'val/psnr': psnr_s / max(n, 1), 'val/ssim': ssim_s / max(n, 1),
                   'val/lpips': lpips_s / max(n, 1), 'val/delta_e': de_s / max(n, 1),
                   'val/n_samples': n}
        print("\n===== VAL (Calibration, end of training) =====")
        print(f"  samples: {n}")
        print(f"  PSNR:  {metrics['val/psnr']:.2f} dB")
        print(f"  SSIM:  {metrics['val/ssim']:.4f}")
        print(f"  LPIPS: {metrics['val/lpips']:.4f}")
        print(f"  ΔE76:  {metrics['val/delta_e']:.2f}")
        print("===============================================\n")
        try:
            for k, v in metrics.items():
                if isinstance(v, (int, float)):
                    self.writer.add_scalar(k, v, global_step=step)
        except Exception:
            pass
        return metrics


__all__ = ["CalibrationTrainer"]
