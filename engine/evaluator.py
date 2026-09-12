"""engine.evaluator: evaluation for a trained AdaptiveISP checkpoint.

Auto-routes on the ckpt's `task` field:

  - `detection` (default) → mAP loop against LOD/COCO. Uses YOLOv3Detection
    as the boundary to the vendored yolov3 model, plus pure-function
    imports (non_max_suppression, ap_per_class, box_iou, scale_boxes)
    from yolov3.utils for NMS + metric computation.
  - `human_quality` / `human` → SSIM / LPIPS / Q + mean rollout length +
    learned-STOP pct on the FiveK val split (same numbers as
    HumanTrainer._run_val).

Both paths write a canary visualization under `experiments/<exp>/visualization/`
(derived from the ckpt path), unless `run_viz=False`.

The `tools/val.py` script is a thin argparse + dispatch wrapper.

Detection metric line (matches yolov3 format):
    all <images> <instances> <P> <R> <mAP50> <mAP75> <mAP50-95>
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from tqdm import tqdm

# `tasks` bootstrap makes yolov3.* importable via sys.path.
import tasks  # noqa: F401

from tasks.detection.implementations.yolov3 import YOLOv3Detection
from tasks.detection.dataloader import create_dataloader_real_hr
from controller.adaptiveisp import AdaptiveISPController
from engine.util import set_seed, load_config
from front_isp import build_front_isp_from_cfg
from isp.registry import build_operator
from pipeline import PipelineExecutor, pipeline_state_from_replay
from search import SearchSpace
from search.priors.action_mask import build_from_config as build_action_mask

# Pure-function yolov3 utilities.
from yolov3.utils.general import non_max_suppression, scale_boxes, xywh2xyxy
from yolov3.utils.metrics import ap_per_class, box_iou


def _process_batch(detections: torch.Tensor, labels: torch.Tensor, iouv: torch.Tensor) -> torch.Tensor:
    """[Npred, Niou] bool: whether each prediction is correct at each IoU threshold."""
    correct = torch.zeros(detections.shape[0], iouv.shape[0], dtype=torch.bool, device=iouv.device)
    iou = box_iou(labels[:, 1:], detections[:, :4])
    same_class = labels[:, 0:1] == detections[:, 5]
    for i in range(len(iouv)):
        x = torch.where((iou >= iouv[i]) & same_class)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
            correct[matches[:, 1].astype(int), i] = True
    return correct


def _viz_root_for(isp_weights: str, save_dir: Path) -> Path:
    """Where canary PNGs should land for this ckpt.

    Layout is `experiments/<exp>/ckpt/<file>.pth`, so parent.parent is the
    experiment root that Trainer wrote into. Fall back to `save_dir` when
    the layout doesn't match (e.g. an ad-hoc ckpt path).
    """
    ckpt_path = Path(isp_weights).resolve()
    if ckpt_path.parent.name == 'ckpt' and ckpt_path.parent.parent.is_dir():
        return ckpt_path.parent.parent
    return save_dir


def _run_canary_viz(
    task: str,
    ckpt_dict: dict,
    cfg,
    viz_root: Path,
    viz_cases: int,
    device: torch.device,
) -> None:
    """Write canary PNGs into `viz_root/visualization/`, print one summary line.

    Detection auto-routes to `run_detection`; human_quality/human routes to
    `run_human`. Visualizer internals print a lot of noise (yolo re-load +
    per-case "wrote" lines); we redirect it and surface a single line.
    """
    import contextlib
    import io
    try:
        viz_dir = viz_root / 'visualization'
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
            if task in ('human_quality', 'human'):
                from tools.visualization.visualizer import run_human
                run_human(ckpt_dict, cfg, viz_root, viz_cases, device)
            else:
                from tools.visualization.visualizer import run_detection
                run_detection(ckpt_dict, cfg, viz_root, viz_cases, device)
        n_written = len(list(viz_dir.glob('case_*.png'))) if viz_dir.exists() else 0
        print(f"visualization: {viz_cases} cases ({n_written} PNGs) → {viz_dir}/")
    except Exception as e:
        print(f"visualization: FAILED ({e})")


def evaluate(
    isp_weights: str,
    weights: str,
    data: str,
    data_name: str,
    imgsz: int,
    batch_size: int,
    steps: int,
    cfg_path: str,
    project: str,
    name: str,
    exist_ok: bool,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    seed: int,
    run_viz: bool = True,
    viz_cases: int = 4,
) -> dict:
    """Auto-routing evaluator entry point.

    Peeks the ckpt's `task` field and dispatches to `_evaluate_detection`
    or `_evaluate_human`. Both write canary PNGs under
    `experiments/<exp>/visualization/` (falls back to `<project>/<name>/`
    if the ckpt is not under a Trainer-produced layout).

    Detection-only args (`weights`, `data`, `data_name`, `conf_thres`,
    `iou_thres`, `max_det`) are ignored on the Human path — the ckpt's
    accompanying cfg (`human_quality:` block) drives dataset selection.
    """
    set_seed(seed, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = load_config(cfg_path)
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=exist_ok)

    # Peek the checkpoint once — decides which branch runs, then reused by
    # both the eval branch and (if enabled) the canary visualizer.
    ckpt_dict = torch.load(isp_weights, map_location=device, weights_only=False)
    if 'controller_model' not in ckpt_dict:
        raise SystemExit(
            f"{isp_weights}: missing 'controller_model' key. "
            "Not a V1 checkpoint — for pre-refactor Agent ckpts, use git tag v0-baseline."
        )
    task = ckpt_dict.get('task', 'detection')

    if task in ('human_quality', 'human'):
        metrics = _evaluate_human(
            ckpt_dict=ckpt_dict, cfg=cfg, device=device,
            imgsz=imgsz, batch_size=batch_size,
            save_dir=save_dir,
        )
    else:
        metrics = _evaluate_detection(
            ckpt_dict=ckpt_dict, cfg=cfg, device=device,
            isp_weights=isp_weights, weights=weights, data=data,
            imgsz=imgsz, batch_size=batch_size, steps=steps,
            conf_thres=conf_thres, iou_thres=iou_thres, max_det=max_det,
            seed=seed, save_dir=save_dir,
        )

    if run_viz:
        _run_canary_viz(
            task=task, ckpt_dict=ckpt_dict, cfg=cfg,
            viz_root=_viz_root_for(isp_weights, save_dir),
            viz_cases=viz_cases, device=device,
        )
    return metrics


def _evaluate_detection(
    *,
    ckpt_dict: dict,
    cfg,
    device: torch.device,
    isp_weights: str,
    weights: str,
    data: str,
    imgsz: int,
    batch_size: int,
    steps: int,
    conf_thres: float,
    iou_thres: float,
    max_det: int,
    seed: int,
    save_dir: Path,
) -> dict:
    """mAP eval loop against LOD/COCO. Returns metrics dict."""
    # 1. Detection backbone (frozen)
    with open('tasks/third_party/yolov3/data/hyps/hyp.scratch-low.yaml') as f:
        hyp = yaml.safe_load(f)
    data_dict = YOLOv3Detection.parse_data_cfg(data)
    nc = int(data_dict['nc'])
    task_model = YOLOv3Detection(
        weights=weights, yolo_cfg='tasks/third_party/yolov3/models/yolov3.yaml',
        hyp=hyp, nc=nc, imgsz=imgsz, device=device,
    )
    task_model.eval()

    # 2. Controller
    ops = {n: build_operator(n).to(device) for n in cfg.operators}
    controller = AdaptiveISPController(
        ops, cfg.operators, obs_hw=64,
        mid_channels=cfg.base_channels, fc1_size=cfg.fc1_size,
        feature_dim=cfg.feature_extractor_dims,
        dropout_keep_prob=cfg.dropout_keep_prob,
        exploration=cfg.exploration, max_steps=cfg.test_steps,
    ).to(device)
    controller.load_state_dict(ckpt_dict['controller_model'])
    controller.eval()
    executor = PipelineExecutor(ops, cfg.operators)
    _am = build_action_mask(cfg.get('action_mask', {}) or {}, cfg.operators)
    search_space = SearchSpace(ops, cfg.operators, priors=[_am] if _am.priors else None)
    # Configurable Front ISP (identity when disabled — E0 parity).
    front_isp = build_front_isp_from_cfg(cfg).to(device)

    # 3. Data
    val_loader, _ = create_dataloader_real_hr(
        data_dict['val'], imgsz, batch_size, task_model.gs, single_cls=False,
        hyp={'anchor_t': hyp.get('anchor_t', 4.0), 'label_smoothing': 0.0},
        cache=False, rect=False, workers=1, pad=0.0, prefix='val: ',
        add_noise=False, seed=seed,
    )

    # 4. Rollout + evaluation
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    niou = iouv.numel()
    stats: list = []
    seen = 0

    with torch.no_grad(), open(save_dir / 'records.txt', 'w') as f_rec:
        f_rec.write(','.join(cfg.operators) + '\n')
        for imgs, targets, paths, shapes, _imgs_hr in tqdm(val_loader, desc='val'):
            imgs = imgs.to(device).float()
            targets = targets.to(device)
            imgs = front_isp(imgs).clamp(0.0, 1.0)
            _, _, height, width = imgs.shape

            state = executor.initial_state(imgs)
            for _ in range(steps):
                out = controller.act(state, search_space.valid_actions(state))
                state = executor.step(state, out.action)
                if state.stopped.all():
                    break
            retouch = state.image

            for b in range(imgs.shape[0]):
                ops_taken = ','.join(str(int(a.op_indices[b].item())) for a in state.history)
                f_rec.write(f'{os.path.basename(paths[b])},{ops_taken}\n')

            preds = task_model.model(retouch)
            targets[:, 2:] *= torch.tensor((width, height, width, height), device=device)
            preds = non_max_suppression(
                preds, conf_thres, iou_thres, labels=[],
                multi_label=True, agnostic=False, max_det=max_det,
            )

            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]
                shape = shapes[si][0]
                correct = torch.zeros(npr, niou, dtype=torch.bool, device=device)
                seen += 1
                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    continue
                predn = pred.clone()
                scale_boxes(imgs[si].shape[1:], predn[:, :4], shape, shapes[si][1])
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_boxes(imgs[si].shape[1:], tbox, shape, shapes[si][1])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct = _process_batch(predn, labelsn, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

    # 5. Aggregate
    stats_np = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)] if stats else []
    metrics = {'seen': seen, 'instances': 0, 'P': 0.0, 'R': 0.0,
               'mAP50': 0.0, 'mAP75': 0.0, 'mAP50-95': 0.0}
    if stats_np and stats_np[0].any():
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(
            *stats_np, plot=False, save_dir=str(save_dir), names=data_dict['names'])
        ap50, ap75, ap_all = ap[:, 0], ap[:, 5], ap.mean(1)
        metrics.update({
            'instances': int(np.bincount(stats_np[3].astype(int), minlength=nc).sum()),
            'P': float(p.mean()), 'R': float(r.mean()),
            'mAP50': float(ap50.mean()), 'mAP75': float(ap75.mean()),
            'mAP50-95': float(ap_all.mean()),
        })

    header = '%22s%11s%11s%11s%11s%11s%11s%11s' % (
        'Class', 'Images', 'Instances', 'P', 'R', 'mAP50', 'mAP75', 'mAP50-95')
    line = '%22s%11i%11i%11.3g%11.3g%11.3g%11.3g%11.3g' % (
        'all', metrics['seen'], metrics['instances'],
        metrics['P'], metrics['R'],
        metrics['mAP50'], metrics['mAP75'], metrics['mAP50-95'],
    )
    print(header)
    print(line)
    with open(save_dir / 'val_log.txt', 'w') as f:
        f.write(header + '\n' + line + '\n')

    return metrics


def _evaluate_human(
    *,
    ckpt_dict: dict,
    cfg,
    device: torch.device,
    imgsz: int,
    batch_size: int,
    save_dir: Path,
) -> dict:
    """FiveK val — SSIM/LPIPS/Q + rollout length + learned-STOP pct.

    Mirrors `HumanTrainer._run_val` (dataset, quality metric, aggregation)
    so a bare ckpt gets the same numbers you would see at the tail of a
    training log. Writes a `val_log.txt` summary next to the mAP one for
    parity with the Detection path.
    """
    from torch.utils.data import DataLoader
    from tasks.human_quality import (
        FiveKDataset, HumanQualityTask, collate_fivek,
        psnr_batch, delta_e_batch,
    )
    # V3.1: Front ISP state in ckpt is loaded via generic
    # `len(state_dict()) > 0` below (learnable mode has params; identity/
    # fixed do not), so no CalibratedFrontISP import is needed here (legacy name of LearnableFrontISP).

    hq_cfg = cfg.get('human_quality', {}) or {}
    fivek_root = hq_cfg.get('fivek_root', '/home/jing/datasets/fivek')
    val_list = hq_cfg.get('val_list', os.path.join(fivek_root, 'val_expert_c.txt'))
    cache_dir = hq_cfg.get('cache_dir', os.path.join(fivek_root, 'cache_expert_c'))

    task_model = HumanQualityTask(
        lambda_ssim=float(hq_cfg.get('lambda_ssim', 1.0)),
        lambda_lpips=float(hq_cfg.get('lambda_lpips', 1.0)),
        lpips_net=hq_cfg.get('lpips_net', 'alex'),
        device=device,
    )

    val_dataset = FiveKDataset(val_list, cache_dir=cache_dir, imgsz=imgsz)
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=2, pin_memory=True, drop_last=False,
        collate_fn=collate_fivek,
    )

    ops = {n: build_operator(n).to(device) for n in cfg.operators}
    controller = AdaptiveISPController(
        ops, cfg.operators, obs_hw=int(cfg.get('obs_hw', 64)),
        mid_channels=cfg.base_channels, fc1_size=cfg.fc1_size,
        feature_dim=cfg.feature_extractor_dims,
        dropout_keep_prob=cfg.dropout_keep_prob,
        exploration=cfg.exploration, max_steps=cfg.test_steps,
    ).to(device)
    controller.load_state_dict(ckpt_dict['controller_model'])
    controller.eval()
    executor = PipelineExecutor(ops, cfg.operators)
    _am = build_action_mask(cfg.get('action_mask', {}) or {}, cfg.operators)
    search_space = SearchSpace(ops, cfg.operators, priors=[_am] if _am.priors else None)
    front_isp = build_front_isp_from_cfg(cfg).to(device)
    # V3.1: prefer the Front ISP state bundled in the ckpt over cfg-init
    # (learnable mode; identity/fixed have nothing to load).
    if ckpt_dict.get('front_isp') and len(front_isp.state_dict()) > 0:
        front_isp.load_state_dict(ckpt_dict['front_isp'])

    T = int(cfg.test_steps)
    ssim_sum = lpips_sum = q_sum = 0.0
    len_sum = stop_count = n = 0
    psnr_sum = de_sum = 0.0
    with torch.no_grad():
        for imgs_v, targets_v, cam_v in tqdm(val_loader, desc='val'):
            imgs_v = imgs_v.to(device, non_blocking=True).float()
            targets_v = targets_v.to(device, non_blocking=True).float()
            cam_v = cam_v.to(device)
            imgs_v = front_isp(imgs_v, {'camera_id': cam_v}).clamp(0.0, 1.0)
            state = executor.initial_state(imgs_v)
            lengths = torch.zeros(imgs_v.shape[0], dtype=torch.long, device=device)
            stopped_learned = torch.zeros(imgs_v.shape[0], dtype=torch.bool, device=device)
            for _ in range(T):
                o = controller.act(state, search_space.valid_actions(state))
                is_time_limit = (state.step >= (T - 1))
                stop_learned_this_step = (
                    o.action.is_stop & ~is_time_limit & ~state.stopped
                )
                stopped_learned = stopped_learned | stop_learned_this_step
                lengths = lengths + (~state.stopped).long() * (~o.action.is_stop).long()
                state = executor.step(state, o.action)
                if state.stopped.all():
                    break
            m_v = task_model.compute_metrics(state.image, targets_v)
            b = imgs_v.shape[0]
            ssim_sum += m_v['ssim'].sum().item()
            lpips_sum += m_v['lpips'].sum().item()
            q_sum += m_v['quality'].sum().item()
            psnr_sum += psnr_batch(state.image, targets_v).sum().item()
            de_sum += delta_e_batch(state.image, targets_v).sum().item()
            len_sum += int(lengths.sum().item())
            stop_count += int(stopped_learned.sum().item())
            n += b

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

    lines = [
        "===== VAL (Human Quality) =====",
        f"  samples: {metrics['val/n_samples']}",
        f"  SSIM:  {metrics['val/ssim']:.4f}",
        f"  LPIPS: {metrics['val/lpips']:.4f}",
        f"  PSNR:  {metrics['val/psnr']:.2f} dB   ΔE76: {metrics['val/delta_e']:.2f}",
        f"  Q:     {metrics['val/quality']:+.4f}",
        f"  mean rollout length: {metrics['val/mean_length']:.2f}/{T}",
        f"  pct learned-STOP (before time-limit): "
        f"{100 * metrics['val/pct_learned_stop']:.1f}%",
        "===============================",
    ]
    print("\n" + "\n".join(lines) + "\n")
    with open(save_dir / 'val_log.txt', 'w') as f:
        f.write("\n".join(lines) + "\n")
    return metrics
