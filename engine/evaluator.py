"""engine.evaluator: mAP evaluation for a trained AdaptiveISP checkpoint.

Owns the full val loop. Uses YOLOv3Detection as the boundary to the vendored
yolov3 model, plus pure-function imports (non_max_suppression, ap_per_class,
box_iou, scale_boxes) from yolov3.utils for NMS + metric computation. Does
not depend on `tasks/third_party/yolov3/val_adaptiveisp.py`; that vendored
script is a paper-shipped example and is not on our call path.

The `tools/val.py` script is a thin argparse + dispatch wrapper.

Documented metric line (matches yolov3 format):
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
from engine.trainer import _load_config
from engine.util import set_seed
from isp.registry import build_operator
from pipeline import PipelineExecutor, pipeline_state_from_replay
from search import SearchSpace

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
) -> dict:
    """Run mAP evaluation. Returns a dict of scalar metrics."""
    set_seed(seed, deterministic=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    cfg = _load_config(cfg_path)
    save_dir = Path(project) / name
    save_dir.mkdir(parents=True, exist_ok=exist_ok)

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
    ckpt = torch.load(isp_weights, map_location=device, weights_only=False)
    if 'controller_model' not in ckpt:
        raise SystemExit(
            f"{isp_weights}: missing 'controller_model' key. "
            "Not a V1 checkpoint — for pre-refactor Agent ckpts, use git tag v0-baseline."
        )
    ops = {n: build_operator(n).to(device) for n in cfg.operators}
    controller = AdaptiveISPController(
        ops, cfg.operators, obs_hw=64,
        mid_channels=cfg.base_channels, fc1_size=cfg.fc1_size,
        feature_dim=cfg.feature_extractor_dims,
        dropout_keep_prob=cfg.dropout_keep_prob,
        exploration=cfg.exploration, max_steps=cfg.test_steps,
    ).to(device)
    controller.load_state_dict(ckpt['controller_model'])
    controller.eval()
    executor = PipelineExecutor(ops, cfg.operators)
    search_space = SearchSpace(ops, cfg.operators)

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
