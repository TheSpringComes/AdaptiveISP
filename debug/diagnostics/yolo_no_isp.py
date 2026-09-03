"""diagnostics: YOLO on raw LOD (no ISP) → expected: no detections.

This is the negative control that anchors the mAP claim. If this
script produces non-trivial detections, the reported 71.6 mAP@0.5 is
not attributable to the learned pipeline, and the val flow needs
investigation. Under the standard LOD low-light conditions and the
default 0.001 confidence threshold, the vendored yolov3 backbone
produces zero detections.

Reference: docs/DESIGN.md; the earlier val validation session logged
`stats[0].any() == False` for all 400 LOD val images.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import numpy as np
import torch
from tqdm import tqdm

import tasks  # noqa: F401  bootstrap yolov3 sys.path
sys.path.insert(0, os.path.join(_ROOT, "tasks", "third_party", "yolov3"))

from engine.dataloader import create_dataloader_real_hr
from yolov3.models.experimental import attempt_load
from yolov3.utils.general import (
    TQDM_BAR_FORMAT, colorstr, non_max_suppression, scale_boxes, xywh2xyxy,
)
from yolov3.utils.metrics import ap_per_class, box_iou
from yolov3.utils.torch_utils import select_device


def _process_batch(detections, labels, iouv):
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


def main() -> None:
    device = select_device("")
    model = attempt_load("pretrained/yolov3.pt", device=device, inplace=True, fuse=True)
    model.eval()
    stride = max(int(model.stride.max()), 32)

    from yolov3.utils.general import check_dataset
    data = check_dataset("tasks/third_party/yolov3/data/lod.yaml")

    val_loader, _ = create_dataloader_real_hr(
        data["val"], 512, 1, stride, single_cls=False,
        hyp={"anchor_t": 4.0, "label_smoothing": 0.0},
        cache=False, rect=False, workers=1, pad=0.0, prefix=colorstr("val: "),
        add_noise=False, seed=0,
    )
    iouv = torch.linspace(0.5, 0.95, 10, device=device)
    stats: list = []

    with torch.no_grad():
        for imgs, targets, paths, shapes, imgs_hr in tqdm(val_loader, desc="raw-YOLO", bar_format=TQDM_BAR_FORMAT):
            im = imgs.to(device).float()
            targets = targets.to(device)
            preds = model(im)
            preds = non_max_suppression(preds, 0.001, 0.6, labels=[], multi_label=True, agnostic=False, max_det=300)
            for si, pred in enumerate(preds):
                labels = targets[targets[:, 0] == si, 1:]
                nl, npr = labels.shape[0], pred.shape[0]
                correct = torch.zeros(npr, iouv.numel(), dtype=torch.bool, device=device)
                if npr == 0:
                    if nl:
                        stats.append((correct, *torch.zeros((2, 0), device=device), labels[:, 0]))
                    continue
                predn = pred.clone()
                scale_boxes(im[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])
                if nl:
                    tbox = xywh2xyxy(labels[:, 1:5])
                    scale_boxes(im[si].shape[1:], tbox, shapes[si][0], shapes[si][1])
                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)
                    correct = _process_batch(predn, labelsn, iouv)
                stats.append((correct, pred[:, 4], pred[:, 5], labels[:, 0]))

    if not stats:
        print("diagnostics/yolo_no_isp: PASS — no batches produced detections (as expected)")
        return
    cat = [torch.cat(x, 0).cpu().numpy() for x in zip(*stats)]
    n_correct = int(cat[0].any(axis=1).sum())
    print(f"diagnostics/yolo_no_isp: correct predictions on raw LOD = {n_correct}")
    if n_correct == 0:
        print("diagnostics/yolo_no_isp: PASS (no correct detections — mAP attribution intact)")
    else:
        tp, fp, p, r, f1, ap, ap_class = ap_per_class(*cat, plot=False, save_dir="/tmp", names=data["names"])
        print(f"diagnostics/yolo_no_isp: UNEXPECTED — mAP@0.5 = {ap[:, 0].mean():.4f}")
        print("                          expected zero; verify the val pipeline.")


if __name__ == "__main__":
    main()
