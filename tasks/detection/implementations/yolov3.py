"""YOLOv3 detection implementation.

Wraps the vendored yolov3 (`tasks/third_party/yolov3/`) Model + ComputeLoss
into the `Task` interface. Documented metric keys:

    detect_loss:         [B, 1] per-sample total loss, clipped 0..1
    detect_retouch_loss: alias for detect_loss (name preserved from
                         original train.py for backward-compat logging)
    box_loss, obj_loss, cls_loss: mean loss components (scalars, .detach)
"""
from __future__ import annotations

import os
import sys
from typing import Any

import torch
import torch.nn as nn

# Make the vendored yolov3 subpackage importable via `from yolov3.*`.
_HERE = os.path.dirname(os.path.abspath(__file__))
_THIRD_PARTY = os.path.abspath(os.path.join(_HERE, "..", "..", "third_party"))
if _THIRD_PARTY not in sys.path:
    sys.path.insert(0, _THIRD_PARTY)

from tasks.base import Task, TaskMetrics


class YOLOv3Detection(Task):
    name = "yolov3-detection"

    def __init__(
        self,
        weights: str,
        yolo_cfg: str,
        hyp: dict,
        nc: int,
        imgsz: int,
        device: torch.device,
        *,
        detect_loss_weight: float = 1.0,
    ) -> None:
        from yolov3.models.yolo import Model
        from yolov3.utils.downloads import attempt_download
        from yolov3.utils.general import intersect_dicts
        from yolov3.utils.loss import ComputeLoss, ComputeLossBatch
        from yolov3.utils.torch_utils import torch_distributed_zero_first

        with torch_distributed_zero_first(-1):
            weights_path = attempt_download(weights)
        ckpt = torch.load(weights_path, map_location='cpu', weights_only=False)

        model = Model(
            yolo_cfg or ckpt['model'].yaml,
            ch=3, nc=nc, anchors=hyp.get('anchors'),
        ).to(device)

        exclude = ['anchor'] if (yolo_cfg or hyp.get('anchors')) else []
        csd = ckpt['model'].float().state_dict()
        csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)
        model.load_state_dict(csd, strict=False)

        nl = model.model[-1].nl
        hyp = dict(hyp)
        hyp['box'] *= 3 / nl
        hyp['cls'] *= nc / 80 * 3 / nl
        hyp['obj'] *= (imgsz / 640) ** 2 * 3 / nl
        hyp['label_smoothing'] = 0.0
        model.nc = nc
        model.hyp = hyp

        self.model = model
        self.device = device
        self.hyp = hyp
        self.nc = nc
        self.imgsz = imgsz
        self.detect_loss_weight = float(detect_loss_weight)
        self.gs = max(int(model.stride.max()), 32)

        self._freeze_and_eval_bn()
        self._compute_loss = ComputeLoss(model)
        self._compute_loss_batch = ComputeLossBatch(model, reduction='mean')

    def attach_class_weights(self, labels, nc: int) -> None:
        from yolov3.utils.general import labels_to_class_weights
        self.model.class_weights = labels_to_class_weights(labels, nc).to(self.device) * nc

    def attach_names(self, names) -> None:
        self.model.names = names

    # -----------------------------------------------------------------
    # Static helpers that isolate yolov3 utilities from the rest of the
    # framework. Engine / Controller / Reward / Pipeline call these
    # instead of importing yolov3.* directly.
    # -----------------------------------------------------------------

    @staticmethod
    def parse_data_cfg(yaml_path: str) -> dict:
        """Parse a yolov3-style dataset yaml. Returns dict with keys train/val/test/names/nc/path."""
        from yolov3.utils.general import check_dataset
        return check_dataset(yaml_path)

    def align_imgsz(self, imgsz: int, floor_factor: int = 2) -> int:
        """Round imgsz to a multiple of the detector's grid stride."""
        from yolov3.utils.general import check_img_size
        return check_img_size(imgsz, self.gs, floor=self.gs * floor_factor)

    def train(self) -> "YOLOv3Detection":
        self.model.train()
        self._freeze_and_eval_bn()
        return self

    def eval(self) -> "YOLOv3Detection":
        self.model.eval()
        return self

    def _freeze_and_eval_bn(self) -> None:
        for p in self.model.parameters():
            p.requires_grad = False
        for m in self.model.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def to(self, device) -> "YOLOv3Detection":
        self.model.to(device)
        self.device = device
        return self

    def compute_metrics(
        self,
        images: torch.Tensor,
        targets: Any,
        **kwargs,
    ) -> TaskMetrics:
        preds = self.model(images)
        b = preds[0].shape[0]

        if isinstance(targets, (list, tuple)):
            targets_per_sample = list(targets)
        elif isinstance(targets, torch.Tensor):
            targets_per_sample = [targets[targets[:, 0] == i].clone() for i in range(b)]
        else:
            raise TypeError(f"YOLOv3Detection: unsupported targets type {type(targets)}")

        per_sample_loss, per_sample_components = self._per_sample_loss(preds, targets_per_sample)
        per_sample_loss = torch.clip(per_sample_loss * self.detect_loss_weight, 0.0, 1.0)

        if isinstance(targets, torch.Tensor):
            _, mean_components = self._compute_loss(preds, targets.to(self.device))
        else:
            stacked = []
            for i, t in enumerate(targets_per_sample):
                if t.numel() == 0:
                    continue
                t = t.clone()
                t[:, 0] = i
                stacked.append(t)
            cat = torch.cat(stacked, dim=0) if stacked else torch.zeros((0, 6), device=self.device)
            _, mean_components = self._compute_loss(preds, cat.to(self.device))

        return TaskMetrics(
            values={
                'detect_loss':          per_sample_loss,
                'detect_retouch_loss':  per_sample_loss,
                'box_loss':             mean_components[0].detach(),
                'obj_loss':             mean_components[1].detach(),
                'cls_loss':             mean_components[2].detach(),
                'total_loss_components': per_sample_components,
            },
            extras={'preds': preds},
        )

    def _per_sample_loss(
        self,
        preds,
        targets_per_sample: list[torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b = preds[0].shape[0]
        device = self.device
        lboxs = torch.zeros((b, 1), device=device)
        lobjs = torch.zeros((b, 1), device=device)
        lclss = torch.zeros((b, 1), device=device)
        for i in range(b):
            pred_one = [p[i].unsqueeze(0).to(device) for p in preds]
            target_one = targets_per_sample[i]
            if isinstance(target_one, torch.Tensor) and target_one.numel() > 0:
                target_one = target_one.clone()
                target_one[:, 0] = 0
            elif not isinstance(target_one, torch.Tensor):
                target_one = torch.tensor(target_one, dtype=torch.float32, device=device)
                if target_one.numel() > 0:
                    target_one[:, 0] = 0
            lbox, lobj, lcls = self._compute_loss_batch(pred_one, target_one.to(device))
            lboxs[i] = lbox
            lobjs[i] = lobj
            lclss[i] = lcls
        return lboxs + lobjs + lclss, torch.cat((lboxs, lobjs, lclss), dim=0).detach()


__all__ = ["YOLOv3Detection"]
