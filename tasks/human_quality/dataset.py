"""FiveK cache dataset for the Human Quality task.

Reads `.npz` files produced by the FiveK pre-processing pipeline:

  raw    : (4, H/2, W/2) float16 in [0, 1]  — Bayer-packed 4-plane RGGB
  target : (3, H,   W)   float16 in [0, 1]  — Expert C sRGB

Returns pairs `(image, target)` both in `(3, H/2, W/2) float32` — RAW is
demosaic-averaged (R, avg(G1,G2), B) and target is 2× down-sampled so
resolutions match. This matches what AdaptiveISP operators consume
(3-channel `[0, 1]` NCHW).

Split lists (one absolute path per line):
  train_expert_c.txt  — 4894 files
  val_expert_c.txt    — 100 files
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


def _bayer4_to_rgb3(raw4: np.ndarray) -> np.ndarray:
    """(4, H, W) RGGB -> (3, H, W) by averaging the two green planes."""
    r = raw4[0]
    g = 0.5 * (raw4[1] + raw4[2])
    b = raw4[3]
    return np.stack([r, g, b], axis=0)


def _downsample_target(tgt3: torch.Tensor, factor: int) -> torch.Tensor:
    """(3, H, W) -> (3, H/factor, W/factor) via average pooling."""
    return F.avg_pool2d(tgt3.unsqueeze(0), kernel_size=factor, stride=factor).squeeze(0)


class FiveKDataset(Dataset):
    """MIT-Adobe FiveK Expert-C cache reader.

    Each sample: `(image, target)` — both `(3, H/2, W/2) float32` in `[0, 1]`.
    """

    def __init__(
        self,
        list_file: str,
        cache_dir: Optional[str] = None,
        imgsz: Optional[int] = None,
    ) -> None:
        self.list_file = str(list_file)
        with open(self.list_file) as fh:
            self.paths = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        if cache_dir is not None:
            # Allow overriding the on-disk root, useful if the .txt uses paths that
            # were built on a different machine. Fall back to as-given if the
            # relocated file also doesn't exist.
            cache_dir = str(cache_dir)
            for i, p in enumerate(self.paths):
                candidate = os.path.join(cache_dir, os.path.basename(p))
                if os.path.exists(candidate):
                    self.paths[i] = candidate
        self.imgsz = imgsz  # optional: resize to (imgsz, imgsz) after demosaic

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        data = np.load(path)
        raw4 = data["raw"].astype(np.float32)      # (4, H/2, W/2)
        target3 = data["target"].astype(np.float32)   # (3, H, W)

        rgb3 = _bayer4_to_rgb3(raw4)                # (3, H/2, W/2)
        image = torch.from_numpy(rgb3).clamp_(0.0, 1.0)
        target = torch.from_numpy(target3).clamp_(0.0, 1.0)

        # Match resolutions: target is 2x the raw's spatial size after Bayer pack.
        h, w = image.shape[-2], image.shape[-1]
        th, tw = target.shape[-2], target.shape[-1]
        if (th, tw) != (h, w):
            factor_h = th // max(h, 1)
            factor_w = tw // max(w, 1)
            factor = max(factor_h, factor_w, 1)
            target = _downsample_target(target, factor)
            # If not exactly divisible, crop to raw's grid.
            target = target[:, :h, :w]

        if self.imgsz is not None:
            image = _resize(image, self.imgsz)
            target = _resize(target, self.imgsz)

        return image, target


def _resize(x: torch.Tensor, imgsz: int) -> torch.Tensor:
    """Center-crop-then-resize `(3, H, W)` to `(3, imgsz, imgsz)`."""
    _, h, w = x.shape
    s = min(h, w)
    top = (h - s) // 2
    left = (w - s) // 2
    x = x[:, top:top + s, left:left + s]
    if s != imgsz:
        x = F.interpolate(x.unsqueeze(0), size=(imgsz, imgsz),
                          mode="bilinear", align_corners=False).squeeze(0)
    return x


def collate_fivek(batch: list[tuple[torch.Tensor, torch.Tensor]]
                  ) -> tuple[torch.Tensor, torch.Tensor]:
    """Stack into `(B, 3, H, W)`; assumes all samples have the same spatial
    size (either `imgsz` was set, or the raw resolutions match).
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    return images, targets


__all__ = ["FiveKDataset", "collate_fivek"]
