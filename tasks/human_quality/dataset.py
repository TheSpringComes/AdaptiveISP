"""FiveK cache dataset for the Human Quality task.

Reads `.npz` files produced by `tools/fivek_build_cache.py`:

  raw    : (4, H/2, W/2) float16 in [0, 1]  — Bayer-packed 4-plane
           R, G(code1), G(code3), B, EXIF-upright, active-crop normalized
           (per-CFA black level subtracted, white-level divided)
  target : (3, H,   W)   float16 in [0, 1]  — Expert C sRGB, EXIF-upright

Input adapter chain (see `front_isp/raw_adapter.py`):

    Dataset/Input Adapter → 数据格式解析 → Bayer reconstruction（按
    per-file CFA pattern 还原 full-res mosaic）→ demosaic（0.5*Malvar +
    0.5*Bilinear）→ canonical linear RGB → ISP

The ISP only ever receives standard 3-channel RGB — it never sees Bayer
or packed raw. Each sample returns image and target BOTH as
full-resolution `(3, H, W) float32` in `[0, 1]`; the target is no longer
avg-pooled 2x down to the Bayer-packed grid.

Both planes are already upright — no load-time rotation is needed or
performed. (The legacy `fix_rotation.json` mechanism is retired; the old
cache stored raw sensor-native while target was EXIF-corrected, which the
rebuilt cache fixes at generation time.)

CFA pattern: the cache packs planes by color code, but WHERE each plane
sits in the 2x2 CFA cell differs per camera (FiveK is ~74% RGGB, plus
BGGR/GBRG/GRBG — see `tools/fivek_cfa_scan.py`). `cfa_json`
(auto-discovered as `cfa_pattern.json` next to the cache_dir) maps
`<stem> → "RGGB"|"BGGR"|"GBRG"|"GRBG"` so mosaic reconstruction uses each
file's true sensor pattern instead of a hardcoded arrangement. Stems
missing from the map fall back to `cfa_default`.

Split lists (one absolute path per line):
  train_expert_c.txt  — 4894 files
  val_expert_c.txt   — 100 files

Bad-sample exclusion: `alignment_threshold` (default 0.5) filters the split
against `_alignment.json` inside the cache dir (produced by
`tools/fivek_scan_cache.py`: corr(RGB3(raw), pooled target) per file).
Files whose corr falls below the threshold are dropped at construction —
their Expert-C TIFFs mismatch their DNGs (legacy data-source errors), so
they can never form valid pairs. Files missing from the cache (e.g. the 42
X-Trans / mirrored-flip images skipped at rebuild time) are dropped the
same way. Set `alignment_threshold=0` to disable. `self.excluded` records
what was dropped and why, for logging.

Camera model (V3.1): `camera_json` (auto-discovered as `camera.json` next
to the cache_dir; produced by `tools/fivek_camera_metadata.py`) maps
`<stem> → "<Make> <Model>"` from the DNG EXIF. Samples are tagged with an
integer camera id (index into the sorted name list) so the camera-specific
Calibration front ISP can select its parameter row. A missing file is not
fatal — all samples get camera id 0.

When `return_camera=True` (default) `__getitem__` returns
`(image, target, camera_id)` and `collate_fivek` stacks a third `(B,)`
long tensor. This feeds `front_isp.calibration.camera_specific`.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from front_isp.raw_adapter import to_canonical_rgb


class FiveKDataset(Dataset):
    """MIT-Adobe FiveK Expert-C cache reader.

    Each sample: `(image, target, camera_id)` (default) — image/target are
    full-resolution `(3, H, W) float32` in `[0, 1]` (image = demosaiced
    linear camera RGB, target = Expert-C sRGB), camera_id is an int
    indexing `self.camera_names`. Pass `return_camera=False` for the
    legacy `(image, target)` 2-tuple.
    """

    def __init__(
        self,
        list_file: str,
        cache_dir: Optional[str] = None,
        imgsz: Optional[int] = None,
        camera_json: Optional[str] = None,
        return_camera: bool = True,
        alignment_threshold: float = 0.5,
        cfa_json: Optional[str] = None,
        cfa_default: str = "RGGB",
    ) -> None:
        self.list_file = str(list_file)
        with open(self.list_file) as fh:
            paths = [ln.strip() for ln in fh if ln.strip() and not ln.startswith("#")]
        if cache_dir is not None:
            cache_dir = str(cache_dir)
            # Allow overriding the on-disk root, useful if the .txt uses paths
            # that were built on a different machine.
            for i, p in enumerate(paths):
                candidate = os.path.join(cache_dir, os.path.basename(p))
                if os.path.exists(candidate):
                    paths[i] = candidate
        self.cache_dir = cache_dir
        self.imgsz = imgsz  # optional: resize to (imgsz, imgsz) after demosaic
        self.return_camera = bool(return_camera)

        # ---- Bad-sample + missing-file exclusion (see module docstring) ----
        self.alignment_threshold = float(alignment_threshold)
        self.excluded: list[tuple[str, str]] = []
        self.paths = self._filter_paths(paths)
        if self.excluded:
            n_align = sum(1 for _, why in self.excluded if why == "alignment")
            n_miss = sum(1 for _, why in self.excluded if why == "missing")
            print(f"FiveKDataset[{os.path.basename(self.list_file)}]: "
                  f"kept {len(self.paths)}/{len(paths)} "
                  f"(dropped {n_align} misaligned, {n_miss} not in cache)")

        # ---- V3.1 camera model tags: `<stem> → "<Make> <Model>"` ----
        # Sorted name list fixes the id ↔ name mapping; missing file → all
        # ids 0 (not fatal — the calibration table stays single-camera).
        if camera_json is None and cache_dir is not None:
            candidate = os.path.join(os.path.dirname(cache_dir), "camera.json")
            if os.path.exists(candidate):
                camera_json = candidate
        self.camera_names: list[str] = []
        self._camera_id: list[int] = []
        stem2name: dict[str, str] = {}
        if camera_json and os.path.exists(camera_json):
            try:
                with open(camera_json) as fh:
                    stem2name = {str(k): str(v) for k, v in json.load(fh).items()}
                self.camera_names = sorted(set(stem2name.values()))
                name2idx = {n: i for i, n in enumerate(self.camera_names)}
                self._camera_id = [
                    name2idx.get(stem2name.get(Path(p).stem, ""), 0)
                    for p in self.paths
                ]
            except Exception:
                self.camera_names, self._camera_id = [], []

        # ---- CFA pattern tags: `<stem> → "RGGB"|"BGGR"|"GBRG"|"GRBG"` ----
        # Auto-discovered next to the cache_dir (like camera.json); the
        # map covers the whole cache, `cfa_default` only covers stems the
        # map does not know about.
        if cfa_json is None and cache_dir is not None:
            candidate = os.path.join(os.path.dirname(cache_dir), "cfa_pattern.json")
            if os.path.exists(candidate):
                cfa_json = candidate
        self.cfa_default = str(cfa_default)
        self._cfa: list[str] = []
        if cfa_json and os.path.exists(cfa_json):
            try:
                with open(cfa_json) as fh:
                    stem2pat = {str(k): str(v) for k, v in json.load(fh).items()}
                self._cfa = [
                    stem2pat.get(Path(p).stem, self.cfa_default)
                    for p in self.paths
                ]
            except Exception:
                self._cfa = []
        if not self._cfa:
            # No per-file metadata: single-pattern fallback from config
            # (correct for uniform-CFA datasets, lossy for mixed FiveK).
            self._cfa = [self.cfa_default] * len(self.paths)

    # ------------------------- path filtering -------------------------

    def _filter_paths(self, paths: list[str]) -> list[str]:
        """Drop files missing from the cache and misaligned pairs.

        Alignment scores come from `_alignment.json` in the cache dir
        (auto-discovered; produced by `tools/fivek_scan_cache.py`). A file
        is dropped when its score is below `alignment_threshold` (0 disables
        the check) or when no `.npz` exists for it (X-Trans / mirrored
        images skipped at rebuild time).
        """
        scores: dict[str, float] = {}
        if self.cache_dir is not None and self.alignment_threshold > 0:
            candidate = os.path.join(self.cache_dir, "_alignment.json")
            if os.path.exists(candidate):
                try:
                    with open(candidate) as fh:
                        scores = {str(k): float(v) for k, v in json.load(fh).items()}
                except Exception:
                    scores = {}

        kept: list[str] = []
        for p in paths:
            stem = Path(p).stem
            if not os.path.exists(p):
                self.excluded.append((stem, "missing"))
                continue
            if scores and scores.get(stem, 1.0) < self.alignment_threshold:
                self.excluded.append((stem, "alignment"))
                continue
            kept.append(p)
        return kept

    def __len__(self) -> int:
        return len(self.paths)

    @property
    def n_cameras(self) -> int:
        """Number of distinct camera models seen in this split (≥1)."""
        return max(len(self.camera_names), 1)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        path = self.paths[idx]
        data = np.load(path)
        raw4 = data["raw"].astype(np.float32)      # (4, H/2, W/2)
        target3 = data["target"].astype(np.float32)   # (3, H, W)

        # Input adapter: packed 4-plane → Bayer reconstruction (per-file
        # CFA pattern) → demosaic → canonical linear RGB (3, 2h, 2w).
        rgb3 = to_canonical_rgb(raw4, pattern=self._cfa[idx])
        image = torch.from_numpy(rgb3).clamp_(0.0, 1.0)
        target = torch.from_numpy(target3).clamp_(0.0, 1.0)

        # Explicit spatial alignment: demosaic output is exactly 2x the
        # packed plane size; if the target deviates (odd 1px, future
        # datasets), center-crop both to the common size.
        if image.shape != target.shape:
            th, tw = min(image.shape[-2], target.shape[-2]), \
                min(image.shape[-1], target.shape[-1])
            image = _center_crop(image, th, tw)
            target = _center_crop(target, th, tw)

        if self.imgsz is not None:
            image = _resize(image, self.imgsz)
            target = _resize(target, self.imgsz)

        if self.return_camera:
            return image, target, self._camera_id[idx] if self._camera_id else 0
        return image, target


def _center_crop(x: torch.Tensor, th: int, tw: int) -> torch.Tensor:
    """Center-crop `(3, H, W)` to `(3, th, tw)`."""
    _, h, w = x.shape
    top, left = (h - th) // 2, (w - tw) // 2
    return x[:, top:top + th, left:left + tw]


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


def collate_fivek(batch: list[tuple]) -> tuple:
    """Stack into `(B, 3, H, W)` (+ `(B,)` camera ids when present).

    Assumes all samples have the same spatial size (either `imgsz` was set,
    or the raw resolutions match) and that every sample carries the same
    number of elements (camera ids on/off is a dataset-level flag).
    """
    images = torch.stack([b[0] for b in batch], dim=0)
    targets = torch.stack([b[1] for b in batch], dim=0)
    if len(batch[0]) > 2:
        cam_ids = torch.tensor([b[2] for b in batch], dtype=torch.long)
        return images, targets, cam_ids
    return images, targets


__all__ = ["FiveKDataset", "collate_fivek"]
