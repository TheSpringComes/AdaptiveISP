"""Neural AWB wrapper (Samsung c5 cross-camera illuminant estimator).

The CC-AWB model consumes 2D chroma histograms + edge histograms + uv
coords packed as a (1, 4, 48, 48) tensor. Histograms are computed with
NumPy from the input image (no learnable path), then fed through the
frozen estimator to get an illuminant `(1, 3)`. That is applied as a
per-channel WB gain `gain = G / illum` (i.e. Green channel anchored),
matching Samsung's `raw_to_lsrgb` step but without the CCM.

The histogram / edge computations run under `torch.no_grad()` and do
not need gradient — the wrapper's `apply` (from NeuralISPOperator) still
gets gradient through the alpha blend.
"""
from __future__ import annotations

import sys

import numpy as np
import torch

from isp.learned.base import NeuralISPOperator
from isp.learned.samsung_modular.backend import (
    DEFAULT_AWB_MODEL,
    _samsung_import,
    get_awb,
)
from isp.registry import register


_HIST_BINS = 48
_TARGET_SIZE = (256, 384)  # (H, W) — matches PipeLine._cc_awb_config

# Lazily-loaded, cached references to Samsung numpy helpers. Loading them
# lazily (a) matches the shared backend's on-demand style and (b) lets
# `_samsung_import` swap yolov3's `utils/` package aside so Samsung's
# `utils.img_utils` resolves correctly.
_compute_edges = None
_imresize = None


def _load_helpers() -> None:
    global _compute_edges, _imresize
    if _compute_edges is not None:
        return
    with _samsung_import():
        from utils.img_utils import compute_edges, imresize  # type: ignore
    _compute_edges = compute_edges
    _imresize = imresize


def _build_hist_stats(rgb_np: np.ndarray, model) -> np.ndarray:
    """Return (H_bins, W_bins, 4) stack of {chroma-hist, edge-hist, u-coord, v-coord}."""
    _load_helpers()

    img = _imresize(img=rgb_np, height=_TARGET_SIZE[0], width=_TARGET_SIZE[1])
    hist = np.zeros((_HIST_BINS, _HIST_BINS, 4), dtype=np.float32)

    chroma_rgb, colors_rgb = model.get_hist_colors(img)
    hist[..., 0] = model.compute_histogram(chroma_rgb, rgb=colors_rgb, bins=_HIST_BINS)

    edge_img = _compute_edges(img)
    chroma_e, colors_e = model.get_hist_colors(edge_img)
    hist[..., 1] = model.compute_histogram(chroma_e, rgb=colors_e, bins=_HIST_BINS)

    u_coord, v_coord = model.get_uv_coords(bins=_HIST_BINS)
    hist[..., 2] = u_coord
    hist[..., 3] = v_coord
    return hist


@register("n_awb")
class NeuralAWB(NeuralISPOperator):
    short_name = "nWB"
    input_domain = "raw_linear"
    output_domain = "raw_linear"
    checkpoint = str(DEFAULT_AWB_MODEL)
    runtime_cost = 8.0

    def _forward_neural(self, img: torch.Tensor) -> torch.Tensor:
        model = get_awb(device=img.device)
        b = img.shape[0]
        gains = torch.empty((b, 3), device=img.device, dtype=img.dtype)
        img_np = img.detach().float().cpu().numpy()  # (B, 3, H, W)
        for i in range(b):
            rgb_hwc = np.transpose(img_np[i], (1, 2, 0))
            hist_hwc = _build_hist_stats(rgb_hwc, model)
            hist_chw = np.transpose(hist_hwc, (2, 0, 1))[None, ...]  # (1, 4, H_bins, W_bins)
            hist_t = torch.from_numpy(hist_chw).to(device=img.device, dtype=img.dtype)
            illum = model(hist_t, inference=True)  # (1, 3)
            gains[i] = illum[0]

        # Green-anchored per-channel gain, matching Samsung's raw_to_lsrgb.
        wb_gain = gains[:, 1:2] / (gains + 1e-6)  # (B, 3)
        out = img * wb_gain.view(b, 3, 1, 1)
        return out.clamp(0.0, 1.0)
