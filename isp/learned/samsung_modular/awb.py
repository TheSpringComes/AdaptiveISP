"""Neural AWB wrapper (Samsung c5 cross-camera illuminant estimator).

The CC-AWB model consumes 2D chroma histograms + edge histograms + uv
coords packed as a (B, 4, 48, 48) tensor. Histograms are computed on
`img.device` via torch ops that mirror Samsung's numpy helpers
(`rgb_to_uv`, `get_hist_colors`, `compute_histogram`, `compute_edges`,
`imresize`, `get_uv_coords`) — see
`isp/third_party/modular_neural_isp/{awb_ccm/c5_model.py, utils/img_utils.py}`
for the numpy originals. `imresize` uses `F.interpolate(mode='bilinear',
align_corners=False)` in place of `cv2.INTER_LINEAR`; values differ by
sub-pixel bilinear kernel details but the downstream illuminant is
numerically consistent for well-exposed inputs.

The histogram / edge computations run under `torch.no_grad()` (inherited
from `NeuralISPOperator.apply`) and do not need gradient — the wrapper's
`apply` still gets gradient through the alpha blend downstream.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

from isp.learned.base import NeuralISPOperator
from isp.learned.samsung_modular.backend import (
    DEFAULT_AWB_MODEL,
    get_awb,
)
from isp.registry import register


_HIST_BINS = 48
_TARGET_SIZE = (256, 384)  # (H, W) — matches PipeLine._cc_awb_config
_HIST_BOUNDARY = (-2.85, 2.85)
_EPS = 1e-8  # matches third_party/modular_neural_isp/utils/constants.EPS


def _rgb_to_uv(rgb: torch.Tensor) -> torch.Tensor:
    """(..., 3) -> (..., 2) log-chroma. Matches utils.img_utils.rgb_to_uv."""
    log_rgb = torch.log(rgb + _EPS)
    u = log_rgb[..., 1] - log_rgb[..., 0]
    v = log_rgb[..., 1] - log_rgb[..., 2]
    return torch.stack([u, v], dim=-1)


def _compute_edges_torch(img: torch.Tensor) -> torch.Tensor:
    """(B, C, H, W) -> (B, C, H, W). Sum-of-|diff| over the 8-neighborhood
    with cv2.BORDER_REFLECT padding, divided by 8. Matches
    utils.img_utils.compute_edges.

    Note: cv2.BORDER_REFLECT with pad=1 duplicates the boundary pixel
    (`a|abcde|e`), which equals torch's `mode='replicate'` for pad=1.
    Torch's `mode='reflect'` is cv2.BORDER_REFLECT_101 (`b|abcde|d`) —
    a different mode and NOT what the numpy original uses.
    """
    pad = F.pad(img, [1, 1, 1, 1], mode="replicate")
    _, _, h, w = img.shape
    edges = torch.zeros_like(img)
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            if dx == 0 and dy == 0:
                continue
            shifted = pad[:, :, 1 + dx : 1 + dx + h, 1 + dy : 1 + dy + w]
            edges = edges + (img - shifted).abs()
    return edges / 8.0


def _uv_coords_torch(bins: int, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Match c5_model.get_uv_coords. Returns two (bins, bins) tensors in [0,1]."""
    lin = torch.arange(bins, device=device, dtype=dtype) / (bins - 1)
    u = lin.view(1, -1).expand(bins, -1).contiguous()   # u[i, j] = j / (bins-1)
    v = torch.flip(lin, dims=[0]).view(-1, 1).expand(-1, bins).contiguous()
    return u, v


def _chroma_histogram(img: torch.Tensor) -> torch.Tensor:
    """img: (B, 3, H, W) → (B, bins, bins). Reproduces c5_model.compute_histogram
    on the flattened image with the get_hist_colors mask (sum > EPS)."""
    b = img.shape[0]
    bins = _HIST_BINS
    rgb = img.permute(0, 2, 3, 1).reshape(b, -1, 3)          # (B, N, 3)
    valid_mask = (rgb.sum(dim=-1) > _EPS).to(img.dtype)       # (B, N)
    uv = _rgb_to_uv(rgb)                                      # (B, N, 2)

    hb0, hb1 = _HIST_BOUNDARY
    bin_eps = (hb1 - hb0) / (bins - 1)
    bins_u = torch.linspace(hb0, hb1, bins, device=img.device, dtype=img.dtype)
    bins_v = torch.flip(bins_u, dims=[0])

    # Original numpy sequence:
    #   diff[diff > eps] = 0;  diff[diff != 0] = 1
    # → mask = (|d| <= eps) AND (|d| > 0). Reproduce exactly.
    du = (uv[..., 0].unsqueeze(-1) - bins_u.view(1, 1, -1)).abs()   # (B, N, bins)
    dv = (uv[..., 1].unsqueeze(-1) - bins_v.view(1, 1, -1)).abs()   # (B, N, bins)
    mu = ((du <= bin_eps) & (du > 0)).to(img.dtype)
    mv = ((dv <= bin_eps) & (dv > 0)).to(img.dtype)

    intensity = rgb.pow(2).sum(dim=-1).sqrt() * valid_mask    # (B, N)
    mu = mu * valid_mask.unsqueeze(-1)
    mv = mv * valid_mask.unsqueeze(-1)

    # hist[b, i, j] = sum_n intensity[b, n] * mv[b, n, i] * mu[b, n, j]
    weighted_mv = mv * intensity.unsqueeze(-1)                # (B, N, bins)
    hist = torch.einsum("bni,bnj->bij", weighted_mv, mu)      # (B, bins, bins)

    norm = hist.sum(dim=(-1, -2), keepdim=True) + _EPS
    return torch.sqrt(hist / norm)


def _build_hist_stats_torch(img: torch.Tensor) -> torch.Tensor:
    """img: (B, 3, H, W) in [0,1]. Returns (B, 4, bins, bins) stack of
    {chroma-hist, edge-hist, u-coord, v-coord}."""
    if img.shape[-2:] != _TARGET_SIZE:
        img_r = F.interpolate(img, size=_TARGET_SIZE, mode="bilinear", align_corners=False)
    else:
        img_r = img

    hist_c = _chroma_histogram(img_r)
    edge = _compute_edges_torch(img_r)
    hist_e = _chroma_histogram(edge)

    u_coord, v_coord = _uv_coords_torch(_HIST_BINS, device=img.device, dtype=img.dtype)
    b = img.shape[0]
    u_b = u_coord.unsqueeze(0).expand(b, -1, -1)
    v_b = v_coord.unsqueeze(0).expand(b, -1, -1)
    return torch.stack([hist_c, hist_e, u_b, v_b], dim=1)


@register("n_awb")
class NeuralAWB(NeuralISPOperator):
    short_name = "nWB"
    input_domain = "raw_linear"
    output_domain = "raw_linear"
    checkpoint = str(DEFAULT_AWB_MODEL)
    runtime_cost = 8.0

    def _forward_neural(self, img: torch.Tensor) -> torch.Tensor:
        model = get_awb(device=img.device)
        # Samsung's chroma/edge helpers run in float32 for numerical parity.
        img_f = img.detach().float()
        hist_t = _build_hist_stats_torch(img_f).to(dtype=img.dtype)  # (B, 4, bins, bins)
        gains = model(hist_t, inference=True)                        # (B, 3)

        # Green-anchored per-channel gain, matching Samsung's raw_to_lsrgb.
        wb_gain = gains[:, 1:2] / (gains + 1e-6)                     # (B, 3)
        out = img * wb_gain.view(-1, 3, 1, 1)
        return out.clamp(0.0, 1.0)
