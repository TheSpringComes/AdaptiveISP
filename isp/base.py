"""ISP Operator base class + ParameterSpec + shared math utilities.

`ISPOperator.apply(img, params) -> img` is a pure image transform; no fc
heads, no masking, no state — those live in the Controller. `ParameterSpec`
describes the physical parameter range plus a `regressor` callable that
maps raw NN features to that range.

Math helpers (`tanh_range`, `rgb2lum`, etc.) live here so that per-operator
files at `isp/operators/*.py` can `from isp.base import ...` without an
extra import layer.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Callable, ClassVar

import math
import numpy as np
import torch
import torch.nn as nn


# ------------------------------ shared math utils ---------------------------

def tanh01(x: torch.Tensor) -> torch.Tensor:
    return torch.tanh(x) * 0.5 + 0.5


def tanh_range(left: float, right: float, initial: float | None = None) -> Callable[[torch.Tensor], torch.Tensor]:
    if initial is not None:
        bias = math.atanh(2 * (initial - left) / (right - left) - 1)
    else:
        bias = 0.0

    def _act(x: torch.Tensor) -> torch.Tensor:
        return tanh01(x + bias) * (right - left) + left

    return _act


def rgb2lum(image: torch.Tensor) -> torch.Tensor:
    """NCHW -> N1HW luminance (Rec.601-ish weights)."""
    lum = 0.27 * image[:, 0, :, :] + 0.67 * image[:, 1, :, :] + 0.06 * image[:, 2, :, :]
    return lum[:, None, :, :]


def lerp(a: torch.Tensor, b: torch.Tensor, l: torch.Tensor | float) -> torch.Tensor:
    return (1 - l) * a + l * b


def rgb2hsv(image: torch.Tensor) -> torch.Tensor:
    """NCHW RGB in [0,1) -> NCHW HSV in [0,1)."""
    _eps = 1e-8
    hue = torch.zeros((image.shape[0], image.shape[2], image.shape[3]),
                      dtype=image.dtype, device=image.device)
    max_c = image.max(1)[0]
    min_c = image.min(1)[0]
    diff = max_c - min_c + _eps

    m_b = image[:, 2] == max_c
    m_g = image[:, 1] == max_c
    m_r = image[:, 0] == max_c

    hue[m_b] = 4.0 + ((image[:, 0] - image[:, 1]) / diff)[m_b]
    hue[m_g] = 2.0 + ((image[:, 2] - image[:, 0]) / diff)[m_g]
    hue[m_r] = (0.0 + ((image[:, 1] - image[:, 2]) / diff)[m_r]) % 6

    hue[min_c == max_c] = 0.0
    hue = hue / 6

    sat = (max_c - min_c) / (max_c + _eps)
    sat[max_c == 0] = 0
    val = max_c

    return torch.stack([hue, sat, val], dim=1)


def hsv2rgb(hsv: torch.Tensor) -> torch.Tensor:
    h, s, v = hsv[:, 0, :, :], hsv[:, 1, :, :], hsv[:, 2, :, :]
    h = h % 1
    s = torch.clamp(s, 0, 1)
    v = torch.clamp(v, 0, 1)

    r = torch.zeros_like(h)
    g = torch.zeros_like(h)
    b = torch.zeros_like(h)

    hi = torch.floor(h * 6)
    f = h * 6 - hi
    p = v * (1 - s)
    q = v * (1 - (f * s))
    t = v * (1 - ((1 - f) * s))

    for k, (rr, gg, bb) in enumerate([
        (v, t, p), (q, v, p), (p, v, t),
        (p, q, v), (t, p, v), (v, p, q),
    ]):
        m = hi == k
        r[m] = rr[m]; g[m] = gg[m]; b[m] = bb[m]

    return torch.stack([r, g, b], dim=1)


# ------------------------------ ParameterSpec ------------------------------

@dataclass(frozen=True)
class ParameterSpec:
    """Physical parameter range for one ISP operator.

    dim: number of scalar parameters the operator consumes per batch element.
    low, high: physical bounds (scalar or per-dim). For informational /
        Search Space use — the operator does NOT enforce these; the
        `regressor` already produces values in-range.
    regressor: maps raw feature tensor (shape [B, dim]) to physical params
        (same shape, in-range). Controller uses this to bound its output.
    description: short human label (e.g. "EV in stops").
    """
    dim: int
    low: float | tuple
    high: float | tuple
    regressor: Callable[[torch.Tensor], torch.Tensor]
    description: str = ""


# ------------------------------ ISPOperator base ---------------------------

class ISPOperator(nn.Module, ABC):
    """Base class for all ISP operators.

    Subclasses declare `name` (registry key), `short_name` (legacy display),
    `spec` (ParameterSpec), and `runtime_cost`. They implement
    `apply(img, params)`.
    """
    name: ClassVar[str] = ""
    short_name: ClassVar[str] = ""
    spec: ClassVar[ParameterSpec]
    runtime_cost: ClassVar[float] = 1.0

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def apply(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        """Apply this operator. img: NCHW in [0,1]. params: N,dim (physical range)."""

    def forward(self, img: torch.Tensor, params: torch.Tensor) -> torch.Tensor:
        return self.apply(img, params)


__all__ = [
    "ISPOperator", "ParameterSpec",
    "tanh01", "tanh_range", "rgb2lum", "rgb2hsv", "hsv2rgb", "lerp",
]
