"""Lazy shared loader for Samsung Modular Neural ISP backend networks.

Four backend nets are exposed on demand and cached at module scope:

  - denoiser        NAFNet (RAW-domain generic denoiser)
  - photofinishing  PhotofinishingModule (holds gain/gtm/chroma/gamma sub-nets)
  - awb             c5_model.IllumEstimator (cross-camera illuminant estimator)
  - detail          NAFNet (sRGB detail-enhancement)

All nets are `.eval()` and `requires_grad_(False)` — AdaptiveISP training
does inference only. Wrappers share these via `get_backend()`.

Import mechanics: Samsung modules use `from utils.constants import *` and
similar absolute-name imports rooted at `third_party/modular_neural_isp/`,
so this module puts that directory on `sys.path` before importing them.
The vendored yolov3 code ALSO ships a top-level `utils/` package, so a
naive `import photofinishing.photofinishing_model` after yolov3 has been
imported picks up yolov3's `utils` and fails (`No module named
'utils.constants'`). `_samsung_import` swaps `sys.modules['utils*']`
entries around each Samsung load so both trees coexist.
"""
from __future__ import annotations

import contextlib
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import torch


_THIRD_PARTY_ROOT = (
    Path(__file__).resolve().parent.parent.parent / "third_party" / "modular_neural_isp"
)


def _ensure_sys_path() -> None:
    p = str(_THIRD_PARTY_ROOT)
    if p not in sys.path:
        sys.path.insert(0, p)


@contextlib.contextmanager
def _samsung_import() -> Iterator[None]:
    """Isolate Samsung imports from other trees that ship a `utils/` package.

    Snapshots and removes any `utils*` entries from `sys.modules`, prepends
    Samsung's third-party root to `sys.path`, yields, then restores the
    snapshot. Modules Samsung imports (e.g. `utils.constants`,
    `photofinishing.photofinishing_model`) stay cached under their real
    names on `sys.modules`; only the yolov3 shadow is swapped out.
    """
    _ensure_sys_path()
    saved: dict[str, object] = {}
    for key in list(sys.modules):
        if key == "utils" or key.startswith("utils."):
            saved[key] = sys.modules.pop(key)
    try:
        yield
    finally:
        # Remove any samsung-installed utils entries and restore yolov3's.
        for key in list(sys.modules):
            if key == "utils" or key.startswith("utils."):
                del sys.modules[key]
        sys.modules.update(saved)


# ------------------------------ paths (V1 defaults) --------------------------

DEFAULT_DENOISE_MODEL = _THIRD_PARTY_ROOT / "denoising" / "models" / "generic_lite.pth"
DEFAULT_DENOISE_CONFIG = _THIRD_PARTY_ROOT / "denoising" / "configs" / "lite.json"

DEFAULT_ENHANCE_MODEL = (
    _THIRD_PARTY_ROOT / "enhancement" / "models" / "enhancement_s24-style-0.pth"
)
DEFAULT_ENHANCE_CONFIG = (
    _THIRD_PARTY_ROOT / "enhancement" / "configs" / "enhancement_s24-style-0.json"
)

DEFAULT_PS_MODEL = (
    _THIRD_PARTY_ROOT / "photofinishing" / "models" / "photofinishing_s24-style-0.pth"
)
DEFAULT_PS_CONFIG = (
    _THIRD_PARTY_ROOT / "photofinishing" / "config" / "photofinishing_s24-style-0.json"
)

DEFAULT_AWB_MODEL = (
    _THIRD_PARTY_ROOT / "awb_ccm" / "models" / "model-c5_single_encoder_neutral.pth"
)


# ------------------------------ backend singleton ---------------------------

@dataclass
class SamsungBackend:
    denoiser: Optional[torch.nn.Module] = None
    detail: Optional[torch.nn.Module] = None
    photofinishing: Optional[torch.nn.Module] = None
    awb: Optional[torch.nn.Module] = None


_backend: Optional[SamsungBackend] = None
_backend_device: Optional[torch.device] = None


def _freeze(mod: torch.nn.Module) -> torch.nn.Module:
    mod.eval()
    for p in mod.parameters():
        p.requires_grad_(False)
    return mod


def _load_nafnet(model_path: Path, config_path: Path, device: torch.device) -> torch.nn.Module:
    with _samsung_import():
        from denoising.nafnet_arch import NAFNet  # type: ignore

        with open(config_path) as fh:
            cfg = json.load(fh)
        net = NAFNet(
            width=cfg["width"],
            middle_block_num=cfg["middle_block_num"],
            encoder_block_nums=cfg["encoder_block_nums"],
            decoder_block_nums=cfg["decoder_block_nums"],
        )
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.to(device)
    return _freeze(net)


def _load_photofinishing(model_path: Path, config_path: Path, device: torch.device) -> torch.nn.Module:
    with _samsung_import():
        from photofinishing.photofinishing_model import PhotofinishingModule  # type: ignore

        with open(config_path) as fh:
            cfg = json.load(fh)
        ps = PhotofinishingModule(device=device, use_3d_lut=cfg.get("use_3d_lut", False))
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    ps.load_state_dict(state)
    ps.update_device(device)
    return _freeze(ps)


def _load_awb(model_path: Path, device: torch.device) -> torch.nn.Module:
    with _samsung_import():
        from awb_ccm.c5_model import IllumEstimator  # type: ignore

        net = IllumEstimator(device=device)
    state = torch.load(str(model_path), map_location=device, weights_only=True)
    net.load_state_dict(state)
    net.to(device)
    return _freeze(net)


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_backend(device: Optional[torch.device] = None) -> SamsungBackend:
    """Return the shared backend, loading nets lazily on first request.

    Passing a different `device` on a later call re-hosts already-loaded
    nets onto that device but does not reload weights.
    """
    global _backend, _backend_device
    dev = device or _default_device()
    if _backend is None:
        _backend = SamsungBackend()
        _backend_device = dev
    if dev != _backend_device and _backend_device is not None:
        _move_backend(_backend, dev)
        _backend_device = dev
    return _backend


def _move_backend(backend: SamsungBackend, device: torch.device) -> None:
    for name in ("denoiser", "detail", "awb"):
        mod = getattr(backend, name)
        if mod is not None:
            mod.to(device)
    if backend.photofinishing is not None:
        backend.photofinishing.update_device(device)


# ------------------------------ getters (lazy) ------------------------------

def get_denoiser(device: Optional[torch.device] = None) -> torch.nn.Module:
    b = get_backend(device)
    if b.denoiser is None:
        b.denoiser = _load_nafnet(DEFAULT_DENOISE_MODEL, DEFAULT_DENOISE_CONFIG, _backend_device)
    return b.denoiser


def get_detail(device: Optional[torch.device] = None) -> torch.nn.Module:
    b = get_backend(device)
    if b.detail is None:
        b.detail = _load_nafnet(DEFAULT_ENHANCE_MODEL, DEFAULT_ENHANCE_CONFIG, _backend_device)
    return b.detail


def get_photofinishing(device: Optional[torch.device] = None) -> torch.nn.Module:
    b = get_backend(device)
    if b.photofinishing is None:
        b.photofinishing = _load_photofinishing(DEFAULT_PS_MODEL, DEFAULT_PS_CONFIG, _backend_device)
    return b.photofinishing


def get_awb(device: Optional[torch.device] = None) -> torch.nn.Module:
    b = get_backend(device)
    if b.awb is None:
        b.awb = _load_awb(DEFAULT_AWB_MODEL, _backend_device)
    return b.awb


__all__ = [
    "SamsungBackend",
    "get_backend",
    "get_denoiser",
    "get_detail",
    "get_photofinishing",
    "get_awb",
    "DEFAULT_DENOISE_MODEL",
    "DEFAULT_ENHANCE_MODEL",
    "DEFAULT_PS_MODEL",
    "DEFAULT_AWB_MODEL",
]
