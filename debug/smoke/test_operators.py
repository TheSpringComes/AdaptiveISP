"""smoke: every ISP operator applies cleanly on a dummy batch.

Checks shape, no NaN, in-range params via each op's regressor.
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

import tasks  # noqa: F401
from isp.registry import CANONICAL_ORDER, OPERATORS, build_operator


def test_operators() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img = torch.rand(2, 3, 32, 32, device=device) * 0.5 + 0.25   # avoid 0/1 saturation

    # Extreme raw features to test regressor range clipping.
    raw = torch.randn(4, 16, device=device) * 5

    for name in CANONICAL_ORDER:
        op = build_operator(name).to(device)
        spec = op.spec

        # Regressor produces in-range physical params.
        r = spec.regressor(raw[:, :spec.dim])
        r_flat = r.reshape(-1)
        assert not torch.isnan(r_flat).any(), f"{name}: regressor produced NaN"
        # low/high may be scalar or tuple; wnb+whitebalance use empirical envelopes.
        if isinstance(spec.low, (int, float)) and isinstance(spec.high, (int, float)):
            assert r_flat.min() >= spec.low - 1e-3, f"{name}: below low ({r_flat.min():.4f} < {spec.low})"
            assert r_flat.max() <= spec.high + 1e-3, f"{name}: above high ({r_flat.max():.4f} > {spec.high})"

        # apply() shape + no NaN.
        # Use the first 2 rows of the regressor output as physical params.
        physical = r[:2] if r.dim() == 2 else r[:2].reshape(2, -1)
        out = op.apply(img, physical)
        assert out.shape == img.shape, f"{name}: shape drift {out.shape} vs {img.shape}"
        assert not torch.isnan(out).any(), f"{name}: NaN in output"


if __name__ == "__main__":
    test_operators()
    print(f"smoke/test_operators: PASS  ({len(CANONICAL_ORDER)} operators)")
