"""Smoke test for the 7 Samsung neural ISP operator wrappers.

Verifies for each operator that:
  1. Import + registration succeeded.
  2. The backend network loads.
  3. Forward runs on a small batch, produces a NCHW tensor of the same
     spatial size and 3 channels, in [0, 1], with no NaN/Inf.
  4. alpha = 0 -> output == input  (identity when strength is zero).
  5. alpha = 1 -> output == F(x)   (matches Samsung backend output).

Run from the AdaptiveISP repo root:

    python -m isp.learned.samsung_modular.tests.smoke_test
"""
from __future__ import annotations

import sys
import time

import torch

from isp.registry import OPERATORS, build_operator


NEURAL_OPS = ["n_denoise", "n_awb", "n_gain", "n_gtm", "n_chroma", "n_gamma", "n_detail"]


def _make_batch(device: torch.device, size: int = 96, batch: int = 2) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch, 3, size, size, device=device, dtype=torch.float32)


def _fmt_stats(x: torch.Tensor) -> str:
    return f"shape={tuple(x.shape)} min={x.min().item():.4f} max={x.max().item():.4f}"


def smoke_one(name: str, device: torch.device) -> None:
    print(f"\n== {name} ==")
    assert name in OPERATORS, f"{name} not registered. Registered: {sorted(OPERATORS)}"
    op = build_operator(name).to(device)

    img = _make_batch(device)
    b = img.shape[0]

    t0 = time.perf_counter()
    alpha0 = torch.zeros(b, 1, device=device)
    out0 = op(img, alpha0)
    dt0 = time.perf_counter() - t0

    assert out0.shape == img.shape, f"{name} shape mismatch: {out0.shape} vs {img.shape}"
    assert torch.isfinite(out0).all(), f"{name} produced NaN/Inf"
    assert torch.allclose(out0, img.clamp(0, 1), atol=1e-5), (
        f"{name} alpha=0 not identity (max diff = {(out0 - img.clamp(0,1)).abs().max().item():.2e})"
    )

    alpha1 = torch.ones(b, 1, device=device)
    t1 = time.perf_counter()
    out1 = op(img, alpha1)
    dt1 = time.perf_counter() - t1
    assert torch.isfinite(out1).all(), f"{name} alpha=1 produced NaN/Inf"
    assert 0.0 <= out1.min().item() and out1.max().item() <= 1.0 + 1e-4, (
        f"{name} alpha=1 out of [0,1]: {_fmt_stats(out1)}"
    )
    # alpha=1 should match F(x) exactly (up to input clamp).
    with torch.no_grad():
        fx = op._forward_neural(img.clamp(0, 1))
    diff_fx = (out1 - fx).abs().max().item()
    assert diff_fx < 1e-5, f"{name} alpha=1 does not match backend F(x); max diff {diff_fx:.2e}"

    delta = (out1 - img.clamp(0, 1)).abs().mean().item()
    print(f"  alpha=0 ok ({dt0*1000:.1f} ms); alpha=1 ok ({dt1*1000:.1f} ms); |F(x)-x| mean = {delta:.4f}")
    print(f"  in : {_fmt_stats(img)}")
    print(f"  out: {_fmt_stats(out1)}")


def main() -> int:
    import isp  # noqa: F401  populate registry via side-effect imports

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device = {device}")

    failures: list[tuple[str, str]] = []
    for name in NEURAL_OPS:
        try:
            smoke_one(name, device)
        except Exception as exc:  # noqa: BLE001
            failures.append((name, f"{type(exc).__name__}: {exc}"))
            print(f"  FAILED: {failures[-1][1]}")

    print("\n==================== summary ====================")
    ok = [n for n in NEURAL_OPS if n not in {f[0] for f in failures}]
    print(f"passed ({len(ok)}/{len(NEURAL_OPS)}): {ok}")
    if failures:
        print("failed:")
        for n, msg in failures:
            print(f"  - {n}: {msg}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
