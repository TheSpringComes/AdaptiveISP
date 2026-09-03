"""diagnostics: inspect a Controller checkpoint's schema.

Prints keys, iter, operator list, and shape / norm summary for the
`controller_model` state_dict. Useful when a ckpt fails to load: shows
whether it is a V1 ckpt (`controller_model` key) or a pre-refactor
Agent ckpt (`agent_model` key), and whether the operator list matches
the current registry.
"""
from __future__ import annotations

import argparse
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt", type=str, help="path to DynamicISP_iter_*.pth")
    parser.add_argument("--sample-weights", type=int, default=3,
                        help="how many weight tensors to preview (0 to skip)")
    args = parser.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f"inspect_ckpt: not a file: {args.ckpt}", file=sys.stderr)
        sys.exit(1)

    size_mb = os.path.getsize(args.ckpt) / 1e6
    d = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    print(f"file:            {args.ckpt}")
    print(f"size:            {size_mb:.1f} MB")
    print(f"top-level keys:  {list(d.keys())}")

    if "controller_model" in d:
        sd = d["controller_model"]
        schema = "V1 (Controller)"
    elif "agent_model" in d:
        sd = d["agent_model"]
        schema = "pre-refactor (Agent, v0-baseline)"
    else:
        print(f"unknown schema — keys={list(d.keys())}", file=sys.stderr)
        sys.exit(1)
    print(f"schema:          {schema}")

    if "iter" in d:
        print(f"iter:            {d['iter']}")
    if "operators" in d:
        print(f"operators:       {d['operators']}")

    n_tensors = len(sd)
    n_params = sum(v.numel() for v in sd.values() if hasattr(v, "numel"))
    print(f"state_dict:      {n_tensors} tensors, {n_params / 1e6:.2f}M params")

    if args.sample_weights > 0:
        keys = list(sd.keys())[:args.sample_weights]
        print("sample weights:")
        for k in keys:
            v = sd[k]
            if hasattr(v, "shape"):
                print(f"  {k:60s}  shape={tuple(v.shape)}  norm={v.float().norm().item():.4f}")
            else:
                print(f"  {k:60s}  (non-tensor: {type(v).__name__})")


if __name__ == "__main__":
    main()
