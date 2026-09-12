"""smoke: Configurable Front ISP — registry, build paths, backward compat.

Covers:
  - identity / canonical / infinite_isp / modular_neural_isp are registered
  - build_front_isp honors enabled:false and type:none → IdentityFrontISP
  - canonical front ISP output shape/range on a synthetic batch
  - build_front_isp_from_cfg: front_isp section, legacy canonical_backbone
    fallback, and default (neither key) → identity
  - wrapper types fail with actionable errors when the third-party repo is
    absent (import-time safety: importing front_isp never fails)
"""
from __future__ import annotations

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

from engine.util import Dict, load_config


def test_front_isp() -> None:
    import front_isp as fi
    from front_isp.identity import IdentityFrontISP

    # --- registration ---
    registered = fi.list_front_isps()
    assert set(registered) >= {'none', 'canonical', 'infinite_isp',
                               'modular_neural_isp'}, registered

    # --- build: disabled / none → identity ---
    assert isinstance(fi.build_front_isp({'enabled': False, 'type': 'canonical'}),
                      IdentityFrontISP)
    assert isinstance(fi.build_front_isp({'type': 'none'}), IdentityFrontISP)
    assert isinstance(fi.build_front_isp({}), IdentityFrontISP)

    # --- build: unknown type raises ---
    try:
        fi.build_front_isp({'type': 'nope'})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # --- canonical: shape / range / non-trivial ---
    canonical = fi.build_front_isp({'type': 'canonical'})
    x = torch.rand(2, 3, 32, 48)
    y = canonical(x)
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0
    # a linear image is not idempotent under AWB+CCM+GTM+gamma (it changes)
    assert not torch.allclose(y, x)

    # --- cfg-level: front_isp section ---
    cfg = Dict({'front_isp': {'enabled': True, 'type': 'canonical'}})
    assert type(fi.build_front_isp_from_cfg(cfg)).__name__ == 'CanonicalBackbone'
    cfg = Dict({'front_isp': {'enabled': True, 'type': 'none'}})
    assert isinstance(fi.build_front_isp_from_cfg(cfg), IdentityFrontISP)

    # --- cfg-level: legacy canonical_backbone fallback ---
    cfg = Dict({'canonical_backbone': {'enabled': True}})
    assert type(fi.build_front_isp_from_cfg(cfg)).__name__ == 'CanonicalBackbone'
    cfg = Dict({'canonical_backbone': {'enabled': False}})
    assert isinstance(fi.build_front_isp_from_cfg(cfg), IdentityFrontISP)

    # --- cfg-level: nothing set → identity ---
    cfg = Dict({})
    assert isinstance(fi.build_front_isp_from_cfg(cfg), IdentityFrontISP)

    # --- cfg-level: front_isp wins over legacy canonical_backbone ---
    cfg = Dict({'front_isp': {'enabled': True, 'type': 'none'},
                'canonical_backbone': {'enabled': True}})
    assert isinstance(fi.build_front_isp_from_cfg(cfg), IdentityFrontISP)

    # --- wrapper types: repo absent → actionable error at build time ---
    for t in ('infinite_isp', 'modular_neural_isp'):
        try:
            fi.build_front_isp({'type': t, t: {'repo_path': '/nonexistent'}})
            raise AssertionError(f"expected FileNotFoundError for {t}")
        except (FileNotFoundError, NotImplementedError, ImportError):
            pass

    # --- real config file: front_isp section parses to identity ---
    cfg = load_config('configs/adaptiveisp_human.yaml')
    assert isinstance(fi.build_front_isp_from_cfg(cfg), IdentityFrontISP)


if __name__ == "__main__":
    test_front_isp()
    print("smoke/test_front_isp: PASS")
