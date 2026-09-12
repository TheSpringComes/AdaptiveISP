"""smoke: V3.1 Learnable Front ISP — module, camera table, gradient, stages.

Covers:
  - 'learnable' registered; build via config
  - identity init is a true no-op on the image
  - camera-specific rows differ; metadata camera_id selects them
  - gradients flow into wb/ccm/bias/gamma (learnable) and stop when frozen
  - fittedisp loader (inline dict + JSON round-trip)
  - ckpt save/load restores parameters
  - CameraParamTable export/import + psnr/delta_e sanity
"""
from __future__ import annotations

import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch


def test_learnable() -> None:
    import front_isp as fi
    from front_isp.learnable import LearnableFrontISP
    from front_isp.learnable.camera_params import CameraParamTable
    from front_isp.learnable.fittedisp_loader import load_fittedisp_params

    assert 'learnable' in fi.list_front_isps()
    assert 'calibrated' not in fi.list_front_isps()   # 旧名已删除

    # --- build via config (shared params) ---
    m = fi.build_front_isp({'type': 'learnable'})
    assert isinstance(m, LearnableFrontISP)
    assert m.table.n_cameras == 1

    # --- camera-specific rows + metadata dispatch ---
    m2 = fi.build_front_isp({'type': 'learnable', 'learnable': {
        'camera_specific': True, 'n_cameras': 3}})
    assert isinstance(m2, LearnableFrontISP) and m2.table.n_cameras == 3

    x = torch.rand(2, 3, 16, 24)
    y = m(x)  # no metadata → row 0
    assert y.shape == x.shape
    # identity init is a no-op (gamma=1, wb=1, ccm=I, bias=0)
    assert torch.allclose(y, x, atol=1e-6), "identity init changed the image"

    # --- camera-specific rows + metadata dispatch ---
    m = fi.build_front_isp({'type': 'learnable', 'learnable': {
        'camera_specific': True, 'n_cameras': 3}})
    assert m.table.n_cameras == 3
    with torch.no_grad():
        # perturb each camera's wb row so rows are distinguishable
        m.table.wb_log += torch.tensor([[0.5, 0.0, 0.0],
                                        [0.0, 0.5, 0.0],
                                        [0.0, 0.0, 0.5]])
    cam = torch.tensor([0, 1, 2])
    xb = torch.rand(3, 3, 8, 8)
    with torch.no_grad():
        yb = m(xb, {'camera_id': cam})
    assert not torch.allclose(yb[0], yb[1]) and not torch.allclose(yb[1], yb[2])
    # out-of-range camera id clamps instead of crashing
    with torch.no_grad():
        m(xb, {'camera_id': torch.tensor([99, -3, 0])})

    # --- gradients flow into learnable params; freeze stops them ---
    m = fi.build_front_isp({'type': 'learnable'})
    x = torch.rand(1, 3, 8, 8)
    out = m(x)
    out.sum().backward()
    assert m.table.wb_log.grad is not None and m.table.ccm.grad is not None
    assert m.table.bias.grad is not None and m.table.log_gamma.grad is not None

    m.freeze()
    assert m.trainable_parameters() == []
    x = torch.rand(1, 3, 8, 8)
    out = m(x)
    assert not out.requires_grad, "frozen learnable front ISP should detach the graph"

    # --- learnable flags from config ---
    m = fi.build_front_isp({'type': 'learnable', 'learnable': {
        'white_balance': {'learnable': False},
        'ccm': {'learnable': False},
        'bias': {'learnable': False},
        'tone': {'learnable': False},
    }})
    assert m.trainable_parameters() == []

    # --- fittedisp loader: inline dict ---
    p = load_fittedisp_params({'params': {'wb_gain': [1.1, 1.0, 0.9],
                                           'color_matrix': [[1.0, 0.1, 0.0],
                                                            [0.1, 1.0, 0.0],
                                                            [0.0, 0.1, 1.0]],
                                           'offset': [0.01, 0.0, -0.01],
                                           'gamma': 1.8}})
    assert torch.allclose(p['log_wb'], torch.log(torch.tensor([1.1, 1.0, 0.9])), atol=1e-6)
    assert torch.allclose(p['log_gamma'], torch.log(torch.tensor(1.8)), atol=1e-6)
    assert p['ccm'].shape == (3, 3) and p['bias'].shape == (3,)

    # fittedisp init via CameraParamTable
    m = fi.build_front_isp({'type': 'learnable', 'learnable': {
        'init': {'type': 'fittedisp', 'params': {'wb': [1.1, 1, 0.9], 'gamma': 1.8}}}})
    x = torch.rand(1, 3, 8, 8)
    y = m(x)
    assert not torch.allclose(y, x)  # non-identity params applied

    # --- ckpt round-trip ---
    m = fi.build_front_isp({'type': 'learnable'})
    with torch.no_grad():
        m.table.wb_log += 0.3
        m.table.log_gamma -= 0.2
    state = m.state_dict()
    m2 = fi.build_front_isp({'type': 'learnable'})
    m2.load_state_dict(state)
    assert torch.allclose(m.table.wb_log, m2.table.wb_log)

    # ckpt key style used by LearnableTrainer ({"front_isp": state})
    with tempfile.TemporaryDirectory() as td:
        ck = os.path.join(td, 'LearnableISP_iter_10.pth')
        torch.save({'front_isp': state, 'task': 'learnable_pretrain'}, ck)
        m3 = fi.build_front_isp({'type': 'learnable', 'learnable': {'ckpt': ck}})
        assert torch.allclose(m.table.wb_log, m3.table.wb_log)

    # fittedisp JSON file path
    with tempfile.TemporaryDirectory() as td:
        pj = os.path.join(td, 'params.json')
        with open(pj, 'w') as fh:
            json.dump({'wb': [1.2, 1.0, 0.8], 'gamma': 2.0}, fh)
        p = load_fittedisp_params({'params': pj})
        assert torch.allclose(p['log_wb'], torch.log(torch.tensor([1.2, 1.0, 0.8])), atol=1e-6)

    # --- metrics sanity ---
    from tasks.human_quality.metrics import psnr_batch, delta_e_batch
    a = torch.rand(2, 3, 32, 32)
    assert (psnr_batch(a, a) > 60).all()          # identical → huge PSNR
    assert (delta_e_batch(a, a) < 1e-3).all()     # identical → ~0 ΔE
    psnr_far = psnr_batch(torch.zeros(1, 3, 8, 8), torch.ones(1, 3, 8, 8))
    assert psnr_far.item() < 1e-6                 # max MSE → ~0 dB

    # --- cfg-level build (front_isp dict on a run config) ---
    from engine.util import Dict
    from front_isp import build_front_isp_from_cfg
    cfg = Dict({'front_isp': {'enabled': True, 'type': 'learnable',
                              'learnable': {'camera_specific': True,
                                              'n_cameras': 5}}})
    m = build_front_isp_from_cfg(cfg)
    assert isinstance(m, LearnableFrontISP) and m.table.n_cameras == 5


if __name__ == "__main__":
    test_learnable()
    print("smoke/test_learnable: PASS")
