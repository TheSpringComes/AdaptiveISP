"""smoke: Configurable Front ISP — V3.1 四种统一模式 + registry + 兼容。

Covers:
  - identity / fixed / learnable / external (+ legacy 别名) 全部注册
  - build_front_isp honors enabled:false / type:identity|none → Identity
  - fixed: 模块链构建、前向 shape/range、未知模块快速失败、
    无 modules 报错、自定义模块注册扩展
  - external: backend 分发（合法名构建、未知名报错、缺第三方仓库报
    可操作错误）
  - canonical front ISP output shape/range on a synthetic batch
  - build_front_isp_from_cfg: front_isp section, legacy canonical_backbone
    fallback, and default (neither key) → identity
  - V3.1 两阶段语义：Stage 2 冻结 Front ISP（trainable_parameters 为空）
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
    from front_isp.fixed import FixedFrontISP, list_fixed_modules, \
        register_fixed_module
    from front_isp.identity import IdentityFrontISP

    # --- registration: V3.1 四种模式 + legacy 别名 ---
    registered = fi.list_front_isps()
    for t in ('identity', 'fixed', 'learnable', 'external',          # V3.1
              'none', 'calibrated', 'canonical',                     # legacy
              'infinite_isp', 'modular_neural_isp'):
        assert t in registered, f"{t} not registered: {registered}"

    # --- build: disabled / identity / legacy none → Identity ---
    assert isinstance(fi.build_front_isp({'enabled': False, 'type': 'fixed'}),
                      IdentityFrontISP)
    assert isinstance(fi.build_front_isp({'type': 'identity'}), IdentityFrontISP)
    assert isinstance(fi.build_front_isp({'type': 'none'}), IdentityFrontISP)
    assert isinstance(fi.build_front_isp({}), IdentityFrontISP)

    # --- build: unknown type raises ---
    try:
        fi.build_front_isp({'type': 'nope'})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # --- fixed: 模块链构建 + 前向 ---
    fixed = fi.build_front_isp({'type': 'fixed', 'fixed': {
        'modules': [
            {'name': 'wb', 'auto': 'grayworld'},
            {'name': 'gamma', 'gamma': 0.4545},
        ]}})
    assert isinstance(fixed, FixedFrontISP)
    x = torch.rand(2, 3, 32, 48)
    y = fixed(x)
    assert y.shape == x.shape
    assert y.min() >= 0.0 and y.max() <= 1.0
    assert not torch.allclose(y, x)          # gamma 改变了图像
    assert len(list(fixed.parameters())) == 0  # 无任何可训练参数

    # --- fixed: 未知模块 → 构建期报错 ---
    try:
        fi.build_front_isp({'type': 'fixed', 'fixed': {
            'modules': [{'name': 'no_such_module'}]}})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # --- fixed: 缺 modules → 报错 ---
    try:
        fi.build_front_isp({'type': 'fixed', 'fixed': {}})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass

    # --- fixed: 自定义模块注册扩展（规划要求：手动添加算子即可） ---
    try:
        @register_fixed_module('smoke_test_module')
        def _double(img, cfg):
            return img * float(cfg.get('k', 1.0))
    except ValueError:
        pass  # 重复注册（重跑 smoke）时忽略
    fixed2 = fi.build_front_isp({'type': 'fixed', 'fixed': {
        'modules': [{'name': 'smoke_test_module', 'k': 0.5}]}})
    y2 = fixed2(x)
    assert torch.allclose(y2, (x * 0.5).clamp(0, 1), atol=1e-6)
    assert 'smoke_test_module' in list_fixed_modules()

    # --- fixed: ccm 'cam2rgb' 快捷方式 + bias 模块 ---
    fixed3 = fi.build_front_isp({'type': 'fixed', 'fixed': {
        'modules': [{'name': 'ccm', 'matrix': 'cam2rgb'},
                    {'name': 'bias', 'offset': [0.0, 0.0, 0.0]}]}})
    y3 = fixed3(x)
    assert y3.shape == x.shape and y3.min() >= 0.0

    # --- external: 未知名 backend → 报错；缺仓库 → 可操作错误 ---
    try:
        fi.build_front_isp({'type': 'external',
                            'external': {'backend': 'nope'}})
        raise AssertionError("expected ValueError")
    except ValueError:
        pass
    for backend in ('infinite_isp', 'samsung_isp'):
        try:
            fi.build_front_isp({'type': 'external', 'external': {
                'backend': backend, 'repo_path': '/nonexistent'}})
            raise AssertionError(f"expected FileNotFoundError for {backend}")
        except (FileNotFoundError, NotImplementedError, ImportError):
            pass

    # --- learnable 别名：new type name 与旧名等价 ---
    from front_isp.learnable import LearnableFrontISP, CalibratedFrontISP
    m_new = fi.build_front_isp({'type': 'learnable', 'learnable': {
        'camera_specific': True, 'n_cameras': 3}})
    m_old = fi.build_front_isp({'type': 'calibrated', 'calibration': {
        'camera_specific': True, 'n_cameras': 3}})
    assert isinstance(m_new, LearnableFrontISP) and m_new.table.n_cameras == 3
    assert isinstance(m_old, LearnableFrontISP) and m_old.table.n_cameras == 3

    # --- canonical: shape / range / non-trivial ---
    canonical = fi.build_front_isp({'type': 'canonical'})
    yc = canonical(x)
    assert yc.shape == x.shape
    assert yc.min() >= 0.0 and yc.max() <= 1.0
    assert not torch.allclose(yc, x)

    # --- cfg-level: front_isp section ---
    cfg = Dict({'front_isp': {'enabled': True, 'type': 'canonical'}})
    assert type(fi.build_front_isp_from_cfg(cfg)).__name__ == 'CanonicalBackbone'
    cfg = Dict({'front_isp': {'enabled': True, 'type': 'identity'}})
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
    cfg = Dict({'front_isp': {'enabled': True, 'type': 'identity'},
                'canonical_backbone': {'enabled': True}})
    assert isinstance(fi.build_front_isp_from_cfg(cfg), IdentityFrontISP)

    # --- real config files: 基础配置 → identity；v31 配置 → 对应模式 ---
    cfg = load_config('configs/adaptiveisp_human.yaml')
    assert isinstance(fi.build_front_isp_from_cfg(cfg), IdentityFrontISP)
    cfg = load_config('configs/adaptiveisp_human_v31_pretrain.yaml')
    m = fi.build_front_isp_from_cfg(cfg)
    assert isinstance(m, LearnableFrontISP) and m.table.n_cameras == 24
    cfg = load_config('configs/adaptiveisp_human_v31_fixed.yaml')
    assert isinstance(fi.build_front_isp_from_cfg(cfg), FixedFrontISP)
    cfg = load_config('configs/adaptiveisp_human_v31_stage2.yaml')
    assert isinstance(fi.build_front_isp_from_cfg(cfg), LearnableFrontISP)
    cfg = load_config('configs/adaptiveisp_human_v31_external.yaml')
    # external 配置构建会因第三方仓库缺失而报可操作错误 — 只验证类型名
    try:
        fi.build_front_isp_from_cfg(cfg)
        built = True
    except (FileNotFoundError, NotImplementedError, ImportError):
        built = False
    assert not built, "external repo missing should fail with actionable error"

    # --- V3.1 两阶段语义：Stage 2 冻结（模拟 trainer_human 的冻结逻辑） ---
    m = fi.build_front_isp({'type': 'learnable'})
    for p in m.parameters():
        p.requires_grad_(False)
    assert m.trainable_parameters() == []
    xg = torch.rand(1, 3, 8, 8)
    out = m(xg)
    assert not out.requires_grad, "frozen front ISP should detach the graph"


if __name__ == "__main__":
    test_front_isp()
    print("smoke/test_front_isp: PASS")
