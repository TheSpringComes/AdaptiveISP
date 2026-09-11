"""Build a 4-way side-by-side comparison of Human ablation ckpts (H0/H1/H2/H3).

Each row = one FiveK sample; columns = [Input | H0 out | H1 out | H2 out | H3 out | Expert C target].
Reads the same 4 val samples via FiveKDataset with the same imgsz used by
the visualizer (512).
"""
from __future__ import annotations

import sys, os
sys.path.insert(0, '.')

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from engine.util import load_config
from isp.registry import build_operator
from pipeline import CanonicalBackbone, PipelineExecutor
from search import SearchSpace
from search.priors.action_mask import build_from_config as build_action_mask
from controller.adaptiveisp import AdaptiveISPController
from tasks.human_quality import FiveKDataset
from tasks.human_quality.metrics import ssim_batch, lpips_batch


def load_ckpt_configs(id_):
    cfg_map = {
        'h0': 'configs/adaptiveisp_human.yaml',
        'h1': 'configs/adaptiveisp_human_v3_e1.yaml',
        'h2': 'configs/adaptiveisp_human_v3_e2.yaml',
        'h3': 'configs/adaptiveisp_human_v3_e3.yaml',
    }
    cfg = load_config(cfg_map[id_])
    ckpt = f'experiments/v3_{id_}/ckpt/HumanISP_iter_3000.pth'
    return cfg, ckpt


def run_rollout(cfg, ckpt_path, imgs, device):
    ops = {n: build_operator(n).to(device) for n in cfg.operators}
    executor = PipelineExecutor(ops, cfg.operators)
    am = build_action_mask(cfg.get('action_mask', {}) or {}, cfg.operators)
    ss = SearchSpace(ops, cfg.operators, priors=[am] if am.priors else None)
    ctrl = AdaptiveISPController(
        ops, cfg.operators, obs_hw=int(cfg.get('obs_hw', 64)),
        mid_channels=cfg.base_channels, fc1_size=cfg.fc1_size,
        feature_dim=cfg.feature_extractor_dims,
        dropout_keep_prob=cfg.dropout_keep_prob,
        exploration=cfg.exploration, max_steps=cfg.test_steps,
        min_rollout_length=int(cfg.get('min_rollout_length', 1)),
    ).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    ctrl.load_state_dict(ckpt['controller_model'])
    ctrl.eval()
    bb_cfg = cfg.get('canonical_backbone', {}) or {}
    backbone = CanonicalBackbone().to(device) if bb_cfg.get('enabled') else None

    with torch.no_grad():
        x = imgs.clone()
        if backbone is not None:
            x = backbone(x).clamp(0, 1)
        T = int(cfg.test_steps)
        state = executor.initial_state(x)
        for _ in range(T):
            c = ss.valid_actions(state)
            o = ctrl.act(state, c)
            state = executor.step(state, o.action)
            if state.stopped.all():
                break
    return state.image.clamp(0, 1).cpu().numpy()


def main():
    device = torch.device('cuda')
    # Load same dataset used by visualizer (via H0 cfg to get fivek paths).
    cfg0 = load_config('configs/adaptiveisp_human.yaml')
    hq = cfg0.get('human_quality')
    dataset = FiveKDataset(hq['val_list'], cache_dir=hq['cache_dir'], imgsz=512)

    n_cases = 4
    imgs_list, tgts_list = [], []
    for i in range(n_cases):
        im, tg = dataset[i]
        imgs_list.append(im); tgts_list.append(tg)
    imgs = torch.stack(imgs_list).to(device)
    tgts = torch.stack(tgts_list).to(device)

    # Rollouts for all 4 configs
    outs = {}
    for id_ in ('h0', 'h1', 'h2', 'h3'):
        cfg, ckpt = load_ckpt_configs(id_)
        outs[id_] = run_rollout(cfg, ckpt, imgs, device)
        print(f'{id_} rollout done')

    # Per-case SSIM/LPIPS
    per_case_metrics = {}
    for id_ in ('h0', 'h1', 'h2', 'h3'):
        s = ssim_batch(torch.from_numpy(outs[id_]).to(device), tgts).cpu().numpy()
        l = lpips_batch(torch.from_numpy(outs[id_]).to(device), tgts).cpu().numpy()
        per_case_metrics[id_] = (s.flatten(), l.flatten())

    # Layout: rows = cases, cols = [Input, H0, H1, H2, H3, Target]
    imgs_np = imgs.cpu().numpy()
    tgts_np = tgts.cpu().numpy()
    fig, axs = plt.subplots(n_cases, 6, figsize=(18, 3 * n_cases))
    col_titles = ['Input (RAW-linear)', 'H0 baseline', 'H1 +backbone', 'H2 +mask', 'H3 +PPO', 'Expert C target']

    def _show(ax, img_chw, title):
        img = np.transpose(np.clip(img_chw, 0, 1), (1, 2, 0))
        ax.imshow(img)
        ax.set_xticks([]); ax.set_yticks([])
        if title: ax.set_title(title, fontsize=10)

    for ci in range(n_cases):
        _show(axs[ci, 0], imgs_np[ci], col_titles[0] if ci == 0 else '')
        for j, id_ in enumerate(('h0', 'h1', 'h2', 'h3'), start=1):
            s, l = per_case_metrics[id_][0][ci], per_case_metrics[id_][1][ci]
            subtitle = f'SSIM={s:.3f}  LPIPS={l:.3f}'
            _show(axs[ci, j], outs[id_][ci], (col_titles[j] + '\n' + subtitle) if ci == 0 else subtitle)
            if ci > 0:
                axs[ci, j].set_title(subtitle, fontsize=9)
        _show(axs[ci, 5], tgts_np[ci], col_titles[5] if ci == 0 else '')

    fig.suptitle('V3 Human Ablation — 4 val samples, 4 ckpts × input/target references', fontsize=12, y=1.005)
    fig.tight_layout()
    out_path = 'logs_ablation/human_ablation_comparison.png'
    fig.savefig(out_path, dpi=110, bbox_inches='tight')
    print(f'\nsaved: {out_path}')


if __name__ == '__main__':
    main()
