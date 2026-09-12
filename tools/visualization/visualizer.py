"""Canary-case visualizer for V2-AI trained checkpoints.

Runs the trained Controller in eval-argmax mode on a small set of fixed
samples and dumps two PNGs per case into <exp-dir>/visualization/:

  case_NN_pipeline.png    — full ISP rollout (input + each intermediate
                            + final), each step labeled with op(param)
  case_NN_detection.png   — Detection task: before/after with GT (green)
                            + YOLOv3 predictions (red). Only Detection.
  case_NN_human.png       — Human task: input | processed | Expert C
                            target, with SSIM/LPIPS metrics. Only Human.

Task (Detection vs Human) is auto-detected from the ckpt ('task' field).

Usage:
    # After training, from repo root:
    python -m tools.visualization.visualizer \\
        --exp-dir experiments/lod-v2ai_full \\
        --n-cases 4

    # Explicit ckpt + task override:
    python -m tools.visualization.visualizer \\
        --exp-dir experiments/v2ai_hbase \\
        --ckpt experiments/v2ai_hbase/ckpt/HumanISP_iter_7000.pth \\
        --n-cases 6
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import tasks  # noqa: F401  bootstrap yolov3 sys.path
import isp    # noqa: F401  register operators

from controller.adaptiveisp import AdaptiveISPController
from front_isp import build_front_isp_from_cfg
from isp.registry import build_operator
from pipeline import PipelineExecutor
from search import SearchSpace
from search.priors.action_mask import build_from_config as build_action_mask


# ---------------------------------------------------------------- helpers ----

def _fmt_op_label(name: str, phys: np.ndarray) -> str:
    """Pretty-print op(first-param) for the pipeline panel titles."""
    first = float(phys[0]) if phys.size else 0.0
    if name.startswith("n_"):
        return f"{name}(α={first:.2f})"
    if phys.size > 1:
        return f"{name}({first:.2f},+{phys.size - 1})"
    return f"{name}({first:.2f})"


def _to_display(img: torch.Tensor) -> np.ndarray:
    """(1,3,H,W) or (3,H,W) in [0,1] → (H,W,3) uint8 clipped."""
    if img.dim() == 4:
        img = img[0]
    return (img.clamp(0, 1).permute(1, 2, 0).cpu().numpy() * 255.0).astype(np.uint8)


def _load_config(path: str) -> dict:
    """Load cfg from an experiments/.../adaptiveisp*.yaml — simple dict access."""
    from engine.util import Dict
    with open(path) as fh:
        data = yaml.safe_load(fh)
    cfg = Dict(data)
    if 'num_state_dim' not in cfg:
        cfg.num_state_dim = 3 + len(cfg.operators)
    if 'z_dim' not in cfg:
        cfg.z_dim = 3 + len(cfg.operators) * cfg.get('z_dim_per_filter', 16)
    return cfg


def _latest_ckpt(exp_dir: Path) -> Optional[Path]:
    ckpt_dir = exp_dir / "ckpt"
    if not ckpt_dir.exists():
        return None
    cands = sorted(ckpt_dir.glob("*.pth"), key=lambda p: p.stat().st_mtime)
    return cands[-1] if cands else None


def _find_cfg(exp_dir: Path) -> Optional[Path]:
    """The trainer copies its yaml into experiments/<save_path>/ on startup."""
    for cand in exp_dir.glob("*.yaml"):
        return cand
    return None


# ---------------------------------------------------- rollout with capture ---

def rollout_capturing_stages(
    controller: AdaptiveISPController,
    executor: PipelineExecutor,
    search_space: SearchSpace,
    img: torch.Tensor,
    op_names: list[str],
) -> tuple[list[tuple[str, np.ndarray, torch.Tensor]], torch.Tensor]:
    """Run eval-argmax on `img` (1,3,H,W). Return (stages, final_img).

    `stages` is a list of (label, phys_params, image_after_this_step). The
    initial input is NOT included; only the outputs after each op.
    """
    controller.eval()
    T = controller.max_steps
    stages: list[tuple[str, np.ndarray, torch.Tensor]] = []
    with torch.no_grad():
        state = executor.initial_state(img.clone())
        for _t in range(T):
            constraint = search_space.valid_actions(state)
            out = controller.act(state, constraint)
            if out.action.is_stop[0].item() and not state.stopped[0].item():
                stages.append(("STOP", np.array([]), state.image.clone()))
                break
            idx = int(out.action.op_indices[0].item())
            name = op_names[idx]
            dim = executor.operators[name].spec.dim
            phys = out.action.params[0, :dim].detach().cpu().numpy()
            state = executor.step(state, out.action)
            stages.append((_fmt_op_label(name, phys), phys, state.image.clone()))
            if state.stopped[0].item():
                break
    return stages, state.image


# --------------------------------------------------- pipeline visualization --

def viz_pipeline(
    input_img: torch.Tensor,
    stages: list[tuple[str, np.ndarray, torch.Tensor]],
    save_path: Path,
    title: str = "",
) -> None:
    """Grid of the raw input + each intermediate step."""
    n_stages = len(stages)
    n_panels = 1 + n_stages
    n_cols = min(n_panels, 6)
    n_rows = (n_panels + n_cols - 1) // n_cols

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.4 * n_cols, 3.4 * n_rows))
    axes = np.atleast_2d(axes).flatten()

    axes[0].imshow(_to_display(input_img))
    axes[0].set_title("input (step 0)", fontsize=10)
    axes[0].axis("off")

    for i, (label, _phys, img) in enumerate(stages):
        ax = axes[i + 1]
        ax.imshow(_to_display(img))
        ax.set_title(f"step {i + 1}: {label}", fontsize=10)
        ax.axis("off")

    for j in range(n_panels, len(axes)):
        axes[j].axis("off")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------- Detection: boxes overlay ----

def _draw_boxes(
    ax, boxes_xyxy: np.ndarray, color: str, label: str, cls: Optional[np.ndarray] = None,
    class_names: Optional[list[str]] = None,
) -> None:
    """Draw a batch of xyxy boxes on the given axis, labeled by cls id."""
    for i, box in enumerate(boxes_xyxy):
        x1, y1, x2, y2 = box[:4]
        rect = mpatches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=1.6, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)
        if cls is not None and class_names is not None:
            name = class_names[int(cls[i])] if int(cls[i]) < len(class_names) else str(int(cls[i]))
            ax.text(x1, max(0, y1 - 4), name, color=color, fontsize=7,
                    bbox=dict(facecolor="white", alpha=0.55, boxstyle="round,pad=0.15", edgecolor="none"))
    # legend handle
    ax.plot([], [], color=color, label=label, linewidth=2)


def viz_detection(
    input_img: torch.Tensor,
    final_img: torch.Tensor,
    gt_boxes_xyxy: np.ndarray,
    gt_cls: np.ndarray,
    preds_before: np.ndarray,          # (N, 6) [x1,y1,x2,y2,conf,cls]
    preds_after: np.ndarray,
    class_names: list[str],
    save_path: Path,
    title: str = "",
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    for ax, img, preds, panel_title in zip(
        axes,
        [input_img, final_img],
        [preds_before, preds_after],
        ["before ISP", "after ISP"],
    ):
        ax.imshow(_to_display(img))
        _draw_boxes(ax, gt_boxes_xyxy, "lime", "GT", cls=gt_cls, class_names=class_names)
        if preds is not None and len(preds):
            _draw_boxes(ax, preds[:, :4], "red", "pred",
                        cls=preds[:, 5], class_names=class_names)
        ax.set_title(f"{panel_title}   GT={len(gt_boxes_xyxy)}, preds={0 if preds is None else len(preds)}",
                     fontsize=10)
        ax.axis("off")
        ax.legend(loc="upper right", fontsize=8)
    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------ Human: SSIM/LPIPS overlay --

def viz_human(
    input_img: torch.Tensor,
    final_img: torch.Tensor,
    target: torch.Tensor,
    ssim_before: float, ssim_after: float,
    lpips_before: float, lpips_after: float,
    save_path: Path,
    title: str = "",
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6))
    axes[0].imshow(_to_display(input_img))
    axes[0].set_title(f"input\nSSIM={ssim_before:.3f}  LPIPS={lpips_before:.3f}", fontsize=10)
    axes[0].axis("off")
    axes[1].imshow(_to_display(final_img))
    axes[1].set_title(f"after ISP\nSSIM={ssim_after:.3f}  LPIPS={lpips_after:.3f}", fontsize=10)
    axes[1].axis("off")
    axes[2].imshow(_to_display(target))
    axes[2].set_title("Expert C (target)", fontsize=10)
    axes[2].axis("off")

    delta_ssim = ssim_after - ssim_before
    delta_lpips = lpips_after - lpips_before
    fig.text(0.5, -0.02,
             f"ΔSSIM = {delta_ssim:+.3f}   ΔLPIPS = {delta_lpips:+.3f}",
             ha="center", fontsize=11,
             color="green" if (delta_ssim > 0 and delta_lpips < 0) else "black")

    if title:
        fig.suptitle(title, fontsize=12, y=1.02)
    plt.tight_layout()
    fig.savefig(save_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ================================================================ main =====

def run_detection(
    ckpt: dict, cfg, exp_dir: Path, n_cases: int, device: torch.device,
) -> None:
    """Detection canary: LOD val split, YOLOv3 preds before/after."""
    from tasks.detection.implementations.yolov3 import YOLOv3Detection
    from tasks.detection.dataloader import create_dataloader_real_hr
    from yolov3.utils.general import non_max_suppression, scale_boxes, xywh2xyxy

    hyp_path = "tasks/third_party/yolov3/data/hyps/hyp.scratch-low.yaml"
    with open(hyp_path) as fh:
        hyp = yaml.safe_load(fh)
    # Data cfg — assume standard LOD yaml unless user has a different one; we
    # look inside the experiments dir for a copied one first.
    data_cfg_path = "tasks/third_party/yolov3/data/lod.yaml"
    for name in ("data.yaml", "lod.yaml"):
        cand = exp_dir / name
        if cand.exists():
            data_cfg_path = str(cand)
            break
    data_dict = YOLOv3Detection.parse_data_cfg(data_cfg_path)
    nc = int(data_dict['nc'])
    class_names = list(data_dict['names'].values()) if isinstance(data_dict['names'], dict) \
        else list(data_dict['names'])

    task_model = YOLOv3Detection(
        weights="pretrained/yolov3.pt", yolo_cfg="tasks/third_party/yolov3/models/yolov3.yaml",
        hyp=hyp, nc=nc, imgsz=int(cfg.get('test_steps_imgsz', 512)), device=device,
    )
    task_model.eval()

    val_loader, _ = create_dataloader_real_hr(
        data_dict['val'], 512, 1, task_model.gs, single_cls=False,
        hyp={'anchor_t': hyp.get('anchor_t', 4.0), 'label_smoothing': 0.0},
        cache=False, rect=False, workers=1, pad=0.0, prefix='canary: ',
        add_noise=False, seed=0,
    )

    ops = {n: build_operator(n).to(device) for n in cfg.operators}
    executor = PipelineExecutor(ops, cfg.operators)
    _am = build_action_mask(cfg.get('action_mask', {}) or {}, cfg.operators)
    search_space = SearchSpace(ops, cfg.operators, priors=[_am] if _am.priors else None)
    controller = AdaptiveISPController(
        ops, cfg.operators, obs_hw=int(cfg.get('obs_hw', 64)),
        mid_channels=cfg.base_channels, fc1_size=cfg.fc1_size,
        feature_dim=cfg.feature_extractor_dims,
        dropout_keep_prob=cfg.dropout_keep_prob,
        exploration=cfg.exploration, max_steps=cfg.test_steps,
    ).to(device)
    controller.load_state_dict(ckpt['controller_model'])
    controller.eval()
    front_isp = build_front_isp_from_cfg(cfg).to(device)

    out_dir = exp_dir / "visualization"
    out_dir.mkdir(exist_ok=True)

    for case_idx, (imgs, targets, paths, shapes, _hr) in enumerate(val_loader):
        if case_idx >= n_cases:
            break
        imgs = imgs.to(device).float()
        targets = targets.to(device)
        with torch.no_grad():
            imgs = front_isp(imgs).clamp(0.0, 1.0)
        _, _, h, w = imgs.shape

        # rollout
        stages, final_img = rollout_capturing_stages(
            controller, executor, search_space, imgs, list(cfg.operators),
        )

        # YOLO preds before & after
        with torch.no_grad():
            p_before = task_model.model(imgs)
            p_after = task_model.model(final_img)
        preds_before = non_max_suppression(p_before, 0.25, 0.5, max_det=100)[0]
        preds_after = non_max_suppression(p_after, 0.25, 0.5, max_det=100)[0]
        preds_before_np = preds_before.detach().cpu().numpy() if preds_before is not None else np.zeros((0, 6))
        preds_after_np = preds_after.detach().cpu().numpy() if preds_after is not None else np.zeros((0, 6))

        # GT boxes — targets rows are [img_idx, cls, cx,cy,w,h] normalized
        gt_rows = targets[targets[:, 0] == 0]
        gt_norm = gt_rows[:, 2:6].detach().cpu().numpy()
        gt_cls = gt_rows[:, 1].detach().cpu().numpy()
        gt_xyxy = np.zeros((len(gt_norm), 4), dtype=np.float32)
        if len(gt_norm):
            gt_xyxy[:, 0] = (gt_norm[:, 0] - gt_norm[:, 2] / 2) * w
            gt_xyxy[:, 1] = (gt_norm[:, 1] - gt_norm[:, 3] / 2) * h
            gt_xyxy[:, 2] = (gt_norm[:, 0] + gt_norm[:, 2] / 2) * w
            gt_xyxy[:, 3] = (gt_norm[:, 1] + gt_norm[:, 3] / 2) * h

        base = f"case_{case_idx:02d}"
        title_prefix = f"canary {case_idx}  |  {os.path.basename(paths[0])}"
        viz_pipeline(imgs, stages, out_dir / f"{base}_pipeline.png",
                     title=f"{title_prefix}  |  ISP pipeline")
        viz_detection(imgs, final_img, gt_xyxy, gt_cls,
                      preds_before_np, preds_after_np, class_names,
                      out_dir / f"{base}_detection.png",
                      title=f"{title_prefix}  |  YOLOv3 before / after")
        print(f"  wrote {base}_pipeline.png + {base}_detection.png")


def run_human(
    ckpt: dict, cfg, exp_dir: Path, n_cases: int, device: torch.device,
) -> None:
    """Human canary: FiveK val, SSIM/LPIPS before/after + Expert C target."""
    from tasks.human_quality import FiveKDataset
    from tasks.human_quality.metrics import ssim_batch, lpips_batch

    hq_cfg = cfg.get('human_quality', {}) or {}
    val_list = hq_cfg.get('val_list', '/home/jing/datasets/fivek/val_expert_c.txt')
    cache_dir = hq_cfg.get('cache_dir', '/home/jing/datasets/fivek/cache_expert_c')
    imgsz = int(cfg.get('test_steps_imgsz', 512))

    dataset = FiveKDataset(val_list, cache_dir=cache_dir, imgsz=imgsz)

    ops = {n: build_operator(n).to(device) for n in cfg.operators}
    executor = PipelineExecutor(ops, cfg.operators)
    _am = build_action_mask(cfg.get('action_mask', {}) or {}, cfg.operators)
    search_space = SearchSpace(ops, cfg.operators, priors=[_am] if _am.priors else None)
    controller = AdaptiveISPController(
        ops, cfg.operators, obs_hw=int(cfg.get('obs_hw', 64)),
        mid_channels=cfg.base_channels, fc1_size=cfg.fc1_size,
        feature_dim=cfg.feature_extractor_dims,
        dropout_keep_prob=cfg.dropout_keep_prob,
        exploration=cfg.exploration, max_steps=cfg.test_steps,
    ).to(device)
    controller.load_state_dict(ckpt['controller_model'])
    controller.eval()
    front_isp = build_front_isp_from_cfg(cfg).to(device)

    out_dir = exp_dir / "visualization"
    out_dir.mkdir(exist_ok=True)

    for case_idx in range(min(n_cases, len(dataset))):
        img, target, cam_id = dataset[case_idx]
        img = img.unsqueeze(0).to(device)
        target = target.unsqueeze(0).to(device)
        cam = torch.tensor([cam_id], dtype=torch.long, device=device)
        with torch.no_grad():
            img = front_isp(img, {'camera_id': cam}).clamp(0.0, 1.0)

        stages, final_img = rollout_capturing_stages(
            controller, executor, search_space, img, list(cfg.operators),
        )

        ssim_before = ssim_batch(img, target).item()
        ssim_after = ssim_batch(final_img, target).item()
        lpips_before = lpips_batch(img, target).item()
        lpips_after = lpips_batch(final_img, target).item()

        base = f"case_{case_idx:02d}"
        title_prefix = f"canary {case_idx}"
        viz_pipeline(img, stages, out_dir / f"{base}_pipeline.png",
                     title=f"{title_prefix}  |  ISP pipeline")
        viz_human(img, final_img, target,
                  ssim_before, ssim_after, lpips_before, lpips_after,
                  out_dir / f"{base}_human.png",
                  title=f"{title_prefix}  |  input / after / Expert C")
        print(f"  wrote {base}_pipeline.png + {base}_human.png")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, type=Path,
                        help="experiments/<save_path>/ dir containing ckpt/ and adaptiveisp*.yaml")
    parser.add_argument("--ckpt", type=Path, default=None,
                        help="explicit ckpt path (defaults to newest .pth under exp-dir/ckpt/)")
    parser.add_argument("--cfg", type=Path, default=None,
                        help="explicit config yaml (defaults to any .yaml in exp-dir)")
    parser.add_argument("--n-cases", type=int, default=4)
    parser.add_argument("--task", type=str, default=None, choices=["detection", "human"],
                        help="force task type; otherwise auto-detected from ckpt")
    args = parser.parse_args()

    if not args.exp_dir.exists():
        print(f"exp-dir does not exist: {args.exp_dir}", file=sys.stderr)
        return 1

    ckpt_path = args.ckpt or _latest_ckpt(args.exp_dir)
    if ckpt_path is None or not ckpt_path.exists():
        print(f"no ckpt found under {args.exp_dir}/ckpt/", file=sys.stderr)
        return 1
    cfg_path = args.cfg or _find_cfg(args.exp_dir)
    if cfg_path is None:
        print(f"no cfg .yaml under {args.exp_dir}/", file=sys.stderr)
        return 1

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"loading ckpt: {ckpt_path}")
    print(f"loading cfg:  {cfg_path}")
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = _load_config(str(cfg_path))

    task = args.task or ckpt.get('task', 'detection')
    print(f"task: {task}")
    print(f"n_cases: {args.n_cases}")
    print(f"output dir: {args.exp_dir / 'visualization'}")

    if task == "human_quality" or task == "human":
        run_human(ckpt, cfg, args.exp_dir, args.n_cases, device)
    else:
        run_detection(ckpt, cfg, args.exp_dir, args.n_cases, device)

    print("done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
