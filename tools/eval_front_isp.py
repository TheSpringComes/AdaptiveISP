"""统一评估各实验组 Front ISP 输出质量（V3.1 消融 A–E）。

对每个 config 构建其 Front ISP，在 FiveK val split（100 张，imgsz 可调）
上计算 Front ISP 输出 vs Expert-C target 的 SSIM / LPIPS / PSNR。
identity 组输出 = 输入本身（对照组下界）。

用法：
    python tools/eval_front_isp.py                       # 全部 5 组
    python tools/eval_front_isp.py --cfg configs/adaptiveisp_human_v31_fixed.yaml --tag B_fixed

输出：experiments/front_isp_eval/summary.json（追加式汇总）
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch

DEFAULT_GROUPS = [
    ("A_identity",  "configs/adaptiveisp_human.yaml"),
    ("B_fixed",     "configs/adaptiveisp_human_v31_fixed.yaml"),
    ("C_learnable", "configs/adaptiveisp_human_v31_stage2.yaml"),
    ("D_infinite",  "configs/adaptiveisp_human_v31_external.yaml"),
    ("E_samsung",   "configs/adaptiveisp_human_v31_external_samsung.yaml"),
]

OUT_PATH = os.path.join(_ROOT, "experiments", "front_isp_eval", "summary.json")


def eval_group(tag: str, cfg_path: str, imgsz: int, device) -> dict:
    from engine.util import load_config
    from front_isp import build_front_isp_from_cfg
    from tasks.human_quality import HumanQualityTask
    from tasks.human_quality.dataset import FiveKDataset

    cfg = load_config(cfg_path)
    front = build_front_isp_from_cfg(cfg).to(device).eval()
    for p in front.parameters():
        p.requires_grad_(False)

    hq = cfg.get('human_quality', {}) or {}
    ds = FiveKDataset(list_file=hq.get('val_list', '/home/jing/datasets/fivek/val_expert_c.txt'),
                      cache_dir=hq.get('cache_dir'), imgsz=imgsz, return_camera=True)

    task = HumanQualityTask(lambda_ssim=1.0, lambda_lpips=1.0,
                            lpips_net=hq.get('lpips_net', 'alex'), device=device)

    ssim_sum = lpips_sum = psnr_sum = 0.0
    n = 0
    t0 = time.perf_counter()
    with torch.no_grad():
        for i in range(len(ds)):
            img, tgt, cam = ds[i]
            img = img.unsqueeze(0).to(device).float()
            tgt = tgt.unsqueeze(0).to(device).float()
            out = front(img, {'camera_id': torch.tensor([cam], device=device)}).clamp(0.0, 1.0)
            m = task.compute_metrics(out, tgt)
            ssim_sum += m['ssim'].sum().item()
            lpips_sum += m['lpips'].sum().item()
            from tasks.human_quality import psnr_batch
            psnr_sum += psnr_batch(out, tgt).sum().item()
            n += 1
            if n % 25 == 0:
                print(f"  [{tag}] {n}/{len(ds)} elapsed {time.perf_counter()-t0:.0f}s", flush=True)

    res = dict(tag=tag, cfg=cfg_path, n=n, imgsz=imgsz,
               ssim=ssim_sum / max(n, 1),
               lpips=lpips_sum / max(n, 1),
               psnr=psnr_sum / max(n, 1),
               elapsed_s=round(time.perf_counter() - t0, 1))
    print(f"[{tag}] SSIM={res['ssim']:.4f} LPIPS={res['lpips']:.4f} "
          f"PSNR={res['psnr']:.2f} dB  ({res['elapsed_s']}s)", flush=True)
    return res


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cfg", type=str, default=None, help="只评估单个 config")
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--imgsz", type=int, default=512)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    groups = [(args.tag, args.cfg)] if args.cfg else DEFAULT_GROUPS

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    summary = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as fh:
            summary = json.load(fh)

    for tag, cfg_path in groups:
        if tag in summary:
            print(f"[{tag}] 已有结果，跳过（删除 {OUT_PATH} 中对应项可重算）")
            continue
        try:
            summary[tag] = eval_group(tag, cfg_path, args.imgsz, device)
        except Exception as exc:
            print(f"[{tag}] FAILED: {exc!r}", flush=True)
            summary[tag] = dict(tag=tag, cfg=cfg_path, error=str(exc))
        with open(OUT_PATH, "w") as fh:
            json.dump(summary, fh, indent=2, ensure_ascii=False)

    print("\n===== Front ISP 输出质量汇总（val split）=====")
    for tag, res in summary.items():
        if 'error' in res:
            print(f"  {tag:12s} ERROR: {res['error'][:80]}")
        else:
            print(f"  {tag:12s} SSIM={res['ssim']:.4f}  LPIPS={res['lpips']:.4f}  "
                  f"PSNR={res['psnr']:.2f} dB")


if __name__ == "__main__":
    main()
