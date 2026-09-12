"""用 FittedISP 的拟合方法在 FiveK 上离线拟合 fixed Front ISP 参数。

方法与 `front_isp/FittedISP/fit.py` 一致（IRLS 颜色拟合 + 色调指数
搜索 + 细节强度搜索），适配本项目数据流：

    输入 x = Dataset 输出的 demosaic linear RGB（Malvar 0.5 混合）
    目标 y = MIT-Adobe FiveK Expert-C sRGB

FittedISP 流程中作用于 RAW 域的部分（Bayer 去马赛克、暗角）对本研究
输入不适用：去马赛克已由 Dataset 完成，FiveK 为已校正数据。

用法：
    python tools/fit_front_isp.py                       # 默认 40 张训练图
    python tools/fit_front_isp.py --num_images 60 --region 320

输出：
    configs/front_isp/fitted_fivek.json   # fixed Front ISP 加载的参数
    experiments/fixed_fit/report.json     # 搜索过程与训练集指标
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# 先导入本项目模块（其内部 `import isp.` 依赖项目根在 sys.path 首位）。
from tasks.human_quality.dataset import FiveKDataset  # noqa: E402


def _load_fittedisp():
    """按独立模块名加载 FittedISP 的 isp.py/fit.py。

    FittedISP 顶层 isp.py 与项目 `isp` 包同名，不能通过 sys.path 直接
    导入；这里在 exec 期间临时把 sys.modules['isp'] 指向 FittedISP 的
    isp.py，让 fit.py 顶部的 `from isp import ...` 解析到正确实现。"""
    import importlib.util

    def _load(name, path):
        spec = importlib.util.spec_from_file_location(name, path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod

    fitted_dir = ROOT / "front_isp" / "FittedISP"
    saved = sys.modules.get('isp')
    isp_mod = _load('fittedisp_isp', fitted_dir / 'isp.py')
    sys.modules['isp'] = isp_mod
    try:
        fit_mod = _load('fittedisp_fit', fitted_dir / 'fit.py')
    finally:
        if saved is not None:
            sys.modules['isp'] = saved
        else:
            sys.modules.pop('isp', None)
    return isp_mod, fit_mod


_fitted_isp, _fitted_fit = _load_fittedisp()
color = _fitted_isp.color        # max(x,0)^e @ ccm.T + offset
detail = _fitted_isp.detail      # 亮度细节控制
fit_color = _fitted_fit.fit_color  # IRLS 颜色拟合


def collect_observations(ds, num_images, region, per_image):
    """每张图取 5 处固定区域（同 FittedISP 的位置），等量抽像素。"""
    xs, ys = [], []
    n = min(num_images, len(ds))
    for idx in range(n):
        image, target, _ = ds[idx]
        img = image.permute(1, 2, 0).numpy()          # (H,W,3) [0,1]
        tgt = target.permute(1, 2, 0).numpy()
        h, w = img.shape[:2]
        patch = min(region, h - 16, w - 16) // 2 * 2
        rng = np.random.default_rng(1234 + idx)
        for fy, fx in ((.15, .15), (.15, .85), (.5, .5), (.85, .15), (.85, .85)):
            y = min(max(int(h * fy - patch / 2), 0), h - patch)
            x = min(max(int(w * fx - patch / 2), 0), w - patch)
            xs.append(img[y:y + patch, x:x + patch].reshape(-1, 3))
            ys.append(tgt[y:y + patch, x:x + patch].reshape(-1, 3))
        # 等量抽取，防止单张图主导
        xi = np.concatenate(xs[-5:])
        yi = np.concatenate(ys[-5:])
        ids = rng.choice(len(xi), min(len(xi), per_image), replace=False)
        xs[-5:] = [xi[ids]]
        ys[-5:] = [yi[ids]]
        print(f"读取 {idx + 1}/{n}", flush=True)
    return np.concatenate(xs), np.concatenate(ys)


def collect_patches(ds, num_images, region):
    """整块区域（不去边缘），用于细节强度评估。"""
    patches = []
    n = min(num_images, len(ds))
    for idx in range(n):
        image, target, _ = ds[idx]
        img = image.permute(1, 2, 0).numpy()
        tgt = target.permute(1, 2, 0).numpy()
        h, w = img.shape[:2]
        patch = min(region, h - 16, w - 16) // 2 * 2
        for fy, fx in ((.15, .15), (.5, .5), (.85, .85)):
            y = min(max(int(h * fy - patch / 2), 0), h - patch)
            x = min(max(int(w * fx - patch / 2), 0), w - patch)
            patches.append((img[y:y + patch, x:x + patch],
                            tgt[y:y + patch, x:x + patch]))
    return patches


def robust_score(pred, target):
    """同 fit_isp 的训练评分：截断 |err| 的均值（[0,1] 域）。"""
    return float(np.minimum(abs(np.clip(pred, 0, 1) - target), .04).mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_images", type=int, default=40)
    parser.add_argument("--region", type=int, default=256)
    parser.add_argument("--per_image", type=int, default=12000)
    parser.add_argument("--out", type=Path, default=ROOT / "configs/front_isp/fitted_fivek.json")
    parser.add_argument("--report_dir", type=Path, default=ROOT / "experiments/fixed_fit")
    parser.add_argument("--fivek_root", type=str, default="/home/jing/datasets/fivek")
    parser.add_argument("--cache_dir", type=str, default="/home/jing/datasets/fivek/cache_expert_c")
    parser.add_argument("--train_list", type=str, default="/home/jing/datasets/fivek/train_expert_c.txt")
    args = parser.parse_args()

    ds = FiveKDataset(list_file=args.train_list, cache_dir=args.cache_dir,
                      imgsz=None, return_camera=True)
    print(f"训练集 {len(ds)} 张，使用前 {min(args.num_images, len(ds))} 张", flush=True)

    x, y = collect_observations(ds, args.num_images, args.region, args.per_image)
    print(f"观测像素 {len(x)}", flush=True)

    # ---- 色调指数搜索（候选同 FittedISP::fit_isp）----
    candidates = []
    for exponent in np.unique(np.r_[np.linspace(.35, 1.8, 16), 1, 1 / 2.2]):
        p = fit_color(x, y, exponent)
        score = robust_score(color(x, p), y)
        candidates.append((score, p))
    best = min(candidates, key=lambda v: v[0])[1]['exponent']
    for exponent in np.linspace(max(.35, best - .08), min(1.8, best + .08), 9):
        p = fit_color(x, y, exponent)
        score = robust_score(color(x, p), y)
        candidates.append((score, p))
    chosen = min(candidates, key=lambda v: v[0])[1]
    print(f"色调指数：{chosen['exponent']:.4f}", flush=True)

    # ---- 细节强度搜索（malvar_weight 不适用：输入已完成去马赛克）----
    patches = collect_patches(ds, min(args.num_images, 20), args.region)
    detail_candidates = []
    for strength in (-.5, 0, .4, .8):
        errs, npx = 0.0, 0
        for xi, yi in patches:
            pred = detail(color(xi, chosen), strength)[2:-2, 2:-2]
            delta = abs(np.clip(pred, 0, 1) - yi[2:-2, 2:-2])
            errs += float(np.minimum(delta, .04).sum())
            npx += delta.size
        detail_candidates.append((errs / npx, strength))
        print(f"  detail={strength:+.1f} robust_mae={errs / npx * 255:.4f}/255", flush=True)
    strength = min(detail_candidates)[1]

    # ---- 保存参数与报告 ----
    params = dict(schema_version=1,
                  exponent=float(chosen['exponent']),
                  ccm=chosen['ccm'], offset=chosen['offset'],
                  detail_strength=float(strength),
                  malvar_weight=0.5,   # 记录 Dataset 去马赛克方式；前端不再使用
                  fit_source="fivek_train_fittedisp_method",
                  note="由 tools/fit_front_isp.py 用 FittedISP 方法在 FiveK "
                       "训练集上拟合；malvar_weight/vignette 属 RAW 域，对本输入不适用。")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(params, ensure_ascii=False, indent=2), encoding="utf-8")

    args.report_dir.mkdir(parents=True, exist_ok=True)
    report = dict(
        method="FittedISP fit_color (IRLS) + exponent/detail search",
        num_images=min(args.num_images, len(ds)), region=args.region,
        observations=int(len(x)),
        tone_candidates=[dict(exponent=p['exponent'], robust_mae=s * 255)
                         for s, p in candidates],
        detail_candidates=[dict(strength=s, robust_mae=e * 255)
                           for e, s in detail_candidates],
        chosen=dict(exponent=params['exponent'], detail_strength=params['detail_strength']),
        metrics_scope="训练区域像素（拟合与评分同一数据，非独立验证）")
    (args.report_dir / "report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(dict(params=str(args.out), report=str(args.report_dir / 'report.json'),
                          exponent=params['exponent'], detail_strength=params['detail_strength']),
                     indent=2))


if __name__ == "__main__":
    main()
