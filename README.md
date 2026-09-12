# AdaptiveISP V3 (SDI Refactor)

Re-implementation of **AdaptiveISP** (NeurIPS 2024) with V2-AI and V3 extensions.
V1 is Wang et al. (2024)'s baseline (10 classical operators, Detection task);
V2-AI adds 16 new operators (7 Samsung neural + 9 Infinite-ISP Torch-native),
a learned STOP action, an exponential repeat penalty, a second downstream task
(FiveK Human Quality), and an automatic post-val canary visualization.
V3 introduces structured search space constraints and PPO-based training.

> Wang Y., Xu T., Zhang F., Xue T., Gu J.,
> *AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection*,
> NeurIPS 2024.
> [Paper](https://arxiv.org/pdf/2410.22939) ·
> [Project page](https://openimaginglab.github.io/AdaptiveISP/) ·
> [Original repository](https://github.com/OpenImagingLab/AdaptiveISP)

The original implementation is a flat layout in which the RL agent, the ISP
filters, the detection model, and the training loop share four top-level
files. In this repository, we retain the same operator set, parameter ranges,
and reinforcement-learning formulation of the original code, split them into
six subsystems, and layer the V2-AI extensions on top. `docs/V1DESIGN.md`
gives the V1 file-level mapping and lists the two intentional algorithmic
deviations (see §7 Stage 2 there).

On LOD, we observe **mAP@0.5 = 71.6** at 30 000 training iterations of V1
(single seed, `experiments/lod-adaptiveisp_v1_lod_seed0/ckpt/DynamicISP_iter_30000.pth`),
within 0.2 pts of the 71.4 reported at full training. Under the same
evaluation script, YOLOv3 applied directly to the raw LOD images returns
zero detections, indicating the recovered mAP is attributable to the
learned pipeline rather than the detection backbone.

## V2-AI at a glance

**Operator bank: 10 → 26 ops.** Three provider families sit side-by-side
in the search space, each with a stable prefix so `op_usage` columns don't
drift across families:

| Family | # | Path | Ops |
|---|---:|---|---|
| Classical (V1) | 10 | `isp/operators/` | `exposure gamma ccm sharpen denoise tone contrast saturation wnb whitebalance` |
| Neural (Samsung Modular ISP) | 7 | `isp/learned/samsung_modular/` | `n_denoise n_awb n_gain n_gtm n_chroma n_gamma n_detail` — each `x + α·(F(x)−x)` with frozen backbone F |
| Infinite-ISP-derived (Torch) | 9 | `isp/operators/infinite_isp/` | `inf_awb_{grayworld,norm2,pca} inf_digital_gain inf_ldci inf_unsharp inf_nlm inf_ebf inf_saturation` |

**Learned STOP action + exponential repeat penalty.** The Controller's
select_head outputs `n_ops + 1` logits; the extra column is a STOP action
that submits the current pipeline. A per-op usage counter (`state.op_usage`
is `int64` count, not bool bitmap) drives a repeat penalty
`base × 2^prev_count`, so the second use of an op costs 2×, the third 4×,
and so on. This prevents policy collapse into single-op loops that were
common in V1 with the flat penalty.

**Second task: Human Quality (FiveK + Expert C).** `tasks/human_quality/`
provides a perceptual objective — terminal-only reward
`R = Q(I_T) − Q(I_0)` where `Q = λ_ssim·SSIM − λ_lpips·LPIPS`. Both SSIM
and LPIPS(AlexNet) are frozen. `HumanTrainer` runs full T-step rollouts
per iter (no ReplayMemory).

**Automatic canary visualization.** `tools/val.py` writes both mAP + PNGs
in one shot — see [Evaluation](#evaluation) below.

## V3 Extensions

V3 introduces three structural improvements to the search space and training algorithm:

**A1: Canonical Backbone.** A fixed ISP pipeline (AWB → CCM → GTM → Gamma) runs before the Controller, providing a reliable baseline RGB image. This reduces the search space from "RAW → task-optimized RGB" to "baseline RGB → task-optimized RGB", making the RL problem more tractable. Implemented in `front_isp/canonical.py` (legacy `canonical_backbone.enabled` in config still works; V3.1 supersedes it with the four-mode Front ISP, see below).

**A2: Action Mask.** Dynamic constraints on the Controller's action space:
- **No-repeat mask**: prevents selecting the same operator twice in a rollout
- **Order mask**: enforces pipeline ordering (e.g., denoise before sharpen)
- **Group budget mask**: limits total selections from related operator groups (e.g., at most 2 tonal adjustments)

Implemented in `search/priors/action_mask.py`, controlled by `action_mask` block in config.

**A3: PPO Training.** Replaces REINFORCE with Proximal Policy Optimization (PPO) for more stable training. Uses GAE(λ) advantage estimation, clipped surrogate objective, and value function baseline. Implemented in `controller/adaptiveisp/ppo.py`, controlled by `rl_algo.name: ppo` in config.

### V3 Results

**Human Quality (FiveK + Expert C, 25 epochs):**

| Config | SSIM ↑ | LPIPS ↓ | Q ↑ | Rollout |
|--------|--------|---------|-----|---------|
| H0 (V2 baseline) | 0.593 | 0.407 | +0.186 | 1.00/8 |
| H1 (+backbone) | 0.540 | 0.370 | +0.170 | 1.00/8 |
| H2 (+mask) | 0.588 | 0.423 | +0.165 | 7.00/8 |
| H3 (+PPO) | 0.616 | 0.374 | +0.242 | 6.96/8 |
| **H3-s5-rew** | **0.605** | **0.318** | **+0.287** | 4.00/5 |

H3-s5-rew uses `test_steps=5` with tuned reward shaping (stop_bonus=0.01, early_stop_penalty=2.0), achieving **Q=+0.287** (4.5pt above H3 baseline) with 37.5% less compute.

**Detection (LOD, 25 epochs):** V3 shows negative transfer at this budget (E0=0.686, E1=0.604, E2=0.564, E3=0.597 mAP@0.5). The backbone constrains the Controller's ability to optimize for YOLOv3's specific feature requirements. Longer training (800 epochs) may recover performance.

### V3 Configs and Scripts

```bash
# Detection ablation ladder (E0→E1→E2→E3)
bash scripts/run_v3_ablation.sh

# Human ablation ladder (H0→H1→H2→H3)
bash scripts/run_v3_human_ablation.sh

# H3-s5 variants (test_steps=5, lr/reward tuning)
bash scripts/run_v3_human_s5_variants.sh
```

Configs:
- `configs/adaptiveisp_v3_e{0,1,2,3}.yaml` — Detection ablation
- `configs/adaptiveisp_human_v3_e{0,1,2,3}.yaml` — Human ablation
- `configs/adaptiveisp_human_v3_e3_s5*.yaml` — H3-s5 variants

## V3.1 Front ISP（四种统一模式）

V3.1 重新整理 Front ISP：放在 AdaptiveISP 之前先做一次基础 ISP 处理，再把结果交给 RL Controller。WB / CCM / Gamma 这类基础处理不必让 RL 一步步搜索，可以提前用固定参数、训练参数或现有 ISP 完成；AdaptiveISP 保持原有逻辑，只负责 Front ISP 之后的算子选择与参数优化。

```
RAW → Input Adapter (Dataset layer: Bayer reconstruction per per-file CFA pattern
      → demosaic 0.5*Malvar + 0.5*Bilinear → canonical linear RGB)
    → Front ISP (identity | fixed | learnable | external) → Baseline RGB
    → AdaptiveISP (Adaptive Tail, unchanged) → Task output
```

统一四种模式（`front_isp.type`）：

| type | 说明 | 实现位置 |
|---|---|---|
| `identity` | 不使用 Front ISP（对照组，旧名 `none`） | `front_isp/identity.py` |
| `fixed` | 人工配置 ISP：WB / CCM / Bias / Gamma / Exposure / Smoothstep 等模块，顺序与参数全部由 `fixed.modules` 配置指定，训练不更新；模块注册表开放扩展 | `front_isp/fixed.py` |
| `learnable` | 固定结构 + 可训练参数（WB gain / CCM / Bias / Gamma，camera-specific 参数表，**两阶段训练**：Stage 1 只训 Front ISP 并冻结；Stage 2 跑 AdaptiveISP。不联合训练 | `front_isp/learnable/` |
| `external` | 接入现有开源 ISP：`backend: infinite_isp \| samsung_isp`，wrapper 统一输入输出 | `front_isp/external.py` + wrappers |

- **两阶段训练（learnable）**：Stage 1 `tools/train.py --task learnable`（loss = λ₁L1 + λ_s(1−SSIM) + λ_pLPIPS vs Expert C），保存 ckpt 并冻结；Stage 2 `--task human` 加载冻结的 Front ISP 跑 AdaptiveISP。不做联合训练，避免两部分同时变化后难以归因。
- **消融四路对比**：不用（`adaptiveisp_human.yaml` baseline）/ 人工固定（`v31_fixed.yaml`）/ 学习参数（`v31_stage2.yaml`）/ 开源 ISP（`v31_external.yaml`）。看两个问题：Front ISP 有没有帮助；基础色彩问题前置解决后 RL 是否更容易训练。

```bash
# One-time: camera metadata for FiveK (camera ID per sample)
python tools/fivek_camera_metadata.py \
    --raw-root /home/jing/datasets/fivek/fivek_dataset/raw_photos \
    --out      /home/jing/datasets/fivek/camera.json

# Stage 1: learnable Front ISP pretrain (freeze after)
python tools/train.py --task learnable \
    --cfg configs/adaptiveisp_human_v31_pretrain.yaml \
    --save_path v31_stage1 --epochs 5 --batch_size 8

# Stage 2: AdaptiveISP with frozen learnable Front ISP
python tools/train.py --task human \
    --cfg configs/adaptiveisp_human_v31_stage2.yaml \
    --save_path v31_stage2

# 小规模验证：任意训练加 --max_iters N 截断迭代数
```

Configs: `v31_pretrain.yaml`（Stage 1）、`v31_stage2.yaml`（Stage 2，冻结加载 Stage-1 ckpt）、`v31_fixed.yaml`（fixed）、`v31_external.yaml`（external）。Legacy 类型名（none / canonical / infinite_isp / modular_neural_isp）保留为别名。Deliberately out of scope（future versions）：image-adaptive CCM、neural parameter predictors、RL-searched front-ISP params、local tone mapping。

## Repository layout

```
isp/
├─ base.py, registry.py         operator base + registry + CANONICAL_ORDER (26)
├─ operators/                   10 classical ops
├─ operators/infinite_isp/      9 Infinite-ISP-derived (Torch-native)
├─ learned/samsung_modular/     7 Samsung Modular Neural ISP wrappers
└─ third_party/modular_neural_isp/    gitignored; 172 MB Samsung code
front_isp/                      V3.1 可插拔前置 ISP（四种统一模式）
├─ identity.py                  identity（对照组，旧名 none）
├─ fixed.py                     fixed：配置驱动模块链 + 开放模块注册表
├─ learnable/                   learnable：可训练 WB/CCM/Bias/Gamma（两阶段训练）
├─ canonical.py                 legacy 固定链（V3-A1）
├─ external.py                  external：backend 分发（infinite_isp | samsung_isp）
└─ raw_adapter.py               Input Adapter：Bayer 重建 → demosaic → linear RGB
controller/adaptiveisp/         Controller + STOP head + Reward + HumanReward + PPO
pipeline/                       PipelineState (op_usage: int64) + Executor + TrajectoryBuffer
search/
├─ space.py                     SearchSpace (composes priors)
└─ priors/action_mask.py        NoRepeatMask, OrderMask, GroupBudgetMask
tasks/
├─ detection/                   YOLOv3 wrapper + ReplayMemory + LOD/COCO loaders
├─ human_quality/               FiveK dataset + SSIM/LPIPS metrics + Task
└─ third_party/yolov3/          vendored
engine/
├─ trainer.py                   Detection (1-iter-1-step + ReplayMemory, or PPO)
├─ trainer_human.py             Human Stage 2 (full T-step rollout per iter, or PPO)
├─ trainer_learnable.py         Human Stage 1 (learnable Front ISP pretrain)
└─ evaluator.py                 mAP + auto-viz
configs/                        adaptiveisp.yaml (main) + V3/V3.1 ablation variants
tools/
├─ train.py                     training CLI (`--task {detection,human,learnable}`)
├─ val.py                       mAP + auto-viz CLI
├─ fivek_build_cache.py         Expert-C cache build（4-plane pack）
├─ fivek_cfa_scan.py            全量 DNG CFA pattern 扫描
├─ verify_raw_adapter.py        Input Adapter 验证套件（rawpy 真值对拍）
├─ vis_val_color.py             颜色链路 4 格诊断可视化
├─ preview.py                   单样本 Front ISP 对比预览
└─ visualization/visualizer.py  standalone canary tool
scripts/
├─ run_ablations.sh             V2 ablation orchestrator
├─ run_v3_ablation.sh           V3 Detection ablation ladder
├─ run_v3_human_ablation.sh     V3 Human ablation ladder
├─ run_v3_human_s5_variants.sh  H3-s5 variant sweep
├─ run_v31_ablation.sh          V3.1 Front ISP 四模式消融（支持 --smoke）
└─ val_v3_ablation.sh           V3 val runner
debug/smoke/                    8 smoke tests (imports, ops, pipeline, front_isp,
                                learnable, controller, e2e, human_quality e2e)
docs/V1DESIGN.md                V1 architecture doc
```

## FiveK cache (Human Quality task)

The Expert-C cache is generated by `tools/fivek_build_cache.py` from the local
DNG corpus + the old cache's targets:

```bash
# raw plane: DNG mosaic → per-CFA black/white normalize → R,G,G,B pack →
#            EXIF flip applied (0→k0, 5→k1, 6→k3, 3→k2) → INTER_AREA resize
# target:    copied verbatim from the old cache (EXIF-corrected TIFFs)
python tools/fivek_build_cache.py \
    --old-cache /home/jing/datasets/fivek/cache_expert_c_old \
    --raw-root  /home/jing/datasets/fivek/fivek_dataset/raw_photos \
    --out       /home/jing/datasets/fivek/cache_expert_c

# post-hoc alignment audit → cache_dir/_alignment.json
python tools/fivek_scan_cache.py --cache /home/jing/datasets/fivek/cache_expert_c
```

Both planes are stored upright — no load-time rotation. `FiveKDataset`
auto-filters the split against `_alignment.json` (drops raw/target pairs whose
correlation is below 0.5 — legacy TIFF/DNG mismatches) and files absent from
the cache (42 X-Trans / mirrored-flip images skipped at build). Current
counts: train 4818/4894, val 97/100, 35 cameras (`camera.json`).

**CFA pattern metadata.** The cache packs planes by color code, so the 2×2
CFA arrangement per file is needed at load time for Bayer reconstruction:

```bash
# 全量 DNG pattern 扫描 → /home/jing/datasets/fivek/cfa_pattern.json
python tools/fivek_cfa_scan.py
# FiveK 实测分布（5000 DNG）：RGGB 3715 / BGGR 675 / GBRG 451 / GRBG 111
```

At load time the Input Adapter (`front_isp/raw_adapter.py`) reconstructs the
full-resolution mosaic per the file's true pattern, then demosaics
(0.5×Malvar + 0.5×Bilinear) into canonical linear RGB — so `FiveKDataset`
returns **full-resolution** `(3, H, W)` linear image paired with the
full-res sRGB target. See `tools/verify_raw_adapter.py` for the validation
suite (17/17 checks incl. rawpy ground-truth comparison per pattern).

⚠️ All pre-rebuild experiments (v2ai_*, v3_h*, lod-*, v31_*) were trained on
the misaligned cache and live in `experiments_archive_polluted_data/` /
`experiments/` — their numbers are not comparable to new-data runs.

## Setup

Tested with Python 3.10, torch 2.6+, torchvision 0.25+.

```bash
conda create -n adaptiveisp python=3.10
conda activate adaptiveisp
pip install -r requirements.txt
```

**YOLOv3 backbone (Detection task).** Place at `pretrained/yolov3.pt`
([download link](https://github.com/OpenImagingLab/AdaptiveISP/releases/download/v1.0/yolov3.pt)).

**LOD dataset (Detection).** Download from
[Baidu Drive](https://pan.baidu.com/s/1J0tLRr4IcxPxogcoKKs3Hw?pwd=nips)
or [OneDrive](https://1drv.ms/u/s!Aq1PSygduHX9czHB9WkUNUTUx8o?e=KURDwo),
unzip, set `path:` in `tasks/third_party/yolov3/data/lod.yaml`.

**FiveK + Expert C (Human Quality).** Sample lists at
`<fivek_root>/{train,val}_expert_c.txt` referencing `.npz` frames
(Bayer4-packed raw + full-res Expert-C sRGB target). Path is set via
`human_quality.fivek_root` in the config yaml
(defaults to `/home/jing/datasets/fivek`).

**Samsung Modular Neural ISP (Neural operators).** Vendored under
`isp/third_party/modular_neural_isp/` (gitignored, 172 MB). Only needed
if you enable any `n_*` op in the config.

## Training

### Detection (LOD, V1 baseline recipe with V2-AI extensions active)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py --task detection \
    --data_name lod \
    --data_cfg tasks/third_party/yolov3/data/lod.yaml \
    --batch_size 8 --epochs 800 \
    --save_path adaptiveisp_lod --seed 0
```

Checkpoints land at `experiments/lod-adaptiveisp_lod/ckpt/DynamicISP_iter_*.pth`
every 1 000 iters, in schema `{'controller_model', 'optimizer', 'iter', 'operators'}`.

### Human Quality (FiveK + Expert C)

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py --task human \
    --batch_size 4 --epochs 30 --imgsz 512 \
    --save_path v2ai_human --cfg configs/adaptiveisp_human.yaml
```

Same schema plus `'task': 'human_quality'`. `HumanTrainer` runs a
full-episode rollout per iter (one `.backward()` covering all T steps),
and the terminal `Q(I_T) − Q(I_0)` is the only non-zero task_delta.

### V3 Training

V3 configs enable the canonical backbone, action masks, and/or PPO:

```bash
# V3 Human E3 (backbone + mask + PPO)
CUDA_VISIBLE_DEVICES=0 python tools/train.py --task human \
    --batch_size 4 --epochs 25 --imgsz 512 \
    --save_path v3_h3 --cfg configs/adaptiveisp_human_v3_e3.yaml

# V3 Human E3 with test_steps=5 (faster, tuned reward)
CUDA_VISIBLE_DEVICES=0 python tools/train.py --task human \
    --batch_size 4 --epochs 25 --imgsz 512 \
    --save_path v3_h3_s5 --cfg configs/adaptiveisp_human_v3_e3_s5_rew.yaml

# V3 Detection E3 (backbone + mask + PPO)
CUDA_VISIBLE_DEVICES=0 python tools/train.py --task detection \
    --data_name lod \
    --data_cfg tasks/third_party/yolov3/data/lod.yaml \
    --batch_size 4 --epochs 25 --imgsz 512 \
    --save_path v3_e3 --cfg configs/adaptiveisp_v3_e3.yaml
```

Key config knobs:
- `front_isp.type: identity | fixed | learnable | external` — V3.1 统一 Front ISP 模式（legacy `canonical_backbone.enabled` 仍可用）
- `action_mask.no_repeat.enabled: true` — prevent op reuse
- `action_mask.order.enabled: true` — enforce pipeline ordering
- `action_mask.group_budget.enabled: true` — limit group selections
- `rl_algo.name: ppo` — use PPO instead of REINFORCE
- `test_steps: 5` — rollout length (default 8)
- `min_rollout_length: 3` — minimum steps before STOP is allowed

### What the print block shows

Every `print_freq` iters, both trainers print (windowed since last print):

```
----- iter N/M [HH:MM:SS] elapsed T | X.XX it/s | ETA T -----
  loss     agent=... val=... detect=... reward=+...        # Detection
  loss     agent=... val=... Q_0=+... Q_T=+... ΔQ=+... SSIM=... LPIPS=...  # Human
  reward   task=+... ent=-... use=-... estop=-... ovfl=-... [stop+=+...] [runt=-...]
  policy   entropy=X.XXX/log(n+1)  argmax=XX%  stop=XX% (learned=XX%, timelimit=XX%)
  example  eval-argmax shadow rollouts (canary / batch / fresh)
  ops      window(N) top: exposure:12 whitebalance:8 ...   cum neural NN/NN = X%
```

`reward` breaks the total into 5 components (+ optional `stop+`, `runt`);
`policy` reports mean entropy vs its ceiling, argmax-match rate
(exploring vs locked-in), and STOP breakdown into learned vs time-limit.

## Evaluation

`tools/val.py` runs the eval loop AND writes canary PNGs in one shot.
The eval loop auto-routes on the ckpt's `task` field:

  - `detection` (V1 default) — mAP against LOD/COCO
  - `human_quality` — SSIM / LPIPS / Q + rollout length + learned-STOP pct
    against FiveK val

Text artifacts (`val_log.txt` for both branches, plus `records.txt` for
detection) land under `<project>/<name>/`; the canary PNGs land under the
checkpoint's own experiment folder (`experiments/<exp>/visualization/`),
so they sit next to the ckpt and cfg that produced them.

**Detection:**
```bash
CUDA_VISIBLE_DEVICES=0 python tools/val.py \
    --weights pretrained/yolov3.pt \
    --isp_weights experiments/lod-adaptiveisp_lod/ckpt/DynamicISP_iter_30000.pth \
    --data_name lod \
    --data tasks/third_party/yolov3/data/lod.yaml \
    --imgsz 512 --batch-size 1 --steps 5 \
    --cfg_file configs/adaptiveisp.yaml \
    --project val_results --name my_run --exist-ok
```

Output:
```
                 Class     Images  Instances     P     R  mAP50  mAP75  mAP50-95
                   all        100        250 0.712 0.634  0.706  0.412      0.30
visualization: 4 cases (8 PNGs) → experiments/lod-adaptiveisp_lod/visualization/
```

**Human Quality:** (FiveK paths come from the cfg's `human_quality:` block,
so no `--data`/`--weights`/`--data_name` needed)
```bash
CUDA_VISIBLE_DEVICES=0 python tools/val.py \
    --isp_weights experiments/v2ai_human/ckpt/HumanISP_iter_28000.pth \
    --cfg_file experiments/v2ai_human/adaptiveisp_human.yaml \
    --project val_results --name v2ai_human_iter28000 --exist-ok
```

Output:
```
===== VAL (Human Quality) =====
  samples: 100
  SSIM:  0.xxxx
  LPIPS: 0.xxxx
  Q:     +0.xxxx
  mean rollout length: X.XX/8
  pct learned-STOP (before time-limit): XX.X%
===============================
visualization: 4 cases (8 PNGs) → experiments/v2ai_human/visualization/
```

Flags (both branches): `--skip_viz` (numbers only), `--viz_cases N` (default 4).

**Standalone visualizer** — for a ckpt you don't want to re-eval:
```bash
python -m tools.visualization.visualizer --exp-dir experiments/lod-adaptiveisp_lod --n-cases 4
```

## Ablation infrastructure

`configs/` ships 15 ablation variants covering: rollout length (`test_steps`),
repeat-penalty base, critic multiplier, op-set restrictions (classical only vs
+neural), and exploration/lr/batch/grad-clip sweeps. Two orchestrators:

```bash
bash scripts/run_ablations.sh d5 hbase hnostop            # sequential
bash scripts/run_d_tuning_parallel_v2.sh                   # parallel pairs, OOM-aware
python scripts/summarize_ablations.py                      # markdown table from logs
```

## Reproducibility

`--seed <int>` seeds `random`, `numpy`, `torch` (CPU + CUDA), and
`PYTHONHASHSEED`, and enables cuDNN deterministic mode (in
`engine/util.set_seed`).

The refactor targets experiment-level reproduction: the mAP obtained by
the refactored code on a given seed and configuration is expected to fall
within the variance band reported for the same configuration by the
original implementation, rather than to reproduce loss trajectories
bit-for-bit. See `docs/V1DESIGN.md` §7 Stage 2 for the two deliberate V1
deviations. `configs/adaptiveisp.yaml` is the sole source of
hyperparameters; a copy is written into each experiment directory at
training start.

The two pretrained checkpoints released by Wang et al. (2024)
(`ckpt-lod-df-1.0` and `ckpt-lod-df-0.98`) use the pre-refactor schema
(`agent_model` key) and are not directly loadable by `tools/val.py`. Use
git tag `v0-baseline` to evaluate them.

## Citation

```bibtex
@inproceedings{wang2024adaptiveisp,
    title     = {AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection},
    author    = {Yujin Wang and Tianyi Xu and Fan Zhang and Tianfan Xue and Jinwei Gu},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2024}
}
```

## Acknowledgements

We build directly on [AdaptiveISP](https://github.com/OpenImagingLab/AdaptiveISP)
by Wang et al. (2024). The LOD dataset is from
[LODDataset](https://github.com/ying-fu/LODDataset); the detection
backbone is vendored from
[Ultralytics YOLOv3](https://github.com/ultralytics/yolov3) under
`tasks/third_party/yolov3/`. The Samsung neural operators wrap
[Modular Neural ISP](https://github.com/SamsungLabs/modular-neural-isp)
(Afifi et al., SIGGRAPH Asia 2026); the classical Infinite-ISP-derived
operators are Torch-native reimplementations of algorithms from
[Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP) by
10x-Engineers.
