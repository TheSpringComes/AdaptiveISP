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

**A1: Canonical Backbone.** A fixed ISP pipeline (AWB → CCM → GTM → Gamma) runs before the Controller, providing a reliable baseline RGB image. This reduces the search space from "RAW → task-optimized RGB" to "baseline RGB → task-optimized RGB", making the RL problem more tractable. Implemented in `pipeline/backbone.py`, controlled by `canonical_backbone.enabled` in config.

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

# Generate comparison visualizations
python tools/v3_summary_viz.py
```

Configs:
- `configs/adaptiveisp_v3_e{0,1,2,3}.yaml` — Detection ablation
- `configs/adaptiveisp_human_v3_e{0,1,2,3}.yaml` — Human ablation
- `configs/adaptiveisp_human_v3_e3_s5*.yaml` — H3-s5 variants

## Repository layout

```
isp/
├─ base.py, registry.py         operator base + registry + CANONICAL_ORDER (26)
├─ operators/                   10 classical ops
├─ operators/infinite_isp/      9 Infinite-ISP-derived (Torch-native)
├─ learned/samsung_modular/     7 Samsung Modular Neural ISP wrappers
└─ third_party/modular_neural_isp/    gitignored; 172 MB Samsung code
controller/adaptiveisp/         Controller + STOP head + Reward + HumanReward + PPO
pipeline/                       PipelineState (op_usage: int64) + Executor + Backbone + TrajectoryBuffer
search/
├─ space.py                     SearchSpace (composes priors)
└─ priors/action_mask.py        NoRepeatMask, OrderMask, GroupBudgetMask
tasks/
├─ detection/                   YOLOv3 wrapper + ReplayMemory + LOD/COCO loaders
├─ human_quality/               FiveK dataset + SSIM/LPIPS metrics + Task
└─ third_party/yolov3/          vendored
engine/
├─ trainer.py                   Detection (1-iter-1-step + ReplayMemory, or PPO)
├─ trainer_human.py             Human (full T-step rollout per iter, or PPO)
└─ evaluator.py                 mAP + auto-viz
configs/                        adaptiveisp.yaml (main) + V3 ablation variants
tools/
├─ train.py                     training CLI (`--task {detection,human}`)
├─ val.py                       mAP + auto-viz CLI
├─ visualization/visualizer.py  standalone canary tool
└─ v3_summary_viz.py            V3 ablation comparison figure generator
scripts/
├─ run_ablations.sh             V2 ablation orchestrator
├─ run_v3_ablation.sh           V3 Detection ablation ladder
├─ run_v3_human_ablation.sh     V3 Human ablation ladder
├─ run_v3_human_s5_variants.sh  H3-s5 variant sweep
└─ val_v3_ablation.sh           V3 val runner
debug/smoke/                    5 smoke tests (imports, ops, pipeline, controller, e2e)
docs/V1DESIGN.md                V1 architecture doc
```

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
`<fivek_root>/{train,val}_expert_c.txt` referencing `.npz` Bayer4-packed
frames. Path is set via `human_quality.fivek_root` in the config yaml
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
- `canonical_backbone.enabled: true` — enable fixed ISP backbone
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
