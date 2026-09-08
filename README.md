# AdaptiveISP V2-AI (SDI Refactor)

Re-implementation of **AdaptiveISP** (NeurIPS 2024) with V2-AI extensions.
V1 is Wang et al. (2024)'s baseline (10 classical operators, Detection task);
V2-AI adds 16 new operators (7 Samsung neural + 9 Infinite-ISP Torch-native),
a learned STOP action, an exponential repeat penalty, a second downstream task
(FiveK Human Quality), and an automatic post-val canary visualization.

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

## Repository layout

```
isp/
├─ base.py, registry.py         operator base + registry + CANONICAL_ORDER (26)
├─ operators/                   10 classical ops
├─ operators/infinite_isp/      9 Infinite-ISP-derived (Torch-native)
├─ learned/samsung_modular/     7 Samsung Modular Neural ISP wrappers
└─ third_party/modular_neural_isp/    gitignored; 172 MB Samsung code
controller/adaptiveisp/         Controller + STOP head + Reward + HumanReward
pipeline/                       PipelineState (op_usage: int64) + Executor
search/                         SearchSpace + priors
tasks/
├─ detection/                   YOLOv3 wrapper + ReplayMemory + LOD/COCO loaders
├─ human_quality/               FiveK dataset + SSIM/LPIPS metrics + Task
└─ third_party/yolov3/          vendored
engine/
├─ trainer.py                   Detection (1-iter-1-step + ReplayMemory)
├─ trainer_human.py             Human (full T-step rollout per iter)
└─ evaluator.py                 mAP + auto-viz
configs/                        adaptiveisp.yaml (main) + 15 ablation variants
tools/
├─ train.py                     training CLI (`--task {detection,human}`)
├─ val.py                       mAP + auto-viz CLI
└─ visualization/visualizer.py  standalone canary tool
scripts/                        ablation orchestrators + summarizer
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

`tools/val.py` runs mAP AND writes canary PNGs to
`<project>/<name>/visualization/` in one shot. Task is auto-detected
from the ckpt's `task` field.

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
visualization: 4 cases (8 PNGs) → val_results/my_run/visualization/
```

Flags: `--skip_viz` (mAP only), `--viz_cases N` (default 4).

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
