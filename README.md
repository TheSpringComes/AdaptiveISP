# AdaptiveISP (SDI Refactor)

Re-implementation of **AdaptiveISP** (NeurIPS 2024) organized around a
subsystem layout. The algorithm — a task-driven, scene-adaptive ISP whose
per-image pipeline and parameters are selected by reinforcement learning
to maximize downstream detection performance — is that of Wang et al.
(2024).

> Wang Y., Xu T., Zhang F., Xue T., Gu J.,
> *AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection*,
> NeurIPS 2024.
> [Paper](https://arxiv.org/pdf/2410.22939) ·
> [Project page](https://openimaginglab.github.io/AdaptiveISP/) ·
> [Original repository](https://github.com/OpenImagingLab/AdaptiveISP)

The original implementation is a flat layout in which the RL agent, the ISP
filters, the detection model, and the training loop share four top-level
files. In this repository, we retain the same operator set (10 filters), the
same parameter ranges, and the same reinforcement-learning formulation of the
original code, and separate them into six subsystems. `docs/DESIGN.md` gives
the file-level mapping and lists the two intentional algorithmic deviations
(see §7 Stage 2 there).

On the LOD validation set, we observe **mAP@0.5 = 71.6** at 30 000 training
iterations (single seed, our checkpoint
`experiments/lod-adaptiveisp_v1_lod_seed0/ckpt/DynamicISP_iter_30000.pth`),
which is within 0.2 points of the 71.4 reported by Wang et al. (2024) at
full training. Under the same evaluation script, YOLOv3 applied directly to
the raw LOD images (no ISP) returns zero detections, indicating that the
recovered mAP is attributable to the learned pipeline rather than to the
detection backbone.

## Repository layout

```
isp/         image signal processing operators + registry
             (base, registry, operators/{exposure, gamma, ccm, …})
search/      base search space + constraints + priors
             (space, constraint, priors/identity)
controller/  action-selection algorithms
             (base; adaptiveisp/{agent, network, reward})
pipeline/    state and action objects; single-step executor
             (state, action, executor)
tasks/       downstream task interface and implementations
             (base, detection/implementations/yolov3, third_party/yolov3/)
engine/      training / evaluation orchestration + support
             (trainer, runner, dataloader, dataset, replay, util)
configs/     declarative experiment configuration
             (adaptiveisp.yaml)
tools/       command-line entry points
             (train.py, val.py, dataset preparation scripts)
debug/       diagnostic and verification scripts
docs/        DESIGN.md (design and file-level map)
```

## Setup

Tested with Python 3.10, torch 2.6+, torchvision 0.25+. The environment
differs from the original repository, which specified torch 2.0.1 and
torchvision 0.15.2; three compatibility patches in `isp/operators/sharpen.py`,
the yolov3 `torch.load` calls, and the data-root paths in the vendored yaml
files bridge that gap. Details are in `docs/DESIGN.md` §7 Stage 1.

```bash
conda create -n adaptiveisp python=3.10
conda activate adaptiveisp
pip install -r requirements.txt
```

Pretrained YOLOv3 detection backbone (COCO-trained). Place at
`pretrained/yolov3.pt`; the file is distributed by the original authors as
[yolov3.pt](https://github.com/OpenImagingLab/AdaptiveISP/releases/download/v1.0/yolov3.pt).

LOD dataset (used by the paper for the headline result). Download from
[Baidu Drive](https://pan.baidu.com/s/1J0tLRr4IcxPxogcoKKs3Hw?pwd=nips) or
[OneDrive](https://1drv.ms/u/s!Aq1PSygduHX9czHB9WkUNUTUx8o?e=KURDwo), unzip,
then set `path:` in `tasks/third_party/yolov3/data/lod.yaml` to the local
dataset root. For the SynRAW COCO variant used in the paper's supplementary
experiments, `tools/coco_syn_preprocess.py` and
`tools/make_coco_synraw_lists.py` reproduce the preprocessing.

## Training

The default configuration matches the paper's headline recipe (LOD, batch
size 8, 800 epochs):

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --data_name lod \
    --data_cfg tasks/third_party/yolov3/data/lod.yaml \
    --batch_size 8 --epochs 800 \
    --save_path adaptiveisp_lod \
    --seed 0
```

With the runtime penalty term enabled (encourages shorter pipelines; the
paper reports this variant in its cost/accuracy trade-off study):

```bash
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
    --data_name lod \
    --data_cfg tasks/third_party/yolov3/data/lod.yaml \
    --batch_size 8 --epochs 800 \
    --save_path adaptiveisp_lod_rt \
    --seed 0 \
    --runtime_penalty --runtime_penalty_lambda 5e-3
```

Checkpoints are written every 1 000 iterations to
`experiments/<data_name>-<save_path>/ckpt/DynamicISP_iter_*.pth`, in the
schema `{'controller_model', 'optimizer', 'iter', 'operators'}`. On a single
RTX 4090, we measured 9 h 46 m to reach iteration 30 000 at batch size 8;
extrapolating the same rate to the full 800-epoch schedule (100 000
iterations at 1000 images per epoch and batch 8) gives approximately 33 h,
though we did not run the full schedule in this repository. The mAP@0.5 =
71.6 measurement reported above was taken at iteration 30 000.

## Evaluation

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

`tools/val.py` reads the `controller_model` and `operators` keys from the
checkpoint and instantiates the corresponding Controller, then executes the
five-step rollout using `PipelineExecutor.step` and applies the vendored yolov3
`ap_per_class` metric. The tool writes per-image rollouts and predictions
under `val_results/<name>/`.

The two pretrained checkpoints released by Wang et al. (2024),
`ckpt-lod-df-1.0` and `ckpt-lod-df-0.98`, use the pre-refactor schema
(`agent_model` key) and are not directly loadable by `tools/val.py`. To
evaluate them, we retain the pre-refactor code at git tag `v0-baseline`.

## Reproducibility

`--seed <int>` seeds Python's `random`, `numpy`, `torch` (CPU + all CUDA
devices), and `PYTHONHASHSEED`, and enables cuDNN deterministic mode; the
implementation is in `engine/util.set_seed`.

The refactor targets experiment-level reproduction: the mAP obtained by the
refactored code on a given seed and configuration is expected to fall within
the variance band reported for the same configuration by the original
implementation, rather than to reproduce loss trajectories bit-for-bit. We
verified this on one seed on LOD (see the mAP measurement above). Two
deliberate deviations from the paper's implementation are noted in
`docs/DESIGN.md` §7 Stage 2: (i) the Controller forwards only the operator
chosen at each step, rather than all ten operators followed by a
softmax-weighted combination, and (ii) minor floating-point differences
from the module reorganization. Both are permitted by the reproducibility
target and yielded the 71.6 measurement above.

`configs/adaptiveisp.yaml` is the sole source of hyperparameters; a copy is
written into each experiment directory at training start, so the exact
configuration used to produce a checkpoint is recoverable from the run
directory.

## Citation

Please cite the original AdaptiveISP paper:

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
[LODDataset](https://github.com/ying-fu/LODDataset); the detection backbone
is vendored from
[Ultralytics YOLOv3](https://github.com/ultralytics/yolov3) under
`tasks/third_party/yolov3/`.
