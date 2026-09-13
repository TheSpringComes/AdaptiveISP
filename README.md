# AdaptiveISP（V3.1）

基于强化学习的自适应 ISP 优化框架：RL Controller 逐步从算子库中挑选
ISP 算子并回归其参数，以最大化下游任务指标（目标检测 / 人眼感知质量）。
本仓库是对 [AdaptiveISP](https://github.com/OpenImagingLab/AdaptiveISP)
（Wang et al., NeurIPS 2024）的模块化重写，并叠加了 Front ISP、动作
约束、PPO 等扩展。

> **本文只描述当前版本（V3.1）的使用方式。** 各版本（V1 重构 → V2-AI
> 算子/任务扩展 → V3 搜索结构 → V3.1 Front ISP）的动机、改动与历史
> 实验结论见 [docs/VERSION_HISTORY.md](docs/VERSION_HISTORY.md)。

## 整体流水线

```
RAW → Input Adapter（Dataset 层：按 per-file CFA 重建 Bayer
      → demosaic 0.5×Malvar + 0.5×Bilinear → canonical linear RGB）
    → Front ISP（identity | fixed | learnable | external）→ Baseline RGB
    → AdaptiveISP（RL Controller 逐步选算子/参数 + STOP）→ Task
      ├── Detection: LOD/COCO + YOLOv3 → mAP
      └── Human Quality: FiveK vs Expert C → SSIM/LPIPS/Q
```

Front ISP 先用固定参数、离线拟合参数、可训练参数或现成开源 ISP 产生
稳定的基础 RGB（WB / CCM / Gamma 这类基础处理不交给 RL 搜索）；
AdaptiveISP 只负责其后的增量算子选择与参数优化。

## 环境准备

测试环境：Python 3.10+，torch 2.x + CUDA（本机实测 torch 2.10 / cu128）。

```bash
conda create -n adaptiveisp python=3.10
conda activate adaptiveisp
pip install -r requirements.txt
```

**数据与权重**（`configs/*.yaml` 里的路径均为本机绝对路径，按需修改）：

| 资产 | 用途 | 位置 / 准备方式 |
|---|---|---|
| YOLOv3 权重 | Detection 任务 | `pretrained/yolov3.pt`（[下载](https://github.com/OpenImagingLab/AdaptiveISP/releases/download/v1.0/yolov3.pt)） |
| LOD 数据集 | Detection 任务 | [百度网盘](https://pan.baidu.com/s/1J0tLRr4IcxPxogcoKKs3Hw?pwd=nips) / [OneDrive](https://1drv.ms/u/s!Aq1PSygduHX9czHB9WkUNUTUx8o?e=KURDwo)，路径写在 `tasks/third_party/yolov3/data/lod.yaml` |
| FiveK Expert-C cache | Human 任务 | `human_quality:` 配置块，默认 `/home/jing/datasets/fivek`（构建见[下文](#fivek-数据准备human-任务)） |
| Samsung Modular Neural ISP | `n_*` 算子 / `samsung_isp` 前端 | `git clone https://github.com/SamsungLabs/modular_neural_isp isp/third_party/modular_neural_isp`（含权重，~172 MB，gitignored） |
| InfiniteISP_RAW | `infinite_isp` 前端 | `front_isp/third_party/InfiniteISP_RAW`（上游 Infinite-ISP 整理版 + 已验证基线配置；wrapper 缺仓库时报错会给指引） |

冒烟测试（无数据、无权重、< 60s；改完代码先跑这个）：

```bash
bash debug/smoke/run.sh    # 7 项：imports / operators / pipeline / front_isp
                           # / learnable / controller / e2e
```

## 训练

统一入口 `tools/train.py`，`--task` 选 trainer：

| `--task` | Trainer | 用途 |
|---|---|---|
| `detection` | `engine/trainer.py` | LOD/COCO + YOLOv3，1-step TD + ReplayMemory（或 PPO） |
| `human` | `engine/trainer_human.py` | FiveK vs Expert C，每 iter 完整 T 步 rollout |
| `learnable` | `engine/trainer_learnable.py` | Front ISP Stage 1 预训练（只训 Front ISP 参数） |

产物落在 `experiments/<save_path>/`：`ckpt/`（每 `save_model_freq` iters
一个）、`logs/`（log.txt + TensorBoard）、`images/`、启动时复制的 config
副本；Human 任务结束时额外写 `final_val.json`（val 指标 + 全程算子选择
累计）。Detection 的 `save_path` 会自动加 `data_name-` 前缀。

### Human Quality（FiveK + Expert C）

```bash
# 基线（无 Front ISP，identity）
python tools/train.py --task human \
    --cfg configs/adaptiveisp_human.yaml \
    --save_path my_human --batch_size 4 --epochs 25 --imgsz 512

# fixed Front ISP（FittedISP 离线拟合参数，见下文"重新拟合"）
python tools/train.py --task human \
    --cfg configs/adaptiveisp_human_v31_fixed.yaml --save_path v31_fixed

# learnable Front ISP（两阶段）
python tools/train.py --task learnable \
    --cfg configs/adaptiveisp_human_v31_pretrain.yaml \
    --save_path v31_stage1 --epochs 5 --batch_size 8   # Stage 1: 只训 Front ISP
python tools/train.py --task human \
    --cfg configs/adaptiveisp_human_v31_stage2.yaml \
    --save_path v31_stage2                              # Stage 2: 冻结 Front ISP 跑 RL
                                                        # （ckpt 路径写在 cfg 的
                                                        #  front_isp.learnable.ckpt）

# external Front ISP（infinite_isp / samsung_isp）
python tools/train.py --task human \
    --cfg configs/adaptiveisp_human_v31_external.yaml --save_path v31_external
```

小规模验证：任意训练加 `--max_iters N` 截断迭代数（如 `--max_iters 40`）。

### Detection（LOD）

```bash
python tools/train.py --task detection \
    --data_name lod --data_cfg tasks/third_party/yolov3/data/lod.yaml \
    --cfg configs/adaptiveisp.yaml \
    --batch_size 8 --epochs 800 --save_path my_lod --seed 0
# ckpt → experiments/lod-my_lod/ckpt/DynamicISP_iter_*.pth
```

### 关键配置项（`configs/adaptiveisp.yaml` 为完整参考）

```yaml
front_isp:
  enabled: true
  type: identity | fixed | learnable | external   # 四种模式，见下表
  # 各模式自己的内部配置（fixed.params / learnable.* / external.backend ...）

action_mask:            # V3 动作约束（默认全关 = 无约束）
  no_repeat:  {enabled: false}        # 禁止同一算子二次选择（exempt 白名单）
  order:      {enabled: false, rules: []}   # before/after 顺序约束
  group_budget: {enabled: false, groups: []} # 相关算子组配额

rl_algo:
  name: actor_critic | ppo            # ppo = T-step rollout + GAE + clip

test_steps: 8            # rollout 长度
min_rollout_length: 1    # 最少步数（之前禁用 STOP）
```

四种 Front ISP 模式：

| type | 说明 | 实现 |
|---|---|---|
| `identity` | 不用 Front ISP（对照组；旧名 `none`） | `front_isp/identity.py` |
| `fixed` | FittedISP 拟合方法（色调指数 → CCM+偏置 → 细节控制），全局一套参数，运行期完全固定。参数由 `tools/fit_front_isp.py` 离线拟合，存于 `configs/front_isp/fitted_fivek.json` | `front_isp/fixed.py`（方法来自 `front_isp/FittedISP/`） |
| `learnable` | 可训练 WB gain / CCM / Bias / Gamma，camera-specific 参数表。**两阶段**：Stage 1 只训 Front ISP 并冻结，Stage 2 跑 AdaptiveISP，不联合训练 | `front_isp/learnable/` |
| `external` | 现成开源 ISP：`backend: infinite_isp \| samsung_isp`，wrapper 统一输入输出 | `front_isp/external.py` + wrappers |

legacy 类型名（`none` / `canonical` / `infinite_isp` / `modular_neural_isp`）
与旧键 `canonical_backbone.enabled` 仍可用。

### 训练日志怎么看

每 `print_freq` iters 打一个块（Human 分支；Detection 格式相同，loss 行
为 `agent/val/detect/reward`，example 行为 canary/batch/fresh 三条
eval-argmax shadow rollout）：

```
----- iter N/M [HH:MM:SS] elapsed T | X.XX it/s | ETA T -----
  loss     agent=... val=... Q_0=+... Q_T=+... ΔQ=+... SSIM=... LPIPS=...
  reward   task=+... ent=-... use=-... estop=-... ovfl=-... [stop+=+...] [runt=-...]
  policy   entropy=X.XXX/log(n+1)  argmax=XX%  stop=XX% (learned=XX%, timelimit=XX%)
  example  sample 0  Q_0=+... → Q_T=+... (ΔQ=+...)
           picks: exposure(0.42) → n_gamma(0.71) → STOP
  ops      window(N) top: exposure:12 whitebalance:8 ...
           cum neural NN/NN = X%
```

- `reward`：总奖励按 task 增量 / 熵惩罚 / 使用惩罚 / 早停惩罚 / 溢出
  惩罚（+可选 stop bonus、runtime 惩罚）分解；
- `policy`：熵 vs 上限、argmax 命中率（探索 vs 锁定）、STOP 里 learned
  vs 到时强制停的占比；
- `example`：sample 0 的完整算子序列（带首参数）；
- `ops`：窗口内选择频次 top6 + neural 算子累计占比。

## 评估

`tools/val.py` 按 ckpt 的 `task` 字段自动路由，一次跑出指标 + canary
可视化 PNG（落盘在 ckpt 所在实验目录的 `visualization/` 下）：

```bash
# Human（FiveK 路径来自 ckpt 目录里的 cfg 副本，无需 --data/--weights）
python tools/val.py \
    --isp_weights experiments/v31_fixed/ckpt/HumanISP_iter_1000.pth \
    --cfg_file experiments/v31_fixed/adaptiveisp_human_v31_fixed.yaml \
    --project val_results --name v31_fixed_val --exist-ok

# Detection
python tools/val.py \
    --isp_weights experiments/lod-my_lod/ckpt/DynamicISP_iter_30000.pth \
    --weights pretrained/yolov3.pt \
    --data tasks/third_party/yolov3/data/lod.yaml --data_name lod \
    --cfg_file configs/adaptiveisp.yaml \
    --project val_results --name lod_val --exist-ok
```

Human 输出（val_log.txt 同步落盘）：

```
===== VAL (Human Quality) =====
  samples: 97
  SSIM:  0.xxxx
  LPIPS: 0.xxxx
  PSNR:  XX.XX dB   ΔE76: XX.XX
  Q:     +0.xxxx
  mean rollout length: X.XX/8
  pct learned-STOP (before time-limit): XX.X%
===============================
```

通用 flag：`--skip_viz`（只要数字）、`--viz_cases N`（默认 4 张）。

**免重评估的可视化**（对已有 ckpt 单独出 canary 图）：

```bash
python -m tools.visualization.visualizer --exp-dir experiments/v31_fixed --n-cases 4
# --case-start 10 可增量渲染后续样本；--ckpt/--cfg 可显式指定
```

## 消融设施（V3.1 五组 A–E）

`scripts/run_ablation_v31.sh` 串行跑五组（约 9h，统一 Stage 2 预算
1000 iters）：A identity / B fixed / C learnable 两阶段 / D external-infinite /
E external-samsung；每组结束自动落盘 `final_val.json`。

```bash
bash scripts/run_ablation_v31.sh
python tools/eval_front_isp.py              # 各组 Front ISP 输出质量 →
                                            # experiments/front_isp_eval/summary.json
python scripts/summarize_ablation_v31.py    # 汇总三张表 → experiments/ablation_summary.md
```

当前 cache 上这套消融的完整结果与五条主要发现见
[docs/VERSION_HISTORY.md §4](docs/VERSION_HISTORY.md#4-v31--front-isp-四模式当前版本)。

仍随库发行的 V3 历史消融脚本：`scripts/run_v3_ablation.sh`（detection
E0–E3）、`scripts/run_v3_human_ablation.sh`（human H0–H3）、
`scripts/run_v3_human_s5_variants.sh`（H3-s5 变体）、
`scripts/val_v3_ablation.sh`、`scripts/summarize_ablations.py`。

## FiveK 数据准备（Human 任务）

Human 任务读 `human_quality:` 配置块指向的 cache（`.npz`：raw 4-plane
Bayer pack + 全分辨率 Expert-C sRGB target，均为 EXIF-upright）。
首次构建三步：

```bash
# 1. 重建 cache（raw 平面按 per-CFA 黑白电平归一化 + EXIF 翻转；
#    target 从旧 cache 原样复制）+ 对齐审计 _alignment.json
python tools/dataset/fivek_build_cache.py \
    --old-cache <旧cache目录> \
    --raw-root  /home/jing/datasets/fivek/fivek_dataset/raw_photos \
    --out       /home/jing/datasets/fivek/cache_expert_c
python tools/dataset/fivek_scan_cache.py --cache /home/jing/datasets/fivek/cache_expert_c

# 2. 相机型号表 camera.json（stem → "Make Model"，learnable 模式
#    camera-specific 参数表 & dataset 相机 id 依赖它）
python tools/dataset/fivek_camera_metadata.py \
    --raw-root /home/jing/datasets/fivek/fivek_dataset/raw_photos \
    --out      /home/jing/datasets/fivek/camera.json

# 3. CFA 分布扫描（5000 DNG：RGGB 3715 / BGGR 675 / GBRG 451 / GRBG 111）
#    产出 per_file.json，复制到 cache 上级目录作为 cfa_pattern.json
#    （dataset 自动发现，Bayer 重建按每文件真实 pattern）
python tools/dataset/fivek_cfa_scan.py
cp experiments/fivek_cfa_scan/per_file.json /home/jing/datasets/fivek/cfa_pattern.json
```

`FiveKDataset` 加载时自动按 `_alignment.json` 过滤错配样本、跳过 cache
缺失文件（当前计数：train 4818/4894，val 97/100，35 台相机），
Input Adapter（`front_isp/raw_adapter.py`）做 Bayer 重建 + demosaic，
输出全分辨率 `(3, H, W)` 线性 RGB。

数据链路诊断工具：

| 工具 | 用途 |
|---|---|
| `tools/dataset/verify_raw_adapter.py` | Input Adapter 验证套件（17/17 检查，含 rawpy 真值对拍） |
| `tools/dataset/check_fivek_shapes.py` | raw/target 尺寸 2× 对齐审计 |
| `tools/visualization/preview.py` | 抽样 raw/target 对 + corr 直方图 |
| `tools/visualization/vis_val_color.py` | 颜色链路 4 格诊断（Adapter 直出 / 各 Front ISP / target） |
| `tools/visualization/vis_size_match.py` | 像素级尺寸对齐可视化 |

fixed Front ISP 重新拟合（改了训练集/拟合超参后）：

```bash
python tools/fit_front_isp.py    # → configs/front_isp/fitted_fivek.json
                                #   + experiments/fixed_fit/report.json
```

## 目录结构

```
isp/                             算子库（26 算子 + 注册表 + CANONICAL_ORDER）
├─ operators/                    10 经典算子 + infinite_isp/ 9 个衍生算子
├─ learned/samsung_modular/      7 个 Samsung neural 算子 wrapper
└─ third_party/modular_neural_isp/   gitignored，需自行 clone
front_isp/                       可插拔前置 ISP（V3.1 四模式）
├─ identity.py fixed.py external.py   三种模式 + legacy canonical.py
├─ learnable/                    learnable 模式（两阶段训练）
├─ infinite_isp/ modular_neural_isp/  external 后端 wrapper
├─ FittedISP/                    fixed 模式的拟合方法（独立可运行）
├─ raw_adapter.py                Input Adapter：Bayer 重建 → demosaic → linear RGB
└─ third_party/                  gitignored 第三方 clone（InfiniteISP_RAW 等）
controller/adaptiveisp/          Controller + STOP head + Reward + HumanReward + PPO
pipeline/                        PipelineState / PipelineExecutor / TrajectoryBuffer
search/                          SearchSpace + priors（action_mask 三类约束）
tasks/
├─ detection/                    YOLOv3 wrapper + ReplayMemory + LOD/COCO loader
├─ human_quality/                FiveKDataset + SSIM/LPIPS/PSNR/ΔE76 + Task
└─ third_party/yolov3/           vendored YOLOv3
engine/                          trainers（detection/human/learnable）+ evaluator + util
configs/                         adaptiveisp.yaml（全量参考）+ human/v31/v3 变体
tools/                           train.py / val.py / eval_front_isp.py / fit_front_isp.py
│                                dataset/（cache·camera·cfa·验证） visualization/
scripts/                         消融驱动与汇总脚本
debug/smoke/                     7 项冒烟测试（run.sh 一键）
docs/                            VERSION_HISTORY.md（版本历史）· V1DESIGN.md
```

## 可复现性

`--seed` 统一播种 `random` / `numpy` / `torch`（CPU+CUDA）/
`PYTHONHASHSEED` 并启用 cuDNN deterministic（`engine/util.set_seed`）。
实验级复现：同 seed 同配置的 mAP 落在原实现的方差带内即可，不追求
逐位一致的 loss 轨迹。`configs/*.yaml` 是实验定义的唯一来源，训练启动
时自动复制进实验目录；环境类参数（`--workers` / `--seed` / 权重与数据
路径）留在 CLI。

## 引用

```bibtex
@inproceedings{wang2024adaptiveisp,
    title     = {AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection},
    author    = {Yujin Wang and Tianyi Xu and Fan Zhang and Tianfan Xue and Jinwei Gu},
    booktitle = {Advances in Neural Information Processing Systems},
    year      = {2024}
}
```

## 致谢

直接构建于 [AdaptiveISP](https://github.com/OpenImagingLab/AdaptiveISP)
（Wang et al., 2024）。LOD 数据集来自
[LODDataset](https://github.com/ying-fu/LODDataset)；检测 backbone
vendored 自 [Ultralytics YOLOv3](https://github.com/ultralytics/yolov3)；
Samsung neural 算子与 `samsung_isp` 前端包装
[Modular Neural ISP](https://github.com/SamsungLabs/modular_neural_isp)；
`infinite_isp` 前端与 `inf_*` 算子源自
[Infinite-ISP](https://github.com/10x-Engineers/Infinite-ISP)（10x-Engineers）；
fixed 前端使用内置的 FittedISP 拟合工具（`front_isp/FittedISP/`，
NumPy 实现的 IRLS 全局 ISP 拟合方法）。
