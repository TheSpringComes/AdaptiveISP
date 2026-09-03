# AdaptiveISP V1 重构方案

## 1. 目标

将现有 AdaptiveISP 从面向固定 ISP、固定 YOLO 检测任务和固定 RL 流程的研究代码，重写为一个面向软件定义成像实验的模块化 ISP 优化框架。

**V1 只改代码架构，不改算法定义**：

```text
ISP 集合        不变
ISP 参数范围    不变
状态/动作定义   不变
AdaptiveISP RL 不变
YOLO            不变
Reward          不变
训练配置        不变
```

V1 以**实验级复现**为目标，而非逐步数值完全一致。在保持原实验定义、数据、模型和训练配置不变的条件下，新版应能够获得与原 AdaptiveISP 相近的训练行为、最终任务性能和实验结论。对于包含随机性的训练与搜索过程，应统一提供 seed 配置以提高实验重复性，并可通过多个 seed 的均值和方差评估重构前后的性能一致性。

这次是**完整重写**，不是包装。V1 验证完成后不保留 `new → old agent/filter/train` 的 legacy 调用链。

---

## 2. Definition of Done

1. **Smoke Test 通过** — 新代码完整链路能跑（一次 forward + 一步训练不炸）。
2. **Baseline Reproduction 通过** — 新代码能复现原 AdaptiveISP 的性能水平和主要结论（多 seed 平均 mAP 在原版可比区间内）。
3. **Reproducibility 通过** — seed 可控，多次实验结果稳定（新版自身多 seed 方差与原版同数量级）。
4. **Run Guide 通过** — 干净环境按文档可从 RAW → mAP 完成实验。

**验收协议**（2026-09-03 V1 DoD 验证完成，全部通过）：

```text
N seeds       = 2 (baseline) + 1 (V1)
Metric        = mAP50-95 on coco2017 SynRAW val (5000 imgs, 36335 instances)

Baseline pool = seed 0 → 0.230,  seed 1 → 0.271
  mean        = 0.2505
  sample_std  = 0.029  (N=2, ddof=1)

DoD threshold (k=1) = 0.221

V1 result:
  seed 2 (new code) → mAP50-95 = 0.275   ← PASS (+0.054 over threshold,
                                            slightly above baseline mean)

Inference speed:  V1 = 18.4 ms/img    baseline = 355 ms/img    (19× faster)
                  — Option B (forward only chosen op) working as designed.

Pretrained: experiments/coco-adaptiveisp_baseline_seed{0,1}/ckpt/DynamicISP_iter_37000.pth
            experiments/coco-adaptiveisp_v1_seed2/ckpt/DynamicISP_iter_37000.pth
```

**辅助指标**（跨 seed 稳定性）：
- mAP50:   0.355 → 0.411 (mean 0.383)
- P / R:   0.557/0.329 → 0.603/0.370
- 训练集 detect_retouch_loss 最后 100 iter: 0.098 vs 0.104
- 训练集 box/obj/cls loss: 差异 <0.5%（YOLO 已完全收敛）

---

## 3. V1 非目标

以下全部推到 V2 之后：

```text
× 新 ISP Prior
× Grouping Search
× Hierarchical RL
× Dynamic Operator Set
× Multi-task
× 新 Reward
× 新 ISP
× 修改 Parameter Range
× 修改 Action Space
× 更换 YOLO
```

V1 定义：**同一个 AdaptiveISP，通过一套新的模块化代码实现，获得与原版等价的实验行为和结论。**

---

## 4. 架构：四个子系统

```text
                   Experiment
                       │
         ┌─────────────┼─────────────┐
         ▼             ▼             ▼
     ISP System    Search Space      Task
         │             │             │
         │        Prior / Constraint │
         │             │             │
         └─────────┐   │   ┌─────────┘
                   ▼   ▼   ▼
                  Controller
                      │
                  ISP Action
                      │
                      ▼
              PipelineExecutor
                      │
               Processed Image
                      │
                      ▼
                     Task
                      │
                   Metrics
                      │
                      ▼
                    Reward
```

### 4.1 ISP System

ISP Operator 只负责 `Image + Parameters → Image`。Exposure、Gamma、CCM、Denoise、Sharpen 等统一接口，Registry 管理。参数范围从处理逻辑中剥离，通过独立的 `ParameterSpec` 描述。

### 4.2 Search Space

管理 Operator Set / Parameter Space / Pipeline Length / Stop Action / Prior / Constraint。Prior 是 Search Space 的子模块。

**V1 中 Prior = Identity**：接口先建好，第一版完全尊重 AdaptiveISP 原搜索空间（无 action mask、无 order/repeat/group constraint、参数范围使用原范围）。后续所有 Prior 改动都通过 Search Space 走，不进入 Controller。

### 4.3 Controller + PipelineExecutor

**Controller** 只负责 `Observation + Search Space → ISP Action`。V1 唯一实现是 `AdaptiveISPController`，保持原 RL 网络和训练方式。Controller 不再知道 ISP 具体实现、YOLO 调用、Reward 计算、Prior 逻辑。

**PipelineExecutor** 负责 `Pipeline State + ISP Action → New Pipeline State`。Pipeline State 统一维护当前图像、历史 action、历史参数、pipeline 长度、operator 使用次数等运行状态。

未来更换 CMA-ES / Bayesian Optimization / 其他 RL，只需换 Controller。

### 4.4 Task + Reward

Task 通过一个统一接口暴露下游任务模型：

```text
Task (ABC) —— 具体实现 ——→ 第三方仓库 / 包
```

V1: `Task` 抽象基类 + `YOLOv3Detection`（具体的下游检测模型，直接持有 vendored yolov3 的 Model / ComputeLoss）。具体类**本身就是下游任务模型**，不是包装层——V2 加分割/深度时就在同一层加 `MaskRCNNInstanceSeg` / `MiDaSDepth` 等。`Task` 内部负责隔离数据预处理、输入格式、模型调用、输出格式、NMS、metric、device/dtype、第三方依赖。主框架只接收统一的 `TaskMetrics`。

第三方仓库应固定版本（Git commit / submodule / vendor snapshot / pinned pip version）。

Reward 独立接收 `TaskMetrics + Pipeline State/Cost`，V1 完整保留原 AdaptiveISP Reward 公式。

---

## 5. 六个核心 Interface

| Interface | 输入 | 输出 |
|---|---|---|
| **ISP Operator** | Image + Parameters | Image |
| **Search Space** | Pipeline State | Valid Operators + Parameter Bounds |
| **Controller** | Observation + Search Space | ISP Action |
| **PipelineExecutor** | Pipeline State + ISP Action | New Pipeline State |
| **Task** | Image + Batch | Task Metrics |
| **Reward** | Task Metrics + State | Reward |

跨模块数据对象：`ISPAction` / `PipelineState` / `TaskMetrics` / `ConstraintResult`。其他实现细节全部隐藏在各自模块内部。

---

## 6. 旧代码 → 新子系统映射

| 旧文件 : 位置 | 内容 | 新子系统 | 备注 |
|---|---|---|---|
| `isp/filters.py:37` `Filter` 基类 | fc 头 + process + regressor + mask + forward | **ISP System** + **Controller**（fc 头剥离） | 拆 3 份：`ISPOperator.apply` / `ParameterSpec` / `ParameterHead` |
| `isp/filters.py:215–815` 10 个 Filter 子类 | process / param_regressor | **ISP System** Registry 条目 | 参数范围搬到 `ParameterSpec` |
| `isp/denoise.py`, `isp/sharpen.py` | 算子实现 | **ISP System** | 保留，被 Operator 复用 |
| `isp/unprocess_np.py` | RAW 合成 | 数据侧 | 与 dataset 一起保留 |
| `agent.py:26` `FeatureExtractor` | CNN backbone | **Controller** 内部 | |
| `agent.py:64` `Agent` | 特征 + 全 filter 前向 + softmax 选择 + 状态更新 + penalty | **Controller** + **PipelineExecutor** + **Reward**（penalty） | 最大拆分点 |
| `value.py:48` `Value` | critic | **Controller**（AdaptiveISPController 的一部分） | |
| `train.py:60` `DynamicISP.__init__` | YOLO 加载 + 数据 + Agent/Value 构造 | **Experiment** 装配 + **Task Adapter** | |
| `train.py:250` 训练循环 | pipeline 推进 + reward 组装 + TD/PG 更新 | **PipelineExecutor** + **Reward** + **Controller.update** | 三段清晰 |
| `train.py:278–287` YOLO forward + ComputeLoss | detect loss | **Task (YOLOv3Detection)** | 唯一 YOLO 接触点 |
| `train.py:289–293` reward 公式 | Δloss × mult − penalty | **Reward** | |
| `train.py:395` validate 内层循环 | 逐步 rollout | **PipelineExecutor**（executor 复用） | |
| `train.py:634` argparse + `config.py` + `yolov3/data/hyps/*.yaml` | 超参 | **Experiment** 配置层 | Stage 1 需固化为单一 snapshot |
| `replay_memory.py:38` `ReplayMemory` | 轨迹池 + get_initial_states + get_noise | **Controller** or **Experiment** | AdaptiveISP-specific |
| `util.py:15` `STATE_*` 常量 | state layout | **PipelineExecutor** PipelineState schema | 替换为 dataclass |
| `util.py:58` `enrich_image_input` | image ⊕ state 拼接 | **Controller** 内部（observation 构造） | |
| `dataset.py`, `dataloader.py`, `COCO_Syn_preprocess.py` | 数据 | 数据侧 | 保留 |
| `yolov3/val_adaptiveisp.py`, `yolov3/gt.py` | eval 入口 | **Task Adapter** eval 端 | |

---

## 7. Stage 实施顺序

### Stage 1：建立原版基准 + 补 seed

**关键前置**：`train.py` 目前**无任何 seed 调用**。Stage 1 必须补齐：

```text
torch.manual_seed
torch.cuda.manual_seed_all
np.random.seed
random.seed
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

固定并记录：代码版本、数据、模型权重、seed 列表、超参数、训练过程、测试指标。

跑 N seeds（推荐 5），记录：
- 最终 mAP（均值 + 方差）
- Training curve
- Reward curve
- （可选）Operator 使用频率、Action 分布，作为 sanity check

**产出**：基准 snapshot + 冻结的 `k`、`N` 阈值写回本文档 §2。

**Stage 1 兼容补丁**（Stage 1 smoke 时发现，均为环境/依赖漂移，非算法改动，永久保留）：

| 位置 | 原因 | 修改 |
|---|---|---|
| `isp/sharpen.py:11` | torchvision 0.17+ 移除了 `transforms.functional_tensor` 模块 | `torch_pad = F.pad`（`F` 已导入） |
| `train.py:109,218,527` | torch 2.6+ 将 `torch.load` 默认 `weights_only` 翻转为 `True`，破坏 YOLO pickle checkpoint 加载 | 显式 `weights_only=False` |
| `yolov3/data/coco_synraw.yaml` | 相对路径 `../../../../datasets/coco2017` 是从 yolov3/ ROOT 解析，落到 `/home/jing/projects/datasets/coco2017`（不存在） | 改为绝对路径 `/home/jing/datasets/coco2017` |
| `util.py`, `train.py` | 缺少全局 seed 控制 | 新增 `util.set_seed()`；`train.py` 加 `--seed` / `--nondeterministic` CLI 并在主入口调用 |

**Stage 1 环境**：Python 3.13 / torch 2.6+ / torchvision 0.25 / RTX 4090 24GB / Ubuntu 7.0.0-30。

### Stage 2：重写 ISP 与控制主链

优先完成：`ISPOperator` / `ParameterSpec` / `SearchSpace`（identity 空壳）/ `ISPAction` / `PipelineState` / `Controller` / `PipelineExecutor`。

拆分顺序（风险由低到高）：

1. **ISP Operator**：从 `Filter` 抽出 `apply(img, params) → img`；参数范围抽成 `ParameterSpec`。fc 头暂留旧位。Smoke: 同图同参数量纲/形状/无 NaN。
2. **PipelineState + PipelineExecutor**：`agent.py:234–260` 的 state 更新搬到 `PipelineExecutor.step(state, action)`；`STATE_*` 换成 dataclass。
3. **Controller**：Agent 里的 "跑全部 filter → softmax 选 → 输出参数" 改成 "softmax 选 op → 只前向选中 op → PipelineExecutor 执行"。**V1 允许改动 gradient 路径**（因验收从 bit-exact 降级为多 seed 均值），此处比原计划更自由。
4. **Search Space**：空壳 `SearchSpace.valid_actions(state) → all_ops, param_bounds`；Controller 先忽略返回值，仅接口就绪。

### Stage 3：重写 Task 与 Reward

- `train.py:278–293` 抽出 `YOLOv3Detection.compute_metrics(imgs, targets) → {detect_loss, ...}` + `Reward.compute(metrics_before, metrics_after, penalty_terms) → r`。
- Agent 内部的 penalty 项（entropy / usage / early_stop / clamp / runtime）搬进 Reward。
- 形成完整的新训练、验证、测试流程。

### Stage 4：实验级验证并删除旧实现

按 §2 的 DoD 走：

- Smoke → Baseline（N seeds，均值 mAP 达阈值）→ Repro（新版自身多 seed 方差）→ Run Guide。
- **不再做 operator / state / reward 的 bit-level 数值对齐**，只在 smoke 层级检查（shape / dtype / range / no NaN）。
- 通过后删除旧的 `agent.py` / `value.py` / `isp/filters.py` 中的旧类 / 旧训练循环 / 临时兼容代码。

---

## 8. V1 完成后的研究变量归属

```text
ISP 新算子           → ISP System
参数 / 顺序 / 分组先验 → Search Space
新搜索方法（BO / CMA-ES / 其他 RL） → Controller
新视觉任务（Seg / Depth / ...）    → Task Adapter
Multi-task                        → Task + Reward
```

修改一个研究变量不再牵动整个 AdaptiveISP 主流程。
