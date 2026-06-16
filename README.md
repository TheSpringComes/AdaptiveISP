# 基于互信息的暗光 RAW 图像 ISP 参数优化方法研究

本项目面向**暗光 RAW 图像目标检测**，实现一种基于检测后验信息的 ISP 参数优化框架。代码以 AdaptiveISP 的任务驱动 ISP 强化学习框架为基础进行改造：保留“冻结下游检测器，仅优化前端 RAW-to-RGB ISP 控制策略”的核心思想，但将优化目标从单纯依赖检测损失 / mAP 的任务奖励，扩展为**检测性能奖励 + 检测后验信息奖励**。

与传统低光增强方法不同，本项目不以人眼视觉质量作为唯一目标，而关注 ISP 输出是否降低冻结检测器对目标存在性和类别判断的不确定性，从而提升暗光场景下的目标检测性能。

## 研究目标

暗光 RAW 图像经过默认 ISP 转换为 RGB 后，常出现亮度不足、噪声增强、颜色响应不稳定、局部对比度下降和小目标纹理丢失等问题。这些退化会导致目标检测器漏检、误检或定位不准确。

本研究的目标是：

1. 固定预训练 YOLOv3 目标检测器，不更新检测器参数。
2. 使用强化学习策略网络为每张 RAW 图像自适应选择 ISP 操作和参数。
3. 在原检测性能奖励之外，引入类别后验熵和 objectness 后验熵构成的后验信息奖励。
4. 通过更密集的任务不确定性反馈，缓解 mAP-only reward 稀疏、非平滑和单图像反馈不稳定的问题。

## 与 AdaptiveISP 的关系

本项目**借鉴并改造**了 AdaptiveISP 的代码结构和训练范式。AdaptiveISP 的核心是使用强化学习自动选择 ISP pipeline 和参数，以提升冻结检测器的检测性能。本项目在此基础上做了如下变化：

- 研究对象聚焦于暗光 RAW 目标检测。
- 优化目标从 detection-only reward 扩展为 posterior-information-guided reward。
- 新增 `posterior_info.py`，从 YOLO 检测头中计算类别后验熵、objectness 后验熵和归一化后验信息奖励。
- 在 `train.py` 中将后验信息奖励与原检测奖励融合，形成新的强化学习 reward。
- README、实验命令和消融说明均围绕本研究主题重新组织。

原 AdaptiveISP 论文引用见文末致谢与引用部分。

## 方法概述

给定暗光 RAW 图像 $x$，参数化 ISP 管线 $G(\cdot; a)$ 将其转换为 RGB 图像：

```math
y_a = G(x; a),
```

其中 $a$ 表示 ISP 参数或由强化学习 agent 逐步选择的 ISP 操作与参数。

固定预训练目标检测器 $F$，检测器对 ISP 输出图像 $y_a$ 产生候选框、objectness 概率和类别后验分布：

```math
F(y_a)=\{(b_i, q_i, \mathbf{p}_i)\}_{i=1}^{N_a}.
```

其中：

- $b_i$ 是第 $i$ 个预测框；
- $q_i$ 是 objectness 概率；
- $\mathbf{p}_i$ 是类别后验分布；
- $N_a$ 是候选框数量。

本研究将暗光 ISP 参数优化理解为任务相关信息最大化问题：

```math
I(Z_a;T)=H(T)-H(T|Z_a),
```

其中 $Z_a=\Phi_F(y_a)$ 表示冻结检测器从 ISP 输出中提取的任务表示，$T$ 表示检测任务变量，包括目标存在性、类别和位置。

对于固定数据集，$H(T)$ 与 ISP 参数无关，因此最大化互信息等价于最小化条件熵：

```math
\arg\max_a I(Z_a;T)=\arg\min_a H(T|Z_a).
```

由于目标检测任务变量是结构化变量，真实 $H(T|Z_a)$ 难以直接计算，因此代码使用检测器输出后验分布构造代理：如果某组 ISP 参数使冻结检测器对目标存在性和类别判断更加确定，则认为该 ISP 输出保留了更多检测器可利用的任务信息。

## 后验信息奖励

### 类别后验熵

对候选框 $i$，类别后验熵定义为：

```math
H_{\mathrm{cls}}^{(i)}=-\sum_{c=1}^{C}p_i(c)\log(p_i(c)+\epsilon),
```

其中 $C$ 为类别数，$\epsilon$ 为数值稳定常数。

### Objectness 后验熵

Objectness 后验熵定义为：

```math
H_{\mathrm{obj}}^{(i)}=-q_i\log(q_i+\epsilon)-(1-q_i)\log(1-q_i+\epsilon).
```

### Objectness 加权聚合

为了突出更可能包含目标的候选框，代码使用 objectness 权重：

```math
w_i=\frac{q_i}{\sum_{j=1}^{N_a}q_j+\epsilon}.
```

聚合后的类别熵和 objectness 熵为：

```math
\mathcal{H}_{\mathrm{cls}}(a)=\sum_i w_iH_{\mathrm{cls}}^{(i)},
```

```math
\mathcal{H}_{\mathrm{obj}}(a)=\sum_i w_iH_{\mathrm{obj}}^{(i)}.
```

检测后验熵为：

```math
\mathcal{H}_{\mathrm{det}}(a)=\mathcal{H}_{\mathrm{cls}}(a)+\beta\mathcal{H}_{\mathrm{obj}}(a),
```

其中 $\beta$ 控制类别不确定性和目标存在性不确定性的相对权重。

### 归一化后验信息奖励

后验信息奖励定义为：

```math
R_{\mathrm{post}}(a)=1-\frac{\mathcal{H}_{\mathrm{cls}}(a)+\beta\mathcal{H}_{\mathrm{obj}}(a)}{\log C+\beta\log 2}.
```

该值越大，表示 ISP 输出使检测器后验不确定性越低。

### 最终强化学习奖励

原 detection-only reward 可写为：

```math
R_{\mathrm{old}}=R_{\mathrm{det}}.
```

本项目使用新的 reward：

```math
R_{\mathrm{new}}=\lambda_{\mathrm{det}}R_{\mathrm{det}}+\lambda_{\mathrm{info}}R_{\mathrm{post}}-P_{\mathrm{penalty}}.
```

其中：

- $R_{\mathrm{det}}$ 来自检测损失改善或检测性能估计；
- $R_{\mathrm{post}}$ 来自检测器后验不确定性；
- $P_{\mathrm{penalty}}$ 沿用 ISP 处理中的惩罚项，用于抑制过度增强、极端参数或额外代价；
- $\lambda_{\mathrm{det}}$ 和 $\lambda_{\mathrm{info}}$ 控制两类 reward 的权重。

## 代码结构

| 文件 / 目录 | 作用 |
| --- | --- |
| `train.py` | 训练入口；加载冻结 YOLOv3、Adaptive ISP agent、value network，并计算 detection reward 与 posterior-information reward。 |
| `posterior_info.py` | 后验信息奖励实现；负责展平 YOLO detection heads，计算 $H_{\mathrm{cls}}$、$H_{\mathrm{obj}}$、$\mathcal{H}_{\mathrm{det}}$ 和 $R_{\mathrm{post}}$。 |
| `config.py` | ISP filter、强化学习、replay memory 和网络结构配置。 |
| `isp/` | ISP 操作模块，包括曝光、伽马、锐化、降噪、tone mapping、白平衡等。 |
| `agent.py` | ISP 策略网络，根据图像状态选择 ISP filter 和参数。 |
| `value.py` | 强化学习 value network。 |
| `replay_memory.py` | 训练样本与中间 ISP 状态的 replay memory。 |
| `yolov3/` | 冻结目标检测器代码、模型配置和数据集 YAML。 |
| `docs/posterior_information_experiment.md` | 后验信息实验命令和消融设置补充说明。 |

## 环境安装

```bash
conda create -n adaptiveisp python=3.10
conda activate adaptiveisp
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

准备 YOLOv3 权重：

```bash
mkdir -p pretrained
# 将 yolov3.pt 放到 pretrained/yolov3.pt
```

## 数据准备

### LOD

1. 下载 LOD 低光 RAW 目标检测数据集。
2. 解压数据集。
3. 修改 `yolov3/data/lod.yaml` 中的 `train`、`val`、`test` 路径。

### COCO-Syn

仓库提供 `yolov3/data/coco_synraw.yaml` 用于合成暗光 RAW 风格实验。使用前需要确保其中的数据路径指向你的 COCO-Syn 图像和标签。

### OnePlus RAW / 其他 RAW 数据

如果使用 OnePlus RAW 或其他真实暗光 RAW 数据，需要准备 YOLO 格式标注，并新增类似 `yolov3/data/oneplus_raw.yaml` 的数据集配置文件。

## 运行实验

### 1. Detection-only / mAP-only baseline

该设置不启用后验信息奖励，用于对比原任务驱动 ISP reward：

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --batch_size=8 \
    --data_name=lod \
    --data_cfg=yolov3/data/lod.yaml \
    --save_path=map-only \
    --lambda_det=1.0 \
    --lambda_info=0.0
```

### 2. 本文方法：Posterior-information-guided ISP

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --batch_size=8 \
    --data_name=lod \
    --data_cfg=yolov3/data/lod.yaml \
    --save_path=posterior-info \
    --lambda_det=1.0 \
    --lambda_info=0.25 \
    --posterior_beta=1.0 \
    --posterior_topk=1000
```

### 3. COCO-Syn 合成暗光 RAW 实验

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --batch_size=8 \
    --data_name=coco \
    --data_cfg=yolov3/data/coco_synraw.yaml \
    --save_path=coco-syn-posterior-info \
    --lambda_det=1.0 \
    --lambda_info=0.25 \
    --posterior_beta=1.0 \
    --posterior_topk=1000
```

### 4. OnePlus RAW / 自定义真实 RAW 数据实验

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
    --batch_size=8 \
    --data_name=lod \
    --data_cfg=yolov3/data/oneplus_raw.yaml \
    --save_path=oneplus-posterior-info \
    --lambda_det=1.0 \
    --lambda_info=0.25 \
    --posterior_beta=1.0 \
    --posterior_topk=1000
```

## 重要参数

| 参数 | 含义 |
| --- | --- |
| `--lambda_det` | 检测性能奖励权重，对应 $\lambda_{\mathrm{det}}$。 |
| `--lambda_info` | 后验信息奖励权重，对应 $\lambda_{\mathrm{info}}$；设为 `0.0` 表示 detection-only baseline。 |
| `--posterior_beta` | objectness 熵权重，对应 $\beta$；设为 `0.0` 时主要使用类别后验熵。 |
| `--posterior_topk` | 使用 objectness 最高的 top-k 候选框计算后验熵；设为 `0` 表示使用全部候选框。 |
| `--runtime_penalty` | 启用 ISP 操作运行代价惩罚。 |
| `--runtime_penalty_lambda` | 运行代价惩罚权重。 |
| `--steps` | 测试或可视化时 ISP agent 的最大处理步数。 |

## 日志与结果分析

训练日志保存在：

```text
experiments/<data_name>-<save_path>/logs
```

可以使用 TensorBoard 查看：

```bash
tensorboard --logdir experiments
```

重点关注以下指标：

- `reward/det_reward`: 检测奖励部分。
- `posterior/h_cls`: objectness 加权类别后验熵。
- `posterior/h_obj`: objectness 加权 objectness 后验熵。
- `posterior/h_det`: 检测后验总熵。
- `posterior/reward`: 归一化后验信息奖励 $R_{\mathrm{post}}$。
- `agent_loss`: 策略网络损失。
- `value_loss`: value network 损失。
- `detect_loss`: 冻结 YOLOv3 检测损失。

后续可计算 $-\mathcal{H}_{\mathrm{det}}$ 与 mAP、AP50、Recall 的 Pearson / Spearman 相关系数，用于验证后验熵作为任务信息代理的合理性。

## 推荐消融实验

1. `--lambda_info=0.0`：去掉后验信息奖励，只保留检测奖励。
2. `--posterior_beta=0.0`：主要使用类别后验熵。
3. 增大 `--posterior_beta`：增强 objectness 后验熵影响。
4. 扫描 `--lambda_info`，例如 `0.05 / 0.1 / 0.25 / 0.5`。
5. 开启或关闭 `--runtime_penalty`，观察是否能减少过度增强和复杂 ISP pipeline。
6. 改变 `--posterior_topk`，分析候选框数量对训练稳定性和显存占用的影响。

## 引用与致谢

本项目代码结构和强化学习式 ISP 控制流程借鉴自 AdaptiveISP：

```bibtex
@article{wang2024adaptiveisp,
      title={AdaptiveISP: Learning an Adaptive Image Signal Processor for Object Detection},
      author={Yujin Wang and Tianyi Xu and Fan Zhang and Tianfan Xue and Jinwei Gu},
      booktitle={Conference on Neural Information Processing Systems},
      year={2024}
}
```

同时感谢 LODDataset 与 Ultralytics YOLOv3 项目。
