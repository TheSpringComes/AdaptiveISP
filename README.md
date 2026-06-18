# SDI-Det：基于软件定义成像的暗光目标检测

本仓库已由原始 **AdaptiveISP** 代码整理为报告《基于软件定义成像的暗光目标检测》对应的 **SDI-Det** 实现。SDI-Det 将暗光 RAW/线性图像输入、相机硬件控制代理、Camera ISP 参数搜索和 YOLO 目标检测反馈统一为闭环强化学习问题，使 RAW-to-RGB 成像过程直接服务于暗光检测性能。

## 方法概览

SDI-Det 的目标函数为：

\[
(h^*, \theta^*) = \arg\max_{h,\theta} \mathcal{R}\left(D(I_\theta(x;h))\right)
\]

其中：

- `h`：软件定义的相机硬件控制代理，包括曝光、增益、读出降噪和 ROI 局部结构增强；
- `θ`：可调 Camera ISP 参数，包括曝光、Gamma、CCM、锐化、去噪、色调、对比度、饱和度和白平衡；
- `D`：冻结的 YOLO 检测器；
- `R`：由检测奖励、图像稳定性约束和计算代价共同构成的闭环奖励。

相较于固定 ISP 或仅在 RGB 上做增强的方法，SDI-Det 在成像前端引入任务反馈，减少暗光场景中目标边界、弱纹理和局部对比度在传统 ISP 中被压缩或平滑的问题。

## 与报告一致的代码改动

- **硬件控制闭环**：新增 `HardwareControlFilter`，作为不可微相机硬件的可学习代理，用于模拟曝光时间、模拟/数字增益、读出降噪和 ROI 局部对比度控制。
- **任务驱动奖励**：训练奖励从单一检测损失差值扩展为 `λ_t R_det + (1-λ_t) R_img - β R_cost`，同时约束过曝、欠曝和噪声放大。
- **渐进式奖励调度**：训练早期更强调稳定成像，后期逐渐提高检测奖励权重，降低 RL 早期过曝、过锐化或异常色调映射带来的不稳定。
- **分组搜索约束**：将动作空间划分为硬件/曝光、细节、颜色和色调四组，Agent 每个决策步只在一个功能组内搜索，从而降低高维 ISP 参数空间的采样成本。
- **README 更新**：项目说明、训练命令和实验结果均改为 SDI-Det 报告设定。

## 环境安装

```bash
conda create -n sdi-det python=3.10
conda activate sdi-det
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 数据准备

本项目支持报告中的低光 RAW/线性检测设置：

1. **LOD**：真实暗光检测数据集，修改 `yolov3/data/lod.yaml` 中的路径。
2. **Raw COCO / COCO SynRAW**：由 COCO 通过逆 ISP 或合成 RAW 退化流程得到，修改 `yolov3/data/coco_synraw.yaml` 中的路径。
3. **OnePlus**：真实手机传感器夜间场景，可按现有 YAML 模板添加数据路径。

## 训练

### LOD 真实暗光 RAW 设置

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --batch_size=8 \
  --data_name=lod \
  --data_cfg=yolov3/data/lod.yaml \
  --save_path=sdi-det-lod \
  --runtime_penalty \
  --runtime_penalty_lambda=5e-3
```

### Raw COCO / COCO SynRAW 设置

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --batch_size=8 \
  --data_name=coco \
  --data_cfg=yolov3/data/coco_synraw.yaml \
  --save_path=sdi-det-coco \
  --add_noise=False
```

关键配置位于 `config.py`：

- `cfg.progressive_reward`：是否启用渐进式奖励调度；
- `cfg.reward_lambda_min/max/k/t0`：检测奖励权重的 Sigmoid 调度参数；
- `cfg.use_grouped_search` 与 `cfg.action_groups`：是否启用分组动作搜索；
- `cfg.hardware_*`：硬件控制代理的曝光、增益、读出降噪和 ROI 增强范围；
- `cfg.filter_runtime_penalty`：是否启用计算代价惩罚。

## 验证与可视化

```bash
CUDA_VISIBLE_DEVICES=0 python yolov3/val_adaptiveisp.py \
  --project=results \
  --isp_weights=experiments/lod-sdi-det-lod/ckpt/DynamicISP_iter_*.pth \
  --data_name=lod \
  --data=yolov3/data/lod.yaml \
  --batch-size=1 \
  --steps=5 \
  --name=sdi-det \
  --save_image \
  --save_param
```

验证脚本会保存每一步的成像结果、动作选择和 ISP 参数，便于分析 Agent 在暗光场景中如何在亮度、噪声、颜色和目标结构之间取得任务友好的折中。

## 报告实验结果

| 数据集 | Dark-YOLO | RAW-Adapter | AdaptiveISP | SDI-Det |
| --- | ---: | ---: | ---: | ---: |
| Raw COCO | 19.6 mAP | 22.0 mAP | 30.0 mAP | **36.5 mAP** |
| LOD | 60.3 mAP | 61.5 mAP | 65.4 mAP | **71.4 mAP** |

这些结果表明，暗光目标检测的性能瓶颈不仅存在于后端检测网络，也存在于相机成像前端。通过软件定义方式对硬件控制代理和 ISP 参数进行检测反馈式优化，可以从源头提升暗光目标的可检测性。

## 代码结构

```text
config.py                 # SDI-Det 滤波器、奖励调度、硬件代理和分组搜索配置
agent.py                  # 强化学习 Agent 与分组动作搜索
train.py                  # 闭环训练、渐进式奖励、图像约束奖励
isp/filters.py            # 硬件控制代理与 ISP 滤波器
yolov3/                   # YOLO 检测器、验证和数据配置
```

## 致谢

本项目基于 AdaptiveISP 与 Ultralytics YOLOv3 代码继续开发，用于复现报告中的软件定义成像暗光检测框架。
