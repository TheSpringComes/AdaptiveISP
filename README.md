# SDI-Det：基于软件定义成像的暗光目标检测

SDI-Det 是一个面向暗光目标检测的软件定义成像框架。项目将暗光 RAW/线性图像、相机硬件控制代理、可调 Camera ISP 和 YOLO 检测器连接为闭环系统，使成像过程不再只是固定前处理，而是能够根据下游检测反馈主动调整。

在夜间驾驶、暗光监控和端侧视觉感知等场景中，目标外观常受到低信噪比、低对比度、颜色偏移、运动模糊和暗部细节缺失影响。SDI-Det 的核心思想是：暗光检测的性能瓶颈不仅存在于检测网络，也存在于相机成像前端。因此，项目通过强化学习 Agent 搜索任务友好的硬件控制与 ISP 参数，从 RAW-to-RGB 转换阶段保留更有利于目标检测的边界、纹理和局部对比度信息。

## 方法概览

SDI-Det 将软件定义成像建模为任务驱动优化问题：

```text
(h*, θ*) = argmax_{h,θ} R(D(I_θ(x; h)))
```

其中：

- `x` 表示暗光 RAW/线性输入；
- `h` 表示软件定义的相机硬件控制代理，包括曝光、增益、读出降噪和 ROI 局部结构增强；
- `θ` 表示可调 Camera ISP 参数，包括曝光、Gamma、CCM、锐化、去噪、色调、对比度、饱和度和白平衡；
- `I_θ(x; h)` 表示由硬件控制代理和 ISP 参数共同决定的成像结果；
- `D` 表示冻结的 YOLO 检测器；
- `R` 表示由检测奖励、图像稳定性约束和计算代价共同构成的闭环奖励。

训练时，Agent 根据图像状态、历史动作和检测反馈选择下一步成像动作。系统先通过硬件控制代理调节 RAW/线性信号，再通过可调 ISP 生成检测输入，最后由检测器反馈奖励。为提升训练稳定性和搜索效率，SDI-Det 使用渐进式奖励调度和分组动作搜索：训练早期更关注稳定成像，后期逐渐提高检测奖励权重；动作空间按硬件/曝光、细节、颜色和色调分组，避免在全部 ISP 参数上进行无约束高维搜索。

## 代码结构

```text
config.py                 # 滤波器、硬件代理、奖励调度和分组搜索配置
agent.py                  # 强化学习 Agent 与分组动作搜索
train.py                  # 闭环训练、渐进式奖励、图像约束奖励
isp/filters.py            # 硬件控制代理与 ISP 滤波器
yolov3/                   # YOLO 检测器、验证脚本和数据配置
```

## 环境安装

```bash
conda create -n sdi-det python=3.10
conda activate sdi-det
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.8 -c pytorch -c nvidia
pip install -r requirements.txt
```

## 数据准备

本项目支持低光 RAW/线性目标检测设置：

1. **LOD**：真实暗光检测数据集，修改 `yolov3/data/lod.yaml` 中的数据路径。
2. **Raw COCO / COCO SynRAW**：由 COCO 通过逆 ISP 或合成 RAW 退化流程得到，修改 `yolov3/data/coco_synraw.yaml` 中的数据路径。
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

常用配置位于 `config.py`：

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

验证脚本会保存每一步的成像结果、动作选择和 ISP 参数，便于观察 Agent 如何在暗光场景中平衡亮度、噪声、颜色和目标结构。

## 实验结果

| 数据集 | Dark-YOLO | RAW-Adapter | AdaptiveISP | SDI-Det |
| --- | ---: | ---: | ---: | ---: |
| Raw COCO | 19.6 mAP | 22.0 mAP | 30.0 mAP | **36.5 mAP** |
| LOD | 60.3 mAP | 61.5 mAP | 65.4 mAP | **71.4 mAP** |

结果表明，通过软件定义方式对硬件控制代理和 ISP 参数进行检测反馈式优化，可以从成像源头提升暗光目标的可检测性。

## 致谢

本项目基于 AdaptiveISP 与 Ultralytics YOLOv3 代码继续开发，用于研究软件定义成像在暗光目标检测中的应用。
