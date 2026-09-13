# 版本历史（V1 → V2-AI → V3 → V3.1）

本文记录各版本的动机、改动与实验结论。**当前版本（V3.1）的使用方式见
[README.md](../README.md)**，本文只做历史存档。

> ⚠️ 数据兼容性：FiveK Expert-C cache 于 **2026-09-12 重建**（旧 cache 中
> ~1/3 样本的 raw/target 存在旋转错位，见
> `tools/dataset/fivek_build_cache.py` 模块注释）。**V2-AI、V3 及更早的
> 全部 Human Quality 数字（含下文 H 系列、v2ai_\*）都训练于旧 cache，
> 与当前 cache 上的数字不可比较。** 当前可比较的基准只有 2026-09-13 的
> V3.1 五组消融 A–E（见 §4）。

---

## 1. V1 — 代码重构（对齐原版 AdaptiveISP）

**目标**：把原版 AdaptiveISP（Wang et al., NeurIPS 2024）的扁平研究代码
（agent / filter / train / detect 四个顶层文件）完整重写为模块化框架，
只改架构、不改算法定义。原版的算子集、参数范围、RL 形式化全部保留。

六个子系统：ISP System（算子注册表）、Search Space（prior 组合）、
Controller（RL 策略）、PipelineExecutor + PipelineState、Task（下游任务
边界）、Reward。旧文件 → 新子系统的逐行映射与两处刻意偏差记录在
[V1DESIGN.md](V1DESIGN.md)（与 [DESIGN.md](DESIGN.md) 内容相同，后者为
初版存档）。

**验收（DoD，2026-09-03 全部通过）**：

- COCO SynRAW 检测复现：V1（seed 2）mAP50-95 = **0.275**，
  baseline 池（seed 0/1）均值 0.2505 ± 0.029，阈值 0.221 → PASS；
- 推理提速 **19×**（18.4 vs 355 ms/img，"只前向被选中算子"生效）；
- LOD 复现：mAP@0.5 = **71.6** @ 30 000 iters（seed 0），论文报告 71.4；
  同一评估脚本下 YOLOv3 直接吃 RAW 为零检出，说明 mAP 来自学习到的管线。

评估 Wang et al. 官方发布的两个 pre-refactor ckpt（`agent_model` schema）
请用 git tag `v0-baseline`。

## 2. V2-AI — 算子库与任务扩展

在 V1 架构上叠四个扩展（数字基于旧 cache，仅供趋势参考）：

- **算子库 10 → 26**：
  - 10 个经典算子（V1 原有，`isp/operators/`）；
  - 7 个 Samsung Modular Neural ISP 算子 `n_*`（`x + α·(F(x)−x)`，冻结
    backbone F，`isp/learned/samsung_modular/`）；
  - 9 个 Infinite-ISP 衍生的 Torch 原生算子 `inf_*`
    （`isp/operators/infinite_isp/`）。
  每族有稳定前缀，`op_usage` 列不漂移。
- **可学习 STOP 动作 + 指数重复惩罚**：select_head 输出 `n_ops + 1` logits，
  附加列为 STOP（提交当前管线）；重复惩罚 = base × 2^prev_count，防止
  策略塌缩成单算子循环。
- **第二任务 Human Quality**（FiveK + Expert C）：终端稀疏奖励
  `R = Q(I_T) − Q(I_0)`，`Q = λ_ssim·SSIM − λ_lpips·LPIPS`；
  `HumanTrainer` 每 iter 跑完整 T 步 rollout（无 ReplayMemory）。
- **自动 canary 可视化**：`tools/val.py` 一次跑出指标 + PNG。

## 3. V3 — 搜索空间结构与 PPO

三个结构改进（H 系列数字基于旧 cache，仅供趋势参考）：

- **A1 Canonical Backbone**：Controller 之前固定跑一条
  AWB → CCM → GTM → Gamma 链，把搜索空间从 "RAW → 任务优化 RGB" 缩小为
  "baseline RGB → 任务优化 RGB"。*V3.1 已用四模式 Front ISP 取代该机制*
  （`canonical` 保留为 legacy 别名）。
- **A2 Action Mask**（`search/priors/action_mask.py`）：no-repeat /
  order / group-budget 三类动态动作约束。
- **A3 PPO**（`controller/adaptiveisp/ppo.py`）：GAE(λ) + clipped
  surrogate + value baseline，替代 REINFORCE / 1-step TD。

**Human Quality 25 epochs（旧 cache）**：

| Config | SSIM ↑ | LPIPS ↓ | Q ↑ | Rollout |
|--------|--------|---------|-----|---------|
| H0 (V2 baseline) | 0.593 | 0.407 | +0.186 | 1.00/8 |
| H1 (+backbone) | 0.540 | 0.370 | +0.170 | 1.00/8 |
| H2 (+mask) | 0.588 | 0.423 | +0.165 | 7.00/8 |
| H3 (+PPO) | 0.616 | 0.374 | +0.242 | 6.96/8 |
| H3-s5-rew | 0.605 | 0.318 | **+0.287** | 4.00/5 |

H3-s5-rew 用 `test_steps=5` + 奖励塑形（stop_bonus=0.01,
early_stop_penalty=2.0），Q 比 H3 高 4.5pt 且省 37.5% 计算。

**Detection（LOD，25 epochs）负迁移**：E0=0.686 → E1=0.604 → E2=0.564 →
E3=0.597 mAP@0.5。Backbone 限制了 Controller 为 YOLOv3 特征做定向优化的
空间；800 epochs 长训可能追回。

V3 的配置与脚本仍随库发行，可复跑：
`configs/adaptiveisp_v3_e{1,2,3}.yaml`、
`configs/adaptiveisp_human_v3_e{1,2,3}*.yaml`、
`scripts/run_v3_ablation.sh`、`scripts/run_v3_human_ablation.sh`、
`scripts/run_v3_human_s5_variants.sh`、`scripts/val_v3_ablation.sh`。

## 4. V3.1 — Front ISP 四模式（当前版本）

V3.1 的设计与当前用法见 [README.md](../README.md)。此处存档
**2026-09-13 五组消融（A–E，新 cache）**的完整结果——这是当前 cache 上
唯一一套可比较基准。汇总由 `scripts/summarize_ablation_v31.py` 生成，
本机同步存于 `experiments/ablation_summary.md`（`experiments/` 不入库）。

统一预算：C 组 Stage 1 预训练 1500 iters（batch 8），五组 Stage 2 各
1000 iters（batch 4, imgsz 512, T=8），val = 97 张 Expert-C。

### 表 1 — Front ISP 输出质量（AdaptiveISP 之前）

| 组 | Front ISP | SSIM↑ | LPIPS↓ | PSNR↑ |
|---|---|---|---|---|
| A | identity | 0.3188 | 0.4460 | 11.13 |
| B | fixed (FittedISP) | 0.8504 | 0.1257 | 20.03 |
| C | learnable (2-stage) | 0.3621 | 0.4716 | 11.30 |
| D | external-infinite | 0.6888 | 0.3803 | 18.62 |
| E | external-samsung | 0.8026 | 0.1890 | 17.85 |

### 表 2 — AdaptiveISP 最终 val 指标（RL 之后）

| 组 | Front ISP | SSIM↑ | LPIPS↓ | PSNR↑ | ΔE76↓ | Q↑ | mean len |
|---|---|---|---|---|---|---|---|
| A | identity | 0.3746 | 0.4575 | 11.50 | 30.78 | -0.0829 | 1.14 |
| B | fixed (FittedISP) | 0.7606 | 0.2996 | 18.94 | 15.53 | **+0.4610** | 3.22 |
| C | learnable (2-stage) | 0.3780 | 0.4623 | 11.26 | 31.18 | -0.0843 | 1.00 |
| D | external-infinite | 0.4827 | 0.5298 | 13.79 | 25.71 | -0.0471 | 5.60 |
| E | external-samsung | 0.1850 | 0.6095 | 8.94 | 36.84 | -0.4245 | 3.00 |

### 表 3 — 基础算子选择频率（全程累计，%）

| 组 | exposure | gamma | ccm | whitebalance | tone | contrast | saturation | sharpen | denoise | wnb | neural% | total picks |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A | 12.1 | 7.5 | 6.8 | 6.3 | 5.9 | 6.2 | 5.9 | 6.3 | 6.1 | 6.0 | 31.1 | 31904 |
| B | 13.6 | 6.8 | 5.7 | 6.3 | 8.1 | 4.8 | 6.8 | 5.8 | 5.9 | 5.2 | 31.2 | 31904 |
| C | 12.6 | 5.9 | 5.4 | 5.8 | 6.8 | 6.7 | 7.3 | 7.1 | 5.9 | 6.3 | 30.2 | 31912 |
| D | 13.1 | 5.7 | 6.3 | 5.7 | 7.5 | 5.9 | 6.2 | 6.7 | 6.4 | 5.8 | 30.7 | 31832 |
| E | 11.6 | 6.5 | 6.7 | 6.2 | 6.0 | 5.5 | 7.3 | 6.0 | 6.2 | 6.6 | 31.4 | 31980 |

（训练/验证曲线见本机 `experiments/ablation_curves.png`。）

### 五条主要发现

1. **起点质量决定 RL 可训性（本预算下最根本的发现）**：低起点组
   （A 0.319 / C 0.362）的策略收敛到"立即 STOP"（A: 99% learned-STOP,
   len 1.14；C: 100%, len 1.00），val SSIM 仅微升——弱起点上探索期内
   大多数算子动作是负收益，策略学到的最优解就是不动。
2. **fixed（FittedISP 拟合）是本预算下唯一全面成功的组**：起点 SSIM
   0.850（表 1 最高），RL 后 0.761 / Q +0.461，策略积极优化
   （len 3.22）且最终指标全面领先。
3. **external 前端起点好但短预算 RL 反而退化**：D 起点 0.689 → RL 后
   0.483（len 5.60，45% 学会 STOP 但仍净损伤）；E 起点 0.803 → 0.185
   （严重退化）。共同点：前端输出风格与 Expert-C 目标差距大（LPIPS
   0.38/0.19 vs fixed 0.126），1000-iter 短预算下策略未学会"何时该停"。
   即：高起点 ≠ 高可训性，奖励塑形与预算的匹配同样关键。
4. **算子选择频率**：五组一致以 exposure 为最高频（11.6–13.6%），印证
   曝光/亮度校正是 AdaptiveISP 首要动作；wb 类算子占比普遍低于 exposure，
   因为多数 Front ISP 已前置处理色偏；neural 算子总占比 ~30–31%，
   各组无显著分化。
5. **对 C 组的解读**：learnable 两阶段在 1500-iter Stage 1 预算下尚未
   追上 fixed 的拟合质量（0.362 vs 0.850）；Stage 1 需要显著更多预算
   （或更强参数化）才能进入"可激活 RL"的起点区间。
