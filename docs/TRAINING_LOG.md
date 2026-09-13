# 训练日志阅读指南（Training Log Guide）

Human 任务的打印块每 `print_freq` iters 输出一次，分
**reward / policy / quality / rollout** 四节（Detection 无 reward/policy/quality
节，只保留 loss + rollout + ops；rollout 节两者一致）：

```text
----- iter 120/1000 | elapsed 0:12:34 | 0.52 it/s | ETA 0:28:24 -----

reward   total=-2.179
         task=-0.801 | use=-1.250 | estop=-0.125 | ent=-0.001 | ovfl=-0.002

policy   loss=-0.0805  value=57.1235
         entropy=2.724/2.773  stop=39%  avg_len=6.4/7
         top_prob: sharpen=7.5%  ccm=7.3%  n_awb=7.1%  STOP=6.9%

quality  Q     +0.8105 → +0.5083   Δ=-0.3023
         SSIM  0.6xxx → 0.7198    Δ=+0.0xxx
         LPIPS 0.3xxx → 0.4230   Δ=+0.0xxx

rollout  canary : sharpen → sharpen → sharpen → tone → n_gain → STOP
         batch  : sharpen → sharpen → ccm → STOP
         fresh  : exposure → n_gamma → STOP
         ops(32): whitebalance=6  ccm=4  sharpen=4
```

```text
==================== Training Log Guide ====================

reward
  total : 本轮总奖励，越高越好
  task  : 任务质量变化，>0 表示整体改善，<0 表示变差
  use   : 重复使用算子惩罚，绝对值过大说明算子重复较多
  estop : 过早停止惩罚
  ent   : exploration / entropy 惩罚
  ovfl  : 像素上溢惩罚，正常情况下应接近 0

policy
  loss    : 策略损失（PPO 为 policy loss；actor_critic 为 agent loss），
            主要观察是否出现异常跳变
  value   : value loss，持续过大或爆炸需要检查 critic
  entropy : 当前策略熵 / 最大熵；初期接近最大值正常，
            训练后应逐渐下降，过快下降可能表示策略过早收敛
  stop    : rollout 中 STOP 的比例，过高可能表示过早停止
  avg_len : 平均 rollout 长度 / 可达上限（= 最大步数 −1，最后一步
            恒为强制 STOP 不执行算子）；过低可能表示过早停止，
            接近上限表示大多数样本跑满步数
  top_prob: 当前概率最高的几个动作及其概率；
            某个动作长期占明显优势时需要检查策略是否塌缩

quality
  Q       : 综合质量分数，↑ 越高越好
  SSIM    : 结构相似度，↑ 越高越好
  LPIPS   : 感知距离，↓ 越低越好
  A → B   : Front ISP / rollout 起点 → AdaptiveISP 最终输出
  Δ       : AdaptiveISP 带来的变化；Q、SSIM 希望 >0，LPIPS 希望 <0

rollout
  canary  : 固定样本的 greedy 路径，用于长期观察策略变化
  batch   : 当前训练 batch 的代表路径
  fresh   : 新样本路径，用于观察泛化情况
  ops(32) : 最近一个打印窗口内各算子的使用次数（top-6）

重点观察：
  1. Q↑、SSIM↑、LPIPS↓：AdaptiveISP 正在改善输入
  2. reward task 长期 <0：当前策略总体在破坏图像
  3. entropy 很快下降 + top_prob 集中：可能发生策略过早收敛
  4. 同一算子连续重复：结合 top_prob 和 usage penalty 判断是否异常
  5. ovfl 明显增大：检查算子参数范围或输出 clamp

============================================================
```

### avg_len 的读法

`avg_len` = 当前统计窗口内 rollout 实际执行算子数的平均值（与 val 指标
`mean rollout length` 同口径）。分母是**可达上限 T−1**：rollout 的
最后一步恒为强制 STOP（`is_stop_time`），该步不执行算子，故 T=8 时
跑满显示 `7.0/7` 而非 `8.0/8`：

| 现象（T=8，上限 7） | 解读 |
|---|---|
| `avg_len ≈ 7` | 基本都跑满，STOP 很少起作用 |
| `avg_len` 很低 | 策略经常提前结束 |
| 训练初期快速从 7 降到 2~3 | 检查 early stop 是否过强 |
| 训练后逐渐下降 | 若同时质量变好，可能说明策略学会了更高效地停止 |

## 统计口径

各节数字的统计方式不同，对比时注意口径：

| 节 | 口径 |
|---|---|
| reward 各项 | **每 rollout 累计**：rollout 内逐步累加、批平均，再按打印窗口内的 iter 数平均。`total = task − use − estop − ent − ovfl`（含激活的 `stop+` / `runt`），各成分与 total 严格可加 |
| policy 的 entropy / stop / top_prob | **逐步决策样本**上的平均或比率（每步每样本记一次） |
| policy 的 avg_len | **每 rollout**：窗口内全部样本实际执行算子数的均值（`op_usage` 行和） |
| policy 的 loss / value | 自训练开始以来的累计均值 |
| quality | 打印时刻所在 iter 的**即时批均值**（非窗口平均） |

`task` 与 `ΔQ` 的对应：全长度 rollout 下
`task = ΔQ × critic_logit_multiplier`（human 配置 ×10）。注意早停样本的
终端项在其停止后的每一步都会重复计入（图像已冻结、值不变），这类样本
会使 `task` 偏大——比较时以 quality 节的 Δ 为准。

## debug 模式

设置环境变量 `ADAPTIVEISP_LOG_DEBUG=1` 后：

- rollout 各步展开为带参数的完整形式（默认只显示算子名）：

```text
rollout  canary : sharpen(4.84) → sharpen(4.85) → tone(1.21,+7) → STOP
```

- PPO 追加一行优化诊断（不进主日志，避免输出过长）：

```text
policy   loss=-0.0805  value=57.1235
         [ppo] kl=0.0123  clipfrac=0.08  n_mb=8
```
