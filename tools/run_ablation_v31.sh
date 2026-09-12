#!/usr/bin/env bash
# V3.1 五组消融（A–E）串行驱动脚本 — 预算约 9 小时。
#
#   A identity   : configs/adaptiveisp_human.yaml
#   B fixed      : configs/adaptiveisp_human_v31_fixed.yaml（FittedISP 拟合参数）
#   C stage1     : v31_pretrain（batch 8, 1500 iters）→ C stage2（加载 ckpt 冻结）
#   D infinite   : v31_external（InfiniteISP_RAW 基线, GPU）
#   E samsung    : v31_external_samsung
#
# 统一预算：Stage 2 各 1000 iters（batch 4, imgsz 512, T=8 rollout）。
# 每组结束自动落盘 experiments/<save_path>/final_val.json（val 指标 +
# 算子选择累计）。Front ISP 输出质量由 tools/eval_front_isp.py 统一评估。
set -u
cd "$(dirname "$0")/.."
mkdir -p experiments/ablation_logs

run() {  # run <save_path> <extra args...>
  local save="$1"; shift
  echo "===== [$(date +%H:%M:%S)] START $save : $* ====="
  python tools/train.py --task human --save_path "$save" --workers 4 "$@" \
    > "experiments/ablation_logs/${save}.log" 2>&1
  local rc=$?
  echo "===== [$(date +%H:%M:%S)] END $save rc=$rc ====="
  return $rc
}

# ---------- C Stage 1：learnable Front ISP 预训练（1500 iters, batch 8） ----------
if [ ! -f experiments/v31_ablation_stage1/ckpt/LearnableISP_iter_1500.pth ]; then
  echo "===== [$(date +%H:%M:%S)] START C-stage1 ====="
  python tools/train.py --task learnable \
    --cfg configs/adaptiveisp_human_v31_pretrain.yaml \
    --save_path v31_ablation_stage1 --batch_size 8 --epochs 24 \
    --max_iters 1500 --workers 4 \
    > experiments/ablation_logs/v31_ablation_stage1.log 2>&1
  echo "===== [$(date +%H:%M:%S)] END C-stage1 rc=$? ====="
else
  echo "stage1 ckpt 已存在，跳过"
fi

# ---------- A / B / C2 / D / E：各 1000 iters ----------
run v31_ablation_A_identity  --cfg configs/adaptiveisp_human.yaml \
    --epochs 8 --batch_size 4 --max_iters 1000

run v31_ablation_B_fixed     --cfg configs/adaptiveisp_human_v31_fixed.yaml \
    --epochs 8 --batch_size 4 --max_iters 1000

run v31_ablation_C_stage2    --cfg configs/adaptiveisp_human_v31_stage2.yaml \
    --epochs 8 --batch_size 4 --max_iters 1000

run v31_ablation_D_infinite  --cfg configs/adaptiveisp_human_v31_external.yaml \
    --epochs 8 --batch_size 4 --max_iters 1000

run v31_ablation_E_samsung   --cfg configs/adaptiveisp_human_v31_external_samsung.yaml \
    --epochs 8 --batch_size 4 --max_iters 1000

# ---------- Front ISP 输出质量统一评估 ----------
python tools/eval_front_isp.py > experiments/ablation_logs/eval_front_isp.log 2>&1

echo "===== [$(date +%H:%M:%S)] ABLATION ALL DONE ====="
