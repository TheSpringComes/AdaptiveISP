#!/usr/bin/env bash
# scripts/run_v31_ablation.sh — V3.1 四组消融（Front ISP 四种模式对比）。
#
#   | 实验     | Front ISP                      | AdaptiveISP |
#   |----------|--------------------------------|-------------|
#   | identity | 无（H-BASE baseline）          | ✓           |
#   | fixed    | 人工固定链（grayworld+gamma）   | ✓           |
#   | learnable| Stage 1 训练 → 冻结（两阶段）   | ✓           |
#   | external | Infinite-ISP（需本地 clone）    | ✓           |
#
# 依赖：/home/jing/datasets/fivek/camera.json（若缺先生成：
#   python tools/fivek_camera_metadata.py \
#     --raw-root /home/jing/datasets/fivek/fivek_dataset/raw_photos \
#     --out    /home/jing/datasets/fivek/camera.json ）
#
# 用法：bash scripts/run_v31_ablation.sh [stage1_epochs] [stage_epochs]
# 小规模冒烟：bash scripts/run_v31_ablation.sh 1 1 --smoke
set -euo pipefail
cd "$(dirname "$0")/.."

E1=${1:-5}      # Stage 1 epochs（learnable Front ISP pretrain）
E2=${2:-25}     # Stage 2 epochs（adaptive）
SMOKE=${3:-}    # 传入 --smoke 时用 --max_iters 40 小规模验证

EXTRA=()
if [ "$SMOKE" = "--smoke" ]; then
    EXTRA+=(--max_iters 40)
    echo "*** SMOKE MODE: --max_iters 40 ***"
fi

CALIB_CFG=configs/adaptiveisp_human_v31_pretrain.yaml
STAGE2_CFG=configs/adaptiveisp_human_v31_stage2.yaml
FIXED_CFG=configs/adaptiveisp_human_v31_fixed.yaml

echo "================================================================"
echo "[1/4] learnable Front ISP Stage 1 — pretrain (calibration)"
echo "================================================================"
python -u tools/train.py --task learnable \
    --cfg "$CALIB_CFG" --save_path v31_stage1 \
    --epochs "$E1" --batch_size 8 "${EXTRA[@]}"

# 最新 Stage-1 ckpt，Stage 2 从它初始化并冻结。
CALIB_CKPT=$(ls -1 experiments/v31_stage1/ckpt/CalibISP_iter_*.pth \
    | sort -t_ -k3 -n | tail -n 1)
echo "stage-1 ckpt: $CALIB_CKPT"

# 把实际 ckpt 路径写进 Stage-2 临时配置（不改动入库的 stage2 配置）。
STAGE2_RUN_CFG=/tmp/v31_stage2_run.yaml
python - "$STAGE2_CFG" "$STAGE2_RUN_CFG" "$CALIB_CKPT" <<'PY'
import re, sys
src, dst, ckpt = sys.argv[1:4]
cfg = open(src).read()
cfg = re.sub(r'ckpt: \S*CalibISP_iter_\d+\.pth', f'ckpt: {ckpt}', cfg)
open(dst, 'w').write(cfg)
PY

echo "================================================================"
echo "[2/4] learnable Front ISP Stage 2 — frozen + AdaptiveISP"
echo "================================================================"
python -u tools/train.py --task human \
    --cfg "$STAGE2_RUN_CFG" --save_path v31_learnable \
    --epochs "$E2" --batch_size 4 "${EXTRA[@]}"

echo "================================================================"
echo "[3/4] fixed Front ISP — manual chain + AdaptiveISP"
echo "================================================================"
python -u tools/train.py --task human \
    --cfg "$FIXED_CFG" --save_path v31_fixed \
    --epochs "$E2" --batch_size 4 "${EXTRA[@]}"

echo "================================================================"
echo "[4/4] external Front ISP — 需第三方仓库，可选；跳过方式见下"
echo "================================================================"
# external 需要 Infinite-ISP 本地 clone（wrapper 会给出指引），默认尝试
# 构建一次并允许失败：
if python -c "
import sys; sys.path.insert(0, '.')
from engine.util import load_config
from front_isp import build_front_isp_from_cfg
build_front_isp_from_cfg(load_config('configs/adaptiveisp_human_v31_external.yaml'))
" 2>/dev/null; then
    python -u tools/train.py --task human \
        --cfg configs/adaptiveisp_human_v31_external.yaml --save_path v31_external \
        --epochs "$E2" --batch_size 4 "${EXTRA[@]}"
else
    echo "external backend 不可用（仓库未 clone），跳过。"
    echo "  git clone https://github.com/10x-Engineers/Infinite-ISP third_party/Infinite-ISP"
fi

echo "V3.1 ablation done:"
echo "  identity : 复用 H-BASE 基线（adaptiveisp_human.yaml）"
echo "  fixed    : experiments/v31_fixed"
echo "  learnable: experiments/v31_stage1（Stage 1） + experiments/v31_learnable（Stage 2）"
echo "  external : experiments/v31_external（若可用）"
echo "汇总指标：PSNR / SSIM / LPIPS / ΔE76（各 run 的 VAL 输出）"
