#!/usr/bin/env bash
# scripts/run_v31_ablation.sh — V3.1 四组消融（§7）。
#
#   | 实验      | Calibration         | AdaptiveISP |
#   |-----------|---------------------|-------------|
#   | V3 base   | Fixed Front ISP     | ✓           |
#   | V3.1-A    | Learnable Calibration (Stage 1 only) | × |
#   | V3.1-B    | Frozen Calibration (Stage 1 → 2)     | ✓ |
#   | V3.1-C    | Joint Fine-tuning (Stage 1 → 3)      | ✓ |
#
# 依赖：/home/jing/datasets/fivek/camera.json（若缺先生成：
#   python tools/fivek_camera_metadata.py \
#     --raw-root /home/jing/datasets/fivek/fivek_dataset/raw_photos \
#     --out    /home/jing/datasets/fivek/camera.json ）
#
# 用法：bash scripts/run_v31_ablation.sh [stage1_epochs] [stage_epochs]
set -euo pipefail
cd "$(dirname "$0")/.."

E1=${1:-5}      # Stage 1 epochs（calibration pretrain）
E2=${2:-25}     # Stage 2/3 epochs（adaptive / joint）

CALIB_CFG=configs/adaptiveisp_human_v31_pretrain.yaml
JOINT_CFG=configs/adaptiveisp_human_v31_joint.yaml

echo "================================================================"
echo "V3.1 Stage 1 — calibration pretrain (V3.1-A)"
echo "================================================================"
python -u tools/train.py --task calibration \
    --cfg "$CALIB_CFG" --save_path v31_stage1 \
    --epochs "$E1" --batch_size 8

# 最新 Stage-1 ckpt，Stage 2/3 从它初始化标定。
CALIB_CKPT=$(ls -1 experiments/v31_stage1/ckpt/CalibISP_iter_*.pth \
    | sort -t_ -k3 -n | tail -n 1)
echo "stage-1 ckpt: $CALIB_CKPT"

echo "================================================================"
echo "V3.1-B — frozen calibration + AdaptiveISP (stage 2)"
echo "================================================================"
python - <<PY
import re
cfg = open('$JOINT_CFG').read()
cfg = cfg.replace('stage: joint_finetune', 'stage: adaptive_train')
cfg = re.sub(r'ckpt: experiments/v31_stage1/ckpt/CalibISP_iter_\d+\.pth',
             'ckpt: $CALIB_CKPT', cfg)
open('configs/adaptiveisp_human_v31_stage2.yaml', 'w').write(cfg)
PY
python -u tools/train.py --task human \
    --cfg configs/adaptiveisp_human_v31_stage2.yaml --save_path v31_b_frozen \
    --epochs "$E2" --batch_size 4

echo "================================================================"
echo "V3.1-C — joint fine-tuning (stage 3, lr_calib = lr×0.01)"
echo "================================================================"
python - <<PY
import re
cfg = open('$JOINT_CFG').read()
cfg = re.sub(r'ckpt: experiments/v31_stage1/ckpt/CalibISP_iter_\d+\.pth',
             'ckpt: $CALIB_CKPT', cfg)
open('configs/adaptiveisp_human_v31_stage3.yaml', 'w').write(cfg)
PY
python -u tools/train.py --task human \
    --cfg configs/adaptiveisp_human_v31_stage3.yaml --save_path v31_c_joint \
    --epochs "$E2" --batch_size 4

echo "V3.1 ablation ladder done:"
echo "  V3 base   : 需在新数据上重训 (旧 v3_h* 已归档至 experiments_archive_polluted_data/)"
echo "  V3.1-A    : experiments/v31_stage1 （Calibration only）"
echo "  V3.1-B    : experiments/v31_b_frozen（Frozen Calibration）"
echo "  V3.1-C    : experiments/v31_c_joint （Joint Fine-tuning）"
echo "汇总指标：PSNR / SSIM / LPIPS / ΔE76（各 run 的 VAL 输出）"
