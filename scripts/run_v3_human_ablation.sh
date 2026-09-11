#!/usr/bin/env bash
# scripts/run_v3_human_ablation.sh — V3 Human ablation ladder H0→H1→H2→H3.
#
# FiveK + Expert C perceptual objective (SSIM + LPIPS). All four runs use
# the SAME data, SAME budget (25 epochs at batch=4 → 3125 iter each), and
# SAME optimizer defaults.
#
# Est. wall-clock (single GPU): ~2h total.
#   H0/H1/H2 ~30m   H3 (PPO K=4) ~40m
#
set -euo pipefail
cd "$(dirname "$0")/.."   # → repo root

mkdir -p logs_ablation
export OMP_NUM_THREADS=2

PY=/home/jing/anaconda3/envs/adaptiveisp/bin/python

DATA_ARGS=(
    --task human
    --mode train
    --imgsz 512
    --workers 2
    --epochs 25
    --batch_size 4
    --lr 3e-5
)

run_one () {
    local id=$1  cfg=$2  save=$3
    local log=logs_ablation/human_${id}.log
    echo ""
    echo "================================================================"
    echo "[$(date '+%F %T')] START  $id  cfg=$cfg  save=$save"
    echo "================================================================"
    "$PY" -u tools/train.py \
        "${DATA_ARGS[@]}" \
        --save_path "$save" \
        --cfg "$cfg" \
        2>&1 | tee "$log"
    echo "[$(date '+%F %T')] DONE   $id"
}

run_one h0  configs/adaptiveisp_human.yaml           v3_h0
run_one h1  configs/adaptiveisp_human_v3_e1.yaml     v3_h1
run_one h2  configs/adaptiveisp_human_v3_e2.yaml     v3_h2
run_one h3  configs/adaptiveisp_human_v3_e3.yaml     v3_h3

echo ""
echo "================================================================"
echo "[$(date '+%F %T')] H0+H1+H2+H3 finished."
echo "================================================================"
