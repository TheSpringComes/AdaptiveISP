#!/usr/bin/env bash
# scripts/run_v3_ablation.sh — V3 ablation ladder E0 → E1 → E2 → E3.
#
# All 4 runs use the SAME data (LOD), the SAME budget (25 epochs), and
# the SAME optimizer defaults (lr=3e-5). Only the config differs, so
# cross-run mAP diffs reflect the config ablation.
#
# Memory footprint tuned (2026-09-11 OOM incident): workers=2 (was 4),
# batch_size=4 (was 8), replay_memory_size=64 (was 128 in each E1/E2/E3
# config). PYTORCH_CUDA_ALLOC_CONF cuts GPU fragmentation.
#
# E0 already completed in the pre-OOM run — this script now skips it
# and only runs E1/E2/E3. (Re-enable the E0 line if starting fresh.)
#
# Est. wall-clock (single GPU): ~5.3h for E1+E2+E3 sequential.
#   E1 ~1h35m   E2 ~1h35m   E3 ~2h10m
#
# After each run finishes:
#   experiments/lod-<save_path>/ckpt/DynamicISP_iter_*.pth  — checkpoints
#   experiments/lod-<save_path>/logs/log.txt                — training log
#   logs_ablation/v3_<id>.log                                — tee'd copy
#
set -euo pipefail
cd "$(dirname "$0")/.."   # → repo root

mkdir -p logs_ablation

# Env: reduce numpy per-thread scratch. (PYTORCH_CUDA_ALLOC_CONF omitted —
# the pinned torch version in this env rejects `expandable_segments`.)
export OMP_NUM_THREADS=2

# Use the adaptiveisp conda env's interpreter directly — the samsung
# neural ISP needs `exiftool`, which only exists there.
PY=/home/jing/anaconda3/envs/adaptiveisp/bin/python

DATA_ARGS=(
    --task detection
    --mode train
    --data_name lod
    --data_cfg tasks/third_party/yolov3/data/lod.yaml
    --imgsz 512
    --workers 2
    --epochs 25
    --batch_size 4
    --lr 3e-5
)

run_one () {
    local id=$1  cfg=$2  save=$3
    local log=logs_ablation/v3_${id}.log
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

# E0 was completed in the pre-OOM run; leave commented for re-runs.
# run_one e0  configs/adaptiveisp.yaml         v3_e0

run_one e1  configs/adaptiveisp_v3_e1.yaml   v3_e1
run_one e2  configs/adaptiveisp_v3_e2.yaml   v3_e2
run_one e3  configs/adaptiveisp_v3_e3.yaml   v3_e3

echo ""
echo "================================================================"
echo "[$(date '+%F %T')] E1+E2+E3 finished."
echo "================================================================"
