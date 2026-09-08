#!/usr/bin/env bash
# Run the 5 remaining Detection tuning configs in parallel pairs.
# dclip01 is already running (PID 3325106) — this script pairs the rest.
#
# Pairs (2 GPUs' worth of memory):
#   pair 1: dlr1e-4  (runs alongside already-running dclip01)
#   pair 2: dlr1e-5  + dbatch16
#   pair 3: dexp05   + dexp01

set -u
cd "$(dirname "$0")/.."   # → repo root

mkdir -p logs_ablation

DATA_ARGS=(
    --task detection
    --mode train
    --data_name lod
    --data_cfg tasks/third_party/yolov3/data/lod.yaml
    --imgsz 512
    --workers 4
    --epochs 60
)

launch () {
    local id=$1  cfg=$2  save=$3
    shift 3
    local extra=("$@")
    local log=logs_ablation/${id}.log
    echo "[$(date '+%F %T')] launching  $id  cfg=$cfg  extras=${extra[*]}  log=$log"
    python -u tools/train.py \
        "${DATA_ARGS[@]}" "${extra[@]}" \
        --save_path "$save" --cfg "$cfg" \
        > "$log" 2>&1 &
    echo "  pid=$!"
    LAST_PID=$!
}

wait_all () {
    for p in "$@"; do
        wait "$p" 2>/dev/null || true
    done
}

# ---- Pair 1: dlr1e-4 runs alongside dclip01 (which is already going) ----
echo "==== Pair 1 (dlr1e-4 alongside existing dclip01 PID 3325106) ===="
launch dlr1e-4 configs/adaptiveisp_steps5.yaml v2ai_dlr1e-4 --batch_size 8 --lr 1e-4
DLR1E4_PID=$LAST_PID

# Wait for BOTH the existing dclip01 and the new dlr1e-4
echo "waiting for dclip01 (3325106) and dlr1e-4 ($DLR1E4_PID)..."
while ps -p 3325106 >/dev/null 2>&1 || ps -p $DLR1E4_PID >/dev/null 2>&1; do
    sleep 30
done
echo "[$(date '+%F %T')] Pair 1 complete"

# ---- Pair 2: dlr1e-5 + dbatch16 ----
echo "==== Pair 2: dlr1e-5 + dbatch16 ===="
launch dlr1e-5  configs/adaptiveisp_steps5.yaml v2ai_dlr1e-5  --batch_size 8  --lr 1e-5
DLR1E5_PID=$LAST_PID
launch dbatch16 configs/adaptiveisp_steps5.yaml v2ai_dbatch16 --batch_size 16 --lr 3e-5
DBATCH16_PID=$LAST_PID
wait_all $DLR1E5_PID $DBATCH16_PID
echo "[$(date '+%F %T')] Pair 2 complete"

# ---- Pair 3: dexp05 + dexp01 ----
echo "==== Pair 3: dexp05 + dexp01 ===="
launch dexp05 configs/adaptiveisp_dexp05.yaml v2ai_dexp05 --batch_size 8 --lr 3e-5
DEXP05_PID=$LAST_PID
launch dexp01 configs/adaptiveisp_dexp01.yaml v2ai_dexp01 --batch_size 8 --lr 3e-5
DEXP01_PID=$LAST_PID
wait_all $DEXP05_PID $DEXP01_PID
echo "[$(date '+%F %T')] Pair 3 complete"

echo ""
echo "All 6 D-tuning runs finished."
