#!/usr/bin/env bash
# Reschedule: dbatch16 goes solo at the end so it doesn't OOM alongside another
# batch=8. Assumes dclip01 (PID 3325106) and dlr1e-4 (PID 3347350) are already
# running as Pair 1.
#
# Pair 2: dlr1e-5 + dexp05   (both batch=8)
# Pair 3: dexp01 solo         (batch=8, alone since only 1 config left)
# Solo:   dbatch16 alone      (batch=16, needs full 24GB)

set -u
cd "$(dirname "$0")/.."

mkdir -p logs_ablation

DATA_ARGS=(
    --task train
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
    echo "[$(date '+%F %T')] launching  $id  extras=${extra[*]}  log=$log"
    python -u tools/train.py \
        "${DATA_ARGS[@]}" "${extra[@]}" \
        --save_path "$save" --cfg "$cfg" \
        > "$log" 2>&1 &
    LAST_PID=$!
    echo "  pid=$LAST_PID"
}

# ---- Wait for existing Pair 1 (dclip01 + dlr1e-4) ----
echo "==== Pair 1 already running: dclip01 (3325106) + dlr1e-4 (3347350) ===="
while ps -p 3325106 >/dev/null 2>&1 || ps -p 3347350 >/dev/null 2>&1; do
    sleep 30
done
echo "[$(date '+%F %T')] Pair 1 complete"

# ---- Pair 2 (parallel, both batch=8): dlr1e-5 + dexp05 ----
echo "==== Pair 2: dlr1e-5 + dexp05 ===="
launch dlr1e-5 configs/adaptiveisp_steps5.yaml v2ai_dlr1e-5 --batch_size 8 --lr 1e-5
DLR1E5_PID=$LAST_PID
launch dexp05  configs/adaptiveisp_dexp05.yaml v2ai_dexp05  --batch_size 8 --lr 3e-5
DEXP05_PID=$LAST_PID
wait $DLR1E5_PID 2>/dev/null || true
wait $DEXP05_PID 2>/dev/null || true
echo "[$(date '+%F %T')] Pair 2 complete"

# ---- Pair 3: dexp01 solo (batch=8) ----
echo "==== Pair 3: dexp01 solo ===="
launch dexp01 configs/adaptiveisp_dexp01.yaml v2ai_dexp01 --batch_size 8 --lr 3e-5
wait $LAST_PID 2>/dev/null || true
echo "[$(date '+%F %T')] Pair 3 complete"

# ---- Solo: dbatch16 alone (batch=16 needs full GPU) ----
echo "==== Solo: dbatch16 alone ===="
launch dbatch16 configs/adaptiveisp_steps5.yaml v2ai_dbatch16 --batch_size 16 --lr 3e-5
wait $LAST_PID 2>/dev/null || true
echo "[$(date '+%F %T')] dbatch16 complete"

echo ""
echo "All 6 D-tuning runs finished."
