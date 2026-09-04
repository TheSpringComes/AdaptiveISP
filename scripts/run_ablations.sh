#!/usr/bin/env bash
# Run V2-AI ablation experiments sequentially, tee log to file.
#
# Usage:
#   bash scripts/run_ablations.sh                  # run all remaining
#   bash scripts/run_ablations.sh d5                # single experiment
#   bash scripts/run_ablations.sh hbase hnostop      # multiple
#
# Runs from repo root (this script cd's into it).

set -euo pipefail
cd "$(dirname "$0")/.."   # → repo root

mkdir -p experiments logs_ablation

DETECTION_ARGS=(
    --data_name lod
    --data_cfg tasks/third_party/yolov3/data/lod.yaml
    --batch_size 8
    --imgsz 512
    --workers 4
    --lr 3e-5
    --epochs 60
    --task train
)

HUMAN_ARGS=(
    --batch_size 4
    --imgsz 512
    --workers 4
    --lr 3e-5
    --epochs 60
    --task train
)

# Ablation registry — (id, kind, cfg, save_path)
declare -A ABL_KIND=(
    [d5]=detection      [hbase]=human      [hnostop]=human
    [hnorepeat]=human    [ht5]=human
)
declare -A ABL_CFG=(
    [d5]=configs/adaptiveisp_steps5.yaml
    [hbase]=configs/adaptiveisp_human.yaml
    [hnostop]=configs/adaptiveisp_human_nostop.yaml
    [hnorepeat]=configs/adaptiveisp_human_norepeat.yaml
    [ht5]=configs/adaptiveisp_human_t5.yaml
)
declare -A ABL_SAVE=(
    [d5]=v2ai_d5
    [hbase]=v2ai_hbase
    [hnostop]=v2ai_hnostop
    [hnorepeat]=v2ai_hnorepeat
    [ht5]=v2ai_ht5
)

run_one () {
    local id=$1
    local kind=${ABL_KIND[$id]}
    local cfg=${ABL_CFG[$id]}
    local save=${ABL_SAVE[$id]}
    local log=logs_ablation/${id}.log

    echo "================================================================"
    echo "[$(date '+%F %T')] START  $id  kind=$kind  cfg=$cfg  save=$save"
    echo "  log → $log"
    echo "================================================================"

    if [[ $kind == detection ]]; then
        python -u tools/train.py \
            "${DETECTION_ARGS[@]}" \
            --save_path "$save" \
            --cfg "$cfg" \
            2>&1 | tee "$log"
    else
        python -u tools/train_human.py \
            "${HUMAN_ARGS[@]}" \
            --save_path "$save" \
            --cfg "$cfg" \
            2>&1 | tee "$log"
    fi

    echo "[$(date '+%F %T')] DONE   $id"
}

# ------------------------- dispatch -------------------------

if [[ $# -eq 0 ]]; then
    RUNS=(d5 hbase hnostop hnorepeat ht5)
else
    RUNS=("$@")
fi

echo "Ablations queued: ${RUNS[*]}"
for id in "${RUNS[@]}"; do
    if [[ -z "${ABL_KIND[$id]:-}" ]]; then
        echo "unknown ablation id: $id  (valid: ${!ABL_KIND[*]})" >&2
        exit 1
    fi
    run_one "$id"
done

echo ""
echo "All requested runs finished. Summarize with:"
echo "  python scripts/summarize_ablations.py"
