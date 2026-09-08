#!/usr/bin/env bash
# Detection tuning batch 3: 6 runs with per-experiment CLI overrides.
# 3 need yaml (clip/explore); 3 need only different CLI (lr/batch).

set -euo pipefail
cd "$(dirname "$0")/.."   # → repo root

mkdir -p experiments logs_ablation

DATA_ARGS=(
    --task detection
    --mode train
    --data_name lod
    --data_cfg tasks/third_party/yolov3/data/lod.yaml
    --imgsz 512
    --workers 4
    --epochs 60
)

# id : (cfg, extra_cli)
run_one () {
    local id=$1  cfg=$2  save=$3
    shift 3
    local extra=("$@")   # e.g. --batch_size 16 --lr 1e-4
    local log=logs_ablation/${id}.log
    echo "================================================================"
    echo "[$(date '+%F %T')] START  $id  cfg=$cfg  save=$save  extras=${extra[*]}"
    echo "================================================================"
    python -u tools/train.py \
        "${DATA_ARGS[@]}" "${extra[@]}" \
        --save_path "$save" --cfg "$cfg" \
        2>&1 | tee "$log"
    echo "[$(date '+%F %T')] DONE   $id"
}

# All 6 —  batch_size + lr default to base (8, 3e-5) unless overridden
run_one dclip01     configs/adaptiveisp_dclip01.yaml     v2ai_dclip01     --batch_size 8  --lr 3e-5
run_one dlr1e-4     configs/adaptiveisp_steps5.yaml       v2ai_dlr1e-4     --batch_size 8  --lr 1e-4
run_one dlr1e-5     configs/adaptiveisp_steps5.yaml       v2ai_dlr1e-5     --batch_size 8  --lr 1e-5
run_one dbatch16    configs/adaptiveisp_steps5.yaml       v2ai_dbatch16    --batch_size 16 --lr 3e-5
run_one dexp05      configs/adaptiveisp_dexp05.yaml       v2ai_dexp05      --batch_size 8  --lr 3e-5
run_one dexp01      configs/adaptiveisp_dexp01.yaml       v2ai_dexp01      --batch_size 8  --lr 3e-5

echo ""
echo "All 6 runs finished. Val each with:"
echo "  for id in dclip01 dlr1e-4 dlr1e-5 dbatch16 dexp05 dexp01; do"
echo "    LATEST=\$(ls -v experiments/lod-v2ai_\${id}/ckpt/DynamicISP_iter_*.pth | tail -1)"
echo "    python tools/val.py ... --isp_weights \$LATEST ..."
echo "  done"
