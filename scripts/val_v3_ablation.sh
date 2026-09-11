#!/usr/bin/env bash
# scripts/val_v3_ablation.sh — run val on every E{0..3} ckpt to get mAP.
#
# Uses each config's LATEST ckpt (iter_6000 for E1/E2/E3, iter_3000 for E0)
# for the head-to-head comparison. Prints one line per config so the diff
# is easy to eyeball.
#
set -euo pipefail
cd "$(dirname "$0")/.."   # → repo root

mkdir -p logs_ablation/val

PY=/home/jing/anaconda3/envs/adaptiveisp/bin/python

# id -> (config-yaml)
declare -A CFG=(
  [e0]="configs/adaptiveisp.yaml"
  [e1]="configs/adaptiveisp_v3_e1.yaml"
  [e2]="configs/adaptiveisp_v3_e2.yaml"
  [e3]="configs/adaptiveisp_v3_e3.yaml"
)

for id in e0 e1 e2 e3; do
    save=lod-v3_${id}
    latest=$(ls -v experiments/${save}/ckpt/DynamicISP_iter_*.pth | tail -1)
    log=logs_ablation/val/v3_${id}.log
    echo ""
    echo "================================================================"
    echo "[$(date '+%F %T')] VAL  $id  ckpt=$(basename "$latest")"
    echo "================================================================"
    "$PY" -u tools/val.py \
        --isp_weights "$latest" \
        --cfg_file "${CFG[$id]}" \
        --weights pretrained/yolov3.pt \
        --data tasks/third_party/yolov3/data/lod.yaml \
        --data_name lod \
        --imgsz 512 \
        --batch-size 4 \
        --steps 8 \
        --skip_viz \
        --project val_v3 \
        --name "$id" \
        --exist-ok \
        2>&1 | tee "$log"
done

echo ""
echo "================================================================"
echo "V3 ablation val done.  mAP summaries:"
echo "================================================================"
for id in e0 e1 e2 e3; do
    line=$(grep -E "^\s*all\s+[0-9]+" logs_ablation/val/v3_${id}.log 2>/dev/null | tail -1)
    printf "  %-4s  %s\n" "$id" "$line"
done
