#!/usr/bin/env bash
# debug/regression/val_lod.sh — regression check for LOD mAP@0.5.
#
# Runs tools/val.py on a saved V1 checkpoint and compares the final
# mAP@0.5 to the threshold in expected.yaml (currently 68.0). Passes
# if the measured value is >= threshold. Reference: paper reports 71.4;
# our seed-0 iter-30000 checkpoint measured 71.6.
#
# Usage:
#   bash debug/regression/val_lod.sh <path/to/ckpt.pth>
# Default ckpt: experiments/lod-adaptiveisp_v1_lod_seed0/ckpt/DynamicISP_iter_30000.pth

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

CKPT="${1:-experiments/lod-adaptiveisp_v1_lod_seed0/ckpt/DynamicISP_iter_30000.pth}"
if [[ ! -f "$CKPT" ]]; then
  echo "regression/val_lod: ckpt not found: $CKPT" >&2
  exit 2
fi

THRESHOLD=$(python -c "import yaml; print(yaml.safe_load(open('debug/regression/expected.yaml'))['regression_threshold_mAP50'])")
NAME="regression_$(basename "$CKPT" .pth)"
LOG="val_results/${NAME}.log"
mkdir -p val_results
echo "regression/val_lod: threshold=${THRESHOLD}, ckpt=${CKPT}"

python tools/val.py \
  --weights pretrained/yolov3.pt \
  --isp_weights "$CKPT" \
  --data_name lod \
  --data tasks/third_party/yolov3/data/lod.yaml \
  --imgsz 512 --batch-size 1 --steps 5 \
  --cfg_file configs/adaptiveisp.yaml \
  --project val_results --name "$NAME" --exist-ok \
  > "$LOG" 2>&1

# Parse "all N M P R mAP50 mAP75 mAP50-95" line.
LINE=$(tr '\r' '\n' < "$LOG" | grep -E "^\s+all\s+" | tail -1)
if [[ -z "$LINE" ]]; then
  echo "regression/val_lod: FAIL — no mAP line in $LOG" >&2
  exit 1
fi
MAP50=$(echo "$LINE" | awk '{print $6}')

pass=$(python -c "print(1 if float('$MAP50') * 100 >= float('$THRESHOLD') else 0)")
echo "regression/val_lod: mAP@0.5 = $(python -c "print(f'{float(\"$MAP50\") * 100:.2f}')")   threshold = ${THRESHOLD}"
if [[ "$pass" == "1" ]]; then
  echo "regression/val_lod: PASS"
  exit 0
else
  echo "regression/val_lod: FAIL (below threshold)"
  exit 1
fi
