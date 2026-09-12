#!/usr/bin/env bash
# scripts/val.sh — mAP + canary-viz on a trained Detection checkpoint.
#
# Auto-discovers the latest ckpt under `experiments/<exp>/ckpt/` and the
# config yaml copied into `experiments/<exp>/` at training start, so the
# common case is one positional arg:
#
#     bash scripts/val.sh lod-adaptiveisp_v1_lod_seed0
#
# The visualization lands under `experiments/<exp>/visualization/`
# (see engine.evaluator; that path derives from the ckpt location). The
# mAP artifacts (val_log.txt, records.txt) land under
# `val_results/<exp>/`, `--project`/`--name` overridable via env.
#
# Overrides (env vars):
#     WEIGHTS      YOLOv3 pretrained (default: pretrained/yolov3.pt)
#     DATA         dataset yaml      (default: tasks/third_party/yolov3/data/lod.yaml)
#     DATA_NAME    dataset name      (default: lod)
#     IMGSZ        image size        (default: 512)
#     BATCH        batch size        (default: 1)
#     STEPS        ISP rollout steps (default: from ckpt's cfg test_steps, fallback 5)
#     PROJECT      mAP out root      (default: val_results)
#     NAME         mAP out subdir    (default: <exp>)
#     CKPT         override ckpt path (default: latest DynamicISP_iter_*.pth)
#     CFG          override cfg path  (default: <exp>/*.yaml)
#     EXTRA        extra CLI args passed through to tools/val.py
#                  (e.g. EXTRA="--skip_viz" or EXTRA="--viz_cases 8")
#
# Human-Quality checkpoints are NOT supported here: engine.evaluator is
# Detection-only (mAP). The end-of-training self-val already runs for
# Human trainers; use the standalone visualizer if you want more PNGs:
#     python -m tools.visualization.visualizer --exp-dir experiments/<exp>

set -euo pipefail
cd "$(dirname "$0")/.."   # → repo root

if [[ ${1:-} == "-h" || ${1:-} == "--help" || $# -lt 1 ]]; then
    sed -n '2,/^$/p' "$0" | sed 's/^# \{0,1\}//'
    exit 0
fi

EXP=$1
shift || true

EXP_DIR="experiments/${EXP}"
if [[ ! -d $EXP_DIR ]]; then
    echo "no experiment dir: $EXP_DIR" >&2
    exit 1
fi

# --- Resolve ckpt: latest DynamicISP_iter_*.pth unless CKPT override --------
CKPT=${CKPT:-}
if [[ -z $CKPT ]]; then
    CKPT=$(ls -1 "${EXP_DIR}/ckpt/"DynamicISP_iter_*.pth 2>/dev/null \
        | sort -t_ -k3 -n | tail -n 1 || true)
fi
if [[ -z $CKPT || ! -f $CKPT ]]; then
    echo "no DynamicISP_iter_*.pth ckpt under ${EXP_DIR}/ckpt/" >&2
    echo "(if this is a Human-Quality run, use the standalone visualizer instead)" >&2
    exit 1
fi

# --- Resolve cfg: first .yaml in the experiment root unless CFG override ---
CFG=${CFG:-}
if [[ -z $CFG ]]; then
    CFG=$(ls -1 "${EXP_DIR}"/*.yaml 2>/dev/null | head -n 1 || true)
fi
if [[ -z $CFG || ! -f $CFG ]]; then
    echo "no cfg .yaml under ${EXP_DIR}/; pass CFG=path/to.yaml" >&2
    exit 1
fi

WEIGHTS=${WEIGHTS:-pretrained/yolov3.pt}
DATA=${DATA:-tasks/third_party/yolov3/data/lod.yaml}
DATA_NAME=${DATA_NAME:-lod}
IMGSZ=${IMGSZ:-512}
BATCH=${BATCH:-1}
PROJECT=${PROJECT:-val_results}
NAME=${NAME:-${EXP}}
EXTRA=${EXTRA:-}

# --- Resolve STEPS: pull test_steps from cfg unless STEPS override ---------
STEPS=${STEPS:-}
if [[ -z $STEPS ]]; then
    STEPS=$(awk -F: '/^test_steps:/ {gsub(/ /,"",$2); sub(/#.*/,"",$2); print $2; exit}' "$CFG")
fi
STEPS=${STEPS:-5}

echo "================================================================"
echo "val: $EXP"
echo "  ckpt   : $CKPT"
echo "  cfg    : $CFG"
echo "  data   : $DATA ($DATA_NAME)"
echo "  imgsz  : $IMGSZ   batch: $BATCH   steps: $STEPS"
echo "  mAP → $PROJECT/$NAME/    viz → $EXP_DIR/visualization/"
echo "================================================================"

python -u tools/val.py \
    --weights "$WEIGHTS" \
    --isp_weights "$CKPT" \
    --data "$DATA" \
    --data_name "$DATA_NAME" \
    --imgsz "$IMGSZ" --batch-size "$BATCH" --steps "$STEPS" \
    --cfg_file "$CFG" \
    --project "$PROJECT" --name "$NAME" --exist-ok \
    $EXTRA "$@"
