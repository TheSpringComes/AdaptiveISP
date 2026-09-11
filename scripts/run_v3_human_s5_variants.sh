#!/bin/bash
# Run 3 H3 variants with test_steps=5 and different hyperparams
# Compare to H3 baseline (test_steps=8)

set -e

# Use the adaptiveisp conda env's Python directly
PYTHON=/home/jing/anaconda3/envs/adaptiveisp/bin/python

CONFIGS=(
    "adaptiveisp_human_v3_e3_s5"
    "adaptiveisp_human_v3_e3_s5_lr1e4"
    "adaptiveisp_human_v3_e3_s5_rew"
)

for cfg in "${CONFIGS[@]}"; do
    echo "=========================================="
    echo "Training: $cfg"
    echo "=========================================="
    $PYTHON tools/train.py \
        --task human \
        --cfg configs/${cfg}.yaml \
        --save_path experiments/v3_human_${cfg} \
        --epochs 25 \
        --batch_size 4 \
        --workers 4 \
        --seed 0
    echo "Done: $cfg"
    echo ""
done

echo "All variants trained. Run val_v3_human_ablation.sh to evaluate."
