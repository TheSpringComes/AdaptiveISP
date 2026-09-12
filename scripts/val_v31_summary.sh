#!/usr/bin/env bash
# scripts/val_v31_summary.sh — V3.1 消融收尾：评估 V3 baseline + 汇总四组表。
#
# 在 run_v31_ablation.sh 完成后运行（等 ladder 进程结束后自动执行）：
#   1. 用 tools/val.py（engine.evaluator human 路径，含 PSNR/ΔE76）评估
#      已有 V3 ckpt：v3_h0（V2 基线）与 v3_h3（V3 完全体 = Fixed Front ISP）
#   2. 打印 V3 / V3.1-A / V3.1-B / V3.1-C 四组对照表
#
# 用法：bash scripts/val_v31_summary.sh
set -euo pipefail
cd "$(dirname "$0")/.."

latest_ckpt() {  # <exp_dir>
    ls -1 "$1/ckpt/"HumanISP_iter_*.pth 2>/dev/null | sort -t_ -k3 -n | tail -n 1
}

val_exp() {  # <exp_dir> <out_name>
    local exp=$1 name=$2
    local ckpt cfg
    ckpt=$(latest_ckpt "$exp")
    cfg=$(ls -1 "$exp"/*.yaml 2>/dev/null | head -n 1)
    echo "---- val: $exp ($ckpt) ----"
    python -u tools/val.py \
        --isp_weights "$ckpt" --cfg_file "$cfg" \
        --imgsz 512 --batch-size 4 --skip_viz \
        --project experiments/val_results --name "$name" --exist-ok 2>&1 | grep -E "samples|SSIM|LPIPS|PSNR|Q:"
}

echo "================================================================"
echo "V3 baselines（Fixed Front ISP）— 同口径 PSNR/SSIM/LPIPS/ΔE76"
echo "================================================================"
val_exp experiments/v3_h0 v31_base_h0
val_exp experiments/v3_h3 v31_base_h3

echo
echo "================================================================"
echo "V3.1 四组消融（VAL on FiveK val split, 100 samples）"
echo "================================================================"
echo "  V3.1-A (calibration only) : experiments/v31_stage1  — 训练结束时的 VAL 输出"
echo "  V3.1-B (frozen)           : experiments/v31_b_frozen — 训练结束时的 VAL 输出"
echo "  V3.1-C (joint fine-tune)  : experiments/v31_c_joint — 训练结束时的 VAL 输出"
echo "  各组完整指标见各实验目录 logs/log.txt 的 '===== VAL' 块，"
echo "  baseline 见 experiments/val_results/v31_base_*/val_log.txt"
echo "================================================================"
