python tools/train.py --task human --workers 4     --cfg configs/adaptiveisp_human_v31_fixed.yaml     --save_path experiments/v31_ablation_B_fixed     --epochs 128 --batch_size 4 --max_iters 200000

python tools/val.py     --isp_weights experiments/experiments/v31_ablation_B_fixed/ckpt/HumanISP_iter_7000.pth     --cfg_file experiments/experiments/v31_ablation_B_fixed/adaptiveisp_human_v31_fixed.yaml     --name v31_ablation_B_fixed --exist-ok
