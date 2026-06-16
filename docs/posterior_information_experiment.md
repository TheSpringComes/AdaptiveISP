# Posterior-information-guided AdaptiveISP experiments

This repository now supports the experiment design for low-light RAW object detection with a posterior-information reward.  The detector is kept frozen and the AdaptiveISP policy is optimized with

```text
R_new = lambda_det * R_det + lambda_info * R_post
R_post = 1 - (H_cls + beta * H_obj) / (log(C) + beta * log(2))
```

where `H_cls` is the objectness-weighted class-posterior entropy and `H_obj` is the objectness-posterior entropy computed from YOLO detection heads.

## Main experiments

### mAP-only AdaptiveISP baseline

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --batch_size=8 \
  --data_name=lod \
  --data_cfg=yolov3/data/lod.yaml \
  --save_path=map-only \
  --lambda_det=1.0 \
  --lambda_info=0.0
```

### Posterior-information-guided AdaptiveISP

```bash
CUDA_VISIBLE_DEVICES=0 python train.py \
  --batch_size=8 \
  --data_name=lod \
  --data_cfg=yolov3/data/lod.yaml \
  --save_path=posterior-info \
  --lambda_det=1.0 \
  --lambda_info=0.25 \
  --posterior_beta=1.0 \
  --posterior_topk=1000
```

Use the same commands with `--data_cfg` set to the corresponding OnePlus RAW or COCO-Syn YAML file to reproduce the three-dataset protocol.  TensorBoard logs include `posterior/h_cls`, `posterior/h_obj`, `posterior/h_det`, `posterior/reward`, and `reward/det_reward`, which can be used for the entropy-performance correlation analysis.

## Ablations

* Class/objectness entropy ablations can be approximated by changing `--posterior_beta`:
  * `--posterior_beta=0.0` emphasizes class entropy only.
  * Larger values increase the contribution of objectness entropy.
* Reward-weight trade-off is controlled by `--lambda_info`.
* The mAP-only baseline is `--lambda_info=0.0`.
* Candidate selection cost is controlled by `--posterior_topk`; pass `--posterior_topk=0` to use all YOLO candidates.
