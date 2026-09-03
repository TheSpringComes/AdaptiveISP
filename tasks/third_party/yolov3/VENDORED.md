# Vendored YOLOv3

This directory contains a snapshot of the yolov3 codebase used as the
detection backbone in AdaptiveISP. It is called only from
`tasks/detection/implementations/yolov3.py::YOLOv3Detection`. No other
module in the framework may import `yolov3.*` directly (see the boundary
policy in `docs/DESIGN.md`).

## Provenance

- **Upstream repository**: [OpenImagingLab/AdaptiveISP](https://github.com/OpenImagingLab/AdaptiveISP)
  (Wang et al., NeurIPS 2024), which itself vendors
  [ultralytics/yolov3](https://github.com/ultralytics/yolov3) plus paper-specific
  additions (notably `val_adaptiveisp.py`, `gt.py`, and the SynRAW / LOD
  data YAMLs).
- **Snapshot taken**: 2026-09-01 (before the V1 refactor started).
- **Distribution method**: vendor snapshot (no git submodule). Version
  frozen at this commit; upgrades happen only via an explicit resnap
  documented in this file.

## Local modifications relative to the AdaptiveISP upstream snapshot

Each change is a compatibility or boundary patch, not an algorithmic edit.

| File | Change | Reason |
|---|---|---|
| `models/experimental.py` | `torch.load(..., weights_only=False)` at the `attempt_load` call site | torch ≥ 2.6 flipped the `torch.load` default to `True`; the vendored YOLO checkpoint is a pickle containing custom class references and cannot be loaded under the new default |
| `data/coco_synraw.yaml` | `path:` changed from a relative-to-yolov3-ROOT layout to the absolute local dataset path | The vendored `check_dataset` resolves paths against yolov3's own `ROOT`, which no longer matches this machine's dataset layout after the repository move |
| `data/lod.yaml` | same as `coco_synraw.yaml` | same |

## Files removed from the upstream snapshot

| File | Reason |
|---|---|
| `val_adaptiveisp.py` | Reimplemented as `engine/evaluator.py` (with `tools/val.py` as the entry point). The rewrite reads pure-function utilities (`non_max_suppression`, `ap_per_class`, `box_iou`, `scale_boxes`, `xywh2xyxy`) from `yolov3.utils.{general,metrics}` but owns the full val loop. Regression: mAP@0.5 = 71.60 on both the deleted script and the rewrite, bit-identical. |
| `gt.py` | Not on the call path from any framework module (only referenced by the deleted `val_adaptiveisp.py`). |

## Checkpoint provenance

The `pretrained/yolov3.pt` referenced by the framework (via a symlink) is
the COCO-trained yolov3 backbone distributed by the upstream AdaptiveISP
release:
- Release: <https://github.com/OpenImagingLab/AdaptiveISP/releases/tag/v1.0>
- Filename: `yolov3.pt`
- SHA-256: not verified here — pin at first-use if reproducibility across
  hosts becomes a concern.

## When to upgrade

Do not casually `git pull` from ultralytics upstream. Trigger a fresh
snapshot only when:

- A CVE in a vendored dependency requires it, or
- The framework needs a feature that upstream added (verify equivalent
  behavior via `debug/regression/val_lod.sh` after the resnap), or
- The paper releases new AdaptiveISP-specific code that we want to track.

When upgrading, follow this sequence:

1. Record the new upstream commit hash in this file.
2. Re-apply each row in the "Local modifications" table.
3. Run `bash debug/smoke/run.sh` (import + shape checks).
4. Run `bash debug/regression/val_lod.sh` on the current V1 seed-0
   checkpoint. The mAP@0.5 must stay above the threshold in
   `debug/regression/expected.yaml`.
