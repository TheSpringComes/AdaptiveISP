# debug/

Development-time verification scripts, split by purpose. Run from the
project root.

## Layout

    debug/
    ├── smoke/          fast checks (< 60 s) — run after every code change
    ├── regression/     end-to-end mAP verification against paper numbers
    └── diagnostics/    targeted probes for specific failure modes

## smoke/

Objective: confirm the codebase still assembles and the key seams still
connect. No real data, no pretrained weights beyond what registry imports
touch. Target wall-clock < 60 seconds on a single GPU.

| Script | Checks |
|---|---|
| `test_imports.py` | Every subsystem module imports; `OPERATORS` populated with 10 ops |
| `test_operators.py` | Every operator's `apply()` produces the right shape without NaN; each regressor stays in its declared range |
| `test_pipeline.py` | `PipelineExecutor.step` advances image / step / op_usage correctly; stopped samples are frozen; replay-format round-trip preserves state |
| `test_controller.py` | `Controller.act` output shapes; gradient flows to `select_head`, `select_features`, and `value_net`; eval mode is deterministic |
| `test_end_to_end.py` | Full rollout on dummy images: Controller → PipelineExecutor → Reward |

Run all:

    bash debug/smoke/run.sh

Exit 0 = all passed. Exit 1 = at least one failed (name is printed).

## regression/

Objective: verify the refactored code reproduces the paper-level mAP@0.5
on LOD. Requires a trained checkpoint and the LOD dataset (see the main
README). Expected numbers live in `expected.yaml` and are cited to the
paper.

    bash debug/regression/val_lod.sh [path/to/ckpt.pth]

Passes when the measured mAP@0.5 is at or above
`regression_threshold_mAP50` (currently 68.0, roughly three sigma below
our seed-0 measurement of 71.6 at iter 30 000).

## diagnostics/

Targeted probes, invoked when a specific concern needs isolating rather
than a full end-to-end check.

| Script | Purpose |
|---|---|
| `yolo_no_isp.py` | Negative control: YOLOv3 applied to raw LOD (no ISP). Expected: zero correct detections. Anchors the mAP-attribution claim. |
| `inspect_ckpt.py` | Prints schema (`controller_model` vs `agent_model`), iter, operator list, and a preview of weight tensors. Use when a ckpt fails to load. |

## Workflow

After a non-trivial code change:

    bash debug/smoke/run.sh                   # < 60 s

After a change that could affect training dynamics (Controller, Reward,
Runtime, or data path):

    bash debug/regression/val_lod.sh          # ~30 s on saved ckpt

If regression fails, run diagnostics to narrow it down (`inspect_ckpt`
for schema issues; `yolo_no_isp` if the mAP is suspiciously high — it
should be zero without the learned pipeline).
