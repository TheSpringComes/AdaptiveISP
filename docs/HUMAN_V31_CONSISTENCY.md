# V3.1 fixed-front consistency experiment

This is a controlled follow-up to the reward/mask/validation audit. It does not
change PPO's update equations, network dimensions, ISP operators, Front ISP,
Lab_ab definition, Q weights, or minimum rollout length.

## Findings

- The supplied completed run used `lambda_lab_ab=0.02`; the current base uses
  `0.1`. Old `final_val.json` files cannot establish the current run's quality.
  New Human runs save `resolved_config.yaml` with the merged configuration.
- On the current iter-0 checkpoint and four real validation images at 128px,
  recomputing action log-probabilities without an optimizer update differed by
  up to 0.116 in train mode (Dropout and batch-dependent BN). In eval mode the
  same-batch error was zero. This is an implementation consistency finding,
  not evidence of a trained policy's final performance.
- A forced STOP used to store the sampled operator's log-probability, while
  replay looked up STOP. The same check measured an error of 0.087.
- LPIPS affected reward values but its default `no_grad` path supplied no
  gradient to deterministic operator-parameter training. SSIM and Lab_ab did.
- STOP's dummy operator index was counted as exposure in usage logs.

## Changes

`SearchSpace.policy_mask` now constructs the complete support once for action
selection and replay. Forced/already-stopped states have only STOP available;
minimum length, `stop_allowed`, and operator priors are respected.

`configs/adaptiveisp_human_v31_fixed_consistent.yaml` enables two independent
switches (both default off for existing configurations):

- `deterministic_features`: disables feature Dropout and holds BN running
  statistics fixed in training as well as evaluation. Conv, BN affine, and
  head parameters still learn; action sampling remains stochastic. Existing
  checkpoint shapes are compatible. From scratch, BN starts with its default
  running statistics; convergence must be checked in the controlled run.
- Human Quality operator parameters always optimize the parameter quality
  objective as `-quality_scale * mean(Q(final)-Q(front))` with LPIPS input
  gradients enabled. This is the telescoped dense quality objective,
  including early-stopped samples. Existing per-step invalid/overflow guards
  remain. PPO still uses the original dense reward values. Only one LPIPS
  gradient graph is retained per rollout; this adds computation and memory
  compared with detached LPIPS.

Routing diagnostics use first-step distributions, where inputs have the same
history/mask: `input_JS = H(mean(pdf)) - mean(H(pdf))`, unique argmax count, and
largest argmax share. Identical high-entropy policies still have zero input_JS.
These are diagnostics, not diversity rewards or guarantees of good routing.
Validation computes them over the complete validation set. Training usage
counts exclude STOP and padding, and policy summaries exclude forced STOP.
The Human CLI now applies its existing `--seed` option.

## Controlled run

Use a new output directory; the running old process retains its already loaded
code. Start without `--resume` for a clean comparison:

```bash
python tools/train.py --task human --workers 4 \
  --cfg configs/adaptiveisp_human_v31_fixed_consistent.yaml \
  --save_path v31_fixed_consistent_seed0 \
  --epochs 32 --batch_size 4 --imgsz 512 --max_iters 500 --seed 0
```

Compare with the original fixed config under the same seed, sample budget,
resolution, and Q weights, in a separate output directory. To isolate effects,
run the new config with `full_quality_param_loss: false`, then enable it.
Do not compare Q across the old 0.02 and current 0.1 color weights.

Judge improvements using paired validation mean delta Q, the improved/degraded
sample counts, SSIM/LPIPS/Lab_ab changes, and visual inspection. Routing diversity
alone is not an objective. If color artifacts remain, investigate cache color
profiles and residual alignment before changing Lab_ab or its scale. The fixed
Front ISP includes a fitted tone exponent and CCM targeting Expert-C RGB; the
code does not show a bare linear RGB output being used directly in this path.
The existing alignment filter does not guarantee subpixel registration.

## Verification

```bash
python -m unittest debug.regression.test_policy_consistency \
  debug.regression.test_human_v31_audit -v
```

Tests cover minibatch-independent probability replay, forced STOP, gradients
from the complete Q, routing diagnostics, priors, and paired aggregation.
A real-data CPU smoke run completed three training iterations and paired
validation on three samples at 128px with finite parameters. Its mean delta Q
was -0.0059 (1/3 improved); this is execution validation, **not evidence of a
quality improvement**. A matched training comparison is still required.
