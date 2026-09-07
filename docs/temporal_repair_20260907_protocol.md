# A5 causal HR candidate repair, 2026-09-07

This experiment tests an additional trained temporal head on frozen formal A5
step 6000. It does not overwrite A0-A5, change their training protocol, or
constitute a new independently trained full-model ablation.

Observed source-level constraints are: history is hard-vetoed by consistency
with the potentially erroneous current base depth; history contributes only
features; the two multiplicative depth residuals are each bounded by exp(0.25).
The existing opportunity audit compares an independent HR previous prediction
against an A4 endpoint, while the original A5 memory consumes a different LR
state. Consequently its oracle rate does not prove original LR memory could
have produced those improvements.

The proposed head can select a directly transported HR candidate, using current
stereo photometric evidence, causal history photometry, discrepancy, confidence,
and local depth variation. It does not accept GT depth, native dynamic labels,
match/detail labels, or any future frame. Invalid history falls back exactly to
A5. Original A5 output validity is retained, so rejection cannot improve scores.
This is still a known-camera-pose stereo-video model.

Training uses 1,152 evenly selected endpoints from the original train split,
with its deterministic epoch-zero random crops. Each endpoint yields 4,096
uniform pixels and 4,096 disagreement pixels. The loss weights these two groups
3:1. Entire train sequences (sorted IDs at positions 0, 7, 14, ...) are reserved
for development/calibration, and never used in optimizer updates. The head has
6,273 parameters, trained for a fixed 3,000 AdamW steps, seed 42. Calibration
selects a logit shift only on the internal train-development sequences.
The full 1,294 endpoint validation domain is used after training and calibration.

Acceptance was specified before full validation:

- All-GT penalized EPE improves by at least 1% against the exact frozen A5.
- GT temporal-delta penalized EPE improves by at least 1%.
- EPE on available HR-history opportunities improves by at least 5%.
- Dynamic and high-detail penalized EPE each worsen by no more than 2%.
- Prediction coverage is unchanged and all outputs are finite and positive.

The temporal metric reprojects the REPAIRED previous output using its repaired
depth, and compares against the separately warped previous GT. It does not
compare a repaired current output to an unrepaired previous output. Predictions
use independent strict prefixes ending at t, t-1 and t-2. Each endpoint is counted
once despite FSDP sampler padding. Opportunity masks are frozen from baseline
A5 and the available HR candidate, not recomputed to favor the repair.

Report actual raw and penalized errors, 99% coverage EPE, completeness, native
Spring detail/match/boundary partitions, dynamic/static residuals, and both the
largest rescues and regressions. Report checkpoint, manifest, source and cache
hashes. A failed acceptance remains a failed experiment.

## Completed evidence

Both v1 (capped L1) and a matched v2_raw (uncapped L1) head actually completed
3,000 updates. Internal train-development penalized EPE selected v1 (0.524928)
over v2_raw (0.528738), recorded before reading full-validation results. Both
pass the six stated checks. The selected v1 achieves 0.330162 all-GT penalized
EPE and 0.258275 GT temporal residual, versus the same A5's 0.351058/0.315694.
Independent A4 evaluates at 0.355078/0.319920 on the same 1,294 endpoints.

The final coverage report uses the original contract's valid_probability score,
4,096 bins and all configured coverage points. Motion partitions use its frozen
thresholds. Aligning these supplementary metrics did not change any previously
reported spatial, temporal, regional or opportunity metric. Raw-image inference
matches cached repair predictions exactly on 8 endpoints (3,145,728 pixels).
All 65 base runtime source files match the A5 training receipt.

The complete result, reproducible commands, training receipts, source snapshots,
paired sequence confidence intervals, successful rescues and severe remaining
failures are in `runs/metric_stereo_video/temporal_repair_20260907/REPORT.md`.
