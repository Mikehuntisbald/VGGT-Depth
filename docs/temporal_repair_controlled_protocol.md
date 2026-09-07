# Controlled history recovery, rejection, matching and alignment

Baseline: immutable A5-HRRepair-v1 checkpoint SHA-256
`60fe5a02263c2d39a622e0a0f64db381154c01e01bb33c9dcc21dd295556f269`.
Original A5 and the existing v1 train/validation caches remain immutable.

All controlled heads start from v1, take 3,000 optimizer updates, use the same
seed 42, sampled training pixels, sequence-held-out TRAIN development split,
optimizer and learning-rate schedule. The continuation control retains v1's
loss. Supervision changes are tested first with the exact v1 architecture.
Matching/alignment/ambiguity components are then added individually under the
same chosen supervision. Added feature weights initialize to zero, retaining
exact v1 output at initialization. No end-to-end claim is made for head-only
experiments; an end-to-end integration is considered only after component gates.

The recovery task uses current-error > 1 px and history-error < 1 px; rejection
uses current-error < 0.1 px and history-error > 1 px. Their losses are separately
normalized and logged. Regret is measured against the frozen current A5 error,
with explicit hinge penalties for degradation beyond 0.1, 1 and 5 px. GT defines
training targets and evaluation masks only, never inference inputs.

Fixed validation domains (all derived from original A5/v1-cache candidates):

- Opportunity: valid GT and valid original history, E_A5 - E_original_history > 0.1 px.
- Good current: valid GT and E_A5 < 0.1 px; damage means E_new - E_A5 > 0.1 px.
- Large degradation: E_new - E_A5 > 1 px and > 5 px, denominator all valid GT.
- Also report newly introduced and eliminated > 5 px degradation versus v1.
- Foreground/background ambiguity: fixed native boundary mask, original
  disparity gap > 5 px, and at least one original candidate within 1 px of GT.
  Report inaccurate interpolation between these candidates separately.

Acceptance requires no increase versus v1 in all-GT penalized EPE or penalized
GT temporal residual, lower good-current damage and > 5 px large-degradation
rates, and greater recovery (> 0.1 px gain over A5) on the FIXED opportunity
mask. Report the > 1 px rate, raw EPE, 99% coverage, native detail/dynamic/boundary
partitions and coverage as safeguards. No change of masks, denominators, or
validity is allowed to manufacture improvements.

Train-development data choose hyperparameters/checkpoints; final-step models
and all failed arms are retained. Each component is evaluated independently.
Any combined head is explicitly separate from those ablations. All claims
require actual completed training and full 1,294-endpoint evaluation with the
repaired previous prediction reprojected using its own corrected depth.

Local stereo matching, integer history-position correction and ambiguity
features are hypotheses to be evaluated. Integer candidate gathering avoids
blending foreground and background disparities while constructing history
candidates; the learned output mixture is still measured for mixing failures.

## Additional controlled checks and outcome

A 0.05-spaced TRAIN-development calibration grid was also evaluated, with
zero additional optimizer updates. This did not produce a passing replacement.
All original and recalibrated checkpoints and their complete metrics are kept.

Boundary inspection found that simply making the soft gate more conservative
could increase erroneous intermediate disparities on the FIXED foreground/
background domain. A further categorical-routing hypothesis was trained under
both original and mild supervision: for an inference-observed disparity gap
above 5 px it selects one endpoint, retaining soft blending below 5 px. It uses
straight-through gradients in training and ordinary discrete choices in
inference. This arm shares v1 WEIGHT initialization but intentionally changes
the initial output policy; it is the exception to initial output equivalence.
Zero spurious mixing alone is not acceptance: wrong-endpoint errors, temporal
accuracy, good-pixel harm and fixed recovery are still evaluated.

A further matched block freezes the v1 head and learns an 8,065-parameter logit
residual, initially zero. Its control retains the original loss; its guarded
loss adds regret relative to v1, a good-current error hinge, and broadens the
rejection training domain to E_history > E_current + 0.1 where E_current < 0.1.
No-evidence, matching-evidence and ambiguity-evidence variants were each trained
under both losses. All these variants initialize to the exact v1 prediction.

In total 16 weight-trained arms completed 3,000 updates each and full 1,294-
endpoint validation. None passes the five core user requirements simultaneously,
even before the additional 1-px/severity/recovery-to-1-px safeguards. v1 remains
the main model. End-to-end integration is deferred because the independently
tested components have not cleared the joint gate. This outcome does not prove
that the broader designs cannot work at other budgets or with other evidence.

Receipts, all weights, complete results and both recovered/regressed cases are
under `runs/metric_stereo_video/temporal_controlled_20260907/`.
