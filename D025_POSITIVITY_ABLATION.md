# D-025 separate positivity ablation

This is an independent Stage-B fine-tune lineage, not a modification or
replacement of the formal Stage-B checkpoint.  Use
`configs/ablations/d025_positivity_t3.yaml` with a new output directory and a
declared Stage-A initialization checkpoint.  It is intentionally absent from
the baseline config defaults, so Stage-A/B/C resolved configs and checkpoint
fingerprints retain their prior form.

The code-path audit is:

1. Source validity requires finite, strictly positive disparity and finite
   confidence.  Previously a finite negative FFS value could be sanitized only
   for non-finiteness and then selected by the deterministic all-invalid
   fallback.
2. With this ablation enabled, every invalid source disparity is set to exact
   zero before source mixing.  The same sanitized FFS tensor feeds the anchor.
   Zero remains invalid (`d > 0` is still required for point-cloud use); no
   epsilon and no softplus makes it a completed point.
3. The LR mixture plus bounded residual is clamped at exactly zero before
   convex upsampling.  This prevents a negative LR neighborhood from being
   propagated by convex weights.  The HR residual output is clamped at the
   same boundary before the FFS confidence anchor.
4. A squared hinge penalty on both *pre-clamp* taps supplies gradient to the
   LR residual and HR correction whenever either goes negative.  The logged
   `positivity_penalty` is already weighted by the two config coefficients.

The high-confidence anchor remains its normal correction rule:
`d_final = d_ffs + gate * (d_raw - d_ffs)`, with `gate = 0.1` at confidence
one.  Because valid FFS disparity and the bounded raw output are nonnegative,
this convex combination remains nonnegative.  An all-invalid pixel becomes
exactly zero, not a positive pseudo-depth.

Suggested invocation (select new, empty paths):

```bash
python train.py \
  --config configs/ablations/d025_positivity_t3.yaml \
  --init-from /path/to/stage_a_final.pt \
  --manifest /path/to/manifest.jsonl \
  --observation-cache-root /path/to/observation_cache \
  --teacher-cache-root /path/to/teacher_cache \
  --derived-cache-root /path/to/derived_geometry_cache \
  --output-dir /path/to/new_d025_positivity_output
```

Report this lineage separately and retain raw, clamped, and `d > 0` validity
metrics.  Do not use an improved clamped validity count as a completeness
claim.
