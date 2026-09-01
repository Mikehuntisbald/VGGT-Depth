# D-025 separate positivity ablation

This is an independent, controlled **full 15,000-update Stage-B rerun** from
the final T=1 Stage-A checkpoint. It is not a Stage-B short fine-tune, warm
start, modification, or replacement of the formal Stage-B checkpoint. The
separate config records that protocol and is intentionally absent from the
baseline defaults, so Stage-A/B/C resolved configs and checkpoint fingerprints
retain their prior form.

`train.py` permits a new temporal run only with `--init-from` a T=1 Stage-A
checkpoint. Its `--resume` path restores a checkpoint only when the complete
resolved config matches, so it cannot convert the formal Stage-B run into this
positivity ablation. A shortened run can be created only as a new Stage-A
initialized diagnostic and must not be compared or described as this fair
15k rerun.

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

First run the CPU-only data/lineage preflight. This reads the real manifest,
cache receipts, one causal data sample, and the Stage-A model state; it does
not start a DataLoader, GPU operation, optimizer, or training output.

```bash
/home/haoyi/miniconda/envs/env-tsr/bin/python tools/preflight_d025_positivity.py \
  --config configs/ablations/d025_positivity_t3.yaml \
  --stage-a-checkpoint outputs/ffs_omega_tsr_x2/stage_a/final.pt \
  --stage-a-summary outputs/ffs_omega_tsr_x2/stage_a/run_summary.json \
  --manifest /home/haoyi/ffs_omega_cache/manifests/train_video_isolated.jsonl \
  --observation-cache-root /home/haoyi/ffs_omega_cache/m1_formal_train/observation \
  --teacher-cache-root /home/haoyi/ffs_omega_cache/m1_formal_train/teacher \
  --derived-cache-root /home/haoyi/ffs_omega_cache/m2_formal_train/derived \
  --receipt reports/m4/d025_positivity_preflight.json
```

Only after a `PREFLIGHT_PASS` receipt, use a new, empty output directory for
the declared full rerun:

```bash
python train.py \
  --config configs/ablations/d025_positivity_t3.yaml \
  --init-from outputs/ffs_omega_tsr_x2/stage_a/final.pt \
  --manifest /home/haoyi/ffs_omega_cache/manifests/train_video_isolated.jsonl \
  --observation-cache-root /home/haoyi/ffs_omega_cache/m1_formal_train/observation \
  --teacher-cache-root /home/haoyi/ffs_omega_cache/m1_formal_train/teacher \
  --derived-cache-root /home/haoyi/ffs_omega_cache/m2_formal_train/derived \
  --output-dir /path/to/new_empty_d025_positivity_full15k
```

Report this lineage separately and retain raw, clamped, and `d > 0` validity
metrics.  Do not use an improved clamped validity count as a completeness
claim. Do not pass a formal Stage-B checkpoint to `--resume` or label any
shortened diagnostic as this arm.

After the full rerun, audit its training directory with
`tools/audit_training_run.py`. Its checkpoint-resolved config enables the
strict nine-term D-025 log schema: the eight baseline terms plus exactly one
finite, non-negative `positivity_penalty`. The audit reports a separate
rolling statistic for that weighted penalty and rejects missing, extra, or
negative terms. Baseline runs remain constrained to exactly the original eight
terms.
