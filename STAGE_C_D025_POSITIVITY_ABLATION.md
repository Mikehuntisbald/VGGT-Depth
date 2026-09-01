# Controlled Stage-C positivity after D-025

This is a future, opt-in ablation. It does **not** repair or replace the
canonical Stage-C result. The completed canonical run remains
`STAGE_C_M5_GATE_FAIL`: raw invalid disparity rose from 2.78447% to 4.41968%
and invalid-region completeness fell 23.1346%.

The read-only quantitative diagnosis is bound in
`reports/m6/stage_c_output_health_root_cause.json`. Its key conclusion is a
domain mismatch: the epipolar correction is nonzero over 99.9898% of the
candidate-valid domain (98.1985% of the image), while its disparity and
correction losses operate only on trusted positive-target pixels. The finite,
unsaturated negative correction tail can therefore cross near-zero base
disparity. NaN/Inf, checkpoint misbinding, correction saturation and
correspondence OOB are not the primary cause.

## Immutable experiment boundary

The arm may start only after the independent D-025 Stage-B rerun:

1. reaches exactly 15,000 updates from final Stage A;
2. has a `PASS` / `TRAINING_COMPLETE` training audit with the D-025 positivity
   loss schema;
3. completes the canonical 244/240/238 held-out evaluation;
4. receives `D025_FINAL_CONTROLLED_COMPARISON_PASS` from the independent
   `tools/audit_d025_evaluation.py` cross-audit against canonical Stage B;
5. passes the original temporal, low-confidence, completeness and trusted
   gates; and
6. has raw invalid, negative and NaN rates each below 0.5% (Inf remains part
   of the raw invalid definition).

At the time this protocol was added, D-025 was still in progress and had no
final checkpoint or held-out result. Its current checkpoint is deliberately
rejected. The existing canonical Stage-C refiner is bound to a different base
and must never initialize this arm. The only legal initialization is a fresh
zero-output epipolar head over the passing D-025 final base.

## Physical correction

For the non-negative D-025 base, retain the normal candidate mask and bounded
correction:

```text
d_pre       = d_base + delta
delta_safe  = max(delta, -d_base)       only where any candidate is valid
delta_safe  = 0                         where all candidates are invalid
d_refined   = d_base + delta_safe
```

The projection is evaluated in FP32. It guarantees `d_refined >= 0` without
changing the `[-2,+2]` upper correction bound. Exact zero remains invalid for
depth, point clouds and completeness; there is no epsilon fill, softplus or
other fabricated positive point. A squared hinge on `d_pre` over the complete
candidate-any-valid domain supplies gradient when the hard projection is hit:

```text
L_stage_c_positive = lambda * mean(relu(-d_pre)^2)
```

The D-025 base remains frozen/no-grad, its FFS anchor remains unchanged, and
the epipolar candidate mask/row-coordinate contract are unchanged. Default
Stage-C construction has no extra tensor taps, loss term, config section or
checkpoint field; old checkpoints retain their exact behavior and state dict.

## CPU-only prerequisite check

Do not create a Stage-C output directory until this passes:

```bash
/home/haoyi/miniconda/envs/env-tsr/bin/python \
  tools/preflight_stage_c_d025_positivity.py \
  --config configs/ablations/d025_stage_c_positivity.yaml \
  --d025-base-checkpoint /path/to/d025/final.pt \
  --d025-training-audit /path/to/d025_training_audit_final.json \
  --d025-evaluation-audit /path/to/d025_evaluation_audit_final.json \
  --receipt /path/to/stage_c_d025_preflight.json
```

The preflight is read-only and uses no GPU. It binds the exact D-025 checkpoint
SHA to both receipts, hash-checks every artifact named by the formal evaluation
audit, and reruns that independent cross-audit. A missing, intermediate,
canonical-Stage-B, health-failing or lineage-mismatched artifact fails closed.

After `PREFLIGHT_PASS`, start a new run:

```bash
/home/haoyi/miniconda/envs/env-tsr/bin/python train_epipolar.py \
  --config configs/ablations/d025_stage_c_positivity.yaml \
  --init-from /path/to/d025/final.pt \
  --d025-training-audit /path/to/d025_training_audit_final.json \
  --d025-evaluation-audit /path/to/d025_evaluation_audit_final.json \
  --manifest /home/haoyi/ffs_omega_cache/manifests/train_video_isolated.jsonl \
  --observation-cache-root /home/haoyi/ffs_omega_cache/m1_formal_train/observation \
  --teacher-cache-root /home/haoyi/ffs_omega_cache/m1_formal_train/teacher \
  --derived-cache-root /home/haoyi/ffs_omega_cache/m2_formal_train/derived \
  --rectification-audit reports/m6/epipolar_rectification_audit.json \
  --output /path/to/new_empty_stage_c_d025_positivity \
  --device cuda
```

Its checkpoint is labeled `CONTROLLED_D025_STAGE_C_ABLATION`, embeds the
prerequisite receipt, and always records `formal_training_complete=false` and
`canonical_stage_c_replacement=false`. A completed arm instead records the
separate `controlled_ablation_training_complete` field.

A full matched evaluation is labeled
`CONTROLLED_ABLATION_EVALUATION_COMPLETE`; its canonical
`acceptance_eligible` remains false and it records
`canonical_stage_c_replacement=false`. Thus the arm can produce complete
ablation evidence without relabeling either the historical Stage-C result or
the paper-facing canonical track as passed.

## Required matched evaluation

Report raw and clamp-zero rows, but only raw owns gates. In addition to existing
metrics, retain a sign-transition matrix, base-disparity bins crossed by each
correction sign, trusted/untrusted and hole cross-tabs, candidate-boundary/OOB
cross-tabs, and the positivity projection-hit rate. A zero produced by the
projection is still invalid and cannot improve completeness.

If completeness remains deficient, any teacher-positive hole reweighting is a
second factor, not silently bundled with positivity. It must remain restricted
to `teacher_trusted & !ffs_valid & candidate_any_valid`, and its same-family
pseudo-GT limitation must be explicit.

## Optional high-VRAM execution profile

The separate `d025_stage_c_positivity_high_vram.yaml` changes only the batch
split from 2x4 to 4x2. It is not enabled by this protocol and cannot start
formal training until its exact CUDA/BF16 optimizer-step memory probe produces
a bound PASS receipt. See `STAGE_C_D025_HIGH_VRAM_PREFLIGHT.md`; OOM or less
than 2 GiB measured headroom falls back to this standard 2x4 config.
