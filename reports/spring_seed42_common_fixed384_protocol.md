# Spring seed=42 common fixed384 evaluation protocol

This is the comparison protocol for the seven Spring arms. It is separate
from the existing full-resolution exploratory report and does not overwrite
any earlier result.

## Frozen population

- Validation manifest: `runs/spring_seed42_primary/manifests/validation.jsonl`
- Manifest SHA-256: `6c016bdb4aa7f4a2be07c713cf9ae90b8dccdeab31c76c71d9d0e96b1a8bc45e`
- Endpoint index: `runs/spring_seed42_primary/manifests/common_fixed384_endpoints.json`
- Endpoint-index SHA-256: `9fa81b1ca652fdd1f33634f83213c28e826db91562125430aedb0beef4f5c838`
- Endpoint identity SHA-256: `aa6ba30295b8d5ab0e1b4326a14fae61f9c8ec42641801cd8442097bc3ab5b57`
- Endpoint identity hash: `sha256(canonical_json([manifest_index,sequence_id,frame_id,timestamp]))`
- Population: 1,302 endpoint windows, sequence-disjoint validation, seed 42
- Temporal context: causal T=3 student frames and five causal VGGT pairs
- Selection rule: build the complete dataset/cache first, then filter by the
  original manifest endpoint index; missing or mismatched endpoints are fatal.

## Image/crop coordinate contract

The model's HR/image grid is `1080×1920`; the Spring disparity source is a
separate `2160×3840` 4K grid. The common model-space crop is:

```text
crop_mode = fixed
crop_hr = [height=384, width=768]
origin_hr_xy = [x=576, y=348]
```

`[1536,888]` is not a valid origin for the current model grid. It is a
4K-grid coordinate and must not be passed directly to the current evaluator.
The equivalent 4K rectangle, if needed only for documentation, is
`[x=1152, y=696, width=1536, height=768]`.

## Pose/depth isolation

| comparison | pose | VGGT depth | epipolar refiner | purpose |
|---|---|---|---|---|
| S4 vs S5 | GT vs VGGT | yes | no | isolate pose error |
| S5 base vs S6 refined | VGGT | yes | no vs yes | isolate Stage-C correction |
| S2 vs S3 | GT | no | no | isolate top-K/history selection |

GT-pose and VGGT-pose values are never pooled into one temporal number.

## Metric contract

Use the same native Spring metric implementation and masks for every arm:

- overall EPE and Bad-1px;
- high-detail EPE/Bad-1px, low-detail EPE;
- matched EPE;
- unmatched completion at 1px and 2px;
- rigid and non-rigid temporal residual error;
- boundary EPE;
- FFS trusted-measurement error;
- negative, zero, and invalid rates.

Aggregate pixel-valued quantities from global numerators/counts (not an
unweighted mean of frame means). Report valid-pixel counts and denominators.
Top-K diagnostics use the same endpoint population: age-2 survival, unique-age
fraction, phase variance, candidate-depth spread, attention entropy, and the
fractional-phase/camera-motion buckets.

## S6 decision rule

The primary S6 claim is a paired raw-output comparison on the same 1,302
windows and crop. `T3_VGGT_base` is the frozen S5 output; it is compared with
`T3_VGGT_epipolar`. The clamp-to-zero variant is diagnostic only and never
replaces the raw output.

Accept S6 as an improvement only when the paired EPE/Bad-1 (and preferably
trusted-region EPE) gain is accompanied by no material invalid/negative/zero
regression and no unacceptable low-confidence or temporal-residual regression.
Report per-sequence paired deltas/95% sequence-block bootstrap intervals before
making a final claim.

Current paired evidence (same fixed crop and 1,302 windows) is:

```text
EPE              0.444291 -> 0.441873  (-0.54%)
Bad-1            0.049999 -> 0.049703  (-0.59%)
trusted EPE      0.141278 -> 0.135864  (-3.83%)
low-confidence   0.711929 -> 0.712242  (+0.044%)
invalid rate     0.01113808 -> 0.01113812 (negligible)
correction used  92.25%; candidate coverage 97.08%; saturated 0%
```

The existing S5 fixed-crop receipt
`runs/spring_seed42_primary/corrected_plan/arms/S5/eval_fixed_crop_20260902/metrics.json`
and the S6 receipt's `T3_VGGT_base` method have identical aggregate values,
pixel counts, and numerators for the common pseudo-GT fields. This establishes
the base side of the S5↔S6 pair without rerunning the old checkpoint.

This is a fixed-crop paired refinement result, not a full-resolution ranking
against the old S5 full-frame number.

## Re-evaluation commands

For S1--S5, append the common endpoint list and explicit model-space crop:

```bash
--spring-endpoint-index-list \
  runs/spring_seed42_primary/manifests/common_fixed384_endpoints.json \
--crop-mode fixed --crop-origin 576 348 --limit 1302
```

S0's standalone baseline adapter currently has only contiguous `start/limit`
selection and no crop-origin argument. For a strict seven-arm table, either
add the same endpoint/crop contract to `tools/eval_spring_baseline.py`, or use
the `bilinear` row emitted by a fixed-crop S1/S5 evaluation as the common S0
reference. Do not mix the existing full-resolution S0 row into this table.

For S6, use the same endpoint list and `--limit 1302`; its Stage-C adapter
already enforces fixed `384×768` crop and center selection. Keep the output in a
new directory (for example `eval_common_fixed384_20260902`) so the existing
receipts remain immutable.
