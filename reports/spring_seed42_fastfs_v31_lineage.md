# Spring seed 42 / FastFS v3.1 lineage preparation

This note records the immutable input preparation for the Fast-
FoundationStereo Spring matrix.  The historical manifests and caches are
not modified.

## Backbone pin

| Item | Value |
|---|---|
| Fast-FoundationStereo commit | `a290ba04c1b3ad1ec41a33974a157b2917b624d4` |
| observation checkpoint | `checkpoints/ffs/20-30-48/model_best_bp2_serialize.pth` |
| observation SHA-256 | `98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692` |
| teacher checkpoint | `checkpoints/ffs/23-36-37/model_best_bp2_serialize.pth` |
| teacher SHA-256 | `af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990` |

## Strict v3.1 calibration input

The original Spring manifests do not contain `K_right`, `P_left`, `P_right`,
or per-row metadata.  Two explicit routes are available:

1. `tools/augment_spring_v31_manifest.py` writes a new immutable manifest and
   deterministic YAML camera-info metadata.  This is the strict augmented
   route.  Because the manifest SHA changes, raw VGGT caches must be rebuilt
   against the augmented manifest.
2. `tools/build_stereo_calibration.py --spring-native` derives the same
   rectified `[I|-baseline]` contract from the original Spring `K` and
   baseline, writing synthetic metadata beside the sidecar.  This route keeps
   the original manifest and allows the existing raw VGGT caches to be reused;
   the sidecar receipt is explicitly marked `spring_rectified_native_v1`.

The native route has been smoke-built and loaded for the original primary
manifests:

```text
runs/spring_seed42_primary_fastfs_v31/inputs/calibration_native/train.jsonl
runs/spring_seed42_primary_fastfs_v31/inputs/calibration_native/validation.jsonl
```

Both receipts are `PASS`; the sidecar SHA-256 values are recorded in the
corresponding receipt files.

Current active-root sidecar hashes:

```text
train:      4e1bb64515a50d8b38fb52371d87ceb061c203dcefc88d14f7d03cdcfa3c2841
validation: c8c59bc555984ad785876855044715421d3cdc72625366d739176181e52e97c6
```

For the active FastFS root, the native sidecars are reproducible with:

```bash
PY=/home/CNF2026527811/miniconda3/envs/trtllm/bin/python
ROOT=runs/spring_v31_ffs
mkdir -p "$ROOT/sidecars"

"$PY" tools/build_stereo_calibration.py \
  --manifest runs/spring_seed42_primary/manifests/train.jsonl \
  --pixel-audit reports/spring_epipolar_rectification_primary.json \
  --output "$ROOT/sidecars/train_calibration.jsonl" \
  --receipt "$ROOT/sidecars/train_calibration.receipt.json" \
  --spring-native

"$PY" tools/build_stereo_calibration.py \
  --manifest runs/spring_seed42_primary/manifests/validation.jsonl \
  --pixel-audit reports/spring_epipolar_rectification_primary.json \
  --output "$ROOT/sidecars/validation_calibration.jsonl" \
  --receipt "$ROOT/sidecars/validation_calibration.receipt.json" \
  --spring-native
```

The augmentation route was also exercised on the full primary manifests.  The
current tool output (under the disposable `aug_new` subdirectory) is:

```text
train manifest SHA-256: abc705fd26f8e89c41a07cc125062449a8f489bc8b366b35da6aef8f7062b9a8
validation manifest SHA-256: 45c0f4eebcb4dbc73c451137a5c2bac373386fad5e888b0c0cebaf240f698d9f
```

## Augmented route command

```bash
PY=/home/CNF2026527811/miniconda3/envs/trtllm/bin/python
ROOT=runs/spring_seed42_primary_fastfs_v31
mkdir -p "$ROOT/manifests" "$ROOT/inputs/spring_v31_metadata"

"$PY" tools/augment_spring_v31_manifest.py \
  --input runs/spring_seed42_primary/manifests/train.jsonl \
  --output "$ROOT/manifests/train.jsonl" \
  --metadata-root "$ROOT/inputs/spring_v31_metadata/train" \
  --receipt "$ROOT/manifests/train.augmentation.json"

"$PY" tools/augment_spring_v31_manifest.py \
  --input runs/spring_seed42_primary/manifests/validation.jsonl \
  --output "$ROOT/manifests/validation.jsonl" \
  --metadata-root "$ROOT/inputs/spring_v31_metadata/validation" \
  --receipt "$ROOT/manifests/validation.augmentation.json"
```

The augmentation tool is immutable: a repeated invocation accepts only
byte-identical metadata/manifest/receipt artifacts, and `--output` may not
equal `--input`.

## FFS cache commands

Use separate train/validation roots so both GPUs can run concurrently without
shard-merging races.  Restrict each process with `CUDA_VISIBLE_DEVICES` and
pass `--device cuda`.

```bash
PY=/home/CNF2026527811/miniconda3/envs/trtllm/bin/python
MANIFEST_ROOT=runs/spring_seed42_primary/manifests
ROOT=runs/spring_v31_ffs
OBS=checkpoints/ffs/20-30-48/model_best_bp2_serialize.pth
TEACH=checkpoints/ffs/23-36-37/model_best_bp2_serialize.pth
REPO=third_party/Fast-FoundationStereo

# F1--F7 half-resolution observation (train on GPU0, validation on GPU1)
CUDA_VISIBLE_DEVICES=0 "$PY" tools/cache_ffs.py \
  --manifest "$MANIFEST_ROOT/train.jsonl" \
  --output "$ROOT/cache/train/ffs_half" --checkpoint "$OBS" \
  --checkpoint-label 20-30-48 --role observation \
  --repo "$REPO" --device cuda --right-left-check

CUDA_VISIBLE_DEVICES=1 "$PY" tools/cache_ffs.py \
  --manifest "$MANIFEST_ROOT/validation.jsonl" \
  --output "$ROOT/cache/validation/ffs_half" --checkpoint "$OBS" \
  --checkpoint-label 20-30-48 --role observation \
  --repo "$REPO" --device cuda --right-left-check

# F0 full-resolution observation (validation is sufficient for the baseline)
CUDA_VISIBLE_DEVICES=1 "$PY" tools/cache_ffs.py \
  --manifest "$MANIFEST_ROOT/validation.jsonl" \
  --output "$ROOT/cache/validation/ffs_full416" --checkpoint "$OBS" \
  --checkpoint-label 20-30-48 --role observation --scale 1 \
  --allow-full-resolution-observation --repo "$REPO" --device cuda \
  --right-left-check --max-disp 416
```

Use `--max-disp 416` for F0.  The checkpoint is serialized with native
`max_disp=416`; `192` is the half-resolution observation setting and may clip
high-disparity Spring pixels when used at full resolution.

The active common-domain run keeps this full-resolution cache in
`runs/spring_v31_ffs/cache/validation/ffs_full416/observation`; the older
`runs/spring_v31_ffs/cache/validation/ffs_full/observation` tree used
`max_disp=192` and is exploratory only.  The Spring GT teacher remains
independent from FFS and is already bound to the original validation manifest.
Audit the half-res and full-res observation caches with
`tools/audit_ffs_cache.py`, passing `--observation-scale 2` and
`--observation-scale 1` respectively.  The active full-res cache is now
complete and audited:

```text
records: 1350/1350
run_receipt SHA-256: 38cf960c7fcd42f5217586d8a52f2ad98bb5df2be50384ea660c1eb272cf75ef
cache_manifest SHA-256: e82586da0589e6b95415d8a649894a27f1d55d240f32b3c9395c678d759ff18b
audit: runs/spring_v31_ffs/audits/ffs_full416_validation.json (PASS, 9/1350 sampled)
```

The receipt binds `scale=1`, `max_disp=416`, `iterations=4`,
`volume_backend=pytorch1`, right-left checking, checkpoint SHA
`98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692`, and
FastFS commit `a290ba04c1b3ad1ec41a33974a157b2917b624d4`.

An exhaustive manifest/index audit also found exact key coverage for all
three active FFS trees: train half `3650/3650`, validation half `1350/1350`,
and validation full416 `1350/1350`, with no duplicate keys, missing files, or
cache-manifest paths outside their roots.

## Common evaluation domain

All F0--F6 results must use one manifest-bound endpoint file and the same
resolution/crop contract.  The active file is:

```text
runs/spring_v31_ffs/manifests/common_endpoints.json
endpoint_count: 1302
endpoint_id_sha256: aa6ba30295b8d5ab0e1b4326a14fae61f9c8ec42641801cd8442097bc3ab5b57
file_sha256: 9fa81b1ca652fdd1f33634f83213c28e826db91562125430aedb0beef4f5c838
```

Passing a smaller `--limit` to only the temporal arms changes the evaluated
endpoint set and invalidates direct F0--F6 comparisons; formal runs should
omit that limit (or use the same explicit subset for every arm).

## Tests

```bash
PYTHONPATH=src "$PY" -m pytest -q \
  tests/test_augment_spring_v31_manifest.py \
  tests/test_stereo_calibration.py \
  tests/test_cache_spring_ffs.py
```

The local targeted suite passes (`12 passed`):

```text
tests/test_augment_spring_v31_manifest.py
tests/test_spring_native_calibration.py
tests/test_stereo_calibration.py
tests/test_cache_spring_ffs.py
```

The repository-wide regression suite currently passes `661 passed, 1 skipped`
(the skip is the optional system-FFmpeg test).  F0/F1/F2 have completed on the
common endpoint domain; F2's T1 validity-gate behavior is documented in
`reports/spring_v31_ffs_f2_t1_diagnosis.md`.  F4's first canary attempts exposed
two fail-closed input checks (inverse-transform homogeneous-row rounding and
GT-pose override residual metadata); both fixes are covered by the current
calibration/lineage tests, and the isolated `canary500_v2` is rerunning with
the corrected contracts.  Failed canary directories are retained as evidence
and must not be treated as formal arm results.
