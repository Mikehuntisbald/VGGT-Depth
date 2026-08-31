# Runbook

All commands below run from `/home/haoyi/ffs_omega_tsr`. Do not install an
upstream package before installing the selected cu128-or-newer PyTorch build.

## 1. Initialize source

```bash
git submodule update --init --recursive
git submodule status
git -C third_party/Fast-FoundationStereo status --short
git -C third_party/vggt-omega status --short
```

Expected pins are recorded in `third_party/LOCK.json`. A different SHA, remote,
or dirty submodule is a failed provenance check.

## 2. Recreate the three environments

```bash
conda create -y -n env-ffs python=3.12 pip
conda create -y -n env-vggt python=3.12 pip
conda create -y -n env-tsr python=3.12 pip

conda run -n env-ffs python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n env-vggt python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128
conda run -n env-tsr python -m pip install \
  torch==2.10.0 torchvision==0.25.0 \
  --index-url https://download.pytorch.org/whl/cu128

conda run -n env-ffs python -m pip install \
  -r third_party/Fast-FoundationStereo/requirements.txt ninja
conda run -n env-vggt python -m pip install \
  -r third_party/vggt-omega/requirements.txt \
  -e third_party/vggt-omega
conda run -n env-tsr python -m pip install -e '.[dev]'
```

Do not copy the FFS README's cu124 command onto this RTX 5090 machine. The
system's default `nvcc` is currently CUDA 13.3; pure PyTorch cu128 works, but set
an explicit compatible `CUDA_HOME` before building future custom extensions.

## 3. Verify each environment

```bash
conda run -n env-ffs python tools/check_env.py \
  --profile ffs --expect-env env-ffs \
  --json-out reports/m0/env_ffs.json
conda run -n env-vggt python tools/check_env.py \
  --profile vggt --expect-env env-vggt \
  --json-out reports/m0/env_vggt.json
conda run -n env-tsr python tools/check_env.py \
  --profile tsr --expect-env env-tsr \
  --json-out reports/m0/env_tsr.json
```

Strict PASS requires CUDA availability, Torch CUDA >=12.8, RTX 5090/SM120, all
profile imports, and real finite FP16/BF16 CUDA matmuls.

## 4. Place checkpoints

Expected production files:

```text
checkpoints/ffs/20-30-48/model_best_bp2_serialize.pth  # observation
checkpoints/ffs/20-30-48/cfg.yaml
checkpoints/ffs/23-36-37/model_best_bp2_serialize.pth  # teacher
checkpoints/ffs/23-36-37/cfg.yaml
checkpoints/vggt/vggt_omega_1b_512.pt
```

FFS weights come from the folder linked by the official repository. The
teacher is also mirrored byte-for-byte at
`Miayan/stereo-matching-weights` on Hugging Face; always validate the final
hash rather than trusting a mirror filename. VGGT-Ω comes from the
gated `facebook/VGGT-Omega` Hugging Face repository; request access in the
browser, wait for approval, then use the authenticated CLI:

```bash
hf download facebook/VGGT-Omega vggt_omega_1b_512.pt \
  --local-dir checkpoints/vggt
```

Never put a Hugging Face token in this repository or a receipt.

Expected FFS identities on this workstation:

```text
20-30-48  62078956 bytes  sha256 98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692
23-36-37  71098210 bytes  sha256 af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990
```

This workstation already has an equivalent ModelScope cache artifact. Reuse it
without duplicating 4.58 GB:

```bash
mkdir -p checkpoints/vggt
ln -s \
  /home/haoyi/.cache/modelscope/hub/models/facebook/VGGT-Omega/vggt_omega_1b_512.pt \
  checkpoints/vggt/vggt_omega_1b_512.pt
sha256sum checkpoints/vggt/vggt_omega_1b_512.pt
```

Expected SHA-256:
`c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934`.

The M0 FFS interface probe used a locally available `20-26-39` artifact at:

```text
checkpoints/ffs/20-26-39/model_best_bp2_serialize.pth
```

Its recorded hash is in `REPORT.md`. It is not a substitute for the final
fast/teacher pair.

An official signed NGC v1.2 artifact is also available from the stable catalog
endpoint:

```bash
mkdir -p checkpoints/ffs/ngc-v1.2
wget -c \
  -O checkpoints/ffs/ngc-v1.2/model_best_bp2_serialize.pth \
  'https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/fast-foundationstereo/v1.2/download-file?path=model_best_bp2_serialize.pth'
wget -q \
  -O checkpoints/ffs/ngc-v1.2/cfg.yaml \
  'https://catalog.ngc.nvidia.com/orgs/nvidia/tao/models/fast-foundationstereo/v1.2/download-file?path=cfg.yaml'
sha256sum checkpoints/ffs/ngc-v1.2/*
```

The NGC checkpoint lacks `model.args.normalize`, while current GitHub source
accesses that field directly. Strict smoke must fail. The recorded compatibility
probe explicitly supplies `true`, matching the upstream volume helper's legacy
default; it is never inferred silently.

## 5. Run backbone smokes

FFS uses the upstream demo stereo pair without the destructive/interactive
official demo wrapper. The canonical M0 receipt uses the NGC artifact:

```bash
conda run -n env-ffs python tools/smoke_ffs.py \
  --checkpoint checkpoints/ffs/ngc-v1.2/model_best_bp2_serialize.pth \
  --iters 4 --max-disp 192 --volume-backend pytorch1 \
  --missing-normalize true \
  --json-out reports/m0/smoke_ffs.json
```

Run without `--missing-normalize true` to enforce the strict checkpoint/source
contract. The local middle-tier candidate can be probed separately without that
fallback; see `reports/m0/smoke_ffs_local_20-26-39.json`.

The `pytorch1` helpers still contain upstream `torch.compile` decorators, so the
first measured forward includes compilation. `--disable-dynamo` is an explicit,
recorded correctness fallback; do not silently use it for performance claims.

VGGT-Ω requires ten real images in this exact causal order:

```bash
conda run -n env-vggt python tools/smoke_vggt.py \
  --checkpoint checkpoints/vggt/vggt_omega_1b_512.pt \
  --sequence-metadata reports/m0/vggt_smoke_window.json \
  --images \
    /data/L_t-4.png /data/R_t-4.png \
    /data/L_t-3.png /data/R_t-3.png \
    /data/L_t-2.png /data/R_t-2.png \
    /data/L_t-1.png /data/R_t-1.png \
    /data/L_t.png   /data/R_t.png \
  --input-mode balanced --image-resolution 512 \
  --json-out reports/m0/smoke_vggt.json
```

Do not repeat one image ten times and call the result an M0 PASS.

The executed M0 window uses `samples_val/000033` through `000037` from
`xd_stereo_flashocc_teacher_video_5fps`. Their metadata confirms one source
video, 0.2-second spacing, declared rectification, and a 0.116124 m baseline.

## 6. Run project tests

```bash
conda run -n env-tsr pytest -q
```

## 7. Build the video-isolated manifests

```bash
export XD_DATA_ROOT=/home/haoyi/Downloads/xd/datasets/xd_stereo_flashocc_teacher_video_5fps
export CACHE_ROOT=/home/haoyi/ffs_omega_cache

conda run -n env-tsr python tools/build_manifest.py \
  --data-root "$XD_DATA_ROOT" \
  --source-csv "$XD_DATA_ROOT/manifest.csv" \
  --include-source-video-stem uvc_20260706_172530_741_2560x800_60fps \
  --include-source-video-stem uvc_20260706_191627_418_2560x800_60fps \
  --output "$CACHE_ROOT/manifests/train_video_isolated.jsonl"

conda run -n env-tsr python tools/build_manifest.py \
  --data-root "$XD_DATA_ROOT" \
  --source-csv "$XD_DATA_ROOT/manifest_val.csv" \
  --include-source-video-stem uvc_20260706_183939_677_2560x800_60fps \
  --output "$CACHE_ROOT/manifests/val_video_isolated.jsonl"
```

The checked-in report records the exact resulting hashes. Never use the CSV's
unrectified `left.jpg/right.jpg` paths.

## 8. Generate and audit FFS caches

Run each command once for train and validation by changing manifest/output.

```bash
conda run -n env-ffs python tools/cache_ffs.py \
  --manifest "$CACHE_ROOT/manifests/train_video_isolated.jsonl" \
  --output "$CACHE_ROOT/m1_formal_train" \
  --checkpoint checkpoints/ffs/20-30-48/model_best_bp2_serialize.pth \
  --checkpoint-label 20-30-48 --role observation --right-left-check

conda run -n env-ffs python tools/cache_ffs.py \
  --manifest "$CACHE_ROOT/manifests/train_video_isolated.jsonl" \
  --output "$CACHE_ROOT/m1_formal_train" \
  --checkpoint checkpoints/ffs/23-36-37/model_best_bp2_serialize.pth \
  --checkpoint-label 23-36-37 --role teacher --right-left-check

conda run -n env-tsr python tools/audit_ffs_cache.py \
  --manifest "$CACHE_ROOT/manifests/train_video_isolated.jsonl" \
  --observation-root "$CACHE_ROOT/m1_formal_train/observation" \
  --teacher-root "$CACHE_ROOT/m1_formal_train/teacher" \
  --samples 11 --json-out reports/m1/cache_audit_train.json
```

Re-running without `--overwrite` safely reuses only exact identity/source
matches. A mismatch is an error, not a cache miss.

## 9. Generate raw VGGT caches

```bash
conda run -n env-vggt python tools/cache_vggt.py \
  --manifest "$CACHE_ROOT/manifests/val_video_isolated.jsonl" \
  --output "$CACHE_ROOT/m2_formal_val/vggt" \
  --checkpoint checkpoints/vggt/vggt_omega_1b_512.pt \
  --context-pairs 5 --causal --input-mode balanced
```

The raw cache is intentionally arbitrary-scale and stores VGGT's unbounded
confidence score. It is not safe for temporal warping until the derived
baseline/photometric/depth quality pipeline marks `pose_valid=true`.

## 10. Derive metric geometry and enforce quality gates

Run the same command for train and validation by changing the three cache
roots and report name:

```bash
conda run -n env-tsr python tools/derive_geometry_manifest.py \
  --vggt-root "$CACHE_ROOT/m2_formal_train/vggt" \
  --ffs-root "$CACHE_ROOT/m1_formal_train/observation" \
  --output "$CACHE_ROOT/m2_formal_train/derived" \
  --cache-dtype float32 --start-window 0 \
  --report reports/m2/derive_geometry_train.json
```

The default gate uses baseline CV, stereo rotation, photometric reprojection,
weighted disparity MAE, and median absolute disparity error. Pointwise relative
error is recorded but is not a default gate because trusted far-range
disparities approach zero. A rejected pose is represented by a false validity
scalar and zero temporal extrinsics; a rejected static prior has zero disparity,
confidence, and mask. The batch command safe-loads every output and audits this
zero contract before publishing its receipt.

## 11. Train and evaluate Stage A (T=1)

Start formal training only from a committed source tree so checkpoint
`git_hash` is never `unknown`:

```bash
conda run --no-capture-output -n env-tsr python train.py \
  --config configs/mvp_x2.yaml \
  --manifest "$CACHE_ROOT/manifests/train_video_isolated.jsonl" \
  --observation-cache-root "$CACHE_ROOT/m1_formal_train/observation" \
  --teacher-cache-root "$CACHE_ROOT/m1_formal_train/teacher" \
  --output-dir outputs/ffs_omega_tsr_x2/stage_a \
  --device cuda

conda run --no-capture-output -n env-tsr python eval.py \
  --config configs/mvp_x2.yaml \
  --checkpoint outputs/ffs_omega_tsr_x2/stage_a/final.pt \
  --manifest "$CACHE_ROOT/manifests/val_video_isolated.jsonl" \
  --observation-cache-root "$CACHE_ROOT/m1_formal_val/observation" \
  --teacher-cache-root "$CACHE_ROOT/m1_formal_val/teacher" \
  --output outputs/ffs_omega_tsr_x2/stage_a_eval \
  --device cuda
```

These metrics use a confidence/LR-consistency-filtered HR FFS teacher and are
engineering pseudo-GT evidence, not paper accuracy. Preserve `final.pt`,
`train.jsonl`, `run_summary.json`, `metrics.json`, and `metrics.csv` together.

## 12. Train Stage B (causal T=3)

Stage B requires the exact Stage-A checkpoint and all three per-time derived
records in each causal window:

```bash
conda run --no-capture-output -n env-tsr python train.py \
  --config configs/temporal_x2.yaml \
  --manifest "$CACHE_ROOT/manifests/train_video_isolated.jsonl" \
  --observation-cache-root "$CACHE_ROOT/m1_formal_train/observation" \
  --teacher-cache-root "$CACHE_ROOT/m1_formal_train/teacher" \
  --derived-cache-root "$CACHE_ROOT/m2_formal_train/derived" \
  --init-from outputs/ffs_omega_tsr_x2/stage_a/final.pt \
  --output-dir outputs/ffs_omega_tsr_x2/stage_b \
  --device cuda
```

Resume with `--resume outputs/ffs_omega_tsr_x2/stage_b/latest.pt` and omit
`--init-from`; the saved initialization lineage, cache identities, deterministic
sampler epoch, and micro-batch cursor are restored exactly. If this stage OOMs,
first set `train.micro_batch_size=1 train.grad_accumulation=8` so the effective
batch remains eight.

## Status and exit codes

| Status | Exit | Meaning |
|---|---:|---|
| `PASS` | 0 | Every required check actually ran and passed. |
| `PASS_WITH_FALLBACK` | 0 | A declared backend or compatibility fallback ran and passed. |
| `FAIL` | 1 | Inputs were present but execution or an invariant failed. |
| `BLOCKED` | 3 | A repository, package, checkpoint, authorization, or real input is missing. |
| `NOT_RUN` | 3 | There is no current receipt. |

Receipts are written even on `BLOCKED` or `FAIL`. Random tensors, dummy models,
and visually plausible output are never accepted as milestone evidence.
