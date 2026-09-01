# Stage-C D-025 high-VRAM preflight

Status: implementation-only controlled execution option. It is not a result,
does not make the current Stage-C run pass, and must not start until the full
D-025 Stage-B training and final controlled-comparison audit both pass.

## Scope

The candidate config is
`configs/ablations/d025_stage_c_positivity_high_vram.yaml`. It changes only the
micro-batch split from `2 x 4` to `4 x 2`; effective batch size remains 8. The
model, objective, crop, optimizer, D-025 base, Stage-C positivity protocol, and
5,000-update schedule remain unchanged. Canonical Stage C and the standard
D-025 Stage-C config do not inherit `stage_c_high_vram`.

The candidate is fail-closed. A formal run needs a PASS receipt from one real
CUDA/bf16 optimizer update of the exact resolved config. The probe is isolated:
it runs two micro-steps of batch size 4, completes forward, backward, gradient
validation, and one optimizer update, then discards that state. It must not
advance a formal run, write a checkpoint, or reuse the probe-updated refiner.

## Required receipt contract

The producer and formal-run validator must bind all four identities:

- canonical SHA-256 of the exact resolved high-VRAM config;
- D-025 base checkpoint path, SHA-256, and completed step 15,000;
- Stage-C runtime source bundle Git hash and bundle SHA-256;
- the complete, already validated D-025 Stage-B prerequisite identity.

The execution section must prove `device_type=cuda`, bf16, micro-batch 4,
accumulation 2, exactly two completed micro-steps, exactly one completed
optimizer step, finite loss/gradient/updated parameters, no CUDA OOM, no formal
training start, no checkpoint write, and discarded probe state.

The memory section records integer bytes for:

- `peak_cuda_allocated_bytes`;
- `peak_cuda_reserved_bytes`;
- `cuda_total_bytes`;
- `headroom_bytes = cuda_total_bytes - peak_cuda_reserved_bytes`;
- `cuda_free_before_bytes` and `cuda_free_after_bytes` as diagnostics.

PASS additionally requires positive headroom of at least 2 GiB. Reset CUDA peak
statistics immediately before the first probe micro-step and synchronize before
reading the final counters. The receipt is only valid for the same config,
base, runtime bundle, and D-025 prerequisite used by the formal launch.

## OOM and fallback

CUDA OOM never produces a receipt or PASS. The probe discards its in-memory
state, reports the fallback explicitly, and falls back exactly to:

```text
train.micro_batch_size=2
train.grad_accumulation=4
train.effective_batch_size=8
stage_c_high_vram.enabled=false
```

The fallback is the existing standard D-025 Stage-C execution profile; it is
not an automatic retry inside the high-VRAM output directory. Use a fresh,
empty output directory. Any missing, malformed, stale, OOM, CPU, non-bf16,
non-finite, insufficient-headroom, or identity-mismatched receipt blocks formal
high-VRAM execution.

## Required launch order

First generate the receipt with a real CUDA/BF16 two-micro-step probe. This
does not create or advance a training output directory:

```bash
python train_epipolar.py \
  --config configs/ablations/d025_stage_c_positivity_high_vram.yaml \
  --dry-run --device cuda \
  --high-vram-preflight-receipt reports/m6/d025_stage_c_high_vram_preflight.json \
  --init-from /path/to/d025/final.pt \
  --d025-training-audit /path/to/d025_training_audit_final.json \
  --d025-evaluation-audit /path/to/d025_final_controlled_audit.json \
  --manifest /home/haoyi/ffs_omega_cache/manifests/train_video_isolated.jsonl \
  --observation-cache-root /home/haoyi/ffs_omega_cache/m1_formal_train/observation \
  --teacher-cache-root /home/haoyi/ffs_omega_cache/m1_formal_train/teacher \
  --derived-cache-root /home/haoyi/ffs_omega_cache/m2_formal_train/derived \
  --rectification-audit reports/m6/epipolar_rectification_audit.json
```

Only after that exact receipt says `PREFLIGHT_PASS`, launch a fresh formal run
with the same arguments, omit `--dry-run`, and add a new empty `--output`.
The trainer reopens and revalidates the receipt before creating that output.

If the probe OOMs or has less than 2 GiB headroom, use
`configs/ablations/d025_stage_c_positivity.yaml` (the unchanged 2x4 profile)
and a different empty output directory. No CUDA probe has been run as part of
this implementation commit; the 4x2 candidate therefore remains blocked.
