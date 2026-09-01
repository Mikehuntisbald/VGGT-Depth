# FFS-Ω-TSR

FFS-Ω-TSR is a causal x2 disparity super-resolution project with three explicit
owners:

- frozen Fast-FoundationStereo owns the current-frame metric disparity;
- frozen VGGT-Ω supplies long-horizon geometry, camera motion, and hole priors;
- a lightweight RGB/temporal reconstruction head below the 12M hard ceiling
  produces HR disparity and uncertainty. The current exact MVP is 1.62M
  parameters; capacity will only increase if matched training shows underfit.

The MVP uses a 640×360 FFS observation for a 1280×720 output, causal `T=3`, and
five past/current stereo pairs for the VGGT-Ω context. Both large backbones are
offline cache producers and are never optimized with the TSR model.

## Current status

M0 is **PASS_WITH_FALLBACK**, M1 is **PASS**, M2/M3 geometry is complete, and
the formal Stage-A spatial run has a conditional pseudo-GT engineering PASS.
Formal Stage-B is running after a verified exact resume from its last atomic
checkpoint. Complete held-out intermediate evaluations are trend diagnostics;
the 15,000-step final checkpoint and audit own the final Stage-B go/no-go.

- Three isolated cu128 environments pass RTX 5090/SM120 FP16 and BF16 CUDA
  checks; both upstream repositories are clean pinned submodules.
- The exact official `20-30-48` observation and `23-36-37` teacher artifacts
  are present, hashed, and pass strict real inference without compatibility
  injection.
- A video-isolated manifest contains 2,787 train and 244 validation frames.
  All 3,031 frames now have versioned x2 observation and HR teacher caches,
  including entropy, update, LR-consistency, validity, and trusted masks.
- M1 cache footprints are about 9.4/32 GiB for train observation/teacher and
  0.84/2.8 GiB for validation. Sampled identity, source, shape, mask, and
  disparity-scale audits pass.
- Frozen VGGT-Ω raw caches cover all 2,779 train and 240 validation causal
  windows. Metric baseline scaling, robust scale-only alignment, complete pose
  quality, and safe-zero rejection were derived and audited for every window.
  Validation accepts 178/240 poses; train accepts 1,514/2,779, with the large
  per-sequence difference retained in the report rather than averaged away.
- The x2 RGB/ConvGRU/convex reconstruction head has 1,619,882 trainable
  parameters (below the 12M ceiling) and passes a batch-2, 384×768 BF16
  forward/backward smoke on the RTX 5090. The strict causal T=3 loader and
  trainer also pass a real-cache BF16 optimizer step using HR z-buffer
  transport and HR temporal loss.
- A separate 69,905-parameter HR local epipolar refiner, audited same-row pixel
  contract, training entrypoint, and strict held-out evaluator are implemented.
  CPU one-step and CUDA BF16 dry-run artifacts are integration evidence only;
  formal Stage-C training waits for the completed Stage-B checkpoint.
- Formal Stage-A completed 5,000 steps at 3.682 step/s. On all 244 held-out
  frames at full 800×1280 resolution, low-confidence EPE improves 11.996%,
  invalid-region completeness improves 725.3%, and trusted-region EPE improves
  4.269%. Raw disparity still has 3.722% negatives; the explicitly reported
  `clamp_min(0)` deployment row removes negatives without inventing positive
  epsilon depth, leaving those pixels honestly zero/invalid.

The M0 fallback remains historical and confined to the separate NGC interface
probe; it is not used by M1 caches. See [`REPORT.md`](REPORT.md) for exact
artifact identities, receipts, and claim boundaries.

## Non-negotiable geometry contracts

1. Every fused disparity is expressed in HR pixels. For x2,
   `disparity_hr_px = 2 * disparity_lr_px`.
2. VGGT-Ω camera-from-world translations and depth are scaled together from the
   known physical stereo baseline.
3. FFS remains the metric owner; VGGT/history corrections are confidence-gated.
4. Temporal transport is forward splatting with a z-buffer, never a bare
   backward `grid_sample` of disparity.

## Colored point-cloud export

`metrics.export_colored_point_cloud_ply` writes one calibrated camera-frame
ASCII PLY from HR-pixel disparity, the matching real HR intrinsics `K`, and the
physical stereo baseline. It requires RGB colors (uint8 or normalized float),
never invents normals/correspondences or point-to-plane scores, and writes only
finite pixels with strictly positive disparity. Optional finite confidence,
minimum-depth, and maximum-depth masks make visualization filtering explicit.

## Start here

Read [`CODEX_TASK.md`](CODEX_TASK.md), then follow [`RUNBOOK.md`](RUNBOOK.md).
M0 machine-readable evidence is under `reports/m0/`. Later milestone code must
not weaken the receipt status semantics or silently reuse incompatible caches.

## Repository layout

```text
configs/             experiment contracts
third_party/         pinned, read-only upstream submodules
src/backbones/       FFS and VGGT-Ω adapters (M1/M2)
src/geometry/        metric geometry and z-buffer reprojection (M1-M3)
src/data/            manifests and versioned cache readers (M1)
src/models/          trainable TSR components (M4-M6)
src/losses/          training losses (M4-M6)
src/metrics/         evaluation metrics (M4-M7)
tools/               environment and backbone smoke tools
tests/               executable contracts
reports/m0..m6/      structured milestone receipts
```

The GitHub FFS source and VGGT-Ω research materials have non-commercial-use
limits; the NGC FFS artifact has separate model terms. Commercial or deployed
robotics use requires a source-and-weight-specific license review.
