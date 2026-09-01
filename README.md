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
Formal Stage-B completed all 15,000 steps and its final training audit passes.
On the complete held-out pseudo-GT domain, its temporal, low-confidence,
completeness, and trusted-region gates pass; raw output health remains a fail.
Formal Stage-C also completed 5,000 steps with a passing training audit, but
its complete held-out result is **STAGE_C_M5_GATE_FAIL**.

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
  The formal CUDA BF16 run completed 5,000 steps from the audited Stage-B final;
  its training audit passes and final checkpoint SHA-256 is
  `b4e916ac0b8150d374d85efa0389e29b0ad455b5064d0693901d272ac278a31a`.
- Formal Stage-A completed 5,000 steps at 3.682 step/s. On all 244 held-out
  frames at full 800×1280 resolution, low-confidence EPE improves 11.996%,
  invalid-region completeness improves 725.3%, and trusted-region EPE improves
  4.269%. Raw disparity still has 3.722% negatives; the explicitly reported
  `clamp_min(0)` deployment row removes negatives without inventing positive
  epsilon depth, leaving those pixels honestly zero/invalid.
- Formal Stage-B completed 15,000 steps with a strict 15,000-record audit. On
  all 238 held-out causal windows, T3 improves paired temporal error by 22.393%
  over T1; T3+VGGT improves low-confidence EPE by 12.840%, completeness by
  668.7%, and trusted-region EPE by 6.395% over bilinear. These are same-family
  FFS pseudo-GT engineering metrics, not paper accuracy. Raw T3+VGGT still has
  2.784% negative disparity; `clamp_min(0)` removes the sign violation but
  leaves the same 2.784% zero/invalid, so the all-gates result remains a fail.
- On all 238 formal Stage-C windows, raw epipolar refinement improves boundary
  EPE by 2.6815%, Bad-1 by 9.6281%, overall EPE by 5.5280%, low-confidence EPE
  by 6.2917%, and trusted-region EPE by 3.7288%. It nevertheless raises raw
  negative/invalid disparity to 4.41968% and reduces invalid-region completeness
  by 23.1346%, so M5 fails. Clamp-to-zero is diagnostic only and cannot own an
  acceptance gate. No refined TEPE is available or claimed; the target remains
  same-family FFS pseudo-GT, and point-to-plane is unavailable.
- Delivery diagnostics are complete: one 238-frame H.264 temporal flicker MP4,
  four failure criteria with four ranked bundles and one PLY each (16/16), and
  four Stage-C base/refined PLY pairs (8 files). They are visualization/posthoc
  artifacts only; exact hashes and counts are in
  `reports/m7/delivery_artifacts.json` and do not alter formal metrics.

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

## Opt-in temporal flicker video

For a causal T=3 `eval.py` run only, `--temporal-flicker-video` streams one
CPU uint8 MP4 panel per source sequence under `temporal_flicker_videos/`. Each
frame contains RGB, bilinear FFS, T3 without the VGGT prior, T3+VGGT, trusted
pseudo-GT absolute error, and uncertainty variance. Disparity/error/uncertainty
scales are fixed by the temporal evaluation config, so color is comparable over
time; this is a visualization artifact only and never contributes to metrics or
point-to-plane reporting. The default is disabled. The writer prefers optional
`.[video]` imageio support; if that is absent, it probes the explicit system
`/usr/bin/ffmpeg` and streams fixed-size raw RGB24 frames directly, so formal
runs do not require changing the Python environment. If neither encoder path
works, an explicitly enabled run writes `temporal_flicker_video.status=NOT_AVAILABLE`
with the reason in `metrics.json` and does not silently write partial MP4s.

## Frozen-evaluator posthoc Stage-C PLY

`tools/export_epipolar_pointclouds_posthoc.py` is separately labeled
`POSTHOC_DIAGNOSTIC`. It refuses anything except the clean audited frozen
Stage-C evaluator source at `4e6b7eb`, dynamically runs that evaluator without
any lineage/runtime bypass, and appends calibrated base/refined PLY only after
the frozen visualization callback succeeds. Its receipt binds the frozen
evaluator SHA, Stage-B/Stage-C checkpoint hashes, exact endpoint HR `K`,
baseline, and PLY counts. It neither computes nor claims formal accuracy; the
unchanged frozen evaluator remains the sole owner of its formal metrics, and
point-to-plane stays `NOT_AVAILABLE`.

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
reports/m0..m7/      structured milestone and delivery receipts
```

The GitHub FFS source and VGGT-Ω research materials have non-commercial-use
limits; the NGC FFS artifact has separate model terms. Commercial or deployed
robotics use requires a source-and-weight-specific license review.
