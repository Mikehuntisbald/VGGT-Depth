# FFS-Ω-TSR execution report

## M0 verdict

**PASS_WITH_FALLBACK** as of 2026-09-01 (Asia/Shanghai).

The environment and source portions passed. FFS passed with one fully recorded
compatibility fallback (and a second local artifact passed strictly). VGGT-Ω
1B-512 passed a real causal five-pair stereo-window inference. The aggregate
status retains `WITH_FALLBACK` because the NGC FFS checkpoint lacks the
`normalize` argument required by current source.

Machine-readable evidence is stored under `reports/m0/`.

## Hardware and software

| Item | Observed |
|---|---|
| GPU | NVIDIA GeForce RTX 5090 |
| Compute capability | 12.0 (SM120) |
| VRAM | 33,667,612,672 bytes / 32,607 MiB reported |
| NVIDIA driver | 595.84 |
| System CUDA toolkit | 13.3 (`nvcc` 13.3.73) |
| PyTorch wheel runtime | CUDA 12.8 |
| PyTorch | 2.10.0+cu128 |
| torchvision | 0.25.0+cu128 |
| Python | 3.12.14 |

The system toolkit and wheel runtime intentionally differ. No custom project
CUDA extension was built in M0. Future extension builds must set and record an
explicit compatible `CUDA_HOME`.

## Environment receipts

| Environment | Prefix | Imports | FP16 kernel | BF16 kernel | Status | Receipt |
|---|---|---|---|---|---|---|
| env-ffs | `/home/haoyi/miniconda/envs/env-ffs` | all required | finite, sample 256 | finite, sample 256 | PASS | `reports/m0/env_ffs.json` |
| env-vggt | `/home/haoyi/miniconda/envs/env-vggt` | all required | finite, sample 256 | finite, sample 256 | PASS | `reports/m0/env_vggt.json` |
| env-tsr | `/home/haoyi/miniconda/envs/env-tsr` | all required | finite, sample 256 | finite, sample 256 | PASS | `reports/m0/env_tsr.json` |

All three Torch builds include `sm_120` in `torch.cuda.get_arch_list()`.

## Upstream source

| Component | Canonical remote | Pinned commit | Tree | License scope |
|---|---|---|---|---|
| Fast-FoundationStereo | `https://github.com/NVlabs/Fast-FoundationStereo.git` | `a290ba04c1b3ad1ec41a33974a157b2917b624d4` | clean | non-commercial research |
| VGGT-Ω | `https://github.com/facebookresearch/vggt-omega.git` | `282ec70363edeff59424bf43731658092fba3d37` | clean | non-commercial research + FAIR AUP |

Neither submodule was modified. `third_party/LOCK.json` is the executable pin
contract.

## Checkpoint inventory

| Intended role | Artifact | Size | SHA-256 / state | Status |
|---|---|---:|---|---|
| Official M0 FFS interface probe | `checkpoints/ffs/ngc-v1.2/model_best_bp2_serialize.pth` | 71,105,659 B | `7aee85948373da62b0503c2542507129a3e7cab9d97d10e6790d89512a7db214` | official NGC content address; compatibility fallback required |
| M0 FFS interface probe only | `checkpoints/ffs/20-26-39/model_best_bp2_serialize.pth` | 67,971,170 B | `d147587849b3482dd2921eef60dc2cd9e12690735612e91ee2782a92f8d1c59e` | available, local mirror provenance |
| FFS observation | `checkpoints/ffs/20-30-48/model_best_bp2_serialize.pth` | 62,078,956 B | `98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692` | strict PASS; M1 cache owner |
| FFS HR teacher | `checkpoints/ffs/23-36-37/model_best_bp2_serialize.pth` | 71,098,210 B | `af0658f289ec840b292645f8d5538978f06e8cabaa1fd31e84acc91af268e990` | strict PASS; M1 pseudo-GT owner |
| VGGT-Ω 1B-512 | `checkpoints/vggt/vggt_omega_1b_512.pt` → ModelScope cache | 4,576,706,117 B | `c02da418b18bb01d0392598d3f6147366bcde1bb70fd08a5e3bf7925b0667934` | available; strict load and inference PASS |

The `20-26-39` source artifact came from
`/home/haoyi/Downloads/xd/01_code/robot_vision_gitlab/test_gitlab/fast-foundationstereo`.
Its filename matches the official middle tier, but upstream publishes no binary
checksum, so byte identity is not independently confirmed. Its `cfg.yaml` is
182 bytes with SHA-256
`d45afe99b176454d5aff416edf16c8da6a99579f8f374b927f37907442a7d6bc`.

The NGC v1.2 artifact was downloaded from NVIDIA's signed catalog entry. Its
SHA-256 equals the NGC content-addressed blob identifier. The NGC page does not
map v1.2 to a GitHub `20-30-48`/`23-36-37` tier, so no such identity is claimed.
The NGC weight is governed by its NGC model terms, while use of the GitHub source
in this repository remains subject to the source license.

The VGGT-Ω artifact comes from the ModelScope cache model ID
`facebook/VGGT-Omega` at its recorded `master` revision. ModelScope did not
provide an independently published SHA-256, so the local byte hash above is the
identity contract for subsequent caches.

## Fast-FoundationStereo smoke

Command:

```bash
conda run -n env-ffs python tools/smoke_ffs.py \
  --checkpoint checkpoints/ffs/ngc-v1.2/model_best_bp2_serialize.pth \
  --iters 4 --max-disp 192 --volume-backend pytorch1 \
  --missing-normalize true \
  --json-out reports/m0/smoke_ffs.json
```

Result: **PASS_WITH_FALLBACK**.

| Check | Result |
|---|---|
| Source identity | official pin, clean |
| Checkpoint provenance | signed NVIDIA NGC v1.2; SHA matches NGC content address |
| Strict source/checkpoint probe | FAIL: serialized args lack `normalize` |
| Explicit compatibility | injected `normalize=true`, matching upstream helper default |
| Frozen/eval | yes; all parameters `requires_grad=false` |
| Input | official rectified demo pair, RGB float32 0..255 |
| Input shape | `[1,3,540,960]` |
| Padded shape | `[1,3,544,960]`, L/R/T/B = `[0,0,2,2]` |
| Output shape after unpad | `[1,1,540,960]` |
| Output unit | input-image pixels |
| Finite fraction | 1.0 |
| Negative fraction | 0.0 |
| Disparity min / mean / max | 30.4975 / 93.3805 / 137.4900 px |
| Actual backend | `pytorch1`, no fallback |
| First forward including compile | 9.5752 s |
| Peak allocated CUDA memory | 1,095,616,512 B (about 1.02 GiB) |

This is an interface/correctness smoke, not a latency benchmark and not M1
confidence-hook validation. The strict failure is retained at
`reports/m0/smoke_ffs_ngc_strict.json`; the compatibility receipt is
`reports/m0/smoke_ffs_ngc_compat.json`. A second real run of the local
`20-26-39` candidate passed without compatibility fallback and is recorded at
`reports/m0/smoke_ffs_local_20-26-39.json`.

## VGGT-Ω smoke

The checkpoint is reused from
`/home/haoyi/.cache/modelscope/hub/models/facebook/VGGT-Omega/`. The real input
window uses validation tokens `000033`–`000037` from one source video at
21.7–22.5 seconds, with 0.2-second spacing. Each metadata file declares
rectification and baseline 0.1161241667 m.

Command shape:

```bash
conda run -n env-vggt python tools/smoke_vggt.py \
  --checkpoint checkpoints/vggt/vggt_omega_1b_512.pt \
  --sequence-metadata reports/m0/vggt_smoke_window.json \
  --images L33 R33 L34 R34 L35 R35 L36 R36 L37 R37 \
  --input-mode balanced --image-resolution 512 \
  --json-out reports/m0/smoke_vggt.json
```

Result: **PASS**.

| Check | Result |
|---|---|
| Original inputs | ten distinct RGB images, each 1280×800 |
| Causal order | `L[t-4],R[t-4],...,L[t],R[t]` |
| Official balanced/512 input | `[10,3,400,640]`, range `[0,1]` |
| Depth | `[1,10,400,640,1]`, finite fraction 1.0 |
| Depth confidence | `[1,10,400,640]`, finite fraction 1.0 |
| Pose encoding | `[1,10,9]`, finite fraction 1.0 |
| camera-from-world extrinsics | `[1,10,3,4]`, finite fraction 1.0 |
| Predicted intrinsics | `[1,10,3,3]`, diagnostic only, finite fraction 1.0 |
| Camera/register tokens | `[1,10,17,2048]`, finite fraction 1.0 |
| Timed forward | 0.4573 s with existing runtime caches; not a benchmark |
| Peak allocated CUDA memory | 6,035,550,208 B (about 5.62 GiB) |

All ten input hashes, source frame indices, timestamps, calibration summary,
checkpoint hash, and output statistics are in `reports/m0/smoke_vggt.json`.

## M1 — manifests and complete FFS caches

M1 result: **PASS**.

The source dataset has 1280×800 rectified stereo frames. The original train
and validation CSVs interleave frames from the same videos and are unsuitable
as a generalization split. The accepted video-isolated manifests are:

| Split | Source videos | Frames | Manifest SHA-256 |
|---|---:|---:|---|
| train | `172530`, `191627` | 2,787 | `596702933688f695d9aac480d9f01e5764f40a4d7b28d72d73c550eb209b301c` |
| validation | `183939` | 244 | `014bd75de8ffbf74530c64eac76394a30bfc62d65b2da02397de2fb5c984760c` |

Every record uses `left_rect.jpg`/`right_rect.jpg`, real calibrated K, and
baseline 0.11612416665784427 m. Existing legacy `depth_rect*.npy` files are not
treated as GT or silently reused because they do not bind a checkpoint hash.

| Cache | Configuration | Records | Elapsed | Disk |
|---|---|---:|---:|---:|
| train observation | `20-30-48`, LR x2, 4 iters, maxdisp 192, L/R check | 2,787 | 144.75 s | 9.4 GiB |
| val observation | same | 244 | 15.67 s | 0.84 GiB |
| train teacher | `23-36-37`, HR, 8 iters, maxdisp 416, L/R check | 2,787 | 875.81 s | 32 GiB |
| val teacher | same | 244 | 81.25 s | 2.8 GiB |

All cache entries bind upstream commit, checkpoint SHA-256, Torch/CUDA,
configuration hash, full manifest record, and source-image hashes. Safe
`weights_only=True` loading rejects mismatched identities. Eleven train and
nine validation records sampled across sequence boundaries pass shape, source,
finite-value, trusted-mask, and disparity-unit audits. The maximum observed
`d_hr - 2*d_lr` cache discrepancy was 5.96e-8 px (independent float16
quantization); the executable tolerance is 0.02 px. Evidence:
`reports/m1/cache_audit_train.json`, `reports/m1/cache_audit_val.json`.

Real strict smoke receipts are `reports/m1/smoke_ffs_20-30-48.json` and
`reports/m1/smoke_ffs_23-36-37.json`. The teacher smoke used 1280×800,
maxdisp 416, and peaked at 4,198,574,592 CUDA bytes. Negative raw teacher
values are explicitly invalid and cannot enter its trusted pseudo-GT mask.

## M2/M3 implementation evidence

The frozen VGGT adapter reproduces the pinned upstream resize/crop/pad pixels
and records invertible original/model transforms. Its real raw cache smoke is
`/home/haoyi/ffs_omega_cache/m2_smoke/vggt/.../954.pt`; the run receipt binds
the ModelScope checkpoint hash and all ten ordered source-image hashes. Dense
depth/confidence uses only the current left view by default, while all ten
camera/register tokens and poses are retained.

On that real window, the geometry-only baseline/rotation probe measured:

```text
predicted stereo baselines: 0.00181087, 0.00196924, 0.00174944,
                            0.00194728, 0.00186930 VGGT units
baseline CV:                0.0440171
max / median rotation err:  0.3310 / 0.2256 degrees
metric alpha:               62.121638 m per VGGT unit
trusted FFS pixels:         130,188
scale-only alignment MAE:   1.04827 HR pixels
```

The complete derived-quality receipt for this same window additionally reports
photometric median RGB residual 0.019063 (84.0% projected-valid) and aligned
depth/disparity weighted MAE 1.98907 HR px / median absolute error 0.44493 HR
px. It therefore has `pose_valid=true` and `static_prior_valid=true` under the
default absolute-pixel gates. The pointwise median relative error (13.9758) is
retained as a diagnostic but is not a default gate: 55% of this window's
trusted FFS samples are below 0.1 HR px, where division by disparity is
ill-conditioned. An explicit CLI option enables that stricter relative gate
for ablation. Evidence: `reports/m2/derive_geometry_954.json`.

The complete video-isolated validation raw VGGT cache contains 240 causal
five-pair windows, took 88.31 seconds, and occupies about 401 MiB. A subset
reuse probe completes without model inference; subset receipts can no longer
overwrite a more complete canonical run receipt.

All 240 validation windows were joined to their exact FFS observation records
and derived with the default strict absolute-pixel gates. Pose and static-prior
quality passed for 178/240 windows (74.1667%) and was explicitly rejected for
62/240 (25.8333%). Failure-reason counts overlap: 17 windows exceeded baseline
CV 0.10; 45 exceeded depth-consistency weighted MAE 2 HR px; 12 of those also
exceeded median absolute error 2 HR px. The 17 baseline-invalid windows have
photometric/depth diagnostics marked missing rather than implicitly passing.
No available photometric residual exceeded its threshold. The default
pointwise-relative-error gate is disabled and recorded as such, while its
diagnostic percentiles remain in the receipt.

The batch stage then reloaded all 240 derived records with safe
`weights_only=True` loading. Manifest, metadata, and tensor validity agreed for
240/240 records; all floating tensors were finite; temporal extrinsics were
zero for all 62 rejected poses; aligned prior disparity/confidence/masks were
zero for all 62 rejected static priors. This audit is part of receipt
generation and is reproduced with:

```bash
/home/haoyi/miniconda/envs/env-tsr/bin/python \
  /home/haoyi/ffs_omega_tsr/tools/derive_geometry_manifest.py \
  --vggt-root /home/haoyi/ffs_omega_cache/m2_formal_val/vggt \
  --ffs-root /home/haoyi/ffs_omega_cache/m1_formal_val/observation \
  --output /home/haoyi/ffs_omega_cache/m2_formal_val/derived \
  --cache-dtype float32 --start-window 0 \
  --report /home/haoyi/ffs_omega_tsr/reports/m2/derive_geometry_val.json
```

The canonical result, failure histogram, all-window/rejected-window diagnostic
percentiles, safe-zero audit, and raw-input audit are in
`reports/m2/derive_geometry_val.json`. The refreshed canonical derived receipt
SHA-256 is
`1e05c4e081620b9f634e9b3020779c6e4fdb7e21bf1df55365c3d4cde3d50620`;
its unchanged 240-record manifest SHA-256 is
`f9e34fa730e43013aa7b1e19b57b132bc929f9b6ffe72c4d7796cd67fa1cd594`.
The embedded raw audit confirms 240/240 safe loads, finite tensors, exact FFS
and VGGT identities, valid causal targets, and complete canonical raw-manifest
coverage.

The formal train raw VGGT cache subsequently completed all 2,779 causal
windows. Its canonical receipt explicitly records
`available_windows=selected_windows=written_records=2779` and
`reused_records=0`; completion was not inferred from file count. Before any
join, all 2,779 records passed `weights_only=True` loading, exact canonical
identity comparison, causal target validation, and finite-tensor checks. Raw
manifest SHA-256 is
`6adc9f7381d7a8eba1282601558bc4b2e6dca3ceef28fc9b82594eb325b9b08d`;
canonical raw receipt SHA-256 is
`37ea38ef0a35ec816e180091d323d95cbe8b895b90a440c2e8aaf3ec1c64590d`.

Default-gate train derivation then wrote 2,779/2,779 records in 202.56 seconds
and occupies about 10 GiB. Temporal pose passed 1,514/2,779 (54.4800%) and was
rejected for 1,265/2,779 (45.5200%). The independently gated static prior
passed 1,515/2,779 (54.5160%) and was rejected for 1,264/2,779 (45.4840%).
This overall rate hides a large sequence distribution difference:

| Train sequence | Selected | Pose valid | Pose rate | Static prior valid | Static rate |
|---|---:|---:|---:|---:|---:|
| `172530` | 1,162 | 161 | 13.8554% | 161 | 13.8554% |
| `191627` | 1,617 | 1,353 | 83.6735% | 1,354 | 83.7353% |

Therefore the train/validation aggregate-rate difference is not interpreted
as a single stationary quality change; downstream sampling and reporting must
retain sequence identity.

Failure-reason counts overlap: baseline CV 26, stereo rotation 2, depth
weighted MAE 1,051, depth median absolute error 486, insufficient depth
samples 141, photometric residual 2, and photometric projected-valid fraction
3. All-window and rejected-window diagnostic percentiles are retained in the
machine-readable receipt; pointwise relative disparity error remains a
diagnostic with its default gate disabled.

The post-run audit safely reloaded 2,779/2,779 derived records, found every
floating tensor finite, verified manifest/metadata/tensor validity agreement,
confirmed zero temporal extrinsics for all 1,265 rejected poses, and confirmed
zero aligned disparity/confidence/masks for all 1,264 rejected static priors.
The exact reproducible command is stored in the receipt and is:

```bash
/home/haoyi/miniconda/envs/env-tsr/bin/python \
  /home/haoyi/ffs_omega_tsr/tools/derive_geometry_manifest.py \
  --vggt-root /home/haoyi/ffs_omega_cache/m2_formal_train/vggt \
  --ffs-root /home/haoyi/ffs_omega_cache/m1_formal_train/observation \
  --output /home/haoyi/ffs_omega_cache/m2_formal_train/derived \
  --cache-dtype float32 --start-window 0 \
  --report /home/haoyi/ffs_omega_tsr/reports/m2/derive_geometry_train.json
```

Evidence: `reports/m2/derive_geometry_train.json`.

The x2 TSR model contains 1,619,882 trainable parameters. A real RTX 5090
batch-2 384×768 BF16-autocast forward/backward produced finite 384×768 output
and peaked at 3,419,121,152 allocated CUDA bytes. This validates execution,
not accuracy.

The strict causal trainer joins three independently derived endpoint records,
uses one shared crop, performs previous-output-detached forward splatting and
z-buffering on the HR grid, and computes temporal consistency on HR winners.
A real validation-cache Stage-B optimizer step completed with finite loss and
gradient. Its machine-readable smoke summary measured 1.36292 optimizer
steps/s, 4,533,681,664 peak allocated CUDA bytes, and 4,838,129,664 peak
reserved bytes. This uses a smoke Stage-A initializer and is execution evidence
only (`reports/m4/stage_b_training_smoke.json`). The independent local HR
epipolar block adds 69,905 parameters and passes
an RTX 5090 BF16 forward/backward smoke; it is not yet enabled in a formal run.

### M4 — formal Stage-A spatial result

The committed Stage-A run completed all 5,000 optimizer steps on the RTX 5090
in 1,357.94 seconds (3.6821 steps/s). Peak CUDA allocation/reservation was
3,414,233,600 / 3,925,868,544 bytes. The final checkpoint SHA-256 is
`6bb7ff116236ea4bb86a010c03f14ba96fb69ac24fc0affdb01e7e561e90c254`
and it records Git commit `3e0df0b543f5000d0bf8490740b5f34ad979e3b6`.
The log has exactly 5,000 finite optimizer records. A separate read-only audit
also verifies continuous steps from one, the exact cosine/warmup learning-rate
schedule, finite model/optimizer/scheduler/RNG state, 1,619,882 parameter
entries, final scheduler epoch 5,000 and zero terminal learning rate. The 1,589
pre-clip gradient norms above 1.0 are retained as diagnostics; the trainer
clips them and the auditor does not misclassify them as post-clip violations.
Evidence: `reports/m4/stage_a_training.json`,
`reports/m4/stage_a_training_audit.json`, and the ignored reproducibility
artifacts under `outputs/ffs_omega_tsr_x2/stage_a/`.

All 244 video-isolated validation frames were evaluated both at the fixed
384×768 crop and at their full 800×1280 resolution. Against trusted HR FFS
pseudo-GT, the full-resolution raw T1 model versus raw bilinear LR FFS gives:

| Engineering metric | Bilinear | T1 | Relative change |
|---|---:|---:|---:|
| low-confidence EPE | 0.224465 px | 0.197538 px | **-11.9962%** |
| invalid-region completeness | 0.082552 | 0.681310 | **+725.31%** |
| trusted-region EPE | 0.100102 px | 0.095828 px | **-4.2694%** |
| overall EPE | 0.168531 px | 0.152000 px | -9.8085% |
| boundary EPE | 1.415507 px | 0.866139 px | -38.8106% |
| Bad-1 | 1.4782% | 0.9330% | -36.884% |

Thus the three Stage-A engineering gates pass on the complete held-out video:
low-confidence EPE improves by at least 10%, completeness improves by at least
15%, and the trusted region improves rather than degrading. The raw model has
no NaN/Inf outputs but 3.7218% negative disparity, so the raw nonnegative-output
gate fails. The declared physical postprocess `clamp_min(0)` reduces negative
rate to zero and slightly improves EPE to 0.150757 px, but honestly reports the
same 3.7218% as zero/invalid rather than injecting epsilon; it does not change
the completeness numerator. Both raw and postprocessed rows remain in the
machine report. This is a conditional engineering PASS, not paper accuracy.
Evidence: `reports/m4/stage_a_eval.json` and the hashed full metrics/CSV named
there.

The causal evaluator now enforces the complete validation accounting of 244
manifest frames, 240 derived endpoints, and 238 evaluable T=3 windows. It
audits the raw VGGT receipt/identity, video-disjoint holdout lineage, per-record
derived linkage, and the exact Stage-A initialization SHA. T1/T3 temporal error
uses the intersection of their HR visibility/static/collision/geometry-safe
masks. A one-window smoke produced finite paired metrics but is explicitly
`NON_HOLDOUT_SMOKE_PASS` because its one-step T3 checkpoint was trained on the
same validation cache; its -4.14% number is not an acceptance result. Evidence:
`reports/m4/stage_b_eval_smoke.json`.

A separate read-only audit then traversed every formal Stage-B training input:
2,787 observation records, 2,787 teacher records, 2,779 raw VGGT windows and
2,779 derived-geometry records. Every cache used `weights_only=True`, all
5,574 current source images matched their recorded SHA-256, and raw/derived
receipts, manifests and per-record links closed exactly. The two train
sequences contain 1,166 and 1,621 frames and are disjoint from the 244-frame
`183939` validation sequence. Requiring all three student times to have
derived geometry leaves exactly 2,775 causal T=3 training endpoints (1,160 +
1,615), with no future index/timestamp access or sequence crossing. The input
audit receipt SHA-256 is
`4326f0b002d136dddc847a3d71c8bd8c1c32cbeab2eb24b6cbe7d669547006d8`;
evidence: `reports/m4/stage_b_input_audit.json`.

The first formal intermediate checkpoint at step 2,500 was then evaluated on
all 238 held-out causal windows with the refreshed raw-to-derived receipt.
Paired HR temporal disparity error fell from 0.369828 px for independently run
T1 to 0.330770 px for history-only T3, a 10.5613% improvement and the first
crossing of the internal temporal gate. On the same endpoints, the retained T1
spatial baseline improves low-confidence EPE by 10.5905%; T3 improves invalid
region completeness by 194.53% and improves, rather than degrades, trusted
region EPE by 2.229%. Enabling the quality-gated VGGT prior improves T3 overall
EPE by another 1.5366%, but its low-confidence EPE improvement versus bilinear
is 8.1068% at this intermediate point.

This is not the final Stage-B result: the declared 15,000-step run remains
active. Raw T3 also still emits 4.7392% negative disparity. The declared
`clamp_min(0)` physical variant has zero NaN/negative outputs but retains those
pixels honestly as 4.7392% zero/invalid, with no epsilon fill. Evidence and
artifact hashes: `reports/m4/stage_b_eval_step2500.json`.

At step 5,000, paired temporal error improved further from 0.364126 px (T1)
to 0.319402 px (T3), or 12.2825%. The quality-gated VGGT prior now improves
T3 overall EPE by 8.3640%, versus 1.5366% at step 2,500; the complete VGGT-on
branch improves low-confidence EPE by 9.3834%, invalid-region completeness by
957.57%, and trusted-region EPE by 3.3663% relative to bilinear. The no-prior
history-only row trades spatial accuracy for temporal stability at this point:
its low-confidence EPE is 1.7329% worse than bilinear even though its paired
TEPE passes. Raw T3 negative rate fell from 4.7392% to 1.8198%, but still does
not meet the raw physical gate. The formal run therefore continues. The
stratified visualization set now contains two valid-geometry/history examples
and two explicit pose-rejected fail-closed examples. Evidence:
`reports/m4/stage_b_eval_step5000.json`.

At step 7,500, the paired temporal improvement reaches 14.2522%
(0.367722 px T1 to 0.315313 px T3). The complete VGGT-on branch now crosses
the low-confidence spatial gate as well: versus bilinear it improves
low-confidence EPE by 10.6805%, invalid-region completeness by 295.42%, and
trusted-region EPE by 4.8329%; the prior improves T3 overall EPE by 9.0305%.
Thus every declared accuracy/completeness gate passes on all 238 held-out
windows at this intermediate checkpoint. Raw output validity remains a
separate non-pass: T3/VGGT-on negative rates are 5.6320%/5.1186%, showing a
non-monotonic regression from step 5,000. The declared clamp-to-zero physical
variants have zero NaN/negative outputs but retain those pixels as explicit
zero/invalid values. Training continues because this is neither the final
15,000-step result nor independent-GT evidence. Evidence:
`reports/m4/stage_b_eval_step7500.json`.

A full 238-window hook diagnostic isolates the negative-disparity regression.
Of the 3,592,681 reproduced negative pixels, 94.53% occur where bilinear FFS
is below 0.25 HR px, and 92.05% have magnitude below 0.1 px; 24,603 pixels are
still below -1 px and cannot be dismissed as sign-rounding noise. The LR source
mix has 309,734 negative elements, almost exactly the all-sources-invalid
count, but the bounded LR residual raises that to 1,770,626 and introduces
1,465,842 new negatives. Convex upsampling propagates those LR neighborhoods;
the HR residual and FFS anchor repair more negatives than they introduce.
Pose-rejected/history-invalid regions are much worse, but all three T3
branches regress together and VGGT-on actually has fewer negatives than
history-only, so disabling VGGT is not supported by the evidence. The current
formal trajectory remains unchanged through 15,000; 10k/12.5k/final snapshots
must track stage-wise and disparity-stratified sign health. `clamp_min(0)` plus
an explicit `d>0` point-cloud mask remains the fail-closed deployment baseline,
but its 5.1186% zero/invalid rate is not a completeness fix. The explanatory
hook run differs from the formal evaluator by three near-zero BF16 boundary
pixels and explicitly has no persisted script/source hash; claim boundaries
remain in `reports/m4/stage_b_negative_diagnostic_step7500.json`.

At step 10,000, all 238 held-out windows remain coverage-valid but the
checkpoint is explicitly `final_acceptance_eligible=false`. Paired TEPE now
falls 23.4307% from T1 (0.363206 px) to T3 (0.278104 px). VGGT-on improves
low-confidence EPE by 12.0683%, invalid-region completeness by 485.01%, and
trusted-region EPE by 5.4580% versus bilinear; the prior improves T3 overall
EPE by 16.4269%. In contrast, history-only T3 low-confidence EPE is 11.0857%
worse than bilinear, so the quality-gated static prior has become important to
the spatial result. Raw VGGT-on negative/invalid rate falls from 5.1186% at
7,500 to 2.7824%, but still misses the 0.5% output-health target. The new
reproducible sign-health evaluator confirms zero non-finite values at every
tap: source mix 1.7651% negative, post-LR residual 9.7137%, post-convex
9.8765%, post-HR residual 7.2138%, and post-anchor final 2.7824%. Thus the
original LR-residual diagnosis still holds while the HR residual/anchor have
recovered 1,639,724 final negative pixels since 7,500. Evidence:
`reports/m4/stage_b_eval_step10000.json`.

At step 12,500, paired TEPE improvement eases slightly to 21.8563% but remains
well above the gate. VGGT-on spatial metrics continue improving: versus
bilinear, low-confidence EPE falls 13.3269%, invalid-region completeness rises
642.95%, and trusted-region EPE falls 6.4198%; the prior improves T3 overall
EPE by 16.8660%. History-only low-confidence EPE remains 9.7600% worse than
bilinear. Raw VGGT-on negative/invalid rate ticks up from 2.7824% to 3.0041%
and still fails output health. The five negative rates are 1.7651% source mix,
9.9279% post-LR residual, 10.1169% post-convex, 7.6393% post-HR residual, and
3.0041% post-anchor, with zero non-finite values. This non-monotonic sign
trajectory reinforces the decision to judge the unchanged run only at the
declared final checkpoint. Evidence: `reports/m4/stage_b_eval_step12500.json`.

The training process was later found externally terminated after its last
atomic checkpoint at step 7,000. The interrupted log contained complete but
uncheckpointed records through step 7,144 and a partial JSON record for step
7,145. The original log was archived, the formal log was rolled back to its
checkpoint boundary, and training resumed from the preserved step-7,000
artifact using a detached worktree at the original training commit
`3e0df0b543f5000d0bf8490740b5f34ad979e3b6`. Optimizer state, scheduler,
all RNG states, and the deterministic data cursor were restored. Recomputed
steps 7,001–7,144 match the interrupted trajectory exactly for learning rate,
gradient norm, and every loss component (144/144 records, maximum absolute
difference zero). Only wall-clock `elapsed_seconds` restarts at the resume
boundary. The resumable artifact SHA-256 is
`da502de3f0d2b39982e10005a217f49f78bfb8b51ea96e6117069ce6dc2dbd29`;
evidence: `reports/m4/stage_b_resume_step7000.json`. A read-only audit after
the next atomic save found step 7,500 in the checkpoint and 7,617 strict,
continuous, finite log records. It independently passes checkpoint identity,
state finiteness, exact learning-rate schedule, and identifies the sole resume
boundary at step 7,001; it remains correctly labeled `IN_PROGRESS` because no
completion receipt exists (`reports/m4/stage_b_training_audit_step7500.json`).

Stage-C now has a separate training entrypoint that runs the full frozen
VGGT-on Stage-B endpoint through the exact three-step causal unroll, loads the
rectified right endpoint with the identical HR crop, and optimizes only the
69,905-parameter local epipolar refiner. A non-holdout CPU integration smoke
used 1,649 trusted teacher pixels, produced finite loss 0.0422251, saved every
required model/optimizer/scheduler/scaler/config/Git field, and reproduced all
18 refiner tensors bit-for-bit across two same-seed runs. This is execution
evidence only because its one-step Stage-B base was trained on the validation
cache; formal Stage-C training waits for the completed Stage-B checkpoint.
Evidence: `reports/m6/stage_c_integration_smoke.json`.

A subsequent calibration audit caught an important coordinate discrepancy:
all manifests advertise `K_right.cy-K_left.cy=+5.4` HR px, while actual
rectified JPEG correspondences are near the same row. Across 96 deterministic
train/validation frames, 98,095 ratio-test matches produced 71,436 RANSAC
inliers; global median `right_y-left_y` is -0.0723 px and p95 absolute residual
is 2.0239 px. All three sequences pass predeclared 1.25 px median / 3.0 px p95
gates, whereas metadata disagrees with observed pixels by 5.4723 px. Stage C
therefore binds `audited_same_row_rectified_pixels_v1`, uses explicit row
scale 1 / offset 0, retains `K_right` only as a diagnostic, and rejects the
legacy smoke checkpoint because it predates this receipt binding. Evidence:
`reports/m6/epipolar_rectification_audit.json`.

After binding that receipt, the current Stage-C producer adds full long-run
transactions: periodic atomic `latest.pt`, durable per-step JSONL, exact
optimizer/scheduler/RNG/data-cursor resume, tail reconciliation, finite-state
checks, and `final.pt/run_summary.json` publication only at the configured
5,000-step boundary. An unbounded formal run refuses any base other than the
completed canonical Stage-B 15,000/15,000 checkpoint. A real train-cache smoke
used the step-7,500 base, ran a fresh CPU optimizer step, resumed its complete
state for step two, and produced log steps `[1,2]`, scheduler epoch two and the
exact next data cursor; it correctly published neither final checkpoint nor
run summary and records `formal_training_complete=false`. The strict Stage-C
evaluator loaded that checkpoint and labeled its one-window held-out run
`LIMITED_SMOKE_ONLY`. An independent RTX 5090 CUDA dry-run binds the same
source bundle, capability 12.0, CUDA 12.8 and native BF16 autocast; it makes no
optimizer update. A first full-config optimizer probe exposed PyTorch's
nondeterministic CUDA `grid_sample` backward and was rejected as the formal
path. The same-row sampler now uses mathematically equivalent FP32 floor/ceil
gather interpolation, while the wrapper rejects nonzero row mappings. Two
independent full 384×768, micro-2/accumulation-4 CUDA runs then produced the
same loss 0.113453, gradient norm 0.980817, and bit-identical model/optimizer/
scheduler tensors with zero deterministic warnings; peak allocation/reservation
is 6.41/7.72 GiB. The current 52-file producer/evaluator receipt additionally
requires `warn_only=false`, cuBLAS workspace `:4096:8`, deterministic cuDNN and
benchmark disabled. A real strict CUDA optimizer checkpoint and strict limited
evaluation pass those runtime checks but remain acceptance-ineligible because
the base and Stage C are incomplete. Supervised-domain non-finite refined
disparity/correction/confidence/correlation now fail immediately instead of
being silently removed by a finite mask. The evaluator also reports horizontal
correspondence OOB without changing any loss, mask, or accuracy metric; a
strict two-window smoke measures 0.268% OOB where at least one discrete
candidate is valid and 50.29% inside the narrow candidate-boundary band, with
zero non-finite coordinates. Evidence:
`reports/m6/stage_c_geometry_smoke.json`.

## Tests

```text
conda run -n env-tsr pytest -q
........................................................................ [ 26%]
........................................................................ [ 53%]
........................................................................ [ 80%]
......................................................                   [100%]
272 passed
```

The current suite includes receipt/cache identity, manifest/crop/intrinsics,
disparity scale, FFS hooks, VGGT preprocessing, baseline scale, robust depth
alignment, camera convention, z-buffer identity/translation/collision/OOV,
model shapes/anchor/recurrence, strict temporal holdout/cache lineage,
paired-domain TEPE, physical clamp reporting, visualization, and empty-safe
losses. It also covers completed/in-progress training receipt distinction,
strict JSON/log continuity, checkpoint finite-state validation, and exact
learning-rate schedule auditing. Stage-C coverage now includes runtime/device
eligibility, right-image SHA lineage, same-domain refined/base metrics,
periodic checkpoint publication, exact optimizer/RNG/data-cursor resume,
crash-tail reconciliation, and the rule that bounded smoke runs cannot publish
a formal completion receipt. It also tests deterministic same-row sampling,
strict CUDA runtime receipts, and supervised-domain non-finite fail-fast.

## Current open work

| Blocker | Required resolution |
|---|---|
| Stage-B training | the formal causal T=3 run is active on the declared 15,000-step schedule; completion receipt and final audit remain pending |
| Final temporal evidence | intermediate full-holdout checkpoints pass the TEPE threshold, but the final T3-vs-T1 acceptance claim waits for completed Stage-B training/evaluation |
| Formal Stage C | the producer/evaluator and geometry/runtime contracts are implemented; formal training waits for the completed 15,000-step Stage-B checkpoint |
| Independent accuracy | current Stage-A metrics use same-family FFS pseudo-GT; add real GT or an independent benchmark before paper claims |

## Claim boundary

M0 proves environment viability, exact source identity, real FFS inference,
and real ten-image VGGT-Ω interface inference. M1 additionally proves the
declared disparity-unit/cache contracts on the formal split. M2 proves metric
baseline scaling, quality-gated priors/poses, and rejected-record zero safety;
M3 proves numerical HR z-buffer conventions. The formal Stage-A artifacts prove
its internal pseudo-GT engineering thresholds and completed-run throughput;
training smokes prove only execution and provenance plumbing. No current
artifact proves the *final* Stage-B go/no-go result, independent-GT accuracy,
point-cloud accuracy, or paper-level performance. The complete step-7,500
holdout diagnostic crosses the internal temporal/accuracy/completeness gates
but does not replace the declared 15,000-step completion receipt.
