# Decision log

## D-001 — Project root

- Date: 2026-09-01
- Status: accepted
- Context: `/home/haoyi` was not an empty repository and no project path was supplied.
- Decision: create the repository at `/home/haoyi/ffs_omega_tsr`.
- Rationale: it matches the agreed project name and avoids modifying the active ComfyUI repository.
- Consequence: all runbook paths are rooted there.
- Revisit trigger: the user explicitly requests relocation.
- Evidence: M0 repository scaffold.

## D-002 — Three isolated environments and cache boundary

- Date: 2026-09-01
- Status: accepted
- Context: FFS and VGGT-Ω have independent upstream dependencies.
- Decision: use `env-ffs`, `env-vggt`, and `env-tsr`; exchange only versioned disk caches.
- Rationale: upstream dependency changes cannot silently alter training.
- Consequence: backbone optimizer state is impossible in the default training environment.
- Revisit trigger: a reproducible unified deployment environment is built after the PyTorch baseline.
- Evidence: `reports/m0/env_*.json`.

## D-003 — Blackwell PyTorch baseline

- Date: 2026-09-01
- Status: accepted
- Context: the FFS README shows an older cu124 recipe unsuitable for this RTX 5090 target.
- Decision: pin M0 to PyTorch 2.10.0+cu128 and torchvision 0.25.0+cu128.
- Rationale: all three environments execute finite FP16/BF16 kernels on SM120.
- Consequence: upstream installation must not resolve or downgrade Torch.
- Revisit trigger: an intentional cu129/cu130 migration with new receipts.
- Evidence: `reports/m0/env_*.json`.

## D-004 — Read-only pinned upstreams

- Date: 2026-09-01
- Status: accepted
- Context: adapters and hooks are required; upstream forks would weaken provenance.
- Decision: use HTTPS submodules pinned in `third_party/LOCK.json` and reject dirty trees in smoke tests.
- Rationale: exact source identity remains auditable.
- Consequence: every extension lives outside `third_party`.
- Revisit trigger: an upstream incompatibility that cannot be wrapped, requiring a separately reviewed patch set.
- Evidence: `.gitmodules`, `third_party/LOCK.json`, smoke receipts.

## D-005 — M0 status semantics; no fake fallbacks

- Date: 2026-09-01
- Status: accepted
- Context: missing gated weights or real data must not look like successful inference.
- Decision: use `PASS`, `PASS_WITH_FALLBACK`, `FAIL`, `BLOCKED`, and `NOT_RUN`; write receipts on every exit.
- Rationale: experiment claims remain tied to executed evidence.
- Consequence: M0 may pass only after both backbones have real receipts; this is now satisfied, with the FFS compatibility fallback kept visible.
- Revisit trigger: never; schemas may version but meanings stay stable.
- Evidence: `tools/_m0_status.py`, `reports/m0/`.

## D-006 — FFS M0 checkpoint is provisional (resolved by D-014)

- Date: 2026-09-01
- Status: resolved historical interface-smoke decision
- Context: the machine already had a `20-26-39` FFS artifact, but not the final `20-30-48` observation and `23-36-37` teacher pair. Upstream publishes no checksums.
- Decision: copy the candidate into ignored project checkpoint storage, record source/size/hash, and use it only for a real M0 interface smoke.
- Rationale: it validates current official code on the 5090 without pretending the final experiment identity exists.
- Consequence: M1 cache generation is blocked on the two intended official model tiers.
- Revisit trigger: official binaries become downloadable and are hashed.
- Evidence: `reports/m0/smoke_ffs_local_20-26-39.json`.

## D-007 — FFS backend and compilation

- Date: 2026-09-01
- Status: accepted
- Context: the upstream Triton volume path is optional; the official `pytorch1` helpers still carry `torch.compile` decorators.
- Decision: request `pytorch1` directly for MVP correctness and keep TSR `compile_model: false`.
- Rationale: this removes the custom Triton volume dependency without mislabeling the upstream compiled helpers as eager execution.
- Consequence: first-forward latency is not a steady-state speed measurement.
- Revisit trigger: a separate matched backend benchmark after correctness.
- Evidence: `configs/mvp_x2.yaml`, `reports/m0/smoke_ffs.json`.

## D-008 — VGGT-Ω interface and checkpoint warning

- Date: 2026-09-01
- Status: accepted
- Context: VGGT-Ω has no causal flag, its preprocessor omits transform metadata, and its 1B checkpoint is gated. The upstream README added a 2026-08-18 benchmark-contamination notice.
- Decision: causality is enforced by the supplied five-pair window; the M2 adapter will reconstruct crop/resize/pad transforms; the released model may be used for unrelated downstream work, but affected Table 1/2 results will not support paper claims.
- Rationale: this matches the real public API and its disclosed limitation.
- Consequence: affected paper benchmarks remain excluded even though the M0 downstream smoke now passes.
- Revisit trigger: upstream publishes a corrected checkpoint or updated notice.
- Evidence: pinned VGGT-Ω README and `reports/m0/smoke_vggt.json`.

## D-009 — Cache location capacity (resolved by D-015)

- Date: 2026-09-01
- Status: resolved
- Context: `/media/haoyi/T9` has only about 90 GiB free while `/home` has about 1.2 TiB free.
- Decision: do not default a new large cache to T9 without an explicit capacity budget; select `CACHE_ROOT` before M1.
- Rationale: offline FFS/VGGT caches can be much larger than checkpoints.
- Consequence: M1 cannot start until dataset size and cache budget are known.
- Revisit trigger: storage is cleared or a dedicated cache volume is supplied.
- Evidence: M0 disk audit.

## D-010 — NGC v1.2 checkpoint compatibility is explicit

- Date: 2026-09-01
- Status: accepted for M0 interface smoke only
- Context: NVIDIA NGC publishes a signed/content-addressed v1.2 FFS checkpoint, but its serialized `args` lacks `normalize`; current GitHub source reads `self.args.normalize` directly.
- Decision: preserve the strict FAIL receipt, then run a separate `PASS_WITH_FALLBACK` probe with `--missing-normalize true`, matching the upstream volume helper's legacy default. Never inject the field silently.
- Rationale: official artifact provenance and source/checkpoint compatibility are two different facts, and both must remain visible.
- Consequence: NGC v1.2 proves the official artifact can execute through the wrapped source but is not assigned to the final GitHub fast/teacher tier identities.
- Revisit trigger: NVIDIA publishes a source-compatible checkpoint, tier mapping, or compatibility guidance.
- Evidence: `reports/m0/smoke_ffs_ngc_strict.json`, `reports/m0/smoke_ffs_ngc_compat.json`.

## D-011 — Reuse the verified ModelScope VGGT-Ω cache

- Date: 2026-09-01
- Status: accepted
- Context: the required 1B-512 checkpoint already exists under the user's ModelScope cache, while the prior Hugging Face account check was not approved.
- Decision: symlink the ignored project checkpoint path to the cached artifact instead of copying it; record its byte size and SHA-256 in every smoke/cache provenance record.
- Rationale: this avoids a redundant 4.58 GB copy and uses the exact artifact the user identified.
- Consequence: local runs depend on the cache path; portability requires restoring a file with the recorded hash.
- Revisit trigger: the cache moves or an upstream checksum/new checkpoint is published.
- Evidence: `reports/m0/smoke_vggt.json`.

## D-012 — M0 VGGT window is a real causal rectified sequence

- Date: 2026-09-01
- Status: accepted for smoke evidence
- Context: an interface PASS requires ten real ordered images rather than repeated placeholders.
- Decision: use validation tokens `val_000033` through `val_000037`, ordered left/right at 21.7, 21.9, 22.1, 22.3, and 22.5 seconds from one source video.
- Rationale: the per-frame metadata records a common source, 12-frame/0.2-second increments, `rectified: true`, and a 0.116124 m baseline.
- Consequence: this window validates M0 I/O only; its pose scale/quality is not M2 evidence.
- Revisit trigger: M2 selects the formal training/evaluation manifest.
- Evidence: `reports/m0/vggt_smoke_window.json`, `reports/m0/smoke_vggt.json`.

## D-013 — Video-isolated formal split

- Date: 2026-09-01
- Status: accepted
- Context: the provided train/validation CSVs contain 0.1-second interleaved frames from the same three videos.
- Decision: train uses videos `172530` and `191627` (2,787 frames); validation uses video `183939` (244 frames).
- Rationale: an interleaved split would leak nearly adjacent views and inflate validation results.
- Consequence: the formal manifests live under `/home/haoyi/ffs_omega_cache/manifests`; legacy CSV split labels are provenance only.
- Revisit trigger: a larger dataset with an independently held-out capture is added.
- Evidence: manifest hashes and sequence ranges in `REPORT.md`.

## D-014 — Exact FFS role identities

- Date: 2026-09-01
- Status: accepted
- Context: the agreed fast and teacher roles require different official family checkpoints.
- Decision: bind observation to `20-30-48` SHA-256 `98b5a9...cf692`, four iterations, LR maxdisp 192; bind teacher to `23-36-37` SHA-256 `af0658...e990`, eight iterations, HR maxdisp 416.
- Rationale: the HR teacher needs roughly twice the disparity search range to preserve the LR observation's physical minimum range.
- Consequence: NGC v1.2 and local `20-26-39` remain M0 probes only and cannot satisfy cache identity.
- Revisit trigger: a declared checkpoint ablation with a new cache namespace.
- Evidence: `reports/m1/smoke_ffs_*.json`, cache run receipts.

## D-015 — Formal cache root and legacy depth exclusion

- Date: 2026-09-01
- Status: accepted
- Context: full FFS auxiliary caches require about 45 GiB and T9 has little headroom; dataset `depth_rect*.npy` lacks checkpoint identity.
- Decision: use `/home/haoyi/ffs_omega_cache`, float16 dense cache fields, and versioned safe-load records. Never reuse legacy depth arrays as GT or the new teacher cache.
- Rationale: `/home` has sufficient capacity and every pseudo-label must bind exact code/config/checkpoint/source identity.
- Consequence: moving caches requires preserving receipts and hashes; a filename-only match is insufficient.
- Revisit trigger: a compact-cache schema is benchmarked and versioned.
- Evidence: `reports/m1/cache_audit_{train,val}.json` and cache receipts.

## D-016 — VGGT confidence and quality gating

- Date: 2026-09-01
- Status: accepted
- Context: upstream `depth_conf` is `1 + exp(logit)`, not a probability, and geometry-only pose checks cannot substitute for photometric/depth agreement.
- Decision: cache the raw unbounded score with that semantic in its name; normalize only in a derived adapter. Overall temporal-pose validity requires baseline CV, stereo rotation, photometric reprojection, and FFS-aligned depth consistency. Missing required diagnostics means invalid.
- Rationale: clamping the raw score to `[0,1]` destroys information, while implicit passes can contaminate temporal warps.
- Consequence: a raw VGGT cache may be valid even when its temporal pose is later rejected.
- Revisit trigger: upstream publishes a calibrated confidence definition or a validated learned quality model.
- Evidence: `reports/m2/smoke_vggt_adapter.json`, derived-geometry tests/receipts.

## D-017 — Initial TSR capacity

- Date: 2026-09-01
- Status: accepted for the x2 MVP
- Context: the specified 96-channel, two-layer ConvGRU architecture naturally totals 1,619,882 trainable parameters, below the broad 8–12M planning estimate and the hard 12M ceiling.
- Decision: validate the smaller exact architecture first; do not add unused blocks merely to hit a parameter count.
- Rationale: capacity should be increased only if matched training evidence shows underfitting.
- Consequence: the paper model size is not fixed until Stage-A/Stage-B results are available.
- Revisit trigger: training/validation curves demonstrate capacity-limited performance.
- Evidence: `tests/test_model_shapes.py` and RTX 5090 forward/backward smoke.

## D-018 — Near-zero-safe VGGT depth consistency gate

- Date: 2026-09-01
- Status: accepted
- Context: in sampled far-range windows, 22–55% of trusted FFS pixels can have disparity below 0.1 HR px. Pointwise `abs(error)/disparity` then becomes arbitrarily large even when the median absolute disagreement is below one pixel.
- Decision: always record median relative error, but gate it only when an explicit CLI threshold is supplied. Default depth consistency gates use weighted MAE and median absolute error in HR pixels, each capped at 2 px.
- Rationale: disparity-domain absolute error remains defined at infinity, while a relative denominator approaching zero does not.
- Consequence: the 954 probe passes default full pose quality with 1.989 px weighted MAE and 0.445 px median absolute error; a strict 10% relative ablation still rejects it and remains reproducible.
- Revisit trigger: a physically justified far-depth floor or calibrated uncertainty-normalized residual is validated across held-out captures.
- Evidence: `tests/test_pose_quality_pipeline.py`, `reports/m2/derive_geometry_954.json`, 12-window validation diagnostic sample.

## D-019 — Preserve sequence-level pose-quality distribution

- Date: 2026-09-01
- Status: accepted
- Context: aggregate train pose validity is 54.48%, but the two source videos differ sharply: `172530` passes 13.86% while `191627` passes 83.67%. Validation passes 74.17%.
- Decision: keep invalid windows as explicit zero-gated training samples, and report pose/static-prior availability by sequence. Do not tune thresholds merely to equalize aggregate split rates.
- Rationale: quality gates protect geometry correctness; collapsing the sequence distribution into one rate would hide a capture-dependent failure mode and make temporal gains hard to interpret.
- Consequence: Stage-B reporting must retain sequence identity and should include source-availability coverage alongside accuracy.
- Revisit trigger: diagnosis identifies a correctable preprocessing/calibration cause for `172530`, or a separately justified threshold ablation is run.
- Evidence: `reports/m2/derive_geometry_train.json`, `reports/m2/derive_geometry_val.json`.

## D-020 — Resolve temporal visibility on the HR grid

- Date: 2026-09-01
- Status: accepted
- Context: the reconstruction model consumes geometry on the LR grid, but downsampling the previous prediction before forward splatting erases thin-surface collisions and changes fractional-offset units.
- Decision: reproject the detached previous HR disparity with calibrated `K_hr`, resolve z-buffer winners/collisions and temporal loss at HR, then deterministically sample winner fields onto the LR model grid. Disparity and fractional offsets retain HR-pixel units.
- Rationale: occlusion is a geometric visibility decision and must be made before spatial reduction; this also matches the declared data flow and loss.
- Consequence: Stage-B uses a small FP32 geometry island and slightly more memory, while the trainable model remains BF16-capable.
- Revisit trigger: an accelerated HR splat is proven numerically equivalent on collision/boundary tests.
- Evidence: `tests/test_temporal_train.py`, `tests/test_zbuffer_identity.py`, Stage-B real-cache smoke receipt.

## D-021 — Formal temporal evaluation requires complete raw-to-derived lineage

- Date: 2026-09-01
- Status: accepted
- Context: a self-consistent subset derived manifest can be useful for smoke tests but cannot support formal temporal acceptance. The validation receipt was regenerated after the audit was implemented and now embeds `raw_input_audit` metadata for all 240 raw-to-derived records.
- Decision: formal evaluation independently requires 244 validation records, all 240 derived endpoints, exactly 238 causal T=3 windows, canonical `start_window=0`/no limit, and exact raw VGGT receipt identity/config/manifest coverage. It audits all per-record raw links and rejects training-sequence/cache overlap. Limited or non-holdout runs are labeled ineligible.
- Rationale: matching derived policy alone does not prove that train and validation used the same frozen VGGT checkpoint/config or that missing windows were not selectively omitted.
- Consequence: smoke fixtures may remain partial only behind an explicit non-holdout smoke flag; formal results fail closed. The raw cache producer now requires `available_windows == selected_windows` for canonical completeness, and the canonical validation-derived receipt cryptographically binds the complete raw receipt.
- Revisit trigger: the cache schema directly embeds and cryptographically binds full raw lineage in every canonical derived receipt.
- Evidence: `tests/test_evaluation.py`, `tests/test_derive_geometry_manifest.py`, `reports/m4/stage_b_eval_smoke.json`.

## D-022 — Stored rectified pixels own the Stage-C row-coordinate contract

- Date: 2026-09-01
- Status: accepted
- Context: every formal manifest reports `K_right.cy - K_left.cy = +5.4` HR px, but the saved `left_rect.jpg`/`right_rect.jpg` pairs do not exhibit that shift. A deterministic 96-frame SIFT plus fundamental-matrix RANSAC audit measured global median `right_y-left_y = -0.0723` px and p95 absolute residual 2.0239 px; all three sequences pass the predeclared 1.25/3.0 px gates. Applying the metadata shift would move real correspondences in the wrong direction.
- Decision: Stage C uses the explicit versioned `audited_same_row_rectified_pixels_v1` runtime contract (`row_scale=1`, `row_offset=0`). `K_right` and `P_right` remain diagnostics and are never silently applied as an image shift. Training and evaluation must bind the exact pixel-audit receipt SHA and reject legacy checkpoints without it.
- Rationale: correspondence must be defined in the coordinate system of the tensors actually sampled. A calibration field that conflicts by 5.4723 px with 71,436 RANSAC inliers cannot override stored-pixel evidence.
- Consequence: the local refiner remains 1D horizontal as specified; residual vertical mismatch is reported rather than learned as disparity. The original Stage-C smoke remains execution evidence only and is not accepted by the strict evaluator because it predates the receipt binding.
- Revisit trigger: the image producer is corrected and regenerated pairs pass a new versioned pixel audit consistent with their calibrated right projection.
- Evidence: `reports/m6/epipolar_rectification_audit.json`, `tests/test_audit_epipolar_rectification.py`, sign tests in `tests/test_epipolar_refiner.py`.

## D-023 — Formal Stage-C training is native CUDA BF16 only

- Date: 2026-09-01
- Status: accepted
- Context: a CPU integration checkpoint and a formal RTX 5090 BF16 training checkpoint have the same model-state shape, so model tensors alone cannot establish the declared execution environment.
- Decision: every Stage-C dry-run/checkpoint records a typed `training_runtime` receipt. Formal eligibility requires an explicit CUDA device, native (not emulated) BF16 support on that exact device, and BF16 autocast. CPU is allowed only with `--allow-cpu-smoke` for a dry-run or exactly one optimizer step and is always acceptance-ineligible.
- Rationale: execution provenance must be machine-checkable; a CPU smoke must never be promoted into a formal 5090 training claim.
- Consequence: the strict evaluator fails closed on missing/malformed runtime receipts and refuses full holdout evaluation of CPU smoke artifacts. A CUDA dry-run validates the environment but not optimization or accuracy.
- Revisit trigger: the formal training precision/device contract changes or a separately declared FP32 reference run is added.
- Evidence: `reports/m6/stage_c_geometry_smoke.json`, `tests/test_epipolar_train.py`, `tests/test_epipolar_evaluation.py`.

## D-024 — Resume formal training only from an atomic checkpoint boundary

- Date: 2026-09-01
- Status: accepted
- Context: Stage-B was externally terminated after atomically saving step 7,000, while its append-only log contained uncheckpointed records through step 7,144 and a partial step-7,145 record.
- Decision: archive the interrupted log, truncate the formal log to the checkpointed step, preserve the checkpoint by hash, and resume with the original training commit in a detached worktree. Treat post-checkpoint log records as diagnostic evidence only until their trajectory is recomputed.
- Rationale: appending directly to an ahead-of-checkpoint or truncated log creates duplicate/corrupt records and falsely implies optimizer state that was never durably saved.
- Consequence: steps 7,001–7,144 were recomputed and matched all recorded LR, gradient and loss values exactly; only per-process elapsed time resets. The formal log remains one continuous optimizer-step sequence.
- Revisit trigger: Stage-B itself gains a journaled log/checkpoint transaction; the separate tested reconciliation tool now covers stopped-run recovery.
- Evidence: `reports/m4/stage_b_resume_step7000.json`, `tools/reconcile_training_log.py`, `tests/test_reconcile_training_log.py`.

## D-025 — Preserve the formal Stage-B trajectory despite the intermediate sign regression

- Date: 2026-09-01
- Status: accepted
- Context: step 7,500 improves TEPE and pseudo-GT EPE but raw VGGT-on negative disparity rises to 5.1186%. A 238-window decomposition attributes the bulk to the ±8 HR-pixel LR residual crossing near-zero disparity and convex propagation, not to VGGT gating or the HR residual.
- Decision: do not change loss, architecture, optimizer, or source gates inside the active 15,000-step run. Preserve `clamp_min(0)` plus an explicit finite-and-positive point-cloud mask as the safe deployment baseline, and monitor raw/clamped sign health at 10k, 12.5k, and 15k. If the final raw gate still fails, create a separate fine-tune/ablation lineage that first addresses all-invalid source sanitization and the LR residual lower bound/negative penalty.
- Rationale: modifying the active run would destroy the declared trajectory, while disabling VGGT is contradicted by branch-matched evidence. Epsilon filling or softplus would manufacture positive depth and inflate completeness.
- Consequence: the clamp row has zero negative outputs but retains 5.1186% honest zero/invalid pixels at step 7,500; it is a safety layer, not a hole-recovery claim. Any corrective fine-tune is reported separately from the formal Stage-B run.
- Revisit trigger: the final 15,000-step sign-stratified audit or an independently controlled positivity ablation.
- Evidence: `reports/m4/stage_b_negative_diagnostic_step7500.json`, `reports/m4/stage_b_eval_step7500.json`.

## D-026 — Stage-C formal sampling and runtime are strictly deterministic

- Date: 2026-09-01
- Status: accepted
- Context: the original local matcher used CUDA `grid_sample` backward, which PyTorch explicitly reports as lacking a deterministic implementation. Warning-only determinism would make a full-state resume operationally exact but would not guarantee the same numerical trajectory.
- Decision: the audited same-row formal path uses FP32 floor/ceil gather plus linear interpolation and rejects any runtime row scale/offset other than exactly 1/0. The generalized vertical-affine `grid_sample` path remains diagnostic-only. Stage-C producer/evaluator require deterministic algorithms with `warn_only=false`, cuBLAS workspace `:4096:8`, deterministic cuDNN and benchmark disabled; all settings are checkpointed and revalidated.
- Rationale: the saved pixel audit already establishes same-row geometry, so a deterministic 1D sampler is both the minimal geometry and the reproducible implementation. Unsupported nondeterministic kernels must fail rather than warn during formal training.
- Consequence: two independent full-config CUDA optimizer steps are bit-identical with zero warnings. Peak reserved memory rises to about 7.72 GiB but remains safe on the RTX 5090. Legacy/warn-only Stage-C checkpoints are evaluation-ineligible, and the runtime source bundle now covers 52 files including `eval_epipolar.py`.
- Revisit trigger: PyTorch provides and verifies a deterministic generalized sampler, or the stored-pixel geometry contract changes away from same-row rectification.
- Evidence: `reports/m6/stage_c_geometry_smoke.json`, `tests/test_epipolar_refiner.py`, `tests/test_epipolar_train.py`, `tests/test_epipolar_evaluation.py`.

## D-027 — Stage-C positivity is a gated post-D-025 controlled ablation

- Date: 2026-09-01
- Status: implementation ready, execution blocked on D-025 final pass
- Context: canonical Stage C improves EPE/Bad-1/boundary metrics but raises raw invalid disparity from 2.78447% to 4.41968% and loses 142,099 positive predictions in the completeness domain. Aggregate evidence excludes NaN/Inf, saturation and correspondence OOB as primary causes; nearly dense additive correction crosses near-zero base disparity outside the direct supervised domain.
- Decision: preserve the failed canonical result. Only after a full D-025 Stage-B checkpoint passes its completed training audit and the independent final controlled-comparison audit may a new Stage-C arm start from that exact base with a freshly initialized refiner. The Stage-C preflight hash-checks and recomputes that audit rather than trusting raw metric fields. The arm uses a pre-projection squared negative hinge and FP32 `delta_safe=max(delta,-d_base)` on candidate-valid pixels; all-invalid correction remains zero.
- Rationale: the canonical base already exceeds the health threshold, so constraining Stage C alone cannot pass. Exact-zero projection prevents negative depth without fabricating positive completion, and the hinge restores gradient through the hard bound.
- Consequence: default and legacy Stage C remain unchanged. Controlled checkpoints cannot claim `formal_training_complete` or replace canonical Stage C. Epsilon, softplus and reuse of the canonical Stage-C refiner are forbidden. Current D-025 intermediate artifacts fail the CPU-only preflight.
- Revisit trigger: D-025 reaches 15,000 updates and its exact final checkpoint passes a full 238-window held-out audit.
- Evidence: `reports/m6/stage_c_output_health_root_cause.json`, `STAGE_C_D025_POSITIVITY_ABLATION.md`, `tests/test_stage_c_d025_positivity.py`.

## D-028 — A 4x2 Stage-C schedule requires its own CUDA memory receipt

- Date: 2026-09-01
- Status: implementation ready, execution blocked on D-025 and CUDA probe
- Context: the RTX 5090 has capacity to try a larger Stage-C micro-batch, but changing 2x4 to 4x2 without measuring a complete backward/optimizer step would turn an execution guess into a formal configuration.
- Decision: keep canonical and standard D-025 Stage C at micro-batch 2 with accumulation 4. The separate high-VRAM config uses 4x2=8 only after two real CUDA/BF16 micro-steps, finite backward/update checks, peak allocated/reserved measurement, and at least 2 GiB `total - peak_reserved` headroom produce an exact lineage-bound PASS receipt. Formal training and evaluation revalidate that receipt; OOM or insufficient headroom returns to the unchanged 2x4 config in a fresh output directory.
- Rationale: effective batch and mathematics stay matched while the memory-risk decision becomes reproducible and fail-closed.
- Consequence: no high-VRAM result or acceptance claim exists until the probe is actually run after D-025 passes. The high-VRAM YAML and receipt are excluded from canonical and standard controlled runtime bundles.
- Revisit trigger: a PASS probe is available on the final D-025 base, or a different memory schedule is proposed as a separately named arm.
- Evidence: `STAGE_C_D025_HIGH_VRAM_PREFLIGHT.md`, `tests/test_stage_c_high_vram.py`.

## D-029 — Architecture v2 is a clean, physical temporal lineage

- Date: 2026-09-01
- Status: implementation ready, untrained
- Context: canonical v1 uses a nearest single z-buffer winner, copies the ConvGRU state without pose transport, reports absolute current/history TEPE, and has no learned physical-validity ownership. Canonical Stage C is not an exact no-op at initialization. Those contracts cannot be changed in place without invalidating existing checkpoints and historical results.
- Decision: create the independent config chain `mvp_x2_v2.yaml -> temporal_x2_v2.yaml -> epipolar_x2_v2.yaml`. Preserve every canonical v1 checkpoint, metric field and go/no-go decision. V2 retains bilinearly splatted top-K candidates from temporal ages 1 and 2 with explicit fractional phase, confidence, age, visibility/collision and z-aware weights. Warp the latest ConvGRU state by the same forward geometry on the calibrated LR grid: projection/index selection is detached, selected feature values retain gradients, and age-2 hidden is not injected twice because age-1 recurrent state already summarizes it. The primary v2 temporal error is `|(d_hat_t-W(d_hat_{t-1}))-(d_ref_t-W(d_ref_{t-1}))|` on the common strict model/reference mask and one shared teacher correspondence, using the trusted teacher until real sequence GT is cache-backed. Keep current/history absolute TEPE explicitly legacy-only. Predict validity and FFS-hole completion separately, parameterize disparity as a non-negative magnitude without epsilon filling, and emit exact zero for invalid pixels. Stage C uses a hard exact no-op forward gate with a straight-through backward path and applies the FP32 base-aware lower bound `delta >= -base`; invalid/zero bases remain zero.
- Rationale: temporal motion should be judged against reference motion rather than penalized directly, recurrent features must live in the current camera frame, and validity must be represented instead of inferred from signed disparity. Exact-zero invalid output and an identity Stage-C initialization prevent apparent completeness gains from fabricated positive depth or an untrained correction.
- Consequence: Stage A v2 must be trained from scratch so the v2-only heads/encoders exist in its state dict. Stage B v2 may initialize only from that exact Stage-A v2 checkpoint, and Stage C v2 only from the resulting Stage-B v2 checkpoint. Strict parameter/state/config checks intentionally reject canonical v1 or D-025 initialization. V2 reports and outputs require new directories and cannot replace canonical fields or results. No v2 metric, acceptance gate, or go/no-go claim exists before complete held-out training/evaluation.
- Revisit trigger: the complete v2 Stage-A/B/C lineage and held-out evaluator receipts exist, or real temporal GT replaces the teacher reference under a separately versioned contract.
- Evidence: `configs/mvp_x2_v2.yaml`, `configs/temporal_x2_v2.yaml`, `configs/epipolar_x2_v2.yaml`, v2 geometry/model/loss/evaluator tests.

## D-030 — Architecture v3 conditions geometry on audited calibration without learned metric scale

- Date: 2026-09-01
- Status: implementation ready; formal ablation training pending
- Context: v2 transports disparity and hidden state correctly only under one time-invariant `K`/baseline contract and does not expose calibrated rays or decomposed motion to the reconstruction head. Rewriting existing manifests or raw caches would invalidate completed v1/v2 lineage. VGGT's predicted stereo rig also cannot replace physical calibration or be allowed to improve its own quality score after a hard constraint.
- Decision: create a separate v3 config/checkpoint lineage and a manifest-bound `rectified_stereo_v1` sidecar. Derive `T_right_rectified_from_left_rectified=[I|-B,0,0]` from audited stored-pixel projection/calibration data, retaining raw optical-frame and `K_right.cy` discrepancies only as diagnostics. Preserve the unconstrained predicted-baseline median scale and every existing pose-quality gate; only for an already-valid window set `E_left_hybrid=E_left_vggt` and `E_right_hybrid=T_right_left E_left_vggt`. Write these tensors under the independent `m2_calibrated_stereo_v2` derived contract while reusing immutable raw FFS/VGGT caches. Store source-frame `K_hr_px` and `baseline_m` in temporal memory and use explicit source/target calibration for top-K disparity/RGB/phase, hidden-state and teacher-correspondence transport, with target disparity recomputed as `fx_target*B_target/Z_target`. Add a parameter-matched, zero-initialized 64-channel conditioner using LR dense unit rays, scale-free static rig rotation-6D/translation direction, and age-1/age-2 rotation-6D/translation direction/`log1p(||t||/B)`; invalid temporal ages are exact zero. Keep FFS HR-pixel disparity as metric owner, retain `Z=fx*B/d`, and add no scale token/head/loss, q-space output or Stage-C search change.
- Rationale: calibration belongs in deterministic projective geometry first and only then as bounded learned context. Separate sidecars avoid raw-cache churn; dual-K/B equations prevent history disparity from silently inheriting the current camera's calibration; decomposed poses expose motion without giving the network a free metric-scale channel.
- Consequence: v1/v2 state dicts and outputs remain byte-contract compatible when v3 is disabled. V3 Stage A instantiates the full matched module and Stage B initializes only from same-seed A3. A0/A1/A2/A3/B0/B1 use distinct outputs under one atomic sequential runner. Because all 3031 current records share one static rig, the static stereo-pose causal effect is `NOT_IDENTIFIABLE`; an exact promoted B1 recipe may keep it enabled only as the evaluated lineage background, never as a standalone effectiveness claim. Learned conditioning has no promotion claim until all declared accuracy, completion, output-health, latency, peak-memory and three-seed clustered-bootstrap gates pass.
- Revisit trigger: the formal runner publishes complete seed-42 evidence and, where screened in, seeds 43/44 plus `decision.json`, or a multi-rig dataset makes static-pose conditioning identifiable.
- Evidence: `src/data/stereo_calibration.py`, `src/geometry/calibration_context.py`, `src/models/calibration_conditioner.py`, `configs/ablations/v3_*.yaml`, `tools/audit_v3_temporal_pose.py`, `tools/run_v3_experiments.py`, `reports/v3/temporal_pose_variation_audit.json`, and v3 sidecar/transport/model/runner tests.
