# Temporal experiments: staged implementation and evidence

The retained baseline is **A5-HRRepair-v1**. The latest surface-mode decoder is a recovery-prioritized candidate with substantial damage costs, not an unconditional replacement. All comparisons below use the existing Spring validation protocol; they are single-seed development evidence, not independent confirmation.

| Stage | Change | Evidence |
|---|---|---|
| 1 | Frozen A5 plus a 6,273-parameter causal HR history selector; 3,000 updates | [v1 and raw-loss control](temporal_repair_20260907/REPORT.md) |
| 2 | Controlled supervision, local matching, alignment, ambiguity and residual variants | [Controlled results](temporal_controlled_20260907/REPORT.md) |
| 3 | CODD-inspired supervision, learned evidence and metric-aligned controls | [Literature comparison](temporal_literature_20260907/REPORT.md), [tradeoff review](temporal_tradeoff_review_20260907/REPORT.md) |
| 4 | Temporal initialization and state before stereo refinement; three 1,000-step arms | [Native temporal controls](native_temporal_stereo_20260907/REPORT.md) |
| 5 | New per-pixel hypothesis geometry decoder; memory/no-memory, 2,000 steps each | [Hypothesis controls](hypothesis_geometry_20260907/REPORT.md) |
| 6 | Matched WTA versus surface probability pooling; each continues the same memory checkpoint for 2,000 steps | [Surface-mode assessment](surface_mode_geometry_20260907/ASSESSMENT.md) |

## Results and interpretation

| Metric | Original A5 | v1 | Surface mode_pool |
|---|---:|---:|---:|
| All-GT penalized EPE, cap 10 px, lower is better | 0.351058 | 0.330162 | 0.321614 |
| GT temporal residual, cap 10 px, lower is better | 0.315694 | 0.258275 | 0.238741 |
| Fixed opportunity recovery %, higher is better | Reference | 35.673887 | 56.118878 |
| Good-current damage %, lower is better | Reference | 1.548202 | 10.242793 |
| More than 5 px regression relative to A5 %, lower is better | Reference | 0.118310 | 0.282347 |

The full evaluation covers 1,294 endpoints. All-GT errors penalize invalid predictions by 10 px. Good-current damage is evaluated among pixels with original A5 error below 0.1 px and counts deterioration above 0.1 px. Large regression uses all GT-valid pixels and counts error increases above 5 px. Fixed opportunities are defined from original A5 and its available original HR history, using a history advantage above 0.1 px; recovery requires an improvement above 0.1 px. These denominators are distinct.

v1 improves overall and temporal errors on all eight validation sequences, with unchanged coverage relative to A5. It still admits some severely incorrect history. Surface pooling has a supported recovery gain over v1, but the paired sequence intervals for its overall and temporal differences versus v1 include zero. Its uncapped valid-region EPE is also worse (0.451717 versus 0.444876). The matched WTA comparison supports an incremental decoder benefit; it does not establish that another module can compensate the damage costs.

v1 obtains its history from unmodified A5 on an independent past prefix. It does not feed repaired predictions back as inference history, and the additional prefix is real computation. The structural decoders use their own causal history. Original A5 and accepted v1 weights remain unchanged.

## Archived evidence and reproduction

Each experiment directory contains an `artifact_manifest.json` with source/archive SHA-256 values, file sizes and final-checkpoint hashes. Small JSON, reports and selected figures are stored directly. JSON larger than 1 MB is losslessly compressed as `.json.gz`; use Python `gzip.open(path, "rt")` with `json.load`, or `gzip -dc`, to read it. Source JSON contents are unchanged. Report links are adapted to the archive, with original hashes retained in the manifests.

Model binaries, training caches, logs and duplicate source snapshots remain under the experiment host's `runs/metric_stereo_video/` directories and are not uploaded to Git. The committed protocols and tools document how to regenerate inputs and run the experiments. A fresh checkout also needs the original A5 checkpoint, frozen backbone dependencies and Spring data; this archive alone is not a self-contained training dataset.

Before staging, the eight related test modules passed all **23 tests** on the experiment host. Python parsing, cross-stage import ordering, compressed evidence round trips and local report links were also checked. This packaging step did not retrain models or alter the measured results.
