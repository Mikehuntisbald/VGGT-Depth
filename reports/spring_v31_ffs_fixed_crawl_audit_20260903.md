# Spring v3.1 + FFS fixed-crop crawl audit (seed 42)

This is a read-only audit of the current arm outputs.  It does not certify a
full validation result unless `records=1302`, `crop_mode=fixed`, and
`hr_crop=[384,768]` all hold.  The temporal rows below are 64-window canaries;
they are useful for smoke comparison only and are not final ranking evidence.

Common domain:

- validation manifest SHA256: `6c016bdb4aa7f4a2be07c713cf9ae90b8dccdeab31c76c71d9d0e96b1a8bc45e`
- endpoint file SHA256: `9fa81b1ca652fdd1f33634f83213c28e826db91562125430aedb0beef4f5c838`
- endpoint ID SHA256: `aa6ba30295b8d5ab0e1b4326a14fae61f9c8ec42641801cd8442097bc3ab5b57`
- fixed crop: HR `384x768`, explicit origin `(x,y)=(576,348)`

## Current rows

| arm | current output | status | crop | windows | primary row | EPE | bad-1 | high-detail EPE | low-detail EPE | matched EPE | unmatched@1 | boundary EPE | rigid residual | non-rigid residual | zero rate | comparability |
| --- | --- | --- | --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| F0 | `arms/F0/eval_common_fixed384` | SCREENING_ONLY | fixed 384x768 | 1302 | FFS full | 0.4941 | 0.0605 | 7.8390 | 0.4076 | 2.5118 | 0.9503 | 2.7046 | n/a | n/a | 0 | full common domain |
| F1 | `arms/F1/eval_common_fixed384` | SCREENING_ONLY | fixed 384x768 | 1302 | FFS half+bilinear | 0.5984 | 0.0705 | 7.9596 | 0.5120 | 2.9165 | 0.9422 | 3.6218 | n/a | n/a | 0 | full common domain |
| F2 | `arms/F2/eval_common_fixed384` | LIMITED_EVALUATION_COMPLETE | fixed 384x768 | 1302 | T1 | 0.5876 | 0.0742 | 7.5996 | 0.5048 | 2.7839 | 0.9386 | 3.3243 | n/a | n/a | 0.010776 | full common endpoint domain; Stage-A final checkpoint |
| F3 | `arms/F3/eval_step2000_64_fixed384` | LIMITED_EVALUATION_COMPLETE | fixed 384x768 | 64 | T3 (GT pose, v2/K=2) | 0.8133 | 0.0526 | 15.3362 | 0.5783 | 2.7027 | 0.9540 | 5.7469 | 0.5144 | 0.1237 | 0.000035 | 2000-step checkpoint; canary only |
| F4 | `arms/F4/eval_canary500_64_fixed384` | LIMITED_EVALUATION_COMPLETE | fixed 384x768 | 64 | T3 (GT pose, v3.1) | 0.9691 | 0.0837 | 16.6810 | 0.7149 | 3.9787 | 0.9450 | 6.4309 | 0.5052 | 0.1390 | 0.000469 | 500-step checkpoint; canary only |
| F5 | `arms/F5/eval_canary500_64_fixed384` | LIMITED_EVALUATION_COMPLETE | fixed 384x768 | 64 | T3_VGGT (depth + GT pose) | 0.8689 | 0.0572 | 16.7157 | 0.6123 | 3.1117 | 0.9484 | 6.2923 | 0.4925 | 0.1176 | 0 | 500-step checkpoint; canary only |
| F6 | `arms/F6/eval_canary500_64_fixed384` | LIMITED_EVALUATION_COMPLETE | fixed 384x768 | 64 | T3_VGGT (depth + VGGT pose) | 0.8549 | 0.0554 | 16.6390 | 0.5993 | 3.0644 | 0.9503 | 6.2855 | 0.4930 | 0.1084 | 0 | 500-step checkpoint; canary only |

The F3--F6 canaries use the same endpoint-selection hash
`452b4b05fe9164317b1ab1f60c714a5ad8ec7b6c28ce309a48e3f03fac89449d`, so their
64-window values are directly comparable to one another.  They are not
comparable to the F0/F1 1302-window averages as estimates of full-domain
performance.  F3 also has a different training budget (durable step 2000)
from F4--F6 (step 500).

## Lineage checks

- F0/F1 fixed reports both resolve to `(576,348)` and 1302 endpoints.
- F3/F4/F5/F6 fixed reports resolve to the same fixed crop and endpoint hash;
  all correctly report `formal_holdout=true`, `coverage_eligible=false`, and
  `final_acceptance_eligible=false` because only 64 windows were selected.
- F3 uses `temporal_pose_source=gt`, `legacy_v1`, top-K=2, and no VGGT depth.
- F4 uses GT pose, calibrated stereo v2, top-K=4, and no VGGT depth.
- F5 uses GT pose plus VGGT depth (`T3_VGGT` primary row).
- F6 uses VGGT pose plus VGGT depth (`T3_VGGT` primary row); its raw VGGT
  receipt is present and video-disjoint from training.
- F0/F1/F3--F6 share the FastFS observation checkpoint SHA
  `98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692` and
  upstream commit `a290ba04c1b3ad1ec41a33974a157b2917b624d4`.
- A read-only invariant pass confirms every current fixed report has
  `hr_crop=[384,768]`, the expected endpoint file SHA, and the expected
  endpoint-ID SHA; F0--F2 evaluate 1302 endpoints, while F3--F6 evaluate the
  same 64-window prefix (hash `452b4b…9449d`).

## Early signal (not a final claim)

On the shared 64-window fixed crop, F3's 2000-step v2 control is ahead of the
500-step v3.1 canary, so this is confounded by training duration.  Among equal
500-step v3.1 canaries, the depth prior (F5) improves T3 EPE from `0.8777` to
`0.8689` (about 1.0%) and paired temporal residual from `0.26570` to `0.26424`.
Adding VGGT pose (F6) improves T3_VGGT EPE further to `0.8549` and non-rigid
residual to `0.1084`, while rigid residual is essentially unchanged.  These
are 64-window canary signals only.

The v3.1 top-K diagnostics are populated for F4--F6 (all over roughly 4.6M
native HR pixels per 64-window run):

| arm | age-2 survival | unique-age fraction | phase variance | depth spread (m) | attention entropy | weighted-minus-rank0 EPE (HR px) |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| F4 | 1.0000 | 0.9998 | 2.60e-7 | 0.3033 | 0.5119 | -0.0785 |
| F5 | 1.0000 | 1.0000 | 2.64e-7 | 0.2647 | 0.5171 | -0.0876 |
| F6 | 1.0000 | 1.0000 | 0.1341 | 0.2661 | 0.8298 | -0.0886 |

The negative weighted-minus-rank0 values show that attention-weighted candidate
fusion beats rank-0 on this canary.  Camera-motion stratification is not
informative yet: all 64 windows fall in the low-motion bucket (the two tertile
boundaries coincide at about `3.03e-6`).

One watch item for the long F6 run is pose-induced phase spread: the F6 canary
phase variance is `0.1341`, versus about `2.6e-7` for F4/F5.  Its EPE and
non-rigid residual improve, but this larger phase variance should be checked at
the first 500/2000-step checkpoints before treating VGGT pose as unconditionally
stable.

## Resource/operational notes

- F4 full-image temporal canary completed, but formal full-image temporal
  evaluation previously hit a ~1.48 GiB top-K allocation with about 1 GiB
  free.  Keep formal F3--F6 evaluation on fixed 384x768 (or implement
  chunking before attempting full resolution).
- F4/F5/F6 500-step canaries completed without NaN/Inf/OOM.  Peak allocated
  memory was about 12.05 GiB; reserved memory 12.8--13.6 GiB on a 24 GiB
  RTX 4090.
- At audit time F6 long training is active on GPU1 from F2 initialization;
  it is a 15,000-step run and has no final checkpoint yet.  The F2 fixed
  evaluation has completed on GPU0; no binary cache was modified.

The helper command used to regenerate the table without modifying artifacts is:

```bash
python /tmp/extract_spring_v31_fixed_metrics.py \
  --run-root /home/CNF2026527811/Documents/VGGT-Depth/runs/spring_v31_ffs
```
