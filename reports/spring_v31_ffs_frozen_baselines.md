# Spring v3.1 + FastFS: frozen F0/F1 common-domain baseline

These two frozen baselines use seed 42, the original primary validation
manifest, and the identical manifest-bound endpoint list (1302 endpoints;
endpoint ID SHA-256
`aa6ba30295b8d5ab0e1b4326a14fae61f9c8ec42641801cd8442097bc3ab5b57`).  Both
reports have complete Spring detail/match map coverage and are screening
reports, not paper-accuracy claims.

| metric (lower is better for EPE/bad-1; higher is better for completion) | F0 full-res FFS | F1 half-res FFS + bilinear |
|---|---:|---:|
| Overall EPE | 0.506059 | 0.563824 |
| Overall bad-1 rate | 0.080156 | 0.084695 |
| High-detail EPE | 5.546149 | 5.448390 |
| High-detail bad-1 rate | 0.540492 | 0.602541 |
| Low-detail EPE | 0.441406 | 0.500992 |
| Matched EPE | 2.087693 | 2.360835 |
| Unmatched completion @1px | 0.930440 | 0.928166 |
| Unmatched completion @2px | 0.973164 | 0.973244 |
| Boundary EPE | 2.041959 | 2.572923 |
| FFS trusted measurement error | 0.191493 | 0.229307 |
| Negative rate | 0.000673 | 0.001112 |
| Zero rate | 6.30e-9 | 5.93e-9 |
| Invalid rate | 0 | 0 |

## Interpretation

Full-resolution FFS improves aggregate and boundary behavior (Overall EPE
−10.3%, Overall bad-1 −5.4%, Matched EPE −11.6%, Boundary EPE −20.6%) and
lowers its trusted measurement error.  High-detail EPE is +1.8%, although its
high-detail bad-1 rate is lower; this mixed signal means the remaining error
is concentrated in a smaller set of large detail outliers.  The next gain is
not obtained by simply enlarging FFS; it should come from the v3.1 temporal
candidate path, calibrated depth prior, or a detail/edge-aware refinement
loss.  The F2/F3/F4/F5/F6 comparisons must retain this exact endpoint list
and report the same fields before selecting the long-training model.

## Input identities

- F0 cache: `runs/spring_v31_ffs/cache/validation/ffs_full416/observation`
  (`scale=1`, `max_disp=416`, 4 iterations, right-left check; receipt SHA
  `38cf960c7fcd42f5217586d8a52f2ad98bb5df2be50384ea660c1eb272cf75ef`).
- F1 cache: `runs/spring_v31_ffs/cache/validation/ffs_half/observation`
  (`scale=2`, `max_disp=192`, 4 iterations, right-left check; receipt SHA
  `8791849d8f2fd514dc1e481806e3a72124dc4409940c4471ad9f5df6f6bce638`).
- Both use Fast-FoundationStereo commit `a290ba04c1b3ad1ec41a33974a157b2917b624d4`
  and observation checkpoint SHA
  `98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692`.

Reports:

- `runs/spring_v31_ffs/arms/F0/eval/metrics.json`
- `runs/spring_v31_ffs/arms/F1/eval/metrics.json`
