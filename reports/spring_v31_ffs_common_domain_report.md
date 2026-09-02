# Spring v3.1 + Fast-FoundationStereo common-domain report

**Report status: PARTIAL** — seed 42; this file is generated from read-only arm receipts.

The canonical comparison contract is fixed HR `384×768`, origin `(x,y)=(576,348)`, over the shared Spring endpoint list. A row marked `canary` or `limited` is not a full common-domain ranking result; its raw numerator/count pairs remain in the JSON.

## Protocol and shared lineage

- protocol: `spring_v31_ffs_common_domain_v1`; seed `42`
- validation manifest: `spring_seed42_primary/manifests/validation.jsonl` (SHA256 `6c016bdb4aa7f4a2be07c713cf9ae90b8dccdeab31c76c71d9d0e96b1a8bc45e`)
- train manifest: `spring_seed42_primary/manifests/train.jsonl` (SHA256 `c0bb8ce75e55f2d71815091a901bc93edc71558bb3fd5cd22fa6919ae27884b6`)
- endpoint file: `manifests/common_endpoints.json` (file SHA256 `9fa81b1ca652fdd1f33634f83213c28e826db91562125430aedb0beef4f5c838`)
- endpoints: `1302`; endpoint-ID SHA256 `aa6ba30295b8d5ab0e1b4326a14fae61f9c8ec42641801cd8442097bc3ab5b57`
- crop: `fixed` HR `[384, 768]` origin `[576, 348]`
- shared FFS observation checkpoint: `98b5a9acf39fbfa795025de8cea95ce123daa40f6b6234d719167751024cf692`; upstream commit `a290ba04c1b3ad1ec41a33974a157b2917b624d4`

## Arm matrix

| Arm | Primary row | Selected output | Status | Records | Fixed384 | Canary | Common eligible | Final ckpt | Overall EPE | 1px | HD EPE | LD EPE | Matched | Unmatched @1/@2 | Boundary | Rigid | Non-rigid |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---:|---:|---:|
| F0 | — | `arms/F0/eval_common_fixed384/metrics.json` | SCREENING_ONLY | 1302 | yes | no | yes | — | 0.491709 | 0.0587674 | 7.61768 | 0.403391 | 2.67 | 0.952414/0.983136 | 2.30236 | — | — |
| F1 | — | `arms/F1/eval_common_fixed384/metrics.json` | SCREENING_ONLY | 1302 | yes | no | yes | — | 0.599472 | 0.0697551 | 7.66815 | 0.511864 | 3.04282 | 0.943056/0.981382 | 2.94974 | — | — |
| F2 | T1 | `arms/F2/eval_common_fixed384/metrics.json` | FINAL_CHECKPOINT_EVALUATION_COMPLETE | 1302 | yes | no | yes | yes | 0.586272 | 0.0726328 | 7.39063 | 0.50194 | 2.90455 | 0.94038/0.977496 | 2.74449 | — | — |
| F3 | T3 | `arms/F3/eval_step2000_64_fixed384/metrics.json` | LIMITED_EVALUATION_COMPLETE | 64 | yes | yes | no | no | 0.813334 | 0.0526224 | 15.1373 | 0.578437 | 2.63923 | 0.954008/0.981331 | 5.63826 | 0.514353 | 0.123686 |
| F4 | T3 | `arms/F4/eval_canary500_64_fixed384/metrics.json` | LIMITED_EVALUATION_COMPLETE | 64 | yes | yes | no | no | 0.969112 | 0.0837311 | 16.4719 | 0.714884 | 3.98918 | 0.945047/0.976852 | 6.32786 | 0.505173 | 0.138979 |
| F5 | T3_VGGT | `arms/F5/eval_canary500_64_fixed384/metrics.json` | LIMITED_EVALUATION_COMPLETE | 64 | yes | yes | no | no | 0.868881 | 0.0571568 | 16.5042 | 0.61248 | 3.03796 | 0.948396/0.980166 | 6.18711 | 0.492496 | 0.117625 |
| F6 | T3_VGGT | `arms/F6/eval_canary500_64_fixed384/metrics.json` | LIMITED_EVALUATION_COMPLETE | 64 | yes | yes | no | no | 0.854883 | 0.0553929 | 16.4282 | 0.599497 | 2.98851 | 0.950297/0.979577 | 6.18092 | 0.493022 | 0.108446 |
| F7 | — | `—` | OPTIONAL_NOT_RUN | — | no | no | no | — | — | — | — | — | — | —/— | — | — | — |

## Output health and FFS trusted measurement

| Arm | FFS trusted error | Negative rate | Zero rate | Invalid rate | Numerator/count details |
|---|---:|---:|---:|---:|---|
| F0 | 0.160589 | 2.85253e-05 | 0 | 0 | `/home/CNF2026527811/Documents/VGGT-Depth/reports/spring_v31_ffs_common_domain_report.json` |
| F1 | 0.195828 | 2.72674e-05 | 0 | 0 | `/home/CNF2026527811/Documents/VGGT-Depth/reports/spring_v31_ffs_common_domain_report.json` |
| F2 | 0.195821 | 0 | 0.0107756 | 0 | `/home/CNF2026527811/Documents/VGGT-Depth/reports/spring_v31_ffs_common_domain_report.json` |
| F3 | 0.267516 | 0 | 3.52859e-05 | 0 | `/home/CNF2026527811/Documents/VGGT-Depth/reports/spring_v31_ffs_common_domain_report.json` |
| F4 | 0.267516 | 0 | 0.000469367 | 0 | `/home/CNF2026527811/Documents/VGGT-Depth/reports/spring_v31_ffs_common_domain_report.json` |
| F5 | 0.267516 | 0 | 0 | 0 | `/home/CNF2026527811/Documents/VGGT-Depth/reports/spring_v31_ffs_common_domain_report.json` |
| F6 | 0.267516 | 0 | 0 | 0 | `/home/CNF2026527811/Documents/VGGT-Depth/reports/spring_v31_ffs_common_domain_report.json` |
| F7 | — | — | — | — | `/home/CNF2026527811/Documents/VGGT-Depth/reports/spring_v31_ffs_common_domain_report.json` |

## Top-K temporal complementarity diagnostics

| Arm | Age-2 survival | Unique-age | Phase variance | Depth spread | Attention entropy | Fractional-phase gain | Camera-motion gain |
|---|---:|---:|---:|---:|---:|---|---|
| F3 | 0.583697 | 0.034841 | 3.85005e-08 | 0.131891 | 0.0358574 | {"phase_0.25_0.5":1.004381635830852,"phase_ge_0.5":0.002351010031361511,"phase_lt_0.25":0.008960931589172105} | {"motion_low_tertile":0.005736882413202693} |
| F4 | 0.999885 | 0.999785 | 2.52595e-07 | 0.308542 | 0.510973 | {"phase_0.25_0.5":2.832263368240092,"phase_ge_0.5":0.06176649905683007,"phase_lt_0.25":0.09605400329746772} | {"motion_low_tertile":0.085656192881288} |
| F5 | 1 | 1 | 2.58416e-07 | 0.275723 | 0.510989 | {"phase_0.25_0.5":1.4774228008463979,"phase_ge_0.5":0.08019945751948399,"phase_lt_0.25":0.09862560330657288} | {"motion_low_tertile":0.0925171799317468} |
| F6 | 0.999973 | 0.999973 | 0.135822 | 0.283557 | 0.826121 | {"phase_0.25_0.5":0.09062019258271903,"phase_ge_0.5":0.0838256142596947,"phase_lt_0.25":0.05849035306772614} | {"motion_low_tertile":0.0894329549773829} |

## Lineage and eligibility notes

- F0/F1 use the top-level frozen-observation `metrics` object; F2–F7 use the declared primary method row, preferring the exact `spring_native_metrics` side channel when present.
- GT pose and VGGT pose are reported independently per arm. VGGT depth and pose switches are read from the resolved checkpoint config (or the arm YAML when no checkpoint exists).
- `cross_arm_common_domain_eligible` requires the fixed crop, the complete 1302-endpoint domain, and coverage eligibility. Canary rows stay visible but are not silently promoted to full validation.
- F7 is optional; a missing F7 receipt is represented as `OPTIONAL_NOT_RUN` with null metrics rather than inferred from F6.
- Metric numerator/count pairs, candidate paths, checkpoint hashes, cache identities, and consistency checks are in the JSON report.

Machine-readable report: `/home/CNF2026527811/Documents/VGGT-Depth/reports/spring_v31_ffs_common_domain_report.json`
