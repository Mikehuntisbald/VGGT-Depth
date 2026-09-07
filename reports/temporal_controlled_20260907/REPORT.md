# Controlled temporal repair: completed results

All 16 arms completed 3,000 optimizer updates and evaluation on all 1,294 original validation endpoints. None passed all five user criteria; v1 is retained. All candidate masks and coverage are fixed to the original A5 cache.

| Arm | all-GT EPE | temporal | good damage % | >5 px degradation % | fixed recovery % |
|---|---:|---:|---:|---:|---:|
| v1 | 0.330162 | 0.258275 | 1.548202 | 0.118310 | 35.673887 |
| none_control | 0.330065 | 0.257967 | 1.661787 | 0.119054 | 37.194383 |
| none_mild | 0.331244 | 0.261205 | 1.368287 | 0.081526 | 34.625570 |
| none_balanced | 0.333396 | 0.267505 | 0.939731 | 0.051448 | 29.613924 |
| none_recover | 0.332652 | 0.262797 | 1.810033 | 0.061334 | 41.016568 |
| none_protect | 0.334159 | 0.267306 | 1.242220 | 0.041865 | 33.827260 |
| matching_mild | 0.330716 | 0.260898 | 1.387131 | 0.082309 | 35.877950 |
| alignment_mild | 0.331596 | 0.261929 | 1.463754 | 0.079538 | 34.694001 |
| ambiguity_mild | 0.330990 | 0.260329 | 1.429908 | 0.080433 | 35.958929 |
| categorical_control | 0.331861 | 0.260708 | 1.755884 | 0.076148 | 37.198312 |
| categorical_mild | 0.332146 | 0.262062 | 1.405103 | 0.051098 | 34.464589 |
| residual_none_control | 0.330478 | 0.258618 | 1.687604 | 0.113966 | 36.538335 |
| residual_none_guarded | 0.330798 | 0.259325 | 1.480636 | 0.110408 | 34.044816 |
| residual_matching_control | 0.329801 | 0.257623 | 1.727221 | 0.119617 | 38.012266 |
| residual_matching_guarded | 0.330548 | 0.259363 | 1.379255 | 0.111749 | 33.353403 |
| residual_ambiguity_control | 0.330365 | 0.257981 | 1.813755 | 0.113816 | 38.294912 |
| residual_ambiguity_guarded | 0.330693 | 0.259129 | 1.464821 | 0.111306 | 33.930507 |

Matching under mild supervision reduced good-current damage and large degradation and increased fixed-opportunity recovery, but regressed overall and temporal errors. Removing foreground/background interpolation by categorical endpoint selection reduced the measured mixing rate to zero but still failed accuracy and temporal acceptance. A residual matching control improved spatial/temporal errors and recovery while increasing good-current damage and large-degradation rate. These are tradeoffs, not accepted replacements.

See controlled_results.json for all metrics and paired sequence-bootstrap intervals, evaluation_*/metrics.json for newly added/removed severe degradations and fixed native regions, and failure_cases/ for paired crops. With only 8 validation sequences and repeated development use, small point differences must not be presented as independent confirmation. Fine calibration used TRAIN development data only and found no jointly feasible setting.

Raw-image inference for matching_mild exactly matched cached evaluation on 8 endpoints / 3,145,728 pixels, including unchanged validity. See raw_inference_parity.json. Six component unit tests passed. Each training phase archives its exact source. These experiments test sampled pixel heads and analytic image cues; they do not establish that published dense fusion, learned matching features, or nonrigid motion methods fail.

Subsequent literature-guided work is documented in docs/temporal_repair_literature_protocol.md and runs/metric_stereo_video/temporal_literature_20260907. It is separate from these 16 arms.
