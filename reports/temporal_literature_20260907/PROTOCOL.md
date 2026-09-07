# Literature-guided temporal repair: evidence and controlled protocol

Date: 2026-09-07. This document was written before running the new CODD component arms.
All existing A0–A5, v1 and the 16 prior controlled experiments remain unchanged.

## Primary sources and official implementations

1. Li et al., **Temporally Consistent Online Depth Estimation in Dynamic Scenes**,
   WACV 2023 (CODD): https://arxiv.org/html/2111.09337v2
   and https://github.com/facebookresearch/CODD .
   Official code pinned at `dabb908f3643cd37d9fad0eba5429b8fa0359e24`.
   Inspected `model/losses/temporal.py`, `model/fusion/fusion.py`, and
   `configs/schedules/schedule_fusion.py`.
   The history coefficient is the product of reset and fusion weights. Reset
   supervision distinguishes error differences above 5 px; fusion uses 1 px and
   a 0.2 regularizer toward 0.5 in ties. Each accept/reject case has its own mean.
   Disparity uses smooth L1. These are actual official code defaults, not guessed
   loss functions. Local stereo costs use learned stereo features at disparity
   offsets -1, 0, +1 in FEATURE-grid units. Pixel-to-patch feature correlations
   and disparity differences provide context. The motion branch estimates
   per-pixel SE(3) with RAFT3D, covering object motion as well as camera motion.
   Official fusion training uses 12,500 updates, Adam, maximum LR 2e-4,
   weight decay 1e-5, gradient clipping 1, and a linear OneCycle schedule.
   Note: the older arXiv prose describes shared convolutions; the released
   implementation has separate `weight_head`/`forget_head` inputs and branches.

2. Zeng et al., **Temporally Consistent Stereo Matching**, ECCV 2024 (TC-Stereo):
   https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04579.pdf
   and https://github.com/jiaxiZeng/Temporally-Consistent-Stereo-Matching .
   Official code pinned at `ad714ad676265d9ed15a8fcd77a3cb35e0e3025f`.
   Inspected `core/corr.py`, `core/tc_stereo.py`, `train_stereo.py`.
   A normalized learned-feature cost volume supplies an ambiguity margin:
   compare the best cost against the best distinct alternative EXCLUDING the
   best index and its immediate neighbors. The official `argmax_disp` code
   uses a 0.3 inference margin. The training cost loss raises the GT similarity
   and suppresses hard negatives outside GT +/- 1.5 feature pixels. This differs
   materially from comparing five RGB residuals between current/history depths.
   Pose-warped temporal disparity completion and recurrent disparity/gradient
   refinement add substantial spatial context. A full TC-Stereo implementation
   is not equivalent to the tested 3x3 integer-history candidate selector.

3. Jing et al., **Stereo Any Video: Temporally Consistent Stereo Matching**,
   ICCV 2025: https://arxiv.org/html/2503.05549v2 .
   Section 3.3 / Eq. 12 explicitly upsamples with disparities t-1, t, t+1;
   temporal 3D-GRU/attention also process video clips. Direct use would change
   our strict-past protocol. Section 3.4 reports an accuracy/consistency tradeoff
   for OPW/TGM losses in their experiments and uses image-based L1 instead.
   This motivates testing representations, but does not prove the same cause
   in our Spring pipeline. Its published metrics are not directly comparable
   with our all-GT penalized and GT-motion-corrected temporal metrics.

## What the existing negative results do and do not establish

The prior 16 heads each completed 3,000 updates and 1,294-endpoint validation.
None met all five user criteria. Analytic RGB/ZNCC/census evidence, one scalar
history gate, +/-1 HR-pixel candidate shifts and categorical endpoints are
limited component hypotheses. Their failures do NOT reject CODD, TC-Stereo,
learned stereo matching, dense spatial fusion or nonrigid motion estimation.
Removing all measured foreground/background interpolation failures also failed
acceptance; correct endpoint selection and representation remain necessary.
Our A5 history warp uses known camera motion only. Consequently dynamic-object
misalignment is a concrete architectural difference from CODD, not yet a
proven causal explanation for every failure case.

## First literature-guided controlled test

Three arms share the same two-output MLP, frozen v1, fixed training samples,
seed 42, batch 32,768 pixels and 12,500 updates:

- `capacity_control`: existing v1 objective, isolating the new two-gate capacity.
- `codd`: official CODD smooth-L1 + separately normalized reset/fusion losses.
- `codd_regret`: same CODD loss plus the already used 0.1-weight regret tail,
  `relu(error-current_error-1)+2*relu(error-current_error-5)`.

All start with the v1 gate factored into two equal square-root gates; zero
residual logits preserve v1 predictions to FP32 tolerance. This initialization
is our compatibility adaptation, not a claim about CODD. Optimizer Adam,
LR 2e-4 linearly decreasing to 1e-8, weight decay 1e-5 and clip 1 are fixed.
The tiny OneCycle warmup is omitted and documented. There is no logit calibration,
validation hyperparameter selection, or replacement of the reference model.
The 12,500-update budget is inspired by the paper; pixel sampling, dataset and
architecture differ, so it is not a reproduction of its effective training budget.
Original valid Spring GT is retained, rather than dropping disparities outside
CODD's default [1,192] range. All thresholds use HR-pixel disparity units.

This isolates SUPERVISION on the same available evidence. It does not test
CODD's dense convolutional fusion, learned stereo features, RAFT3D motion or
recursive fused memory. Those missing components must not be dismissed from
this test's result. No copied third-party implementation is included in the
runtime module; the source points to the equations and pinned official code.

Acceptance is unchanged: preserve v1 all-GT penalized EPE and penalized temporal
residual, reduce good-current damage and >5 px large-degradation rate, and
increase recovery on the ORIGINAL fixed opportunity mask. Report newly added
and eliminated >5 px events separately, plus coverage/native regions. Evaluate
all 1,294 original endpoints and independently repair/reproject the previous
prediction with its own corrected depth. v1 remains the deployed/reference model
unless all criteria pass. The existing validation set has been viewed across
multiple experiments; resulting comparisons are development evidence, not an
untouched confirmatory test.

## Next representation test, ordered by direct relevance

Use the frozen A5 FFS feature extractor for left/right matching, including
feature-space +/-1 costs and a distinct-alternative cost margin. Audit HR to
FFS-half-resolution to feature-grid coordinates before training. Feed spatial
self/cross correlations into dense fusion and isolate feature evidence from
candidate alignment changes. Compare rigid camera warping with a separately
trained object-motion correction only after rigid/dynamic fixed-mask failures
are quantified. Preserve source UV winners and do not average foreground and
background disparities while building candidate locations. TC-Stereo's trained
cost volume and gradient refinement are alternatives if frozen-feature cues
lack discriminative evidence. End-to-end integration comes after component
acceptance, not before it.

## Learned matching evidence control (registered before its cache/train run)

The next three arms repeat the same capacity-control / CODD / CODD-regret losses
and 12,500-update budget, now with 24 extra inference features. The frozen A5 FFS
`feature` output level 0 is the exact 224-channel tensor used by its own groupwise
cost volume and recurrent geometry correlation (`foundation_stereo.py` lines
198–206, 234). Features are captured after the trained A5 checkpoint is loaded,
not from the original pretrained FFS weights. No backbone weights are updated.

At each HR pixel and each fixed A5/history candidate, sample left/right learned
features at offsets -1,0,+1 FEATURE pixels (scale 8 in HR): six cosine similarities,
six normalized L1 distances, six validity indicators. Six more cues describe
full-row cosine best peak, best distinct-alternative margin excluding neighboring
bins, distance of that peak to each original disparity, alternative availability
and history availability. Impossible negative disparities are excluded. The
coordinate mapping uses align_corners=False pixel centers; synthetic known-shift
tests verify the disparity unit conversion. New cues never change A5/history
candidates, source-UV winners, pose, masks, confidence or coverage.

This component uses established learned matching and ambiguity evidence, with
adapted normalization for FFS features. It does not train TC-Stereo's contrastive
cost objective or implement CODD's learned spatial key projection/dense fusion.
The new 55-input MLP has more parameters than the 31-input supervision arms;
its three loss controls share identical capacity. Therefore cross-group gains
are feature-plus-input-capacity evidence, with loss effects isolated within each
group. Original training samples are regenerated with exact base/history/features
assertions; validation candidates are loaded unchanged from the immutable cache.
Full FFS clips contain independently processed frames with frozen BN statistics.
Previous output features are taken from the past frame in the clip; no temporal
attention or cross-frame reduction occurs in the FFS feature extractor.

## One margin adaptation after the six completed component results

The original CODD fusion tie band is |E_current-E_history| <= 1 HR px,
whereas this project's fixed opportunity mask begins at E_current-E_history
> 0.1 HR px. A tie weight near 0.5 can fail the existing recovery >0.1 px test
on small-advantage opportunities even when history is better. This is a
mechanistic mismatch, not proof of the aggregate cause. After the first six
arms failed acceptance, two additional arms change ONLY fusion margin 1 to
0.1 HR px, retain reset margin 5 and all architecture/data/seed/optimizer/12,500
updates/no-calibration settings, under CODD and CODD+regret respectively.
The 0.1 value comes from the pre-existing acceptance definition; there is no
threshold sweep. This is an explicitly post-six-arm development adaptation,
not an untouched replication of the paper. Evaluate the original and adapted
learned heads together, and report recovery in fixed advantage bins
(0.1,0.2], (0.2,1], (1,5], >5 so the suspected mechanism is checkable.
