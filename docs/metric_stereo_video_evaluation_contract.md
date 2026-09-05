# Metric Stereo Video Evaluation Contract

This contract is frozen before independently training the formal ablation
arms. Its machine-readable definition is
`configs/metric_stereo_video/evaluation_contract.yaml`; the arm definitions
are in `configs/metric_stereo_video/formal_ablation_matrix.yaml`.

## Model selection

The learned validity head must not control the support of the primary metric.
Common-valid EPE remains a diagnostic, but it must never be used alone for
checkpoint or main-model selection.

Primary reports include all-GT penalized EPE, EPE at 99% coverage, prediction
coverage, the accuracy-coverage curve, and completeness at 1 px and 2 px. For
all-GT penalized EPE, every GT-valid pixel is counted: a valid prediction
contributes `min(abs_error, 10 px)`, while a rejected or non-finite prediction
contributes 10 px. This makes rejection unable to improve the score.

## Temporal accuracy

Prediction self-consistency metrics are retained, but they do not establish
temporal reconstruction accuracy. The formal temporal residual is computed
from a separate strictly causal prefix ending at `t-1`:

```text
abs((pred_t - warp(pred_t-1)) - (GT_t - warp(GT_t-1)))
```

It is reported on GT-matched pixels overall and by rigid/non-rigid region and
small/medium/large camera motion. Pixels newly visible at `t` do not have a
defined temporal delta target; they are reported separately as disocclusion
current-frame EPE and completion at 1 px and 2 px.

## Spring partitions

Formal reports use Spring's native detail and match maps, plus a GT disparity
boundary band. Required metrics are high-detail EPE/Bad-1, low-detail EPE,
boundary EPE, matched EPE, unmatched completion at 1 px and 2 px, and rigid
and non-rigid temporal residual. Missing required detail or match maps are a
hard evaluation error.

## Pose contract

The current system is a known-pose stereo-video model. It consumes calibrated
intrinsics, stereo extrinsics, and `T_current_from_previous`; VGGT does not
estimate the active camera motion. Pose-source studies must be reported as
separate arms: GT pose, VIO/odometry pose, VGGT pose, corrupted pose, and no
pose/no history. Unavailable pose sources must remain marked blocked and must
not be silently replaced with GT pose.

## Disparity range

The FFS low-resolution maximum disparity of 192 px corresponds to 384 px in
the high-resolution Spring evaluation grid. Before training, the range audit
must report `P(GT disparity > 384 px)` overall, in dynamic regions, and in
high-detail regions for both train and validation splits.

## Ablation attribution

Runtime component toggles on one full-model checkpoint are diagnostics only.
They measure failure under post-training removal and are stored under
`diagnostic_A*` names with `shared_checkpoint_ablation: true`.

Formal A0-A6 comparisons require independent training with the same split,
crop, clip length, optimizer steps, checkpoint rule, seed, validation cadence,
backbone initialization, batch/accumulation, and unrelated loss weights. Each
arm uses all eight GPUs with FSDP. A7 remains optional until Stage C is part of
the joint training graph.

Materialize the seven checked configs with:

```bash
python tools/prepare_metric_stereo_video_formal_ablations.py
```

Each arm must then be launched independently with eight ranks, for example:

```bash
torchrun --standalone --nproc_per_node=8 tools/train_metric_stereo_video.py \
  --config runs/metric_stereo_video/formal_configs/a0.yaml
```
