#!/usr/bin/env bash
set -euo pipefail
cd /mnt/why/VGGT-Depth
cache_root=/tmp/vggt_temporal_repair_20260907
run_root=runs/metric_stereo_video/temporal_repair_20260907
while ! test -f "$cache_root/train/complete.json"; do sleep 10; done
CUDA_VISIBLE_DEVICES=0 python tools/train_temporal_candidate_repair.py \
  --cache "$cache_root" --output-dir "$run_root/head_v1" \
  --steps 3000 --batch-size 32768 > "$run_root/train_v1.log" 2>&1
while ! test -f "$cache_root/validation/complete.json"; do sleep 10; done
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=8 tools/eval_temporal_candidate_repair.py \
  --cache "$cache_root" --checkpoint "$run_root/head_v1/final.pt" \
  --output-dir "$run_root/evaluation_v1" > "$run_root/evaluation_v1.log" 2>&1
OMP_NUM_THREADS=4 torchrun --standalone --nproc_per_node=8 tools/eval_metric_stereo_video.py \
  --checkpoint runs/metric_stereo_video/formal_a4_seed42/checkpoints/step_0006000 \
  --config runs/metric_stereo_video/formal_a4_seed42/resolved_config.yaml \
  --num-workers 2 --output-dir "$run_root/a4_full_evaluation" > "$run_root/a4_full_evaluation.log" 2>&1
