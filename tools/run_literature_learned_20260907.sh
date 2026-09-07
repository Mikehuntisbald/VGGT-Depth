#!/usr/bin/env bash
set -euo pipefail
cd /mnt/why/VGGT-Depth
TASK_RUN=runs/metric_stereo_video/temporal_literature_20260907
TASK_BANK=/tmp/vggt_learned_evidence_20260907
TASK_CACHE=/tmp/vggt_temporal_repair_20260907
TASK_V1=runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt
mkdir -p "$TASK_RUN/learned_source"
cp src/models/learned_temporal_evidence.py src/models/learned_codd_temporal_repair.py tools/cache_learned_temporal_evidence.py tools/train_learned_codd_temporal_repair.py tools/eval_learned_codd_temporal_repair.py docs/temporal_repair_literature_protocol.md "$TASK_RUN/learned_source/"
export OMP_NUM_THREADS=4
export PYTHONPATH=src:.
torchrun --standalone --nproc_per_node=8 tools/cache_learned_temporal_evidence.py --cache "$TASK_CACHE" --output-dir "$TASK_BANK" --config runs/metric_stereo_video/formal_a5_seed42/resolved_config.yaml --checkpoint runs/metric_stereo_video/formal_a5_seed42/checkpoints/step_0006000 > "$TASK_RUN/cache.log" 2>&1
python tools/train_learned_codd_temporal_repair.py --cache "$TASK_CACHE" --bank "$TASK_BANK" --v1-checkpoint "$TASK_V1" --output-dir "$TASK_RUN/learned" > "$TASK_RUN/train_learned.log" 2>&1
torchrun --standalone --nproc_per_node=8 tools/eval_learned_codd_temporal_repair.py --cache "$TASK_CACHE" --bank "$TASK_BANK" --v1-checkpoint "$TASK_V1" --checkpoints "$TASK_RUN/learned/learned_codd_capacity_control/final.pt" "$TASK_RUN/learned/learned_codd_codd/final.pt" "$TASK_RUN/learned/learned_codd_codd_regret/final.pt" --output-dir "$TASK_RUN/evaluation_learned" > "$TASK_RUN/evaluation_learned.log" 2>&1
