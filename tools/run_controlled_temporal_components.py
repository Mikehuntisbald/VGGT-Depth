#!/usr/bin/env python3
"""Run individual component arms after the supervision development decision."""
import json,subprocess,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
run=ROOT/'runs/metric_stereo_video/temporal_controlled_20260907'
bank=Path('/tmp/vggt_controlled_evidence_20260907')
while not (bank/'train/complete.json').exists():time.sleep(5)
selection=json.loads((run/'supervision/selection.json').read_text())
profile=selection['selected_profile']
assert profile in ('mild','balanced','recover','protect')
with (run/'components.log').open('w') as log:
    subprocess.run(['python','tools/train_controlled_temporal_repair.py','--cache','/tmp/vggt_temporal_repair_20260907',
        '--bank',str(bank),'--v1-checkpoint','runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt',
        '--output-dir',str(run/'components'),'--profiles',profile,'--components','matching,alignment,ambiguity'],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
while not (bank/'validation/complete.json').exists():time.sleep(5)
paths=[str(run/'components'/f'{component}_{profile}'/'final.pt') for component in ('matching','alignment','ambiguity')]
with (run/'evaluation_components.log').open('w') as log:
    subprocess.run(['torchrun','--standalone','--nproc_per_node=8','tools/eval_controlled_temporal_repair.py',
        '--cache','/tmp/vggt_temporal_repair_20260907','--bank',str(bank),
        '--v1-checkpoint','runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt',
        '--checkpoints',*paths,'--output-dir',str(run/'evaluation_components')],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
