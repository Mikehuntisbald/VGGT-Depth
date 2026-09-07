#!/usr/bin/env python3
"""Run A5 plus a trained causal HR repair on raw stereo clips.

The training and evaluation entrypoints use the same candidate constructor.
GT/disocclusion/dynamic labels are not passed to the repair feature constructor.
The A5 previous base prediction is deliberately kept as the history owner;
feeding previously repaired outputs back would be a different untrained system.
"""
from __future__ import annotations
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.distributed as dist
from models.temporal_candidate_repair import TemporalCandidateRepair
from tools.cache_temporal_candidate_repair import candidate,cpu,atomic_save
from tools.eval_metric_stereo_video import _load_model,_previous_prefix_batch
from tools.train_metric_stereo_video import _dataset,_loader,_distributed_context,_move_batch,_read_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config',type=Path,required=True)
    p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--repair-checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--max-batches',type=int,default=1)
    args=p.parse_args()
    context=_distributed_context()
    config=_read_config(args.base_config)
    config['data']['num_workers']=1
    dataset=_dataset(config,training=False)
    loader,_=_loader(dataset,config,context,training=False)
    base=_load_model(config,args.base_checkpoint,context)
    state=torch.load(args.repair_checkpoint,map_location='cpu',weights_only=False)
    expected=state['config']['base_lineage']['checkpoint_manifest_sha256']
    if hashlib.sha256((args.base_checkpoint/'manifest.json').read_bytes()).hexdigest()!=expected:
        raise RuntimeError('repair was trained for a different base checkpoint')
    repair=TemporalCandidateRepair().to(context.device)
    repair.load_state_dict(state['model'])
    repair.eval()
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
        for i,raw in enumerate(loader):
            if i>=args.max_batches:break
            batch=_move_batch(raw,context.device)
            with torch.autocast('cuda',dtype=torch.bfloat16):
                current=copy.deepcopy(base(batch))
                previous=base(_previous_prefix_batch(batch))
            inputs=candidate(current,previous,{key:batch[key] for key in ('rgb','K','baseline_m','T_current_from_previous')})
            # Match the audited feature storage precision exactly.
            disparity,gate=repair(inputs['features'].half(),inputs['base'],inputs['history'],inputs['history_valid'],state['logit_shift'])
            index=int(raw['identity_metadata'][0]['dataset_index'])
            if index%context.world_size==context.rank:
                factor=(batch['K'][:,-1,0,0,0]*batch['baseline_m'][:,-1]).reshape(-1,1,1,1)
                atomic_save(cpu({'disparity_hr_px':disparity,'depth_m':factor/disparity,
                    'history_gate':gate,'valid_mask':inputs['base_valid'],'identity':raw['identity_metadata'][0]}),args.output_dir/f'{index:06d}.pt')
    if context.world_size>1:dist.barrier()
    if context.primary:
        (args.output_dir/'receipt.json').write_text(json.dumps({'status':'COMPLETE','base_checkpoint':str(args.base_checkpoint),
            'repair_checkpoint':str(args.repair_checkpoint),'repair_sha256':hashlib.sha256(args.repair_checkpoint.read_bytes()).hexdigest(),
            'base_history_owner':'unmodified A5 predictions','feature_dtype':'FP16 cast then FP32 MLP',
            'output_files':len(list(args.output_dir.glob('*.pt')))},indent=2)+'\n')
    if context.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
