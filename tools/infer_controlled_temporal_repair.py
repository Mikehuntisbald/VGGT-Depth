#!/usr/bin/env python3
"""Raw-input inference for a controlled candidate-repair checkpoint."""
from __future__ import annotations
import argparse,copy,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.distributed as dist
from models.temporal_candidate_repair import TemporalCandidateRepair
from models.controlled_temporal_repair import ControlledTemporalRepair,evidence_bank
from tools.cache_temporal_candidate_repair import candidate,cpu,atomic_save
from tools.eval_metric_stereo_video import _load_model,_previous_prefix_batch
from tools.train_metric_stereo_video import _dataset,_loader,_distributed_context,_move_batch,_read_config


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--base-config',type=Path,required=True);p.add_argument('--base-checkpoint',type=Path,required=True)
    p.add_argument('--v1-checkpoint',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--max-batches',type=int,default=1)
    args=p.parse_args();torch.set_num_threads(4);context=_distributed_context()
    config=_read_config(args.base_config);config['data']['num_workers']=1
    dataset=_dataset(config,training=False);loader,_=_loader(dataset,config,context,training=False)
    original=torch.load(args.v1_checkpoint,map_location='cpu',weights_only=False)
    new=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    if new['config']['initial_checkpoint_sha256']!=hashlib.sha256(args.v1_checkpoint.read_bytes()).hexdigest():raise RuntimeError('v1 initialization mismatch')
    if original['config']['base_lineage']['checkpoint_manifest_sha256']!=hashlib.sha256((args.base_checkpoint/'manifest.json').read_bytes()).hexdigest():raise RuntimeError('A5 base mismatch')
    v1=TemporalCandidateRepair();v1.load_state_dict(original['model'])
    repair=ControlledTemporalRepair(v1,**new['config']['components']).to(context.device).eval();repair.load_state_dict(new['model'])
    base=_load_model(config,args.base_checkpoint,context);base.requires_grad_(False)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    with torch.inference_mode():
      for i,raw in enumerate(loader):
        if i>=args.max_batches:break
        batch=_move_batch(raw,context.device)
        with torch.autocast('cuda',dtype=torch.bfloat16):
            current=copy.deepcopy(base(batch));previous=base(_previous_prefix_batch(batch))
        inputs=candidate(current,previous,{k:batch[k] for k in ('rgb','K','baseline_m','T_current_from_previous')})
        bank=evidence_bank(batch['rgb'][:,-1,0],batch['rgb'][:,-1,1],batch['rgb'][:,-2,0],inputs)
        output=repair(inputs['features'].half(),inputs['base'],inputs['history'],inputs['history_valid'],bank,new['logit_shift'])
        index=int(raw['identity_metadata'][0]['dataset_index'])
        if index%context.world_size==context.rank:
            atomic_save(cpu({'disparity_hr_px':output['prediction'],'gate':output['gate'],'selected_history_hr_px':output['history'],
                'valid_mask':inputs['base_valid'],'identity':raw['identity_metadata'][0]}),args.output_dir/f'{index:06d}.pt')
    if context.world_size>1:dist.barrier()
    if context.primary:(args.output_dir/'receipt.json').write_text(json.dumps({'status':'COMPLETE','samples':len(list(args.output_dir.glob('*.pt'))),
        'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),'gt_provided_to_repair':False,'history_source':'unmodified A5 strict past prefix'},indent=2)+'\n')
    if context.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
