#!/usr/bin/env python3
"""Cache frozen A5 candidates with exact causal prefixes and separate labels."""
from __future__ import annotations
import argparse
import copy
import json
import sys
import time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.utils.data import DataLoader,Subset,DistributedSampler
from data.raw_stereo_video_dataset import collate_raw_stereo_video_samples
from models.temporal_candidate_repair import repair_features
from tools.eval_metric_stereo_video import _load_model,_previous_prefix_batch
from tools.analyze_metric_stereo_video_temporal import _warp_prediction,_warp_gt,_spring_masks,_motion_masks
from tools.train_metric_stereo_video import _dataset,_distributed_context,_move_batch,_read_config,_sha256
from metrics.boundary import disparity_boundary_mask


def candidate(output, previous, batch):
    warp, valid = _warp_prediction(previous,batch)
    base=output.disparity_left_px.float()
    hist=warp.disparity_hr_px.float()
    h,w=base.shape[-2:]
    uv=warp.source_uv.long()
    index=(uv[:,1].clamp(0,h-1)*w+uv[:,0].clamp(0,w-1)).flatten(1)[:,None]
    rgb=batch['rgb'][:,-2,0].float()
    hist_rgb=torch.gather(rgb.flatten(2),2,index.expand(-1,3,-1)).reshape_as(rgb)
    temporal=output.endpoint.temporal
    def up(x):
        return F.interpolate(x.float(),size=(h,w),mode='bilinear',align_corners=False)
    factor=(batch['K'][:,-1,0,0,0]*batch['baseline_m'][:,-1]).reshape(-1,1,1,1)
    features=repair_features(left_rgb=batch['rgb'][:,-1,0],right_rgb=batch['rgb'][:,-1,1],
        base_disparity_hr_px=base,history_disparity_hr_px=hist,history_valid=valid,
        base_confidence=output.confidence,history_confidence=warp.confidence,
        history_rgb=hist_rgb,collision=warp.collision_mask,
        old_gate=up(temporal.learned_gate.mean(1,keepdim=True)),old_valid=up(temporal.valid_mask),
        stereo_disparity_hr_px=up(output.stereo.disparity_left_hr_px_lr_grid[:, -1]),
        stereo_confidence=up(output.stereo.confidence_left_lr[:, -1]),
        internal_history_disparity_hr_px=up(temporal.warped_inverse_depth_pre_consistency_m_inv)*factor)
    return {'features':features,'base':base,'history':hist,'history_valid':valid,
            'base_valid':output.valid_mask,'base_probability':output.valid_probability.float(),
            'base_uncertainty':output.endpoint.uncertainty.float(),'base_confidence':output.confidence.float(),
            'old_valid':up(temporal.valid_mask)>=0.5,
            'internal_history':up(temporal.warped_inverse_depth_pre_consistency_m_inv)*factor,
            'internal_valid':up(temporal.zbuffer_visible_mask)>=0.5,
            'source_uv':warp.source_uv}


def cpu(value):
    if isinstance(value,torch.Tensor):
        return value.detach().cpu().contiguous()
    if isinstance(value,dict):
        return {k:cpu(v) for k,v in value.items()}
    return value


def atomic_save(value,path):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    temporary.replace(path)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',type=Path,required=True)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--train-limit',type=int,default=1152)
    parser.add_argument('--validation-limit',type=int,default=0)
    parser.add_argument('--splits',default='train,validation')
    parser.add_argument('--num-workers',type=int,default=2)
    args=parser.parse_args()
    context=_distributed_context()
    torch.set_num_threads(4)
    config=_read_config(args.config)
    config['data']['num_workers']=args.num_workers
    args.output_dir.mkdir(parents=True,exist_ok=True)
    signature={'schema':1,'base_checkpoint':str(args.checkpoint.resolve()),
        'checkpoint_manifest_sha256':_sha256(args.checkpoint/'manifest.json'),
        'config_sha256':_sha256(args.config),
        'manifests':{key:_sha256(ROOT/config['data'][key]) for key in ('train_manifest','validation_manifest')},
        'features_source_sha256':_sha256(ROOT/'src/models/temporal_candidate_repair.py'),
        'cache_source_sha256':_sha256(Path(__file__)),
        'train_limit':args.train_limit,'validation_limit':args.validation_limit,
        'precision':'A5 BF16; feature storage FP16; disparity and GT FP32',
        'causality':'current prefix t; independent history prefix t-1; previous repair uses prefix t-2',
        'training_pixel_sampling':'4096 uniform + 4096 disagreement pixels per selected training frame',
        'base_frozen':True}
    receipt=args.output_dir/'lineage.json'
    if receipt.exists() and json.loads(receipt.read_text())!=signature:
        raise RuntimeError('cache lineage changed; use a new output directory')
    if context.primary:
        receipt.write_text(json.dumps(signature,indent=2)+'\n')
    if context.world_size>1: dist.barrier()
    model=_load_model(config,args.checkpoint,context)
    model.requires_grad_(False)
    started=time.monotonic()
    for split in args.splits.split(','):
        training=split=='train'
        dataset=_dataset(config,training=training)
        limit=args.train_limit if training else args.validation_limit
        indices=np.linspace(0,len(dataset)-1,min(limit,len(dataset)),dtype=int).tolist() if limit else list(range(len(dataset)))
        directory=args.output_dir/split
        directory.mkdir(exist_ok=True)
        plan={'dataset_length':len(dataset),'indices':indices,'split':split,'crop_mode':dataset.crop_mode}
        if context.primary: (directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        subset=Subset(dataset,indices)
        sampler=DistributedSampler(subset,num_replicas=context.world_size,rank=context.rank,shuffle=False,drop_last=False)
        loader=DataLoader(subset,sampler=sampler,batch_size=1,num_workers=args.num_workers,
            collate_fn=collate_raw_stereo_video_samples,pin_memory=True)
        with torch.inference_mode():
            for i,raw in enumerate(loader):
                batch=_move_batch(raw,context.device)
                index=int(raw['identity_metadata'][0]['dataset_index'])
                # All ranks execute every padded FSDP forward; only the unique owner writes.
                position=indices.index(index)
                owner=position%context.world_size==context.rank
                output_path=directory/f'{index:06d}.pt'
                exists=torch.tensor(int(output_path.exists()),device=context.device)
                if context.world_size>1: dist.all_reduce(exists,op=dist.ReduceOp.MIN)
                if exists.item(): continue
                prefix=_previous_prefix_batch(batch)
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=copy.deepcopy(model(batch))
                    previous=copy.deepcopy(model(prefix))
                current=candidate(output,previous,batch)
                if not training:
                    with torch.autocast('cuda',dtype=torch.bfloat16):
                        previous_previous=model(_previous_prefix_batch(prefix))
                    prev=candidate(previous,previous_previous,prefix)
                if owner:
                    target=batch['disparity_gt_left_px'][:,-1].float()
                    valid=batch['valid_gt_left'][:,-1].bool() & torch.isfinite(target) & (target>0)
                    record=dataset.records[raw['identity_metadata'][0]['endpoint_manifest_index']]
                    identity={'sequence_id':record.sequence_id,'frame_id':record.frame_id,
                              'dataset_index':index,'metadata':raw['identity_metadata'][0]}
                    if training:
                        features=current['features'].permute(0,2,3,1).reshape(-1,current['features'].shape[1])
                        uniform=torch.arange(target.numel(),device=context.device)
                        hard=torch.where((valid & current['history_valid'] & ((current['base']-current['history']).abs()>0.25)).flatten())[0]
                        generator=torch.Generator(device=context.device).manual_seed(42000+index)
                        picks=[uniform[torch.randint(len(uniform),(4096,),device=context.device,generator=generator)]]
                        if hard.numel(): picks.append(hard[torch.randint(len(hard),(4096,),device=context.device,generator=generator)])
                        else: picks.append(picks[0])
                        take=torch.cat(picks)
                        saved={'features':features[take].half(),'target':target.flatten()[take,None],
                            'target_valid':valid.flatten()[take,None],'identity':identity}
                        for key in ('base','history','history_valid','base_valid','old_valid','internal_history','internal_valid'):
                            saved[key]=current[key].flatten()[take,None]
                    else:
                        current['features']=current['features'].half()
                        prev['features']=prev['features'].half()
                        gt_warp=_warp_gt(batch)
                        spring=_spring_masks(raw,dataset)
                        saved={'current':current,'previous':prev,'target':target,'target_valid':valid,
                               'gt_warp':gt_warp.disparity_hr_px,'gt_warp_valid':gt_warp.valid_mask,
                               'dynamic':batch['dynamic_mask_current'],'dynamic_available':batch['dynamic_mask_available'],
                               'detail':spring['high_detail'],'matched':spring['matched'],
                               'boundary':disparity_boundary_mask(target,gradient_threshold_px=1.,radius_px=1),
                               'K':batch['K'][:,-1,0],'K_previous':batch['K'][:,-2,0],
                               'baseline':batch['baseline_m'][:,-1],'baseline_previous':batch['baseline_m'][:,-2],
                               'transform':batch['T_current_from_previous'][:,-1],
                               'rgb':(batch['rgb'][:,-1,0]*255).clamp(0,255).byte(),
                               'identity':identity}
                    atomic_save(cpu(saved),output_path)
                if i%10==0:
                    print(json.dumps({'rank':context.rank,'split':split,'batch':i,'batches':len(loader),'elapsed_s':round(time.monotonic()-started,1)}),flush=True)
        if context.world_size>1: dist.barrier()
        if context.primary:
            missing=[idx for idx in indices if not (directory/f'{idx:06d}.pt').exists()]
            if missing: raise RuntimeError(f'missing cache entries: {missing[:10]}')
            (directory/'complete.json').write_text(json.dumps({'status':'COMPLETE','samples':len(indices),'elapsed_s':time.monotonic()-started})+'\n')
    if context.world_size>1: dist.destroy_process_group()

if __name__=='__main__': main()
