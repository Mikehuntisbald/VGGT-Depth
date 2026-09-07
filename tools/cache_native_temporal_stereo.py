#!/usr/bin/env python3
"""Cache frozen image encodings, never A5 predictions as structural model inputs.

Exact A5 FFS/VGGT encoders are shared by control/new architectures. The iterative
stereo solver, geometry decoder and causal state are executed by the new model.
GT and original FFS outputs are stored separately as labels/parity references.
"""
from __future__ import annotations
import argparse,copy,dataclasses,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader,Subset,DistributedSampler
from data.raw_stereo_video_dataset import collate_raw_stereo_video_samples
from models.native_temporal_stereo.refiner import NativeStereoRefiner
from models.metric_stereo_video_system import left_right_stereo_consistency
from tools.cache_temporal_candidate_repair import cpu,atomic_save
from tools.eval_metric_stereo_video import _load_model,_previous_prefix_batch
from tools.train_metric_stereo_video import _dataset,_distributed_context,_read_config,_move_batch,_sha256,_unwrapped


def snapshot(x):
    if isinstance(x,torch.Tensor):return x.detach().clone()
    if isinstance(x,(list,tuple)):return [snapshot(v) for v in x]
    if isinstance(x,dict):return {k:snapshot(v) for k,v in x.items()}
    return x


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--splits',default='train,validation');p.add_argument('--max-batches',type=int);p.add_argument('--num-workers',type=int,default=2)
    args=p.parse_args();context=_distributed_context();torch.set_num_threads(4);config=_read_config(args.config)
    args.output_dir.mkdir(parents=True,exist_ok=True)
    signature={'schema':1,'checkpoint_manifest_sha256':_sha256(args.checkpoint/'manifest.json'),'config_sha256':_sha256(args.config),
      'sampling_plan_lineage_sha256':_sha256(args.cache/'lineage.json'),'script_sha256':_sha256(Path(__file__)),
      'refiner_source_sha256':_sha256(ROOT/'src/models/native_temporal_stereo/refiner.py'),
      'max_batches':args.max_batches,'encoder_frozen':True,'input_boundary':'FFS encoded feature/cost/context and causal VGGT features; no A5/v1 prediction input',
      'vg_gt_schedule':'independent causal prefixes 1..T, available each frame; shared by structural control and new matcher',
      'training_crop':'same deterministic epoch-zero crops as original cache','torch':torch.__version__,'cuda':torch.version.cuda}
    lineage=args.output_dir/'lineage.json'
    if lineage.exists() and json.loads(lineage.read_text())!=signature:raise RuntimeError('lineage changed; new cache path required')
    if context.primary:lineage.write_text(json.dumps(signature,indent=2)+'\n')
    if context.world_size>1:dist.barrier()
    model=_load_model(config,args.checkpoint,context);model.requires_grad_(False)
    system=_unwrapped(model);ffs=system.stereo_backbone.model
    if context.primary:print(json.dumps({'native_dtype':str(ffs.dtype),'low_memory':bool(ffs.args.get('low_memory',False))}),flush=True)
    # The package contains only the frozen native iterative solver and common
    # geometry decoder initialization. It excludes both image encoder weights.
    with FSDP.summon_full_params(model,recurse=True,writeback=False,rank0_only=False):
        refiner=NativeStereoRefiner(ffs,iterations=config['stereo']['iterations']).to(context.device)
        if context.primary:
            geometry=copy.deepcopy(system.geometry_model).cpu()
            atomic_save({'refiner':copy.deepcopy(refiner).cpu(),'geometry':geometry,'config':config,'volume_channels':int(ffs.volume_dim),'hidden_channels':list(map(int,ffs.args.hidden_dims)),
              'backbone_checkpoint_manifest_sha256':signature['checkpoint_manifest_sha256']},args.output_dir/'initialization.pt')
    captured={};frames=[]
    def hook(key):
        def capture(_module,_inputs,output):
            if key not in captured:captured[key]=snapshot(output)
        return capture
    handles=[ffs.feature.register_forward_hook(hook('features')),ffs.cost_agg.register_forward_hook(hook('volume')),
      ffs.classifier.register_forward_hook(hook('logits')),ffs.stem_2.register_forward_hook(hook('stem'))]
    def update_pre(_module,inputs):
        if 'update' not in captured:captured['update']=snapshot([inputs[0],inputs[1],inputs[3],inputs[4]])
    handles.append(ffs.update_block.register_forward_pre_hook(update_pre))
    original_step=system.geometry_model.forward_step
    def frame_step(frame,state=None):
        frames.append(frame)
        return original_step(frame,state)
    system.geometry_model.forward_step=frame_step
    started=time.monotonic();checks=[]
    for split in args.splits.split(','):
        dataset=_dataset(config,training=split=='train');plan=json.loads((args.cache/split/'plan.json').read_text());indices=plan['indices']
        subset=Subset(dataset,indices);sampler=DistributedSampler(subset,num_replicas=context.world_size,rank=context.rank,shuffle=False)
        loader=DataLoader(subset,sampler=sampler,batch_size=1,num_workers=args.num_workers,collate_fn=collate_raw_stereo_video_samples,pin_memory=True)
        directory=args.output_dir/split;directory.mkdir(exist_ok=True)
        if context.primary:(directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        with torch.inference_mode():
            for i,raw in enumerate(loader):
                if args.max_batches and i>=args.max_batches:break
                index=int(raw['identity_metadata'][0]['dataset_index']);owner=indices.index(index)%context.world_size==context.rank
                batch=_move_batch(raw,context.device);captured.clear();frames.clear()
                with torch.autocast('cuda',dtype=torch.bfloat16):output=model(batch)
                count=batch['rgb'].shape[1]
                if len(frames)!=count:raise RuntimeError('geometry frame capture mismatch')
                feat=captured['features'][0]
                if feat.shape[0]!=2*count:raise RuntimeError('stereo feature ordering changed')
                hidden,context_inputs,seed,attention=captured['update']
                encoded={'left':feat[:count],'right':feat[count:],'volume':captured['volume'],
                    'seed':seed,'hidden':hidden,'context':context_inputs,'attention':attention,'stem':captured['stem']}
                frame_list=list(frames)
                if i==0:
                    decoded_all=refiner(encoded)
                    expected_all=output.stereo.disparity_left_lr_px[0]
                    torch.testing.assert_close(decoded_all['disparity_lr'],expected_all,rtol=1e-5,atol=1e-4)
                    checks.append({'split':split,'index':index,'mode':'same-batch-native-replay','max_abs_px_lr':float((decoded_all['disparity_lr']-expected_all).abs().max())})
                    for t in (0,count-1):
                        def take(v):return [x[t:t+1] for x in v] if isinstance(v,list) else v[t:t+1]
                        decoded=refiner({k:take(v) for k,v in encoded.items()})
                        expected=output.stereo.disparity_left_lr_px[:,t]
                        difference=(decoded['disparity_lr']-expected).abs()
                        if not difference.isfinite().all():raise RuntimeError('nonfinite sequential decoder')
                        checks.append({'split':split,'index':index,'time':t,'max_abs_px_lr':float(difference.max())})
                vggt_frames=[None]*count
                def vggt_snapshot(frame):
                    v=frame.vggt_features
                    return snapshot({'feature':v.feature_map,'inverse_relative':v.inverse_depth_relative,'confidence':v.confidence})
                vggt_frames[-1]=vggt_snapshot(frame_list[-1])
                prefix=batch
                for t in range(count-2,-1,-1):
                    prefix=_previous_prefix_batch(prefix);frames.clear()
                    with torch.autocast('cuda',dtype=torch.bfloat16):unused=model(prefix)
                    vggt_frames[t]=vggt_snapshot(frames[-1])
                if owner:
                    metadata=raw['identity_metadata'][0]
                    old=torch.load(args.cache/split/f'{index:06d}.pt',map_location='cpu',weights_only=False)
                    record={'inputs':{'encoded':encoded,'rgb':(batch['rgb']*255).round().to(torch.uint8),
                      'K':batch['K'],'baseline_m':batch['baseline_m'],'T_current_from_previous':batch['T_current_from_previous'],
                      'right_disparity_lr':output.stereo.disparity_right_lr_px,
                      'stereo_confidence':torch.stack([f.lowres_disparity_confidence for f in frame_list],1),
                      'stereo_valid':torch.stack([f.lowres_disparity_valid_mask for f in frame_list],1),
                      'vggt':{k:torch.stack([v[k] for v in vggt_frames],1) for k in vggt_frames[0]},
                      'T_right_from_left':torch.stack([f.T_right_from_left_m for f in frame_list],1)},
                      'labels':{'target':batch['disparity_gt_left_px'],'valid':batch['valid_gt_left'],
                         'stereo_matched':left_right_stereo_consistency(batch['disparity_gt_left_px'],batch['disparity_gt_right_px'],maximum_error_px=1.,confidence_temperature_px=.5).valid_left_mask & batch['valid_gt_left']},
                      'audit':{'ffs_image_only_disparity_lr':output.stereo.disparity_left_lr_px},
                      'identity':old['identity']}
                    atomic_save(cpu(record),directory/f'{index:06d}.pt')
                if i%20==0:print(json.dumps({'rank':context.rank,'split':split,'batch':i,'batches':len(loader),'elapsed_s':time.monotonic()-started}),flush=True)
        if context.world_size>1:dist.barrier()
        if context.primary:
            actual=len(list(directory.glob('*.pt')));complete=actual==len(indices)
            (directory/('complete.json' if complete else 'smoke_complete.json')).write_text(json.dumps({'status':'COMPLETE' if complete else 'SMOKE_COMPLETE','samples':actual,'expected':len(indices),'elapsed_s':time.monotonic()-started},indent=2)+'\n')
    gathered=[None]*context.world_size
    if context.world_size>1:dist.all_gather_object(gathered,checks)
    else:gathered=[checks]
    if context.primary:(args.output_dir/'native_replay_checks.json').write_text(json.dumps({'status':'PASS','checks':[r for part in gathered for r in part]},indent=2)+'\n')
    if context.world_size>1:dist.destroy_process_group()
if __name__=='__main__':main()
