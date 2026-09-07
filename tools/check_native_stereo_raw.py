#!/usr/bin/env python3
"""Verify raw RGB -> native model equivalence and absence of future dependence."""
import argparse,json,sys,time,types
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'third_party/Fast-FoundationStereo')]
import torch
import torch.distributed as dist
from data.raw_stereo_video_dataset import collate_raw_stereo_video_samples
from models.native_temporal_stereo.encoder import encode_raw_clip
from tools.eval_native_temporal_stereo import load_native,prefix_inputs
from tools.train_native_temporal_stereo import move,digest
from tools.eval_metric_stereo_video import _load_model
from tools.train_metric_stereo_video import _dataset,_distributed_context,_read_config,_unwrapped


def compare(a,b,path=''):
    if isinstance(a,torch.Tensor):
        if a.shape!=b.shape:raise RuntimeError('shape mismatch '+path)
        if a.dtype==torch.bool:delta=float((a!=b).any())
        else:delta=float((a.float()-b.float()).abs().max())
        if delta:raise RuntimeError(f'raw cache mismatch {path}: {delta}')
        return {path:delta}
    items=a.items() if isinstance(a,dict) else enumerate(a);result={}
    for k,v in items:result.update(compare(v,b[k],path+'/'+str(k)))
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--config',type=Path,required=True)
    p.add_argument('--backbone-checkpoint',type=Path,required=True);p.add_argument('--checkpoints',nargs='+',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--indices',default='0,1,2,3,4,5,6,7')
    a=p.parse_args();context=_distributed_context();torch.set_num_threads(1);config=_read_config(a.config)
    frozen=_load_model(config,a.backbone_checkpoint,context).requires_grad_(False);system=_unwrapped(frozen)
    system.forward=types.MethodType(encode_raw_clip,system)
    def forbidden(*args,**kwargs):raise RuntimeError('old A5 geometry was executed by raw encoder')
    system.geometry_model.forward_step=forbidden
    dataset=_dataset(config,training=False);models={}
    for checkpoint in a.checkpoints:
        model,cfg=load_native(a.cache,checkpoint,context.device);models[cfg['config']['variant']]=model
    # One complete independent endpoint on each rank. Each rank makes identical
    # collective calls, including a second pass with changed final-frame pixels.
    indices=list(map(int,a.indices.split(',')))
    if len(indices)!=context.world_size:raise ValueError('one raw check index per rank required')
    index=indices[context.rank];batch=collate_raw_stereo_video_samples([dataset[index]])
    allowed=('rgb','K','baseline_m','T_current_from_previous','T_right_from_left')
    raw=move({k:batch[k] for k in allowed},context.device);started=time.monotonic();checks=[]
    cached=move(torch.load(a.cache/'validation'/f'{index:06d}.pt',map_location='cpu',weights_only=False),context.device)['inputs']
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=frozen(raw)
        differences=compare(encoded,cached)
        for name,model in models.items():
            raw_out=model(encoded)['endpoint'];cache_out=model(cached)['endpoint']
            compare(raw_out.disparity_left_px,cache_out.disparity_left_px,name+'/output')
            compare(raw_out.valid_mask,cache_out.valid_mask,name+'/valid')
            checks.append({'arm':name,'dataset_index':index,'raw_cache_max_abs_px':0.0})
        changed={**raw,'rgb':raw['rgb'].clone()};changed['rgb'][:,-1]=1-changed['rgb'][:,-1]
        changed_encoded=frozen(changed)
        count=raw['rgb'].shape[1]-1
        compare(prefix_inputs(encoded,count),prefix_inputs(changed_encoded,count),'future_pixels_independence')
        for name,model in models.items():
            old=model(encoded);changed_out=model(changed_encoded)
            for t in range(count):compare(old['frames'][t].disparity_left_px,changed_out['frames'][t].disparity_left_px,f'{name}/past_output/{t}')
        independent_raw={k:v[:,:count] for k,v in raw.items()}
        independent_encoded=frozen(independent_raw)
        compare(prefix_inputs(encoded,count),independent_encoded,'independent_raw_prefix')
        for name,model in models.items():
            full=model(encoded);independent=model(independent_encoded)
            compare(full['frames'][count-1].disparity_left_px,independent['endpoint'].disparity_left_px,name+'/independent_raw_prefix_output')
    row={'rank':context.rank,'dataset_index':index,'checks':checks,'raw_encoder_field_max_differences':differences,'future_pixel_perturbation':'both images in last frame inverted; every preceding input field and native output exact','independent_raw_prefix':'past RGB physically sliced and re-encoded; every field and native endpoint exact','elapsed_s':time.monotonic()-started}
    gathered=[None]*context.world_size
    if context.world_size>1:dist.all_gather_object(gathered,row)
    else:gathered=[row]
    if context.primary:
        report={'status':'PASS','samples':context.world_size,'rows':gathered,'source_sha256':digest(Path(__file__)),
          'encoder_source_sha256':digest(ROOT/'src/models/native_temporal_stereo/encoder.py'),'old_A5_geometry_called':False,
          'raw_inputs':list(allowed),'labels_in_forward':False,'original_final_prediction_in_forward':False}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({'status':'PASS','samples':context.world_size}),flush=True)
    if context.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
