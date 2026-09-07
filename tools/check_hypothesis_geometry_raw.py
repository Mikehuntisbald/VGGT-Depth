#!/usr/bin/env python3
"""Raw-image equivalence and strict causal prefixes for replacement decoding."""
import argparse,json,sys,types,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'third_party/Fast-FoundationStereo')]
import torch
import torch.distributed as dist
from data.raw_stereo_video_dataset import collate_raw_stereo_video_samples
from models.hypothesis_geometry import HypothesisGeometry
from models.native_temporal_stereo.encoder import encode_raw_clip
from tools.train_hypothesis_geometry import compact_inputs,move,digest
from tools.eval_native_temporal_stereo import prefix_inputs,load_package
from tools.check_native_stereo_raw import compare
from tools.eval_metric_stereo_video import _load_model
from tools.train_metric_stereo_video import _dataset,_distributed_context,_read_config,_unwrapped

def from_raw(encoded,refiner):
    count=encoded['rgb'].shape[1]
    def pad(v):return torch.cat((v,v[:1].expand(8-count,*v.shape[1:])),0) if count<8 else v
    evidence={k:([pad(x) for x in v] if isinstance(v,list) else pad(v)) for k,v in encoded['encoded'].items()}
    solved=refiner(evidence)['disparity_lr'][:count][None]
    record={'inputs':encoded,'audit':{'ffs_image_only_disparity_lr':solved}}
    return compact_inputs(record)

def load_model(checkpoint,device):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=HypothesisGeometry(use_memory=payload['config']['variant']=='memory').to(device).eval();model.load_state_dict(payload['model'])
    return model,payload

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--backbone-checkpoint',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();ctx=_distributed_context();torch.set_num_threads(1);config=_read_config(a.config)
    frozen=_load_model(config,a.backbone_checkpoint,ctx).requires_grad_(False);system=_unwrapped(frozen);system.forward=types.MethodType(encode_raw_clip,system)
    def forbidden(*args,**kwargs):raise RuntimeError('old geometry decoder executed')
    system.geometry_model.forward_step=forbidden
    refiner=load_package(a.cache)['refiner'].to(ctx.device).eval();models={}
    for checkpoint in a.checkpoints:
        model,payload=load_model(checkpoint,ctx.device);models[payload['config']['variant']]=model
    indices=[0,2,24,29,300,600,900,1200];index=indices[ctx.rank]
    sample=collate_raw_stereo_video_samples([_dataset(config,training=False)[index]])
    allowed=('rgb','K','baseline_m','T_current_from_previous','T_right_from_left');raw=move({k:sample[k] for k in allowed},ctx.device)
    original=torch.load(a.cache/'validation'/f'{index:06d}.pt',map_location='cpu',weights_only=False,mmap=True);cached=move(compact_inputs(original),ctx.device)
    rows=[];started=time.monotonic()
    with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
        encoded=from_raw(frozen(raw),refiner);compare(encoded,cached,'raw_cached_inputs')
        predictions={}
        for name,model in models.items():
            predictions[name]=model(encoded);cached_output=model(cached)
            compare(predictions[name]['endpoint'].disparity_left_px,cached_output['endpoint'].disparity_left_px,name+'/raw_cache_output')
            compare(predictions[name]['endpoint'].valid_mask,cached_output['endpoint'].valid_mask,name+'/raw_cache_validity')
        changed={**raw,'rgb':raw['rgb'].clone()};changed['rgb'][:,-1]=1-changed['rgb'][:,-1]
        altered=from_raw(frozen(changed),refiner);count=raw['rgb'].shape[1]-1
        compare(prefix_inputs(encoded,count),prefix_inputs(altered,count),'future_pixel_independence')
        independent=from_raw(frozen({k:v[:,:count] for k,v in raw.items()}),refiner)
        compare(prefix_inputs(encoded,count),independent,'independent_raw_prefix')
        for name,model in models.items():
            changed_output=model(altered);past=model(independent)
            for t in range(count):compare(predictions[name]['frames'][t].disparity_left_px,changed_output['frames'][t].disparity_left_px,name+'/past_pixels')
            compare(predictions[name]['frames'][count-1].disparity_left_px,past['endpoint'].disparity_left_px,name+'/independent_prefix_output')
            rows.append({'variant':name,'dataset_index':index,'raw_cache_max_abs_px':0.,'future_pixel_past_max_abs_px':0.,'independent_prefix_max_abs_px':0.})
    gathered=[None]*ctx.world_size
    if ctx.world_size>1:dist.all_gather_object(gathered,rows)
    else:gathered=[rows]
    if ctx.primary:
        result={'status':'PASS','samples':ctx.world_size,'rows':[r for shard in gathered for r in shard],'old_geometry_executed':False,'source_sha256':digest(Path(__file__)),'elapsed_s':time.monotonic()-started}
        a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'status':'PASS','samples':ctx.world_size}))
    if ctx.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
