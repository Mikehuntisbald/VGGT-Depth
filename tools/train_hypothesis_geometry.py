#!/usr/bin/env python3
"""Controlled training of a replacement hypothesis-preserving geometry decoder."""
import argparse,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.hypothesis_geometry import HypothesisGeometry
from models.hypothesis_geometry_loss import hypothesis_loss
from tools.train_metric_stereo_video import _distributed_context

def move(x,device):
    if isinstance(x,torch.Tensor):return x.to(device)
    if isinstance(x,dict):return {k:move(v,device) for k,v in x.items()}
    return x

def compact_inputs(record):
    original=record['inputs']
    result={k:original[k] for k in ('rgb','K','baseline_m','T_current_from_previous','T_right_from_left','right_disparity_lr','vggt')}
    result['encoded']={k:original['encoded'][k] for k in ('left','right','seed')}
    # This is the image-only FFS backbone observation, before A5 geometry.
    # Raw inference regenerates it from RGB and verifies exact equivalence.
    result['stereo_lr']=record['audit']['ffs_image_only_disparity_lr']
    return result

def digest(path):
    import hashlib
    return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--variant',choices=('no_memory','memory'),required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--smoke',action='store_true')
    a=p.parse_args();ctx=_distributed_context();torch.set_num_threads(1);torch.manual_seed(42)
    if not (a.cache/'training_labels/complete.json').exists():raise RuntimeError('real last-two-frame GT unavailable')
    identities=json.loads((a.cache/'train/identities.json').read_text());development={'0001','0011','0020','0036','0045'}
    paths=[a.cache/'train'/name for name,row in sorted(identities.items()) if row['sequence_id'] not in development]
    model=HypothesisGeometry(use_memory=a.variant=='memory').to(ctx.device).train();optimizer=torch.optim.AdamW(model.parameters(),lr=1e-4,weight_decay=.01)
    wrapped=DDP(model,device_ids=[ctx.device.index],find_unused_parameters=True) if ctx.world_size>1 else model
    receipt={'architecture':'pixel_hypothesis_geometry','variant':a.variant,'steps':a.steps,'world_size':ctx.world_size,'seed':42,'training_endpoints':len(paths),'development_endpoints':len(identities)-len(paths),
      'cache_lineage_sha256':digest(a.cache/'lineage.json'),'training_labels_receipt_sha256':digest(a.cache/'training_labels/complete.json'),
      'lr':1e-4,'optimizer':'AdamW','weight_decay':.01,'gradient_clip':1.,'trainable_parameters':sum(p.numel() for p in model.parameters()),
      'initialization':'new decoder from scratch; frozen pretrained FFS/VGGT image observations; A5 geometry and v1 not used',
      'full_pixel_supervision':True,'pixel_labels_per_coarse_cell':64,'hypotheses_per_pixel':23,'offset_bound_px':4,'truncated_bptt_frames':2,'clip_length':8,
      'model_source_sha256':digest(ROOT/'src/models/hypothesis_geometry.py'),'loss_source_sha256':digest(ROOT/'src/models/hypothesis_geometry_loss.py'),'trainer_sha256':digest(Path(__file__)),
      'checkpoint_selection':'fixed final step; no validation calibration'}
    a.output_dir.mkdir(parents=True,exist_ok=True)
    if ctx.primary:(a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    generator=torch.Generator().manual_seed(42);indices=torch.randint(len(paths),(a.steps,ctx.world_size),generator=generator);started=time.monotonic();ownership={}
    for step in range(a.steps):
        tick=time.monotonic()
        path=paths[int(indices[step,ctx.rank])]
        # mmap avoids reading the unused large 3-D cost/iterative tensors.
        original=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        record={'inputs':move(compact_inputs(original),ctx.device),'labels':move({k:original['labels'][k].clone() for k in ('target','valid')},ctx.device)}
        labels=torch.load(a.cache/'training_labels'/path.name,map_location='cpu',weights_only=False)
        if labels['dataset_index']!=original['identity']['dataset_index']:raise RuntimeError('label identity mismatch')
        for k in ('target','valid'):record['labels'][k][:,-2:]=labels[k].to(ctx.device)
        loaded=time.monotonic()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=wrapped(record['inputs']);loss,parts=hypothesis_loss(out,record)
        if not loss.isfinite():raise RuntimeError('nonfinite loss')
        forwarded=time.monotonic()
        loss.backward();norm=torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        if not norm.isfinite():raise RuntimeError('nonfinite gradient')
        if step==0:
            ownership={n:float(p.grad.float().norm()) for n,p in model.named_parameters() if p.grad is not None}
            if not any(v>0 for v in ownership.values()):raise RuntimeError('decoder received no gradients')
        optimizer.step();optimizer.param_groups[0]['lr']=1e-4*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/a.steps)))
        if a.smoke:
            torch.cuda.synchronize();print(json.dumps({'rank':ctx.rank,'step':step+1,'load_s':loaded-tick,'forward_s':forwarded-loaded,'backward_update_s':time.monotonic()-forwarded}),flush=True)
            if step==1 and not any(p.grad is not None and bool(p.grad.abs().sum()>0) for n,p in model.named_parameters() if n.startswith('label.') or n.startswith('interactions.')):raise RuntimeError('no gradient reached hypothesis representations')
        if step==0 or (step+1)%25==0 or step+1==a.steps:
            value=torch.tensor([float(loss.detach()),float(norm),*[float(v) for v in parts.values()]],device=ctx.device)
            if ctx.world_size>1:dist.all_reduce(value);value/=ctx.world_size
            row={'step':step+1,'loss':float(value[0]),'gradient_norm':float(value[1]),'parts':dict(zip(parts,map(float,value[2:]))),'elapsed_s':time.monotonic()-started}
            if ctx.primary:
                print(json.dumps(row),flush=True)
                with (a.output_dir/'train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
        del original,record,out,loss,labels
    if ctx.world_size>1:dist.barrier()
    if ctx.primary:
        torch.save({'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'step':a.steps,'config':receipt},a.output_dir/'final.pt')
        summary={'status':'SMOKE_COMPLETE' if a.smoke else 'COMPLETE','steps':a.steps,'variant':a.variant,'elapsed_s':time.monotonic()-started,'checkpoint_sha256':digest(a.output_dir/'final.pt'),'gradient_ownership':ownership}
        (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if ctx.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
