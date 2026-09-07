#!/usr/bin/env python3
"""Train native architectural controls using complete causal image clips."""
from __future__ import annotations
import argparse,hashlib,json,math,os,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'third_party/Fast-FoundationStereo')]
import Utils
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from models.native_temporal_stereo.model import NativeTemporalStereo
from models.native_temporal_stereo.training import native_loss
from tools.train_metric_stereo_video import _distributed_context


def move(value,device):
    if isinstance(value,torch.Tensor):return value.to(device,non_blocking=True)
    if isinstance(value,dict):return {k:move(v,device) for k,v in value.items()}
    if isinstance(value,(tuple,list)):return [move(v,device) for v in value]
    return value


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--cache',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--variant',choices=('late','early_seed','early_state'),required=True);p.add_argument('--steps',type=int,default=1000)
    p.add_argument('--smoke',action='store_true');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();context=_distributed_context();torch.set_num_threads(4);Utils.AMP_DTYPE=torch.bfloat16
    if not a.smoke and not (a.cache/'train/complete.json').exists():raise RuntimeError('training image evidence incomplete')
    if not a.smoke and not (a.cache/'training_labels/complete.json').exists():raise RuntimeError('last-two-frame supervision incomplete')
    package=torch.load(a.cache/'initialization.pt',map_location='cpu',weights_only=False)
    specification=json.loads((a.cache/'structure.json').read_text())
    package.update(hidden_channels=specification['hidden_channels'],volume_channels=specification['volume_channels'])
    torch.manual_seed(a.seed)
    model=NativeTemporalStereo(package,variant=a.variant).to(context.device).train()
    groups=[dict(params=[p for p in model.geometry.parameters() if p.requires_grad],lr=1e-5,name='geometry')]
    if model.initializer is not None:
        if not model.initializer.propagate_hidden:model.initializer.state_fusion.requires_grad_(False)
        groups.append(dict(params=[p for p in model.initializer.parameters() if p.requires_grad],lr=1e-4,name='temporal_matcher'))
    optimizer=torch.optim.AdamW(groups,weight_decay=.01)
    wrapped=DDP(model,device_ids=[context.device.index],find_unused_parameters=True) if context.world_size>1 else model
    paths=sorted((a.cache/'train').glob('*.pt'));development={'0001','0011','0020','0036','0045'}
    # Read identities only once; encoded tensors are loaded on demand below.
    identity_file=a.cache/'train/identities.json'
    if identity_file.exists():identities=json.loads(identity_file.read_text())
    else:
        identities={}
        for path in paths:
            row=torch.load(path,map_location='cpu',weights_only=False);identities[path.name]=row['identity']
        if context.primary:identity_file.write_text(json.dumps(identities,indent=2)+'\n')
    train=[p for p in paths if identities[p.name]['sequence_id'] not in development]
    if a.smoke and not train:train=paths
    if not train:raise RuntimeError('no training sequences after development separation')
    dev=[p for p in paths if identities[p.name]['sequence_id'] in development]
    a.output_dir.mkdir(parents=True,exist_ok=True)
    receipt={'variant':a.variant,'steps':a.steps,'world_size':context.world_size,'seed':a.seed,'training_endpoints':len(train),'development_endpoints':len(dev),
       'training_sequences':sorted({identities[p.name]['sequence_id'] for p in train}),'development_sequences':sorted(development),
       'cache_lineage_sha256':digest(a.cache/'lineage.json'),'initialization_sha256':digest(a.cache/'initialization.pt'),'structure_sha256':digest(a.cache/'structure.json'),
       'optimizer':'AdamW','geometry_lr':1e-5,'matcher_lr':1e-4,'weight_decay':.01,'gradient_clip':1.,'final_checkpoint_selection':'fixed final step, no validation tuning',
       'trainable_parameters':{g['name']:sum(p.numel() for p in g['params']) for g in groups},'full_image_training':True,'clip_length':8,'truncated_bptt_frames':2,
       'frozen_parts':'FFS/VGGT image encoding and pretrained FFS iterative solver; gradients through solver to latent initialization retained',
       'history_source':'this model own preceding geometry/latent state, never a fixed A5 output cache','v1_head_used':False,'A5_prediction_as_input':False,
       'loss_source_sha256':digest(ROOT/'src/models/native_temporal_stereo/training.py'),'model_source_sha256':digest(ROOT/'src/models/native_temporal_stereo/model.py'),'refiner_source_sha256':digest(ROOT/'src/models/native_temporal_stereo/refiner.py'),'trainer_sha256':digest(Path(__file__)),
       'training_labels_receipt_sha256':digest(a.cache/'training_labels/complete.json') if (a.cache/'training_labels/complete.json').exists() else None}
    if context.primary:(a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    generator=torch.Generator().manual_seed(a.seed)
    indices=torch.randint(len(train),(a.steps,context.world_size),generator=generator)
    started=time.monotonic();history=[];gradient_ownership={}
    for step in range(a.steps):
        tick=time.monotonic()
        path=train[int(indices[step,context.rank])]
        record=torch.load(path,map_location='cpu',weights_only=False)
        label_path=a.cache/'training_labels'/path.name
        if label_path.exists():
            label=torch.load(label_path,map_location='cpu',weights_only=False)
            assert label['dataset_index']==record['identity']['dataset_index']
            for key in ('target','valid','stereo_matched'):
                record['labels'][key]=record['labels'][key].clone()
                record['labels'][key][:,-2:]=label[key]
        record=move(record,context.device)
        loaded=time.monotonic()
        if a.smoke and 'stereo_matched' not in record['labels']:record['labels']['stereo_matched']=record['labels']['valid']
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast('cuda',dtype=torch.bfloat16):out=wrapped(record['inputs']);loss,parts=native_loss(out,record)
        if not loss.isfinite():raise RuntimeError('nonfinite structural training loss')
        forwarded=time.monotonic()
        if a.smoke and step==0:
            print(json.dumps({'debug_parts':{k:float(v) for k,v in parts.items()},'valid_fraction':float(record['labels']['valid'].float().mean()),'initializer_requires_grad':{n:p.requires_grad for n,p in model.named_parameters() if 'initializer' in n},'probability_requires_grad':out['details'][-1]['initialization']['probability'].requires_grad if model.initializer is not None else None,'model_training':model.training}),flush=True)
        loss.backward()
        if a.smoke:torch.cuda.synchronize()
        backwarded=time.monotonic()
        if step==0:
            gradient_ownership={name:float(p.grad.float().norm()) for name,p in model.named_parameters() if p.grad is not None and ('initializer' in name or name.startswith('refiner'))}
            if any(name.startswith('refiner') for name in gradient_ownership):raise RuntimeError('frozen stereo solver received parameter gradients')
            if a.smoke:print(json.dumps({'gradient_ownership':gradient_ownership}),flush=True)
            if model.initializer is not None and not any(v>0 for v in gradient_ownership.values()):raise RuntimeError('no gradient reached temporal initialization')
        norm=torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad],1.)
        optimizer.step()
        if a.smoke:print(json.dumps({'rank':context.rank,'step':step+1,'load_s':loaded-tick,'forward_s':forwarded-loaded,'backward_s':backwarded-forwarded,'update_s':time.monotonic()-backwarded}),flush=True)
        for group in optimizer.param_groups:
            base_lr=1e-5 if group['name']=='geometry' else 1e-4
            group['lr']=base_lr*(.1+.9*.5*(1+math.cos(math.pi*(step+1)/a.steps)))
        if step==0 or (step+1)%25==0 or step+1==a.steps:
            values=torch.tensor([float(loss.detach()),float(norm),*[float(v) for v in parts.values()]],device=context.device)
            if context.world_size>1:dist.all_reduce(values);values/=context.world_size
            row={'step':step+1,'loss':float(values[0]),'gradient_norm':float(values[1]),'parts':dict(zip(parts,map(float,values[2:]))),'elapsed_s':time.monotonic()-started}
            if context.primary:
                print(json.dumps(row),flush=True);history.append(row)
                with (a.output_dir/'train.jsonl').open('a') as handle:handle.write(json.dumps(row)+'\n')
        del out,record,loss
    if context.world_size>1:dist.barrier()
    if context.primary:
        payload={'model':{k:v.detach().cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'step':a.steps,'config':receipt}
        torch.save(payload,a.output_dir/'final.pt')
        summary={'status':'SMOKE_COMPLETE' if a.smoke else 'COMPLETE','steps':a.steps,'variant':a.variant,'elapsed_s':time.monotonic()-started,'checkpoint_sha256':digest(a.output_dir/'final.pt'),'gradient_ownership':gradient_ownership,'trainable_parameters':receipt['trainable_parameters']}
        (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps(summary),flush=True)
    if context.world_size>1:dist.destroy_process_group()
if __name__=='__main__':main()
