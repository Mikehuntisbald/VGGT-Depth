#!/usr/bin/env python3
"""Cache actual frozen A5 FFS learned matching evidence on immutable v1 samples."""
from __future__ import annotations
import argparse,copy,hashlib,inspect,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader,Subset,DistributedSampler
from data.raw_stereo_video_dataset import collate_raw_stereo_video_samples
from models.learned_temporal_evidence import learned_evidence,distinct_peak,_sample
from models.temporal_candidate_repair import sample_right
from tools.cache_temporal_candidate_repair import candidate,cpu,atomic_save
from tools.eval_metric_stereo_video import _load_model,_previous_prefix_batch
from tools.train_metric_stereo_video import _dataset,_distributed_context,_read_config,_move_batch,_sha256


def flatten_bank(bank,take):
    return {'learned':bank['learned'].permute(0,2,3,1).reshape(-1,24)[take]}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--config',type=Path,required=True);p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--splits',default='train,validation');p.add_argument('--num-workers',type=int,default=2)
    p.add_argument('--max-batches',type=int)
    args=p.parse_args();context=_distributed_context();torch.set_num_threads(4)
    config=_read_config(args.config)
    signature={'schema':1,'v1_cache_lineage_sha256':_sha256(args.cache/'lineage.json'),
      'base_checkpoint_manifest_sha256':_sha256(args.checkpoint/'manifest.json'),
      'feature_functions_sha256':hashlib.sha256(''.join(inspect.getsource(f) for f in (learned_evidence,distinct_peak,_sample)).encode()).hexdigest(),
      'script_sha256':_sha256(Path(__file__)),'max_batches':args.max_batches,
      'feature_scale_hr':8,'local_offsets_feature_pixels':[-1,0,1],'channels':24,
      'features':'actual A5 FFS matching pyramid 0, normalized cosine/L1 and distinct-peak ambiguity',
      'gt_used_as_input':False,'preserve_v1_training_samples':True,'torch':torch.__version__,'cuda':torch.version.cuda}
    args.output_dir.mkdir(parents=True,exist_ok=True);lineage=args.output_dir/'lineage.json'
    if lineage.exists() and json.loads(lineage.read_text())!=signature:raise RuntimeError('evidence lineage mismatch')
    if context.primary:lineage.write_text(json.dumps(signature,indent=2)+'\n')
    if context.world_size>1:dist.barrier()
    model=None;start=time.monotonic()
    for split in args.splits.split(','):
        training=split=='train'
        dataset=_dataset(config,training=training)
        plan=json.loads((args.cache/split/'plan.json').read_text());indices=plan['indices']
        subset=Subset(dataset,indices)
        sampler=DistributedSampler(subset,num_replicas=context.world_size,rank=context.rank,shuffle=False)
        loader=DataLoader(subset,sampler=sampler,batch_size=1,num_workers=args.num_workers,collate_fn=collate_raw_stereo_video_samples,pin_memory=True)
        directory=args.output_dir/split;directory.mkdir(exist_ok=True)
        if context.primary:(directory/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
        if model is None:
            model=_load_model(config,args.checkpoint,context);model.requires_grad_(False)
            captured=[]
            def capture(_module,_inputs,output):
                if not captured:captured.append(output[0].detach().clone())
            modules=[(n,m) for n,m in model.named_modules() if 'stereo_backbone' in n and n.endswith('model.feature')]
            if len(modules)!=1:raise RuntimeError(f'expected unique FFS feature module: {[n for n,m in modules]}')
            handle=modules[0][1].register_forward_hook(capture)
        written=0
        with torch.inference_mode():
            for i,raw in enumerate(loader):
                if args.max_batches and i>=args.max_batches:break
                index=int(raw['identity_metadata'][0]['dataset_index']);owner=indices.index(index)%context.world_size==context.rank
                batch=_move_batch(raw,context.device)
                old=torch.load(args.cache/split/f'{index:06d}.pt',map_location='cpu',weights_only=False)
                captured.clear()
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    current_output=copy.deepcopy(model(batch))
                feature=captured[0];frames=batch['rgb'].shape[1]
                if feature.shape[0]!=2*frames:raise RuntimeError('FFS stereo feature ordering changed')
                if training:
                    captured.clear()
                    with torch.autocast('cuda',dtype=torch.bfloat16):previous=model(_previous_prefix_batch(batch))
                    current=candidate(current_output,previous,batch)
                    bank=learned_evidence(feature[frames-1:frames],feature[2*frames-1:2*frames],current)
                    gt=batch['disparity_gt_left_px'][:,-1];valid=batch['valid_gt_left'][:,-1].bool()&torch.isfinite(gt)&(gt>0)
                    hard=torch.where((valid&current['history_valid']&((current['base']-current['history']).abs()>.25)).flatten())[0]
                    generator=torch.Generator(device=context.device).manual_seed(42000+index)
                    uniform=torch.randint(gt.numel(),(4096,),device=context.device,generator=generator)
                    focused=hard[torch.randint(len(hard),(4096,),device=context.device,generator=generator)] if len(hard) else uniform
                    take=torch.cat((uniform,focused))
                    if owner:
                        torch.testing.assert_close(current['base'].flatten()[take,None].cpu(),old['base'],rtol=0,atol=0)
                        torch.testing.assert_close(current['history'].flatten()[take,None].cpu(),old['history'],rtol=0,atol=0)
                        old_features=current['features'].permute(0,2,3,1).reshape(-1,31)[take].half().cpu()
                        torch.testing.assert_close(old_features,old['features'],rtol=0,atol=0)
                        saved={'current':flatten_bank(bank,take),'identity':old['identity'],'v1_original_samples_exact':True}
                elif owner:
                    from tools.eval_temporal_candidate_repair import to_device
                    current=to_device(old['current'],context.device);previous=to_device(old['previous'],context.device)
                    saved={'current':learned_evidence(feature[frames-1:frames],feature[2*frames-1:2*frames],current),
                           'previous':learned_evidence(feature[frames-2:frames-1],feature[2*frames-2:2*frames-1],previous),
                           'identity':old['identity']}
                if owner:
                    atomic_save(cpu(saved),directory/f'{index:06d}.pt');written+=1
                if i%20==0:print(json.dumps({'rank':context.rank,'split':split,'batch':i,'batches':len(loader),'elapsed_s':time.monotonic()-start}),flush=True)
        if context.world_size>1:dist.barrier()
        if context.primary:
            actual=len(list(directory.glob('*.pt')));complete=actual==len(indices)
            receipt={'status':'COMPLETE' if complete else 'SMOKE_COMPLETE','samples':actual,'expected':len(indices),'elapsed_s':time.monotonic()-start}
            (directory/('complete.json' if complete else 'smoke_complete.json')).write_text(json.dumps(receipt,indent=2)+'\n')
    if context.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
