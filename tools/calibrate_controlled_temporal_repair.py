#!/usr/bin/env python3
"""Fine calibration on the original TRAIN-development sequences only."""
from __future__ import annotations
import argparse,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from models.temporal_candidate_repair import TemporalCandidateRepair
from models.controlled_temporal_repair import ControlledTemporalRepair
from tools.train_controlled_temporal_repair import stats,merge,finish,selection


def main():
 p=argparse.ArgumentParser(description=__doc__)
 p.add_argument('--cache',type=Path,required=True);p.add_argument('--bank',type=Path,required=True)
 p.add_argument('--v1-checkpoint',type=Path,required=True);p.add_argument('--checkpoints',type=Path,nargs='+',required=True);p.add_argument('--output-dir',type=Path,required=True)
 args=p.parse_args();torch.set_num_threads(4)
 vp=torch.load(args.v1_checkpoint,map_location='cpu',weights_only=False);v1=TemporalCandidateRepair().cuda().eval();v1.load_state_dict(vp['model'])
 ids=vp['config']['development_ids'];rows=[];banks=[]
 for identity in ids:
    name=f"{identity['dataset_index']:06d}.pt"
    r=torch.load(args.cache/'train'/name,map_location='cpu',weights_only=False)
    assert r['identity']['sequence_id'] in vp['config']['development_sequences']
    rows.append(r);banks.append(torch.load(args.bank/'train'/name,map_location='cpu',weights_only=False)['current'])
 data={k:torch.cat([r[k] for r in rows]).cuda() for k in ('features','base','history','history_valid','base_valid','target','target_valid')}
 data['weight']=torch.cat([torch.cat((torch.full((4096,1),1.5),torch.full((4096,1),.5))) for r in rows]).cuda()
 data['bank']={k:torch.cat([b[k] for b in banks]).cuda() for k in banks[0]}
 del rows,banks
 def batch(begin):return {k:({n:x[begin:begin+65536] for n,x in v.items()} if isinstance(v,dict) else v[begin:begin+65536]) for k,v in data.items()}
 total={}
 with torch.inference_mode():
    for begin in range(0,len(data['features']),65536):
        b=batch(begin);prediction,_=v1(b['features'],b['base'],b['history'],b['history_valid'],vp['logit_shift']);merge(total,stats(prediction,b))
 baseline=finish(total);summaries=[];args.output_dir.mkdir(parents=True,exist_ok=True)
 for checkpoint in args.checkpoints:
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=payload['config']
    model=ControlledTemporalRepair(v1,**cfg['components']).cuda().eval();model.load_state_dict(payload['model'])
    # Cache gate logits once; calibration changes only the scalar shift.
    logits=[];histories=[];validity=[]
    with torch.inference_mode():
        for begin in range(0,len(data['features']),65536):
            b=batch(begin);out=model(b['features'],b['base'],b['history'],b['history_valid'],b['bank'],0.)
            logits.append(torch.logit(out.get('soft_gate',out['gate']).clamp(1e-7,1-1e-7)))
            histories.append(out['history']);validity.append(out['history_valid'])
    logits=torch.cat(logits);histories=torch.cat(histories);validity=torch.cat(validity)
    calibration=[]
    with torch.inference_mode():
        for j in range(-20,21):
            shift=j*.05;total={}
            for begin in range(0,len(data['features']),65536):
                b=batch(begin);gate=torch.sigmoid(logits[begin:begin+65536]+shift)*validity[begin:begin+65536]
                if cfg['components'].get('categorical',False):
                    gate=torch.where((histories[begin:begin+65536]-b['base']).abs()>5,(gate>=.5).float(),gate)*validity[begin:begin+65536]
                pred=b['base']+gate*(torch.where(validity[begin:begin+65536],histories[begin:begin+65536],b['base'])-b['base'])
                merge(total,stats(pred,b))
            values=finish(total);feasible,violation,score=selection(values,baseline)
            calibration.append({'shift':shift,'metrics':values,'feasible':feasible,'violation':violation,'score':score})
    chosen=min(calibration,key=lambda r:(not r['feasible'],r['violation'],r['score']))
    # Recompute the selected setting through the real forward, without logit reconstruction.
    exact={}
    with torch.inference_mode():
        for begin in range(0,len(data['features']),65536):
            b=batch(begin);out=model(b['features'],b['base'],b['history'],b['history_valid'],b['bank'],chosen['shift']);merge(exact,stats(out['prediction'],b))
    chosen['metrics']=finish(exact);chosen['feasible'],chosen['violation'],chosen['score']=selection(chosen['metrics'],baseline)
    payload['logit_shift']=chosen['shift'];payload['calibration']=calibration
    payload['fine_calibration']={'data_split':'TRAIN-development only','development_sequences':vp['config']['development_sequences'],
        'source_checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'source':str(checkpoint),
        'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'optimizer_updates':0,'chosen':chosen,'created_unix_s':time.time()}
    outdir=args.output_dir/cfg['arm'];outdir.mkdir(exist_ok=True);torch.save(payload,outdir/'final.pt')
    summary={'arm':cfg['arm'],'status':'CALIBRATED','chosen':chosen,'baseline':baseline,'checkpoint_sha256':hashlib.sha256((outdir/'final.pt').read_bytes()).hexdigest(),'source_checkpoint':str(checkpoint)}
    (outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');summaries.append(summary)
    print(json.dumps(summary),flush=True)
 selected=min(summaries,key=lambda r:(not r['chosen']['feasible'],r['chosen']['violation'],r['chosen']['score']))
 (args.output_dir/'selection.json').write_text(json.dumps({'selected_arm':selected['arm'],'development_feasible':selected['chosen']['feasible'],'validation_used_for_selection':False,'summaries':summaries},indent=2)+'\n')

if __name__=='__main__':main()
