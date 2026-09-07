#!/usr/bin/env python3
"""Train the causal candidate selector exclusively on training-split caches."""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.nn.functional as F
from models.temporal_candidate_repair import TemporalCandidateRepair


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=32768)
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--loss-mode',choices=('capped','raw'),default='capped')
    args=p.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    device=torch.device('cuda')
    if not (args.cache/'train/complete.json').exists(): raise RuntimeError('training cache incomplete')
    files=sorted((args.cache/'train').glob('*.pt'))
    all_data=[]
    for f in files:
        record=torch.load(f,map_location='cpu',weights_only=False)
        all_data.append(record)
    sequences=sorted(set(x['identity']['sequence_id'] for x in all_data))
    # Sequence-held-out development partition inside the TRAIN manifest only.
    dev_sequences=set(sequences[::7])
    if len(sequences)<2: raise RuntimeError('need separate train/development sequences')
    def pack(dev):
        rows=[x for x in all_data if (x['identity']['sequence_id'] in dev_sequences)==dev]
        out={k:torch.cat([x[k] for x in rows]).to(device) for k in ('features','base','history','history_valid','base_valid','target','target_valid','old_valid')}
        # Uniform rows retain the primary role; focused disagreement examples have 1/3 weight.
        weights=[]
        for row in rows:
            n=row['target'].shape[0]
            weights.append(torch.cat((torch.full((n//2,1),1.5),torch.full((n-n//2,1),0.5))))
        out['weight']=torch.cat(weights).to(device)
        return out,[x['identity'] for x in rows]
    training,train_ids=pack(False)
    development,dev_ids=pack(True)
    del all_data
    args.output_dir.mkdir(parents=True,exist_ok=True)
    receipt={'schema':1,'base_frozen':True,'base_lineage':json.loads((args.cache/'lineage.json').read_text()),'cache_lineage_sha256':sha(args.cache/'lineage.json'),
        'cache':str(args.cache),'steps':args.steps,'seed':args.seed,'batch_size':args.batch_size,
        'loss_mode':args.loss_mode,'train_sequences':sorted(set(sequences)-dev_sequences),'development_sequences':sorted(dev_sequences),
        'training_samples':len(train_ids),'development_samples':len(dev_ids),
        'train_ids':train_ids,'development_ids':dev_ids,
        'checkpoint_selection':'fixed final optimizer step; shift chosen only on sequence-held-out TRAIN development data',
        'validation_used_for_training_or_selection':False,'source_sha256':sha(Path(__file__)),
        'model_source_sha256':sha(ROOT/'src/models/temporal_candidate_repair.py')}
    (args.output_dir/'training_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    model=TemporalCandidateRepair(channels=training['features'].shape[1]).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=0.0001)
    history=[]
    start=time.monotonic()
    for step in range(1,args.steps+1):
        index=torch.randint(len(training['features']),(args.batch_size,),device=device)
        b={k:v[index] for k,v in training.items()}
        optimizer.zero_grad(set_to_none=True)
        pred,gate=model(b['features'],b['base'],b['history'],b['history_valid'])
        weights=b['weight']*b['target_valid'].float()
        error=(pred-b['target']).abs()
        primary_error=error if args.loss_mode=='raw' else error.clamp_max(30)
        loss=(primary_error*weights).sum()/weights.sum().clamp_min(1)
        # Supervise which available candidate minimizes error. GT is used here only.
        delta=b['history']-b['base']
        safe=torch.where(delta.abs()>1e-3,delta,torch.ones_like(delta))
        optimal=((b['target']-b['base'])/safe).clamp(0,1)
        risk=delta.abs().clamp_max(10)*b['history_valid'].float()*weights
        selection=(F.binary_cross_entropy(gate.clamp(1e-6,1-1e-6),optimal,reduction='none')*risk).sum()/weights.sum().clamp_min(1)
        total=loss+0.05*selection
        total.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
        optimizer.step()
        lr=0.00005+0.00095*0.5*(1+math.cos(math.pi*step/args.steps))
        optimizer.param_groups[0]['lr']=lr
        if step==1 or step%100==0:
            row={'step':step,'loss':float(total.detach()),'epe_loss':float(loss.detach()),'selection':float(selection.detach()),
                 'gate_mean':float(gate.detach().mean()),'elapsed_s':time.monotonic()-start}
            history.append(row)
            print(json.dumps(row),flush=True)
    model.eval()
    shifts=[-12.,-6.,-4.,-3.,-2.,-1.,0.,1.]
    calibration=[]
    with torch.inference_mode():
        for shift in shifts:
            sums=torch.zeros(5,device=device,dtype=torch.float64)
            for begin in range(0,len(development['features']),65536):
                b={k:v[begin:begin+65536] for k,v in development.items()}
                pred,gate=model(b['features'],b['base'],b['history'],b['history_valid'],shift)
                weight=b['weight']*b['target_valid'].float()
                err=(pred-b['target']).abs()
                base_err=(b['base']-b['target']).abs()
                sums+=torch.stack(((err.clamp_max(10)*weight).sum(),(base_err.clamp_max(10)*weight).sum(),
                    weight.sum(),(err*weight).sum(),(base_err*weight).sum())).double()
            vals=sums.cpu().tolist()
            calibration.append({'shift':shift,'penalized_epe':vals[0]/vals[2],
                'base_penalized_epe':vals[1]/vals[2],'raw_epe':vals[3]/vals[2],'base_raw_epe':vals[4]/vals[2]})
    eligible=[r for r in calibration if r['raw_epe']<=r['base_raw_epe']]
    chosen=min(eligible or calibration,key=lambda r:r['penalized_epe'])
    payload={'model':model.cpu().state_dict(),'optimizer':optimizer.state_dict(),'step':args.steps,
        'config':receipt,'logit_shift':chosen['shift'],'calibration':calibration,
        'parameters':sum(p.numel() for p in model.parameters())}
    torch.save(payload,args.output_dir/'final.pt')
    report={'status':'COMPLETE','steps':args.steps,'elapsed_s':time.monotonic()-start,
        'chosen_development_calibration':chosen,'calibration':calibration,'parameters':payload['parameters'],
        'checkpoint_sha256':sha(args.output_dir/'final.pt')}
    (args.output_dir/'train.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in history))
    (args.output_dir/'training_summary.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report),flush=True)

if __name__=='__main__': main()
