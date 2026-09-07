#!/usr/bin/env python3
"""Train budget-matched history supervision/component arms from immutable v1."""
from __future__ import annotations
import argparse,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from models.temporal_candidate_repair import TemporalCandidateRepair
from models.controlled_temporal_repair import ControlledTemporalRepair,supervision_loss

PROFILES={
 'guarded':dict(recovery=.02,rejection=.02,regret=.05,tail=.1,v1_regret=.2,good_guard=.5,sparse=.001,wide_rejection=True),
 'control':dict(recovery=0.,rejection=0.,regret=0.,tail=0.),
 'mild':dict(recovery=.02,rejection=.02,regret=.05,tail=.1),
 'balanced':dict(recovery=.05,rejection=.05,regret=.15,tail=.3),
 'recover':dict(recovery=.1,rejection=.04,regret=.1,tail=.3),
 'protect':dict(recovery=.04,rejection=.1,regret=.25,tail=.5),
}

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def stats(pred,b):
    target=b['target'];be=(b['base']-target).abs();he=(b['history']-target).abs();error=(pred-target).abs()
    weight=b['weight']*b['target_valid'].float()
    opportunity=b['history_valid']&(be-he>0.1)
    good=be<.1
    bad=b['history_valid']&(be<.1)&(he>1)
    items={
      'penalized_epe':(torch.where(b['base_valid'],error.clamp_max(10),torch.full_like(error,10)),weight),
      'raw_epe':(error,weight),
      'good_damage':((error-be>.1).float(),weight*good),
      'large_1px':((error-be>1).float(),weight),
      'large_5px':((error-be>5).float(),weight),
      'recovery':((be-error>.1).float(),weight*opportunity),
      'opportunity_epe':(error,weight*opportunity),
      'rejection_failure':((error-be>.1).float(),weight*bad),
    }
    return {k:torch.stack(((value*w).double().sum(),w.double().sum())) for k,(value,w) in items.items()}

def merge(total,values):
    for key,value in values.items():total[key]=total.get(key,0)+value


def finish(total):return {k:float(v[0]/v[1].clamp_min(1)) for k,v in total.items()}


def selection(candidate,baseline):
    ratios={k:candidate[k]/max(baseline[k],1e-8) for k in baseline}
    feasible=ratios['penalized_epe']<=1 and ratios['good_damage']<1 and ratios['large_5px']<1 and ratios['recovery']>1
    violation=10*max(ratios['penalized_epe']-1,0)+max(ratios['good_damage']-1,0)+max(ratios['large_5px']-1,0)+2*max(1-ratios['recovery'],0)
    score=ratios['penalized_epe']+.1*ratios['good_damage']+.1*ratios['large_5px']-.1*ratios['recovery']
    return feasible,violation,score


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--bank',type=Path)
    p.add_argument('--v1-checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--profiles',default='control,mild,balanced,recover,protect')
    p.add_argument('--components',default='none');p.add_argument('--steps',type=int,default=3000)
    p.add_argument('--batch-size',type=int,default=32768);p.add_argument('--seed',type=int,default=42)
    args=p.parse_args();torch.set_num_threads(4)
    if not (args.cache/'train/complete.json').exists():raise RuntimeError('incomplete v1 train cache')
    if args.bank and not (args.bank/'train/complete.json').exists():raise RuntimeError('incomplete evidence bank')
    v1_payload=torch.load(args.v1_checkpoint,map_location='cpu',weights_only=False)
    if sha(args.cache/'lineage.json')!=v1_payload['config']['cache_lineage_sha256']:raise RuntimeError('v1 cache mismatch')
    v1=TemporalCandidateRepair().cuda().eval();v1.load_state_dict(v1_payload['model'])
    rows=[]
    for path in sorted((args.cache/'train').glob('*.pt')):
        row=torch.load(path,map_location='cpu',weights_only=False)
        if args.bank:
            bank=torch.load(args.bank/'train'/path.name,map_location='cpu',weights_only=False)
            if bank['identity']['dataset_index']!=row['identity']['dataset_index']:raise RuntimeError('sample identity mismatch')
            row['bank']=bank['current']
        rows.append(row)
    dev_sequences=set(v1_payload['config']['development_sequences'])
    def pack(dev):
        selected=[r for r in rows if (r['identity']['sequence_id'] in dev_sequences)==dev]
        out={k:torch.cat([r[k] for r in selected]).cuda() for k in ('features','base','history','history_valid','base_valid','target','target_valid')}
        out['weight']=torch.cat([torch.cat((torch.full((len(r['target'])//2,1),1.5),torch.full((len(r['target'])-len(r['target'])//2,1),.5))) for r in selected]).cuda()
        if args.bank:out['bank']={k:torch.cat([r['bank'][k] for r in selected]).cuda() for k in selected[0]['bank']}
        return out,[r['identity'] for r in selected]
    training,train_ids=pack(False);development,dev_ids=pack(True);del rows
    def subset(data,index):return {k:({name:value[index] for name,value in val.items()} if isinstance(val,dict) else val[index]) for k,val in data.items()}
    v1_total={}
    with torch.inference_mode():
        for begin in range(0,len(development['features']),65536):
            b=subset(development,slice(begin,begin+65536));pred,_=v1(b['features'],b['base'],b['history'],b['history_valid'],v1_payload['logit_shift'])
            merge(v1_total,stats(pred,b))
    reference=finish(v1_total);args.output_dir.mkdir(parents=True,exist_ok=True)
    summaries=[]
    for component in args.components.split(','):
      for profile in args.profiles.split(','):
        torch.manual_seed(args.seed)
        flags={'matching':component in ('matching','combined','residual_matching','residual_both'),'alignment':component in ('alignment','combined'),'ambiguity':component in ('ambiguity','combined','categorical','residual_ambiguity','residual_both'),'categorical':component=='categorical','residual':component.startswith('residual_')}
        if any(flags.values()) and not args.bank:raise RuntimeError('component requires evidence bank')
        model=ControlledTemporalRepair(v1,**flags).cuda().train()
        optimizer=torch.optim.AdamW(model.parameters(),lr=3e-4,weight_decay=1e-4)
        arm=f'{component}_{profile}';outdir=args.output_dir/arm;outdir.mkdir(exist_ok=True)
        receipt={'arm':arm,'profile':profile,'coefficients':PROFILES[profile],'components':flags,'steps':args.steps,'seed':args.seed,'batch_size':args.batch_size,
            'initial_checkpoint_sha256':sha(args.v1_checkpoint),'cache_lineage_sha256':sha(args.cache/'lineage.json'),
            'bank_lineage_sha256':sha(args.bank/'lineage.json') if args.bank else None,
            'training_ids':train_ids,'development_ids':dev_ids,'validation_used_for_selection':False,
            'learning_rate':3e-4,'weight_decay':1e-4,'initialization':'all arms exactly v1; added feature weights zero',
            'training_source_sha256':sha(Path(__file__)),'model_source_sha256':sha(ROOT/'src/models/controlled_temporal_repair.py')}
        (outdir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        history=[];start=time.monotonic()
        for step in range(1,args.steps+1):
            index=torch.randint(len(training['features']),(args.batch_size,),device='cuda');b=subset(training,index)
            optimizer.zero_grad(set_to_none=True)
            output=model(b['features'],b['base'],b['history'],b['history_valid'],b.get('bank'))
            loss,parts=supervision_loss(output,b,PROFILES[profile]);loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),1.);optimizer.step()
            optimizer.param_groups[0]['lr']=1.5e-5+2.85e-4*.5*(1+math.cos(math.pi*step/args.steps))
            if step==1 or step%500==0:
                log={'step':step,'loss':float(loss.detach()),**{k:float(v.detach()) for k,v in parts.items()},'elapsed_s':time.monotonic()-start}
                print(json.dumps({'arm':arm,**log}),flush=True);history.append(log)
        model.eval();calibration=[]
        with torch.inference_mode():
            for shift in (-1.,-.5,0.,.5,1.):
                total={}
                for begin in range(0,len(development['features']),65536):
                    b=subset(development,slice(begin,begin+65536))
                    out=model(b['features'],b['base'],b['history'],b['history_valid'],b.get('bank'),shift)
                    merge(total,stats(out['prediction'],b))
                result=finish(total);feasible,violation,score=selection(result,reference)
                calibration.append({'shift':shift,'metrics':result,'feasible':feasible,'violation':violation,'score':score})
        chosen=min(calibration,key=lambda r:(not r['feasible'],r['violation'],r['score']))
        payload={'model':{k:v.cpu() for k,v in model.state_dict().items()},'optimizer':optimizer.state_dict(),'step':args.steps,
                 'config':receipt,'logit_shift':chosen['shift'],'development_baseline':reference,'calibration':calibration}
        torch.save(payload,outdir/'final.pt')
        summary={'status':'COMPLETE','arm':arm,'steps':args.steps,'chosen':chosen,'baseline_v1':reference,'checkpoint_sha256':sha(outdir/'final.pt'),'elapsed_s':time.monotonic()-start}
        (outdir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
        (outdir/'train.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in history))
        summaries.append(summary);print(json.dumps(summary),flush=True)
        del model,optimizer
    best=min([r for r in summaries if r['arm']!='none_control'] or summaries,key=lambda r:(not r['chosen']['feasible'],r['chosen']['violation'],r['chosen']['score']))
    selection_receipt={'status':'COMPLETE','selected_arm':best['arm'],'selected_profile':best['arm'].split('_')[-1],
        'development_feasible':best['chosen']['feasible'],'selection_uses_validation':False,'arms':summaries,'created_unix_s':time.time()}
    (args.output_dir/'selection.json').write_text(json.dumps(selection_receipt,indent=2)+'\n')
    print(json.dumps({'selected_arm':best['arm'],'development_feasible':best['chosen']['feasible']}),flush=True)

if __name__=='__main__':main()
