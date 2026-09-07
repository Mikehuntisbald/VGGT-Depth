#!/usr/bin/env python3
"""Collect all completed arms, fixed-domain statistics and paired failure cases."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from models.temporal_candidate_repair import TemporalCandidateRepair
from models.controlled_temporal_repair import ControlledTemporalRepair
from tools.eval_temporal_candidate_repair import to_device


def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--v1-checkpoint',type=Path,required=True)
 args=p.parse_args();torch.set_num_threads(4)
 groups=('supervision','components','categorical','residual')
 reports={g:json.loads((args.run_dir/f'evaluation_{g}'/'metrics.json').read_text()) for g in groups}
 user_keys=('preserve_v1_all_gt_epe','preserve_v1_temporal','reduce_good_pixel_damage','reduce_large_5px','increase_fixed_opportunity_recovery')
 summaries={};confidence={};rows_out=[]
 for group,report in reports.items():
    rows=json.loads((args.run_dir/f'evaluation_{group}'/'per_sample.json').read_text())
    assert len(rows)==1294
    for arm,result in report['acceptance'].items():
        entry={'group':group,'status':result['status'],'five_user_conditions_pass':all(result['checks'][k] for k in user_keys),'checks':result['checks'],
            'metrics':{k:v['value'] for k,v in report['metrics'][arm].items()},'checkpoint':report['models'][arm]['checkpoint']}
        summaries[arm]=entry
        by_sequence={}
        for row in rows:
            data=by_sequence.setdefault(row['sequence_id'],{})
            for metric in ('all_gt_penalized_epe_px','temporal_matched_penalized_delta_epe_px','good_current_damage_rate','large_degradation_5px_rate','fixed_opportunity_recovery_rate'):
                reference=row['v1/'+metric];new=row[arm+'/'+metric]
                assert reference[1]==new[1]
                value=data.setdefault(metric,[0.,0]);value[0]+=reference[0]-new[0];value[1]+=reference[1]
        ci={};rng=np.random.default_rng(42)
        for metric in next(iter(by_sequence.values())):
            data=np.array([v[metric] for v in by_sequence.values()]);draws=rng.integers(0,len(data),(10000,len(data)));totals=data[draws].sum(1)
            gains=totals[:,0]/np.maximum(totals[:,1],1)
            ci[metric]={'v1_minus_arm':float(data[:,0].sum()/max(data[:,1].sum(),1)),'sequence_bootstrap_95pct':np.quantile(gains,[.025,.975]).tolist()}
        confidence[arm]=ci
 baseline={k:v['value'] for k,v in reports['supervision']['metrics']['v1'].items()}
 for group in groups:
    for k,v in reports[group]['metrics']['v1'].items():assert baseline[k]==v['value']
 output={'trained_arms':len(summaries),'updates_per_arm':3000,'validation_endpoints_per_arm':1294,'accepted_arms':[k for k,v in summaries.items() if v['five_user_conditions_pass']],
         'main_model_retained':'v1','baseline_v1':baseline,'arms':summaries,'sequence_bootstrap':confidence}
 (args.run_dir/'controlled_results.json').write_text(json.dumps(output,indent=2)+'\n')
 import csv
 with (args.run_dir/'controlled_results.csv').open('w') as handle:
    keys=['all_gt_penalized_epe_px','temporal_matched_penalized_delta_epe_px','good_current_damage_rate','large_degradation_1px_rate','large_degradation_5px_rate','fixed_opportunity_recovery_rate','recoverable_to_1px_rate','fg_bg_spurious_mixing_rate']
    writer=csv.writer(handle);writer.writerow(['arm','group','accepted',*keys]);writer.writerow(['v1','baseline',True,*[baseline[k] for k in keys]])
    for arm,item in summaries.items():writer.writerow([arm,item['group'],item['five_user_conditions_pass'],*[item['metrics'][k] for k in keys]])
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 payload=torch.load(args.v1_checkpoint,map_location='cpu',weights_only=False);v1=TemporalCandidateRepair().cuda().eval();v1.load_state_dict(payload['model'])
 figures=args.run_dir/'failure_cases';figures.mkdir(exist_ok=True);case_receipt=[]
 with torch.inference_mode():
  for arm,group in (('matching_mild','components'),('categorical_mild','categorical')):
    record=reports[group];checkpoint=Path(record['models'][arm]['checkpoint']);p=torch.load(checkpoint,map_location='cpu',weights_only=False)
    model=ControlledTemporalRepair(v1,**p['config']['components']).cuda().eval();model.load_state_dict(p['model'])
    for kind in ('new_degradation','recovery_over_v1'):
        case=record['cases'][arm][kind][0];index=case['dataset_index']
        data=to_device(torch.load(args.cache/'validation'/f'{index:06d}.pt',map_location='cpu',weights_only=False),'cuda')
        bank=to_device(torch.load(args.bank/'validation'/f'{index:06d}.pt',map_location='cpu',weights_only=False)['current'],'cuda')
        c=data['current'];gt=data['target'];old,_=v1(c['features'],c['base'],c['history'],c['history_valid'],payload['logit_shift'])
        result=model(c['features'],c['base'],c['history'],c['history_valid'],bank,p['logit_shift']);pred=result['prediction']
        x,y=case['x'],case['y'];h,w=gt.shape[-2:];x0,x1=max(x-64,0),min(x+65,w);y0,y1=max(y-64,0),min(y+65,h)
        def array(value):return value.detach().cpu().numpy()
        maps=[('RGB',array(data['rgb'][0].permute(1,2,0))),('GT',array(gt[0,0])),('Original history',array(c['history'][0,0])),('v1',array(old[0,0])),
          (arm,array(pred[0,0])),('v1 error',array((old-gt).abs()[0,0])),('New error',array((pred-gt).abs()[0,0])),('History weight',array(result['gate'][0,0]))]
        fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
        limit=float(np.percentile(maps[1][1][y0:y1,x0:x1],98));errlimit=min(50,max(case['v1_error_px'],case['new_error_px'],2))
        for ax,(name,value) in zip(axes.flat,maps):
            patch=value[y0:y1,x0:x1]
            if name=='RGB':ax.imshow(patch)
            else:
                vmax=1 if name=='History weight' else errlimit if 'error' in name.lower() else limit
                im=ax.imshow(patch,vmin=0,vmax=vmax,cmap='magma' if 'error' in name.lower() else 'viridis');fig.colorbar(im,ax=ax,fraction=.046,pad=.02)
            ax.plot(x-x0,y-y0,'r+');ax.set_title(name);ax.axis('off')
        fig.suptitle(f"{kind}: {arm}, sequence {case['sequence_id']} frame {case['frame_id']} ({x},{y})\nv1 error {case['v1_error_px']:.3f} px → new {case['new_error_px']:.3f} px")
        filename=f'{arm}_{kind}_{index:06d}.png';fig.savefig(figures/filename,dpi=130);plt.close(fig)
        case_receipt.append({**case,'figure':filename})
 (figures/'cases.json').write_text(json.dumps(case_receipt,indent=2)+'\n')
 print(json.dumps({'trained_arms':len(summaries),'accepted_arms':output['accepted_arms'],'retained':'v1'}),flush=True)

if __name__=='__main__':main()
