#!/usr/bin/env python3
"""Create paired per-sequence evidence and scientific failure visualizations."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from models.temporal_candidate_repair import TemporalCandidateRepair


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evaluation',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    args=p.parse_args()
    report=json.loads((args.evaluation/'metrics.json').read_text())
    rows=json.loads((args.evaluation/'per_sample.json').read_text())
    by_sequence={}
    for row in rows:
        seq=by_sequence.setdefault(row['sequence_id'],{})
        for k,v in row.items():
            if '/' in k:
                sums=seq.setdefault(k,[0.,0]);sums[0]+=v[0];sums[1]+=v[1]
    rng=np.random.default_rng(42)
    ci={}
    for metric in ('all_gt_penalized_epe_px','epe_px','temporal_matched_penalized_delta_epe_px','opportunity_epe_px'):
        pairs=[]
        for seq,values in by_sequence.items():
            a=values['A5_frozen/'+metric];b=values['A5_temporal_repair/'+metric]
            if a[1]!=b[1]:raise RuntimeError(f'paired pixel support differs: {metric}, {seq}')
            pairs.append([a[0]-b[0],a[1]])
        array=np.array(pairs)
        draws=rng.integers(0,len(array),(10000,len(array)))
        totals=array[draws].sum(1)
        gains=totals[:,0]/totals[:,1]
        ci[metric]={'gain_px':float(array[:,0].sum()/array[:,1].sum()),
            'sequence_bootstrap_95pct':[float(x) for x in np.quantile(gains,[0.025,0.975])],
            'sequences':len(array),'bootstrap_draws':10000,'seed':42}
    summary={'per_sequence':by_sequence,'paired_gain_confidence_intervals':ci}
    (args.evaluation/'paired_sequence_analysis.json').write_text(json.dumps(summary,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    model=TemporalCandidateRepair().eval()
    model.load_state_dict(ckpt['model'])
    torch.set_num_threads(4)
    output=args.evaluation/'failure_cases';output.mkdir(exist_ok=True)
    selected=[]
    for label in ('top_rescues','top_failures'):
        seen=set()
        for row in report[label]:
            if row['dataset_index'] not in seen:
                selected.append((label,row));seen.add(row['dataset_index'])
            if len(seen)==3:break
    with torch.inference_mode():
        for label,row in selected:
            record=torch.load(args.cache/'validation'/f"{row['dataset_index']:06d}.pt",map_location='cpu',weights_only=False)
            c=record['current'];gt=record['target'];mask=record['target_valid'][0,0].numpy()
            pred,gate=model(c['features'],c['base'],c['history'],c['history_valid'],ckpt['logit_shift'])
            x,y=row['x'],row['y'];height,width=gt.shape[-2:]
            x0,x1=max(0,x-64),min(width,x+65);y0,y1=max(0,y-64),min(height,y+65)
            maps=[('RGB',record['rgb'][0].permute(1,2,0).numpy()),('GT disparity',gt[0,0].numpy()),
                ('A5 disparity',c['base'][0,0].numpy()),('History disparity',c['history'][0,0].numpy()),
                ('Repaired disparity',pred[0,0].numpy()),('A5 absolute error',(c['base']-gt).abs()[0,0].numpy()),
                ('Repair absolute error',(pred-gt).abs()[0,0].numpy()),('History weight',gate[0,0].numpy())]
            fig,axes=plt.subplots(2,4,figsize=(14,7),constrained_layout=True)
            valid_gt=gt[0,0].numpy()[y0:y1,x0:x1][mask[y0:y1,x0:x1]]
            vmax=float(np.percentile(valid_gt,98)) if len(valid_gt) else 100
            for ax,(name,values) in zip(axes.flat,maps):
                patch=values[y0:y1,x0:x1]
                if name=='RGB':ax.imshow(patch)
                else:
                    limit=1 if name=='History weight' else (min(30,max(row['base_error_px'],row['repair_error_px'],1)) if 'error' in name else vmax)
                    picture=ax.imshow(patch,vmin=0,vmax=limit,cmap='magma' if 'error' in name else 'viridis')
                    fig.colorbar(picture,ax=ax,fraction=.046,pad=.02)
                ax.plot(x-x0,y-y0,'r+',markersize=10)
                ax.set_title(name);ax.axis('off')
            fig.suptitle(f"{label}: sequence {row['sequence_id']} frame {row['frame_id']} / ({x},{y})\nA5 error {row['base_error_px']:.3f} px → repair {row['repair_error_px']:.3f} px; history {row['history_error_px']:.3f} px; gate {row['gate']:.3f}")
            fig.savefig(output/f"{label}_{row['dataset_index']:06d}.png",dpi=130)
            plt.close(fig)
    (output/'cases.json').write_text(json.dumps(selected,indent=2)+'\n')
    print(json.dumps(ci,indent=2))

if __name__=='__main__':main()
