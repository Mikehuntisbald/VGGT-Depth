#!/usr/bin/env python3
"""Plot fixed research cases with shared disparity and error scales."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'third_party/Fast-FoundationStereo')]
import Utils
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tools.eval_native_temporal_stereo import load_native
from tools.train_native_temporal_stereo import move
from models.temporal_candidate_repair import TemporalCandidateRepair

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--original-cache',type=Path,required=True);p.add_argument('--v1-checkpoint',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);Utils.AMP_DTYPE=torch.bfloat16
    models={n:load_native(a.cache,a.run_dir/n/'final.pt','cuda')[0] for n in ('late','early_seed','early_state')}
    vp=torch.load(a.v1_checkpoint,map_location='cpu',weights_only=False);v1=TemporalCandidateRepair().cuda().eval();v1.load_state_dict(vp['model'])
    directory=a.run_dir/'failure_cases';directory.mkdir(exist_ok=True);rows=[]
    with torch.inference_mode():
      for index,x,y in ((2,253,147),(24,581,200),(29,570,20)):
        record=move(torch.load(a.original_cache/'validation'/f'{index:06d}.pt',map_location='cpu',weights_only=False),'cuda')
        native=move(torch.load(a.cache/'validation'/f'{index:06d}.pt',map_location='cpu',weights_only=False),'cuda')
        gt=record['target'];c=record['current'];old,_=v1(c['features'],c['base'],c['history'],c['history_valid'],vp['logit_shift'])
        h,w=gt.shape[-2:];x0,x1=max(0,x-48),min(w,x+49);y0,y1=max(0,y-48),min(h,y+49)
        def array(t):return t.detach().float().cpu().numpy()
        def scalar(t):return float(t[0,0,y,x])
        output={}
        for name,model in models.items():
            with torch.autocast('cuda',dtype=torch.bfloat16):output[name]=model(native['inputs'])
        all_maps=[array(t[0,0])[y0:y1,x0:x1] for t in (gt,c['base'],c['history'],old)]
        limit=max(1,float(np.percentile(np.concatenate([v.flatten() for v in all_maps]),99)))
        fig,axes=plt.subplots(4,4,figsize=(13,12),constrained_layout=True)
        rgb=array(record['rgb'][0].permute(1,2,0))
        axes[0,0].imshow(rgb[y0:y1,x0:x1].astype(np.uint8));axes[0,0].set_title('RGB')
        def show(ax,t,title,error=False):
            value=array(t[0,0])[y0:y1,x0:x1];im=ax.imshow(value,vmin=0,vmax=10 if error else limit,cmap='magma' if error else 'viridis')
            ax.set_title(title);fig.colorbar(im,ax=ax,fraction=.045)
        show(axes[0,1],gt,'Real GT');show(axes[0,2],c['base'],'A5');show(axes[0,3],old,'v1')
        row={'dataset_index':index,'x':x,'y':y,'GT':scalar(gt),'A5':scalar(c['base']),'original_history':scalar(c['history']),'v1':scalar(old),'arms':{}}
        for r,(name,out) in enumerate(output.items(),1):
            end=out['endpoint'];detail=out['details'][-1];init=detail['initialization']
            seed=init['seed'] if init is not None else detail['image_seed']
            seed=F.interpolate(seed.float()*8,size=(h,w),mode='nearest-exact')
            solved=F.interpolate(detail['stereo']['disparity_lr'].float()*2,size=(h,w),mode='bilinear',align_corners=False)
            prediction=end.disparity_left_px;error=(prediction-gt).abs()
            show(axes[r,0],seed,name+' initialization (coarse)');show(axes[r,1],solved,'After stereo iterations')
            show(axes[r,2],prediction,'After geometry decoder');show(axes[r,3],error,'Final error (0–10 px) ',True)
            row['arms'][name]={'initialization':scalar(seed),'stereo':scalar(solved),'final':scalar(prediction),'final_error':scalar(error)}
        for ax in axes.flat:ax.plot(x-x0,y-y0,'r+',markersize=8);ax.axis('off')
        fig.suptitle(f'Fixed case {index}, pixel ({x},{y}); shared disparity/error scales; GT={scalar(gt):.3f} px')
        fig.savefig(directory/f'case_{index:06d}.png',dpi=160);plt.close(fig);rows.append(row)
    (directory/'values.json').write_text(json.dumps(rows,indent=2)+'\n')

if __name__=='__main__':main()
