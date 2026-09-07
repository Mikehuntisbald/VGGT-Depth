#!/usr/bin/env python3
"""Scientific view of actual hypothesis contributions in three fixed cases."""
import argparse,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    report=json.loads((a.run_dir/'evaluation/metrics.json').read_text());cases=report['fixed_failure_points']
    colors=['#4477AA','#EE7733',*(['#228833']*9),*(['#AA3377']*3),*(['#663399']*9)]
    fig,axes=plt.subplots(2,3,figsize=(15,8),constrained_layout=True)
    for column,index in enumerate((2,24,29)):
        matching={r['variant']:r for r in cases if r['dataset_index']==index}
        upper=max(120,*[max(r['candidate_predictions']) for r in matching.values()])*1.08
        for row,name in enumerate(('wta','mode_pool')):
            r=matching[name];ax=axes[row,column];values=np.array(r['candidate_predictions']);valid=np.array(r['candidate_valid']);weights=np.array(r['mixture_weights']);positions=np.arange(len(values))
            ax.scatter(values[valid],positions[valid],c=np.array(colors)[valid],s=18,alpha=.35)
            active=valid&(weights>1e-6);ax.scatter(values[active],positions[active],c=np.array(colors)[active],s=40+650*weights[active],edgecolors='black',linewidths=.6)
            ax.axvline(r['GT'],color='black',lw=1.6,label='GT');ax.axvline(r['v1'],color='gray',ls=':',lw=1.5,label='v1');ax.axvline(r['prediction'],color='#CC3311',lw=1.5,label='Output')
            ax.set_xlim(0,upper);ax.set_ylim(23,-1);ax.set_yticks([0,1,6,12,18],['Stereo','VGGT','Coarse modes','Stereo peaks','History'])
            ax.set_xlabel('Disparity (HR pixels)');ax.grid(axis='x',alpha=.2)
            ax.set_title(f"{name}, case {index} ({r['x']},{r['y']})\nerror={abs(r['prediction']-r['GT']):.3f} px; v1 error={abs(r['v1']-r['GT']):.3f} px")
    fig.suptitle('Fixed failure cases: marker area shows actual contribution to final geometry',fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color='black',label='GT'),Line2D([0],[0],color='gray',ls=':',label='v1'),Line2D([0],[0],color='#CC3311',label='Output')],loc='outside lower center',ncol=3)
    fig.savefig(a.run_dir/'fixed_case_contributions.png',dpi=160);plt.close(fig)

if __name__=='__main__':main()
