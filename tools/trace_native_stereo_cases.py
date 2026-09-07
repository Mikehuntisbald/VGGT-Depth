#!/usr/bin/env python3
"""Trace fixed errors through disparity updates, resampling and metric anchors."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'third_party/Fast-FoundationStereo')]
import Utils
import torch
import torch.nn.functional as F
from tools.eval_native_temporal_stereo import load_native
from tools.train_native_temporal_stereo import move,digest
from models.metric_stereo_video_geometry import _resize_scalar

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    torch.set_num_threads(1);Utils.AMP_DTYPE=torch.bfloat16
    models={n:load_native(a.cache,a.run_dir/n/'final.pt','cuda')[0] for n in ('late','early_seed','early_state')};rows=[]
    with torch.inference_mode():
      for index,x,y in ((2,253,147),(24,581,200),(29,570,20)):
        record=move(torch.load(a.cache/'validation'/f'{index:06d}.pt',map_location='cpu',weights_only=False),'cuda');gt=record['labels']['target'][:,-1];size=gt.shape[-2:]
        def up(t,mode='bilinear'):return F.interpolate(t.float(),size=size,mode=mode,**({'align_corners':False} if mode=='bilinear' else {}))
        def point(t,mode='bilinear'):return float(up(t,mode)[0,0,y,x])
        row={'dataset_index':index,'x':x,'y':y,'GT':float(gt[0,0,y,x]),'arms':{}}
        for name,model in models.items():
            captured={};original=model.geometry.forward_step
            def capture_frame(frame,state=None):captured['frame']=frame;return original(frame,state)
            def capture_weights(module,args,output):captured['weights']=torch.softmax(output,dim=1)
            model.geometry.forward_step=capture_frame;handle=model.refiner.spx_gru.register_forward_hook(capture_weights)
            with torch.autocast('cuda',dtype=torch.bfloat16):out=model(record['inputs'])
            model.geometry.forward_step=original;handle.remove()
            frame=captured['frame'];end=out['endpoint'];detail=out['details'][-1];init=detail['initialization'];solved=detail['stereo']
            factor=(frame.intrinsics_left_3x3[:,0,0]*record['inputs']['baseline_m'][:,-1]).reshape(-1,1,1,1)
            confidence=frame.lowres_disparity_confidence.float();sv=frame.lowres_disparity_valid_mask
            vg_conf=_resize_scalar(frame.vggt_features.confidence,sv.shape[-2:]).float()*end.gauge.valid_mask.reshape(-1,1,1,1)
            vg_disp=end.gauge.inverse_depth_m_inv.float()*factor
            sw=confidence*sv;denom=sw+vg_conf
            base=(sw*frame.lowres_disparity_left_px.float()+vg_conf*vg_disp)/denom.clamp_min(1e-8)
            seed=init['seed'] if init is not None else detail['image_seed']
            # Inspect both the update state and the following convex upsampling.
            neighbors=F.unfold(solved['disparity_feature'].float()*8,kernel_size=3,padding=1).reshape(1,9,*solved['disparity_feature'].shape[-2:])
            neighbors=F.interpolate(neighbors,size=solved['disparity_lr'].shape[-2:],mode='nearest')
            weights=captured['weights'].float();ly,lx=y//2,x//2
            trace={'seed_nearest_hr_units':point(seed*8,'nearest-exact'),
              'iteration_states_nearest_hr_units':[point(d*8,'nearest-exact') for d in solved['iterations']],
              'after_FFS_convex_upsampling_hr_units':point(solved['disparity_lr']*2),
              'FFS_upsample_neighbors_at_nearest_LR_pixel':neighbors[0,:,ly,lx].tolist(),
              'FFS_upsample_weights_at_nearest_LR_pixel':weights[0,:,ly,lx].tolist(),
              'stereo_valid_support_bilinear':point(sv.float()),'stereo_confidence_bilinear':point(confidence),
              'aligned_VGGT_disparity_bilinear':point(vg_disp),'aligned_VGGT_confidence_bilinear':point(vg_conf),
              'fixed_blended_anchor_bilinear':point(base),'after_lowres_geometry_decoder':point(end.state.inverse_depth_m_inv*factor),
              'final_geometry':point(end.disparity_left_px),
              'largest_allowed_two_stage_residual_factor':float(torch.exp(torch.tensor(2*model.geometry.inverse_depth_residual_scale)))}
            if init is not None:
                cy,cx=y//8,x//8
                trace.update(candidate_disparities_hr_units=(init['candidates'][0,:,cy,cx]*8).float().tolist(),candidate_valid=init['candidate_valid'][0,:,cy,cx].tolist(),candidate_probabilities=init['probability'][0,:,cy,cx].float().tolist(),selected_candidate=int(init['selected'][0,0,cy,cx]))
            row['arms'][name]=trace
        rows.append(row)
    result={'status':'COMPLETE','cases':rows,'source_sha256':digest(Path(__file__)),'note':'scalar maps bilinearly sampled at exact HR point unless nearest is named; 9 upsample weights listed at nearest LR pixel; no GT in model forward'}
    (a.run_dir/'stage_traces.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
