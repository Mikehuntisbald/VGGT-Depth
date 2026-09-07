#!/usr/bin/env python3
"""Paired full-validation evaluation with GT-owned domains and causal re-warp."""
from __future__ import annotations
import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
import torch.distributed as dist
from models.temporal_candidate_repair import TemporalCandidateRepair
from metrics.metric_stereo_video import MetricAccumulator,AccuracyCoverageHistogram,endpoint_metric_values,temporal_residual_metric_values,scalar_metric
from geometry.zbuffer_reproject import zbuffer_reproject
from tools.train_metric_stereo_video import _distributed_context,_read_config


def to_device(item,device):
    if isinstance(item,torch.Tensor):return item.to(device)
    if isinstance(item,dict):return {k:to_device(v,device) for k,v in item.items()}
    return item


def rewarp(prediction,record):
    previous=record['previous']
    factor=(record['K_previous'][:,0,0]*record['baseline_previous']).reshape(-1,1,1,1)
    depth=factor/prediction.clamp_min(1e-8)
    warp=zbuffer_reproject(prediction,depth,previous['base_confidence'],record['K_previous'],
        torch.eye(4,device=prediction.device)[None],record['transform'],
        intrinsics_current_hr_3x3=record['K'],baseline_previous_m=record['baseline_previous'],
        baseline_current_m=record['baseline'])
    h,w=prediction.shape[-2:]
    uv=warp.source_uv.long()
    index=(uv[:,1].clamp(0,h-1)*w+uv[:,0].clamp(0,w-1)).flatten(1)[:,None]
    valid=torch.gather(previous['base_valid'].flatten(2),2,index).reshape_as(prediction)
    return warp.disparity_hr_px,warp.valid_mask & valid


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--evaluation-contract',type=Path,default=ROOT/'configs/metric_stereo_video/evaluation_contract.yaml')
    args=p.parse_args()
    contract=_read_config(args.evaluation_contract)
    if contract['validity']['accuracy_coverage_ranking_score']!='valid_probability':
        raise RuntimeError('unsupported coverage ranking contract')
    if contract['spring_partitions']['boundary_gradient_threshold_px']!=1.0 or contract['spring_partitions']['boundary_radius_px']!=1:
        raise RuntimeError('cached boundary masks use threshold 1 px and radius 1')
    torch.set_num_threads(4)
    context=_distributed_context()
    device=context.device
    if not (args.cache/'validation/complete.json').exists(): raise RuntimeError('validation cache incomplete')
    ckpt=torch.load(args.checkpoint,map_location='cpu',weights_only=False)
    lineage=hashlib.sha256((args.cache/'lineage.json').read_bytes()).hexdigest()
    if lineage!=ckpt['config']['cache_lineage_sha256']:raise RuntimeError('checkpoint/cache lineage mismatch')
    model=TemporalCandidateRepair().to(device)
    model.load_state_dict(ckpt['model'],strict=True)
    model.eval()
    names=('A5_frozen','A5_temporal_repair')
    accum={n:MetricAccumulator() for n in names}
    curves={n:AccuracyCoverageHistogram(int(contract["validity"]["accuracy_coverage_histogram_bins"]),device) for n in names}
    diagnostics=MetricAccumulator()
    per_sample=[]
    failures=[]
    rescues=[]
    paths=sorted((args.cache/'validation').glob('*.pt'))
    started=time.monotonic()
    with torch.inference_mode():
        for i in range(context.rank,len(paths),context.world_size):
            record=to_device(torch.load(paths[i],map_location='cpu',weights_only=False),device)
            current,previous=record['current'],record['previous']
            pred,gate=model(current['features'],current['base'],current['history'],current['history_valid'],ckpt['logit_shift'])
            prev_pred,_=model(previous['features'],previous['base'],previous['history'],previous['history_valid'],ckpt['logit_shift'])
            if not pred.isfinite().all() or not (pred>0).all():raise RuntimeError('nonfinite/nonpositive repair output')
            warped,warp_valid=rewarp(prev_pred,record)
            target,valid=record['target'],record['target_valid']
            base_error=(current['base']-target).abs()
            repair_error=(pred-target).abs()
            history_error=(current['history']-target).abs()
            support=valid & current['history_valid']
            opportunity=support & (base_error-history_error>0.1)
            lost=opportunity & ~current['old_valid']
            bad_history=support & (history_error-base_error>0.1)
            metrics_by_arm={}
            for name,disparity,hist,hvalid in (
                (names[0],current['base'],current['history'],current['history_valid']),
                (names[1],pred,warped,warp_valid)):
                depth=(record['K'][:,0,0]*record['baseline']).reshape(-1,1,1,1)/disparity.clamp_min(1e-8)
                values=endpoint_metric_values(predicted_depth_m=depth,predicted_disparity_px=disparity,
                    predicted_valid_mask=current['base_valid'],predicted_valid_probability=current['base_probability'],
                    predicted_uncertainty=current['base_uncertainty'],gt_disparity_px=target,gt_valid_mask=valid,
                    intrinsics_left=record['K'],baseline_m=record['baseline'],dynamic_mask=record['dynamic'],
                    dynamic_available=record['dynamic_available'],detail_mask=record['detail'],matched_mask=record['matched'],boundary_mask=record['boundary'])
                transform=record['transform'].float()
                cosine=((transform[:,:3,:3].diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
                score=torch.linalg.vector_norm(transform[:,:3,3],dim=-1)+torch.acos(cosine)
                low=contract['temporal']['small_medium_threshold'];high=contract['temporal']['medium_large_threshold']
                motions={key:mask.reshape(-1,1,1,1).expand_as(target) for key,mask in (('small_motion',score<=low),('medium_motion',(score>low)&(score<=high)),('large_motion',score>high))}
                values.update(temporal_residual_metric_values(current_prediction_disparity_px=disparity,
                    warped_previous_prediction_disparity_px=hist,current_gt_disparity_px=target,
                    warped_previous_gt_disparity_px=record['gt_warp'],current_prediction_valid=current['base_valid'],
                    warped_prediction_valid=hvalid,current_gt_valid=valid,warped_gt_valid=record['gt_warp_valid'],
                    dynamic_mask=record['dynamic'],dynamic_available=record['dynamic_available'],motion_bucket_masks=motions))
                error=(disparity-target).abs()
                for region,mask in (('opportunity',opportunity),('old_gate_rejected_opportunity',lost),('wrong_history',bad_history)):
                    values[region+'_epe_px']=scalar_metric(error,mask)
                    values[region+'_completion_1px']=scalar_metric((error<=1).float(),mask)
                accum[name].update(values)
                penalized=torch.where(current['base_valid'],error.clamp_max(10),torch.full_like(error,10))
                curves[name].update(current['base_probability'],error,valid,error_cap_px=10)
                metrics_by_arm[name]=values
            diagnostics.update({
                'opportunity_rate':scalar_metric(opportunity.float(),valid),
                'opportunity_recovered_rate':scalar_metric((base_error-repair_error>0.1).float(),opportunity),
                'old_gate_rejected_opportunity_recovered_rate':scalar_metric((base_error-repair_error>0.1).float(),lost),
                'wrong_history_materially_harmed_rate':scalar_metric((repair_error-base_error>0.1).float(),bad_history),
                'wrong_history_gate_over_half_rate':scalar_metric((gate>0.5).float(),bad_history),
                'good_base_materially_harmed_rate':scalar_metric((repair_error-base_error>0.1).float(),valid & (base_error<0.1)),
                'all_gt_gain_px':scalar_metric(base_error-repair_error,valid),
                'internal_lr_history_oracle_gain_px':scalar_metric(base_error-(current['internal_history']-target).abs(),valid & current['internal_valid']),
                'hr_history_oracle_gain_px':scalar_metric(base_error-history_error,support),
                'internal_lr_opportunity_rate':scalar_metric(((base_error-(current['internal_history']-target).abs())>0.1).float(),valid & current['internal_valid']),
                'history_gate_mean':scalar_metric(gate,valid),
            })
            row={'sequence_id':record['identity']['sequence_id'],'frame_id':record['identity']['frame_id'],'dataset_index':record['identity']['dataset_index']}
            for name in names:
                for metric in ('all_gt_penalized_epe_px','epe_px','temporal_matched_penalized_delta_epe_px','dynamic_penalized_epe_px','spring_high_detail_epe_px','opportunity_epe_px'):
                    if metric in metrics_by_arm[name]:
                        item=metrics_by_arm[name][metric]
                        row[name+'/'+metric]=[item.numerator,item.count]
            per_sample.append(row)
            for collection,scores,mask in ((rescues,base_error-repair_error,opportunity),(failures,repair_error-base_error,valid)):
                score=torch.where(mask,scores,torch.full_like(scores,-float('inf')))
                value,flat=score.flatten().max(0)
                if torch.isfinite(value):
                    y,x=divmod(int(flat),target.shape[-1])
                    collection.append({'dataset_index':row['dataset_index'],'sequence_id':row['sequence_id'],'frame_id':row['frame_id'],
                        'x':x,'y':y,'change_px':float(value),'base_error_px':float(base_error[0,0,y,x]),
                        'repair_error_px':float(repair_error[0,0,y,x]),'history_error_px':float(history_error[0,0,y,x]),
                        'gate':float(gate[0,0,y,x]),'old_valid':bool(current['old_valid'][0,0,y,x])})
            if (i//context.world_size)%20==0:
                print(json.dumps({'rank':context.rank,'evaluated':i//context.world_size+1,'elapsed_s':time.monotonic()-started}),flush=True)
    payload={'accum':{n:accum[n].values for n in names},'diagnostics':diagnostics.values,'per_sample':per_sample,
             'failures':sorted(failures,key=lambda x:x['change_px'],reverse=True)[:20],
             'rescues':sorted(rescues,key=lambda x:x['change_px'],reverse=True)[:20]}
    gathered=[None]*context.world_size
    if context.world_size>1:dist.all_gather_object(gathered,payload)
    else:gathered=[payload]
    for curve in curves.values():curve.all_reduce_()
    if context.primary:
        merged={n:MetricAccumulator() for n in names}
        diag=MetricAccumulator()
        rows=[]
        for shard in gathered:
            for n in names:merged[n].merge(MetricAccumulator(shard['accum'][n]))
            diag.merge(MetricAccumulator(shard['diagnostics']))
            rows+=shard['per_sample']
        if len(rows)!=len(paths) or len({r['dataset_index'] for r in rows})!=len(paths):raise RuntimeError('duplicate or missing validation endpoints')
        metrics={n:merged[n].finalize() for n in names}
        coverage_results={n:curves[n].finalize(contract['validity']['accuracy_coverage_points']) for n in names}
        for n in names:
            at_99=next(point for point in coverage_results[n]['points'] if point['requested_coverage']==0.99)
            metrics[n]['epe_at_99pct_coverage_px']={'value':at_99['epe_px'],'numerator':None,'count':at_99['effective_count'],'valid':True}
        def value(n,key):return metrics[n][key]['value']
        checks={
            'all_gt_penalized_epe_improves_1pct':value(names[1],'all_gt_penalized_epe_px')<=0.99*value(names[0],'all_gt_penalized_epe_px'),
            'temporal_delta_improves_1pct':value(names[1],'temporal_matched_penalized_delta_epe_px')<=0.99*value(names[0],'temporal_matched_penalized_delta_epe_px'),
            'opportunity_epe_improves_5pct':value(names[1],'opportunity_epe_px')<=0.95*value(names[0],'opportunity_epe_px'),
            'dynamic_penalized_within_2pct':value(names[1],'dynamic_penalized_epe_px')<=1.02*value(names[0],'dynamic_penalized_epe_px'),
            'high_detail_within_2pct':value(names[1],'spring_high_detail_epe_px')<=1.02*value(names[0],'spring_high_detail_epe_px'),
            'coverage_unchanged':abs(value(names[1],'prediction_coverage')-value(names[0],'prediction_coverage'))<1e-12}
        report={'status':'PASS' if all(checks.values()) else 'FAIL','checks':checks,'samples':len(rows),
            'checkpoint':str(args.checkpoint),'checkpoint_sha256':hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
            'cache_lineage_sha256':lineage,'metrics':metrics,'diagnostics':diag.finalize(),
            'accuracy_coverage':coverage_results,
            'evaluation_contract':contract,'evaluation_contract_sha256':hashlib.sha256(args.evaluation_contract.read_bytes()).hexdigest(),
            'evaluation_source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'top_rescues':sorted([x for p in gathered for x in p['rescues']],key=lambda x:x['change_px'],reverse=True)[:20],
            'top_failures':sorted([x for p in gathered for x in p['failures']],key=lambda x:x['change_px'],reverse=True)[:20],
            'epistemic_scope':'Frozen A5 plus trained causal HR selector; not a new independent full A0-A5 training. Known GT camera pose. Native validation labels used only for evaluation.',
            'elapsed_s':time.monotonic()-started}
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output_dir/'per_sample.json').write_text(json.dumps(sorted(rows,key=lambda x:x['dataset_index']),indent=2)+'\n')
        with (args.output_dir/'metrics.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['metric',*names])
            for key in sorted(metrics[names[0]]):writer.writerow([key,*[metrics[n][key]['value'] for n in names]])
        print(json.dumps({'status':report['status'],'checks':checks,'samples':len(rows)}),flush=True)
    if context.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
