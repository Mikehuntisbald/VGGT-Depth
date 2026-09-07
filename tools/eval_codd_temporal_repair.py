#!/usr/bin/env python3
"""Paired validation of fixed opportunities, recovery and safety against v1."""
from __future__ import annotations
import argparse,csv,hashlib,json,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.distributed as dist
from models.temporal_candidate_repair import TemporalCandidateRepair
from models.codd_temporal_repair import CoddTemporalRepair
from metrics.metric_stereo_video import MetricAccumulator,AccuracyCoverageHistogram,endpoint_metric_values,temporal_residual_metric_values,scalar_metric
from tools.eval_temporal_candidate_repair import rewarp,to_device
from tools.train_metric_stereo_video import _distributed_context,_read_config


def safety_values(pred,record,v1_prediction):
    c=record['current'];gt=record['target'];valid=record['target_valid']
    be=(c['base']-gt).abs();he=(c['history']-gt).abs();err=(pred-gt).abs();v1e=(v1_prediction-gt).abs()
    opportunity=valid&c['history_valid']&(be-he>.1)
    recoverable=opportunity&(be>1)&(he<1)
    good=valid&(be<.1);reject=good&c['history_valid']&(he>1)
    large=err-be>5;old_large=v1e-be>5
    gap=(c['history']-c['base']).abs()
    fg_bg=valid&record['boundary']&c['history_valid']&(gap>5)&(torch.minimum(be,he)<1)
    mixed=((pred-c['base']).abs()>1)&((pred-c['history']).abs()>1)&(pred>torch.minimum(c['base'],c['history']))&(pred<torch.maximum(c['base'],c['history']))&(err>1)
    return {
      'good_current_damage_rate':scalar_metric((err-be>.1).float(),good),
      'rejection_task_failure_rate':scalar_metric((err-be>.1).float(),reject),
      'large_degradation_1px_rate':scalar_metric((err-be>1).float(),valid),
      'large_degradation_5px_rate':scalar_metric(large.float(),valid),
      'large_degradation_5px_excess_mean':scalar_metric((err-be-5).clamp_min(0),valid),
      'new_large_5px_vs_v1_rate':scalar_metric((large&~old_large).float(),valid),
      'removed_large_5px_vs_v1_rate':scalar_metric((old_large&~large).float(),valid),
      'error_increase_over_v1_5px_rate':scalar_metric((err-v1e>5).float(),valid),
      'fixed_opportunity_recovery_rate':scalar_metric((be-err>.1).float(),opportunity),
      'fixed_opportunity_epe_px':scalar_metric(err,opportunity),
      'recoverable_to_1px_rate':scalar_metric((err<=1).float(),recoverable),
      'fg_bg_spurious_mixing_rate':scalar_metric(mixed.float(),fg_bg),
      'fg_bg_error_px':scalar_metric(err,fg_bg),
      'boundary_good_current_damage_rate':scalar_metric((err-be>.1).float(),good&record['boundary']),
      'fg_bg_foreground_gt_error_px':scalar_metric(err,fg_bg&((torch.maximum(c['base'],c['history'])-gt).abs()<1)),
      'fg_bg_background_gt_error_px':scalar_metric(err,fg_bg&((torch.minimum(c['base'],c['history'])-gt).abs()<1)),
    }


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--bank',type=Path)
    p.add_argument('--v1-checkpoint',type=Path,required=True);p.add_argument('--checkpoints',type=Path,nargs='+',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    args=p.parse_args();torch.set_num_threads(4);context=_distributed_context();device=context.device
    contract=_read_config(ROOT/'configs/metric_stereo_video/evaluation_contract.yaml')
    if not (args.cache/'validation/complete.json').exists():raise RuntimeError('incomplete validation cache')
    if args.bank and not (args.bank/'validation/complete.json').exists():raise RuntimeError('incomplete validation evidence')
    v1_payload=torch.load(args.v1_checkpoint,map_location='cpu',weights_only=False)
    v1=TemporalCandidateRepair().to(device).eval();v1.load_state_dict(v1_payload['model'])
    models={};configs={};lineage=hashlib.sha256((args.cache/'lineage.json').read_bytes()).hexdigest()
    for checkpoint in args.checkpoints:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=payload['config'];name=cfg['arm']
        if cfg['cache_lineage_sha256']!=lineage:raise RuntimeError('old cache lineage mismatch')
        if cfg['initial_checkpoint_sha256']!=hashlib.sha256(args.v1_checkpoint.read_bytes()).hexdigest():raise RuntimeError('wrong v1 initialization')
        if any(cfg['components'].values()):
            if not args.bank or cfg['bank_lineage_sha256']!=hashlib.sha256((args.bank/'lineage.json').read_bytes()).hexdigest():raise RuntimeError('evidence bank mismatch')
        model=CoddTemporalRepair(v1,v1_shift=cfg['v1_shift']).to(device).eval();model.load_state_dict(payload['model'],strict=True)
        models[name]=(model,payload['logit_shift']);configs[name]={'checkpoint':str(checkpoint),'checkpoint_sha256':hashlib.sha256(checkpoint.read_bytes()).hexdigest(),'config':cfg}
    names=['v1',*models];accumulators={n:MetricAccumulator() for n in names}
    coverage={n:AccuracyCoverageHistogram(4096,device) for n in names}
    paths=sorted((args.cache/'validation').glob('*.pt'));per_sample=[];top=[];started=time.monotonic()
    with torch.inference_mode():
      for i in range(context.rank,len(paths),context.world_size):
        record=to_device(torch.load(paths[i],map_location='cpu',weights_only=False),device)
        bank=to_device(torch.load(args.bank/'validation'/paths[i].name,map_location='cpu',weights_only=False),device) if args.bank else None
        c,prev=record['current'],record['previous'];gt=record['target'];valid=record['target_valid']
        v1p,v1g=v1(c['features'],c['base'],c['history'],c['history_valid'],v1_payload['logit_shift'])
        v1prev,_=v1(prev['features'],prev['base'],prev['history'],prev['history_valid'],v1_payload['logit_shift'])
        row={'sequence_id':record['identity']['sequence_id'],'frame_id':record['identity']['frame_id'],'dataset_index':record['identity']['dataset_index']}
        for name in names:
            if name=='v1':pred,gate,previous_prediction=v1p,v1g,v1prev
            else:
                model,shift=models[name]
                output=model(c['features'],c['base'],c['history'],c['history_valid'],bank['current'] if bank else None,shift)
                previous_output=model(prev['features'],prev['base'],prev['history'],prev['history_valid'],bank['previous'] if bank else None,shift)
                pred,gate,previous_prediction=output['prediction'],output['gate'],previous_output['prediction']
            if not pred.isfinite().all() or not (pred>0).all():raise RuntimeError('nonfinite/nonpositive prediction')
            warped,warped_valid=rewarp(previous_prediction,record)
            factor=(record['K'][:,0,0]*record['baseline']).reshape(-1,1,1,1)
            values=endpoint_metric_values(predicted_depth_m=factor/pred,predicted_disparity_px=pred,
                predicted_valid_mask=c['base_valid'],predicted_valid_probability=c['base_probability'],predicted_uncertainty=c['base_uncertainty'],
                gt_disparity_px=gt,gt_valid_mask=valid,intrinsics_left=record['K'],baseline_m=record['baseline'],
                dynamic_mask=record['dynamic'],dynamic_available=record['dynamic_available'],detail_mask=record['detail'],matched_mask=record['matched'],boundary_mask=record['boundary'])
            transform=record['transform'];cosine=((transform[:,:3,:3].diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
            score=torch.linalg.vector_norm(transform[:,:3,3],dim=-1)+torch.acos(cosine)
            low=contract['temporal']['small_medium_threshold'];high=contract['temporal']['medium_large_threshold']
            motions={k:m.reshape(-1,1,1,1).expand_as(gt) for k,m in (('small_motion',score<=low),('medium_motion',(score>low)&(score<=high)),('large_motion',score>high))}
            values.update(temporal_residual_metric_values(current_prediction_disparity_px=pred,warped_previous_prediction_disparity_px=warped,
                current_gt_disparity_px=gt,warped_previous_gt_disparity_px=record['gt_warp'],current_prediction_valid=c['base_valid'],warped_prediction_valid=warped_valid,
                current_gt_valid=valid,warped_gt_valid=record['gt_warp_valid'],dynamic_mask=record['dynamic'],dynamic_available=record['dynamic_available'],motion_bucket_masks=motions))
            values.update(safety_values(pred,record,v1p));accumulators[name].update(values)
            coverage[name].update(c['base_probability'],(pred-gt).abs(),valid,error_cap_px=10)
            keys=('all_gt_penalized_epe_px','epe_px','temporal_matched_penalized_delta_epe_px','good_current_damage_rate','large_degradation_1px_rate','large_degradation_5px_rate','fixed_opportunity_recovery_rate','recoverable_to_1px_rate','fg_bg_spurious_mixing_rate')
            for key in keys:row[name+'/'+key]=[values[key].numerator,values[key].count]
            if name!='v1':
                delta=(pred-gt).abs()-(v1p-gt).abs()
                for sign,label in ((1,'new_degradation'),(-1,'recovery_over_v1')):
                    value,idx=torch.where(valid,sign*delta,torch.full_like(delta,-float('inf'))).flatten().max(0)
                    y,x=divmod(int(idx),gt.shape[-1]);top.append({'arm':name,'kind':label,'sequence_id':row['sequence_id'],'frame_id':row['frame_id'],'dataset_index':row['dataset_index'],'x':x,'y':y,'change_px':float(value),'v1_error_px':float((v1p-gt).abs()[0,0,y,x]),'new_error_px':float((pred-gt).abs()[0,0,y,x]),'gate':float(gate[0,0,y,x])})
        per_sample.append(row)
        if (i//context.world_size)%30==0:print(json.dumps({'rank':context.rank,'batch':i//context.world_size,'elapsed_s':time.monotonic()-started}),flush=True)
    payload={'metrics':{n:accumulators[n].values for n in names},'rows':per_sample,'cases':top}
    gathered=[None]*context.world_size
    if context.world_size>1:dist.all_gather_object(gathered,payload)
    else:gathered=[payload]
    for curve in coverage.values():curve.all_reduce_()
    if context.primary:
        merged={n:MetricAccumulator() for n in names};rows=[]
        for shard in gathered:
            rows+=shard['rows']
            for n in names:merged[n].merge(MetricAccumulator(shard['metrics'][n]))
        if len(rows)!=1294 or len({r['dataset_index'] for r in rows})!=1294:raise RuntimeError('validation is not exact complete domain')
        metrics={n:merged[n].finalize() for n in names}
        curves={n:coverage[n].finalize(contract['validity']['accuracy_coverage_points']) for n in names}
        for n in names:
            point=next(p for p in curves[n]['points'] if p['requested_coverage']==.99)
            metrics[n]['epe_at_99pct_coverage_px']={'value':point['epe_px'],'numerator':None,'count':point['effective_count'],'valid':True}
        def val(name,key):return metrics[name][key]['value']
        checks={}
        for n in models:
            checks[n]={
              'preserve_v1_all_gt_epe':val(n,'all_gt_penalized_epe_px')<=val('v1','all_gt_penalized_epe_px'),
              'preserve_v1_temporal':val(n,'temporal_matched_penalized_delta_epe_px')<=val('v1','temporal_matched_penalized_delta_epe_px'),
              'reduce_good_pixel_damage':val(n,'good_current_damage_rate')<val('v1','good_current_damage_rate'),
              'reduce_large_5px':val(n,'large_degradation_5px_rate')<val('v1','large_degradation_5px_rate'),
              'reduce_large_1px':val(n,'large_degradation_1px_rate')<val('v1','large_degradation_1px_rate'),
              'preserve_tail_severity':val(n,'large_degradation_5px_excess_mean')<=val('v1','large_degradation_5px_excess_mean'),
              'increase_fixed_opportunity_recovery':val(n,'fixed_opportunity_recovery_rate')>val('v1','fixed_opportunity_recovery_rate'),
              'preserve_recovery_to_1px':val(n,'recoverable_to_1px_rate')>=val('v1','recoverable_to_1px_rate'),
              'coverage_unchanged':val(n,'prediction_coverage')==val('v1','prediction_coverage')}
        cases=[case for p in gathered for case in p['cases']]
        top_cases={n:{kind:sorted([c for c in cases if c['arm']==n and c['kind']==kind],key=lambda c:c['change_px'],reverse=True)[:12] for kind in ('new_degradation','recovery_over_v1')} for n in models}
        report={'samples':1294,'metrics':metrics,'acceptance':{n:{'status':'PASS' if all(checks[n].values()) else 'FAIL','checks':checks[n]} for n in checks},
            'models':configs,'accuracy_coverage':curves,'cases':top_cases,'v1_checkpoint_sha256':hashlib.sha256(args.v1_checkpoint.read_bytes()).hexdigest(),
            'cache_lineage_sha256':lineage,'evaluation_contract':contract,'fixed_masks':'original immutable A5/base and original v1-cache history for every arm',
            'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-started}
        args.output_dir.mkdir(parents=True,exist_ok=True)
        (args.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        (args.output_dir/'per_sample.json').write_text(json.dumps(rows,indent=2)+'\n')
        with (args.output_dir/'metrics.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['metric',*names])
            for k in metrics['v1']:writer.writerow([k,*[val(n,k) for n in names]])
        print(json.dumps(report['acceptance']),flush=True)
    if context.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
