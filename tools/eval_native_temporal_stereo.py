#!/usr/bin/env python3
"""Evaluate native recursive stereo on the immutable original GT-owned domains."""
from __future__ import annotations
import argparse,csv,json,sys,time
from datetime import timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'third_party/Fast-FoundationStereo')]
import Utils
import torch
import torch.distributed as dist
import torch.nn.functional as F
from models.native_temporal_stereo.model import NativeTemporalStereo,warp_map
from models.temporal_candidate_repair import TemporalCandidateRepair
from metrics.metric_stereo_video import MetricAccumulator,AccuracyCoverageHistogram,endpoint_metric_values,temporal_residual_metric_values,scalar_metric
from tools.train_native_temporal_stereo import move,digest
from tools.eval_temporal_candidate_repair import rewarp
from tools.eval_controlled_temporal_repair import safety_values
from tools.train_metric_stereo_video import _distributed_context,_read_config


def prefix_inputs(inputs,count):
    result={}
    for k,v in inputs.items():
        if k=='encoded':result[k]={n:([x[:count] for x in z] if isinstance(z,list) else z[:count]) for n,z in v.items()}
        elif k=='vggt':result[k]={n:z[:,:count] for n,z in v.items()}
        else:result[k]=v[:,:count]
    return result


def load_package(cache):
    package=torch.load(cache/'initialization.pt',map_location='cpu',weights_only=False)
    specification=json.loads((cache/'structure.json').read_text())
    package.update(hidden_channels=specification['hidden_channels'],volume_channels=specification['volume_channels'])
    return package


def load_native(cache,checkpoint,device):
    payload=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=payload['config']
    for key,path in [('cache_lineage_sha256',cache/'lineage.json'),('initialization_sha256',cache/'initialization.pt'),('structure_sha256',cache/'structure.json')]:
        if cfg[key]!=digest(path):raise RuntimeError('native checkpoint lineage mismatch: '+key)
    model=NativeTemporalStereo(load_package(cache),variant=cfg['variant']).to(device).eval()
    model.load_state_dict(payload['model'],strict=True)
    return model,{'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'config':cfg}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cache',type=Path,required=True);p.add_argument('--original-cache',type=Path,required=True)
    p.add_argument('--v1-checkpoint',type=Path,required=True);p.add_argument('--checkpoints',type=Path,nargs='*',default=[])
    p.add_argument('--diagnostics',action='store_true');p.add_argument('--max-samples',type=int);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(1);Utils.AMP_DTYPE=torch.bfloat16
    # Independent inference shards can finish far apart while cold disk reads
    # are in flight. Give final aggregation time to await every complete shard.
    original_init=dist.init_process_group
    def evaluation_init(*args,**kwargs):
        kwargs.setdefault('timeout',timedelta(hours=1));return original_init(*args,**kwargs)
    dist.init_process_group=evaluation_init
    try:context=_distributed_context()
    finally:dist.init_process_group=original_init
    device=context.device
    contract=_read_config(ROOT/'configs/metric_stereo_video/evaluation_contract.yaml')
    if not (a.cache/'validation/complete.json').exists():raise RuntimeError('incomplete native validation cache')
    old_lineage=json.loads((a.cache/'lineage.json').read_text())['sampling_plan_lineage_sha256']
    if digest(a.original_cache/'lineage.json')!=old_lineage:raise RuntimeError('fixed original domain changed')
    v1_payload=torch.load(a.v1_checkpoint,map_location='cpu',weights_only=False)
    v1=TemporalCandidateRepair().to(device).eval();v1.load_state_dict(v1_payload['model'])
    models={};configs={}
    for checkpoint in a.checkpoints:
        model,cfg=load_native(a.cache,checkpoint,device);name=cfg['config']['variant']
        if name in models:raise RuntimeError('duplicate native variant')
        models[name]=model;configs[name]=cfg
    if a.diagnostics:
        for schedule in ('endpoint_only','causal_each_frame'):
            name='zero_'+schedule;models[name]=NativeTemporalStereo(load_package(a.cache),variant='late',vggt_schedule=schedule).to(device).eval()
            configs[name]={'trained':False,'purpose':'disentangle sequential execution and causal VGGT schedule','initialization_sha256':digest(a.cache/'initialization.pt')}
    names=['A5','v1',*models];accumulators={n:MetricAccumulator() for n in names};coverage={n:AccuracyCoverageHistogram(4096,device) for n in names}
    paths=sorted((a.original_cache/'validation').glob('*.pt'))
    if a.max_samples:paths=paths[:a.max_samples]
    rows=[];cases=[];checks=[];started=time.monotonic()
    with torch.inference_mode():
      for i in range(context.rank,len(paths),context.world_size):
        record=move(torch.load(paths[i],map_location='cpu',weights_only=False),device)
        native=move(torch.load(a.cache/'validation'/paths[i].name,map_location='cpu',weights_only=False),device)
        if native['identity']!=record['identity']:raise RuntimeError('sample identity mismatch')
        gt=record['target'];valid=record['target_valid'];c=record['current'];prev=record['previous']
        torch.testing.assert_close(native['labels']['target'][:,-1],gt,rtol=0,atol=0)
        v1p,_=v1(c['features'],c['base'],c['history'],c['history_valid'],v1_payload['logit_shift'])
        v1prev,_=v1(prev['features'],prev['base'],prev['history'],prev['history_valid'],v1_payload['logit_shift'])
        row={k:record['identity'][k] for k in ('sequence_id','frame_id','dataset_index')}
        for name in names:
            extra={};output=None
            if name in ('A5','v1'):
                pred=c['base'] if name=='A5' else v1p;past=prev['base'] if name=='A5' else v1prev
                pred_valid=c['base_valid'];prob=c['base_probability'];uncertainty=c['base_uncertainty']
                warped,warped_valid=rewarp(past,record)
            else:
                model=models[name]
                with torch.autocast('cuda',dtype=torch.bfloat16):
                    output=model(native['inputs'])
                    # Every-frame VGGT is already independently causal. Reuse
                    # the actual recursively computed previous frame, with
                    # independent prefix replay checks on each rank and fixed
                    # failure cases. Endpoint-only diagnostics need a distinct
                    # VGGT endpoint schedule and must always be rerun.
                    replay=model.vggt_schedule=='endpoint_only' or i<context.world_size or row['dataset_index'] in (2,24,29)
                    previous_output=model(prefix_inputs(native['inputs'],native['inputs']['rgb'].shape[1]-1)) if replay else None
                end=output['endpoint'];past=previous_output['endpoint'] if previous_output is not None else output['frames'][-2];pred=end.disparity_left_px.float()
                pred_valid=end.valid_mask;prob=end.valid_probability.float();uncertainty=end.uncertainty.float()
                if model.vggt_schedule=='causal_each_frame' and replay:
                    difference=float((output['frames'][-2].disparity_left_px-past.disparity_left_px).abs().max())
                    if difference!=0:raise RuntimeError(f'causal prefix mismatch: {name}, {difference}')
                    checks.append({'arm':name,'dataset_index':row['dataset_index'],'independent_prefix_max_abs_px':difference})
                wp=warp_map(past.disparity_left_px,past.valid_mask,past.confidence,record['K_previous'],record['K'],record['baseline_previous'],record['baseline'],record['transform'])
                warped,warped_valid=wp.disparity_hr_px,wp.valid_mask
                init=output['details'][-1]['initialization']
                size=gt.shape[-2:]
                solved=F.interpolate(output['details'][-1]['stereo']['disparity_lr'].float()*2,size=size,mode='bilinear',align_corners=False)
                solver_error=(solved-gt).abs();final_error=(pred-gt).abs()
                extra={'native_stereo_epe_px':scalar_metric(solver_error.clamp_max(10),valid),
                  'solver_correct_geometry_wrong_rate':scalar_metric((final_error>1).float(),valid&(solver_error<1)),
                  'solver_wrong_geometry_correct_rate':scalar_metric((final_error<1).float(),valid&(solver_error>1))}
                if init is not None:
                    seed=F.interpolate(init['seed'].float()*8,size=size,mode='nearest-exact')
                    hist=F.interpolate((init['selected']>0).float(),size=size,mode='nearest-exact')>.5
                    extra.update({'history_initialization_rate':scalar_metric(hist.float(),valid),
                      'initializer_correct_solver_wrong_rate':scalar_metric(((solved-gt).abs()>1).float(),valid&((seed-gt).abs()<1)),
                      'initializer_wrong_solver_correct_rate':scalar_metric(((solved-gt).abs()<1).float(),valid&((seed-gt).abs()>1))})
                if row['dataset_index'] in (2,24,29):
                    x,y={2:(253,147),24:(581,200),29:(570,20)}[row['dataset_index']]
                    case={**row,'arm':name,'x':x,'y':y,'gt':float(gt[0,0,y,x]),'A5':float(c['base'][0,0,y,x]),'history_original':float(c['history'][0,0,y,x]),'v1':float(v1p[0,0,y,x]),'prediction':float(pred[0,0,y,x])}
                    if init is not None:case.update(initialized=float(seed[0,0,y,x]),solved_stereo=float(solved[0,0,y,x]),selected_history=bool(hist[0,0,y,x]))
                    cases.append(case)
            if not pred.isfinite().all() or not (pred>0).all():raise RuntimeError('nonfinite/nonpositive native prediction')
            factor=(record['K'][:,0,0]*record['baseline']).reshape(-1,1,1,1)
            values=endpoint_metric_values(predicted_depth_m=factor/pred,predicted_disparity_px=pred,predicted_valid_mask=pred_valid,
                predicted_valid_probability=prob,predicted_uncertainty=uncertainty,gt_disparity_px=gt,gt_valid_mask=valid,
                intrinsics_left=record['K'],baseline_m=record['baseline'],dynamic_mask=record['dynamic'],dynamic_available=record['dynamic_available'],
                detail_mask=record['detail'],matched_mask=record['matched'],boundary_mask=record['boundary'])
            transform=record['transform'];cosine=((transform[:,:3,:3].diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
            score=torch.linalg.vector_norm(transform[:,:3,3],dim=-1)+torch.acos(cosine)
            low=contract['temporal']['small_medium_threshold'];high=contract['temporal']['medium_large_threshold']
            motions={k:m.reshape(-1,1,1,1).expand_as(gt) for k,m in (('small_motion',score<=low),('medium_motion',(score>low)&(score<=high)),('large_motion',score>high))}
            values.update(temporal_residual_metric_values(current_prediction_disparity_px=pred,warped_previous_prediction_disparity_px=warped,
                current_gt_disparity_px=gt,warped_previous_gt_disparity_px=record['gt_warp'],current_prediction_valid=pred_valid,warped_prediction_valid=warped_valid,
                current_gt_valid=valid,warped_gt_valid=record['gt_warp_valid'],dynamic_mask=record['dynamic'],dynamic_available=record['dynamic_available'],motion_bucket_masks=motions))
            values.update(safety_values(pred,record,v1p));values.update(extra);accumulators[name].update(values)
            coverage[name].update(prob,(pred-gt).abs(),valid,error_cap_px=10)
            for key,value in values.items():row[name+'/'+key]=[value.numerator,value.count]
            if output is not None:del output,previous_output
        rows.append(row)
        if (i//context.world_size)%20==0:print(json.dumps({'rank':context.rank,'completed_per_rank':i//context.world_size+1,'elapsed_s':time.monotonic()-started}),flush=True)
        del native,record
    payload={'metrics':{n:accumulators[n].values for n in names},'rows':rows,'cases':cases,'checks':checks};gathered=[None]*context.world_size
    a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save(payload,a.output_dir/f'completed_shard_{context.rank:02d}.pt')
    if context.world_size>1:dist.all_gather_object(gathered,payload)
    else:gathered=[payload]
    for curve in coverage.values():curve.all_reduce_()
    if context.primary:
        merged={n:MetricAccumulator() for n in names};rows=[]
        for shard in gathered:
            rows+=shard['rows']
            for n in names:merged[n].merge(MetricAccumulator(shard['metrics'][n]))
        expected=len(paths)
        if len(rows)!=expected or len({r['dataset_index'] for r in rows})!=expected:raise RuntimeError('incomplete unique evaluation domain')
        if not a.max_samples and expected!=1294:raise RuntimeError('not the full original benchmark')
        metrics={n:merged[n].finalize() for n in names};curves={n:coverage[n].finalize(contract['validity']['accuracy_coverage_points']) for n in names}
        report={'status':'SMOKE_COMPLETE' if a.max_samples else 'COMPLETE','samples':expected,'metrics':metrics,'models':configs,'accuracy_coverage':curves,
          'fixed_cases':[c for s in gathered for c in s['cases']],'causal_prefix_checks':[c for s in gathered for c in s['checks']],
          'fixed_masks':'original immutable A5 and its original history for all arms; native predictions use their OWN recursive validity, confidence and geometry for temporal warp',
          'acceptance':'component tradeoffs reported; no five-way intersection requirement; benchmark reused for development',
          'previous_prediction':'actual causal recursive frame; independent sliced replay checked on first endpoint per rank and fixed cases; endpoint-only schedule always recomputed',
          'v1_checkpoint_sha256':digest(a.v1_checkpoint),'cache_lineage_sha256':digest(a.cache/'lineage.json'),'original_lineage_sha256':old_lineage,
          'source_sha256':digest(Path(__file__)),'evaluation_contract':contract,'elapsed_s':time.monotonic()-started}
        a.output_dir.mkdir(parents=True,exist_ok=True);(a.output_dir/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
        (a.output_dir/'per_sample.json').write_text(json.dumps(sorted(rows,key=lambda r:r['dataset_index']),indent=2)+'\n')
        with (a.output_dir/'metrics.csv').open('w') as handle:
            writer=csv.writer(handle);writer.writerow(['metric',*names])
            for key in metrics['v1']:writer.writerow([key,*[metrics[n][key]['value'] for n in names]])
        print(json.dumps({'status':report['status'],'samples':expected}),flush=True)
    if context.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
