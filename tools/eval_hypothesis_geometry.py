#!/usr/bin/env python3
"""Full immutable-domain evaluation of replacement geometry decoding."""
import argparse,csv,json,sys,time
from datetime import timedelta
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'third_party/Fast-FoundationStereo')]
import torch
import torch.distributed as dist
from models.hypothesis_geometry import HypothesisGeometry,HISTORY_START
from models.native_temporal_stereo.model import warp_map
from models.temporal_candidate_repair import TemporalCandidateRepair
from metrics.metric_stereo_video import MetricAccumulator,AccuracyCoverageHistogram,endpoint_metric_values,temporal_residual_metric_values,scalar_metric
from tools.train_hypothesis_geometry import compact_inputs,move,digest
from tools.eval_native_temporal_stereo import prefix_inputs
from tools.eval_temporal_candidate_repair import rewarp
from tools.eval_controlled_temporal_repair import safety_values
from tools.train_metric_stereo_video import _distributed_context,_read_config

def metric_values(pred,valid_pred,prob,uncertainty,warped,warped_valid,record,v1p,contract):
    gt=record['target'];valid=record['target_valid'];factor=(record['K'][:,0,0]*record['baseline']).reshape(-1,1,1,1)
    values=endpoint_metric_values(predicted_depth_m=factor/pred,predicted_disparity_px=pred,predicted_valid_mask=valid_pred,
      predicted_valid_probability=prob,predicted_uncertainty=uncertainty,gt_disparity_px=gt,gt_valid_mask=valid,intrinsics_left=record['K'],baseline_m=record['baseline'],
      dynamic_mask=record['dynamic'],dynamic_available=record['dynamic_available'],detail_mask=record['detail'],matched_mask=record['matched'],boundary_mask=record['boundary'])
    tr=record['transform'];cosine=((tr[:,:3,:3].diagonal(dim1=-2,dim2=-1).sum(-1)-1)/2).clamp(-1,1)
    motion=torch.linalg.vector_norm(tr[:,:3,3],dim=-1)+torch.acos(cosine)
    low=contract['temporal']['small_medium_threshold'];high=contract['temporal']['medium_large_threshold']
    masks={k:m.reshape(-1,1,1,1).expand_as(gt) for k,m in (('small_motion',motion<=low),('medium_motion',(motion>low)&(motion<=high)),('large_motion',motion>high))}
    values.update(temporal_residual_metric_values(current_prediction_disparity_px=pred,warped_previous_prediction_disparity_px=warped,
      current_gt_disparity_px=gt,warped_previous_gt_disparity_px=record['gt_warp'],current_prediction_valid=valid_pred,warped_prediction_valid=warped_valid,
      current_gt_valid=valid,warped_gt_valid=record['gt_warp_valid'],dynamic_mask=record['dynamic'],dynamic_available=record['dynamic_available'],motion_bucket_masks=masks))
    values.update(safety_values(pred,record,v1p));return values

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--original-cache',type=Path,required=True)
    p.add_argument('--checkpoints',type=Path,nargs='+',required=True);p.add_argument('--v1-checkpoint',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--max-samples',type=int)
    a=p.parse_args();torch.set_num_threads(1)
    original_init=dist.init_process_group
    def evaluation_init(*args,**kwargs):kwargs.setdefault('timeout',timedelta(hours=1));return original_init(*args,**kwargs)
    dist.init_process_group=evaluation_init
    try:ctx=_distributed_context()
    finally:dist.init_process_group=original_init
    device=ctx.device;contract=_read_config(ROOT/'configs/metric_stereo_video/evaluation_contract.yaml')
    lineage=digest(a.cache/'lineage.json')
    if json.loads((a.cache/'lineage.json').read_text())['sampling_plan_lineage_sha256']!=digest(a.original_cache/'lineage.json'):raise RuntimeError('original benchmark lineage mismatch')
    if not (a.cache/'validation/complete.json').exists():raise RuntimeError('incomplete encoded benchmark')
    vp=torch.load(a.v1_checkpoint,map_location='cpu',weights_only=False);v1=TemporalCandidateRepair().to(device).eval();v1.load_state_dict(vp['model'])
    models={};configs={}
    for checkpoint in a.checkpoints:
        payload=torch.load(checkpoint,map_location='cpu',weights_only=False);cfg=payload['config'];name=cfg['variant']
        if cfg['cache_lineage_sha256']!=lineage or cfg['model_source_sha256']!=digest(ROOT/'src/models/hypothesis_geometry.py'):raise RuntimeError('trained source/input lineage changed')
        model=HypothesisGeometry(use_memory=name=='memory').to(device).eval();model.load_state_dict(payload['model'])
        models[name]=model;configs[name]={'checkpoint':str(checkpoint),'checkpoint_sha256':digest(checkpoint),'config':cfg}
    names=['A5','v1',*models];acc={n:MetricAccumulator() for n in names};coverage={n:AccuracyCoverageHistogram(4096,device) for n in names}
    paths=sorted((a.original_cache/'validation').glob('*.pt'))
    if a.max_samples:paths=paths[:a.max_samples]
    rows=[];cases=[];checks=[];started=time.monotonic()
    with torch.inference_mode():
      for i in range(ctx.rank,len(paths),ctx.world_size):
        original=torch.load(a.cache/'validation'/paths[i].name,map_location='cpu',weights_only=False,mmap=True)
        inputs=move(compact_inputs(original),device);record=move(torch.load(paths[i],map_location='cpu',weights_only=False),device)
        if original['identity']!=record['identity']:raise RuntimeError('identity mismatch')
        torch.testing.assert_close(original['labels']['target'][:,-1].to(device),record['target'],atol=0,rtol=0)
        gt=record['target'];valid=record['target_valid'];c=record['current'];pv=record['previous']
        v1p,_=v1(c['features'],c['base'],c['history'],c['history_valid'],vp['logit_shift'])
        v1prev,_=v1(pv['features'],pv['base'],pv['history'],pv['history_valid'],vp['logit_shift'])
        row={k:record['identity'][k] for k in ('sequence_id','frame_id','dataset_index')}
        for name in names:
            extra={};output=None
            if name in ('A5','v1'):
                pred=c['base'] if name=='A5' else v1p;past=pv['base'] if name=='A5' else v1prev
                pred_valid=c['base_valid'];prob=c['base_probability'];uncertainty=c['base_uncertainty'];warped,wv=rewarp(past,record)
            else:
                with torch.autocast('cuda',dtype=torch.bfloat16):output=models[name](inputs)
                end=output['endpoint'];past=output['frames'][-2];pred=end.disparity_left_px.float();pred_valid=end.valid_mask;prob=end.valid_probability;uncertainty=end.uncertainty
                if i<ctx.world_size or row['dataset_index'] in (2,24,29):
                    with torch.autocast('cuda',dtype=torch.bfloat16):independent=models[name](prefix_inputs(inputs,inputs['rgb'].shape[1]-1))
                    difference=float((past.disparity_left_px-independent['endpoint'].disparity_left_px).abs().max())
                    if difference!=0:raise RuntimeError('independent causal prefix changed')
                    checks.append({'variant':name,'dataset_index':row['dataset_index'],'max_abs_px':difference})
                transported=warp_map(past.disparity_left_px,past.valid_mask,past.confidence,record['K_previous'],record['K'],record['baseline_previous'],record['baseline'],record['transform'])
                warped,wv=transported.disparity_hr_px,transported.valid_mask
                detail=output['details'][-1];errors=(detail['hypotheses']-gt).abs().masked_fill(~detail['candidate_valid'],float('inf'));oracle=errors.amin(1,keepdim=True)
                be=(c['base']-gt).abs();he=(c['history']-gt).abs();fixed=valid&c['history_valid']&(be-he>.1)
                extra={'candidate_oracle_epe_px':scalar_metric(oracle.clamp_max(10),valid),
                  'fixed_opportunity_candidate_recovery_rate':scalar_metric((be-oracle>.1).float(),fixed),
                  'correct_candidate_chosen_rate':scalar_metric(((pred-gt).abs()<1).float(),valid&(oracle<1)),
                  'history_chosen_rate':scalar_metric((detail['selected']>=HISTORY_START).float(),valid),
                  'VGGT_chosen_rate':scalar_metric((detail['selected']==1).float(),valid)}
                if row['dataset_index'] in (2,24,29):
                    x,y={2:(253,147),24:(581,200),29:(570,20)}[row['dataset_index']]
                    cases.append({**row,'variant':name,'x':x,'y':y,'GT':float(gt[0,0,y,x]),'A5':float(c['base'][0,0,y,x]),'v1':float(v1p[0,0,y,x]),'prediction':float(pred[0,0,y,x]),
                      'selected_candidate':int(detail['selected'][0,0,y,x]),'candidate_base_values':detail['candidates'][0,:,y,x].tolist(),
                      'candidate_predictions':detail['hypotheses'][0,:,y,x].tolist(),'candidate_valid':detail['candidate_valid'][0,:,y,x].tolist(),'probabilities':detail['probabilities'][0,:,y,x].tolist()})
            if not pred.isfinite().all() or not (pred>0).all():raise RuntimeError('invalid prediction values')
            values=metric_values(pred,pred_valid,prob,uncertainty,warped,wv,record,v1p,contract);values.update(extra);acc[name].update(values)
            coverage[name].update(prob,(pred-gt).abs(),valid,error_cap_px=10)
            for key,v in values.items():row[name+'/'+key]=[v.numerator,v.count]
            if output is not None:del output
        rows.append(row)
        if (i//ctx.world_size)%30==0:print(json.dumps({'rank':ctx.rank,'completed_per_rank':i//ctx.world_size+1,'elapsed_s':time.monotonic()-started}),flush=True)
        del original,inputs,record
    payload={'metrics':{n:acc[n].values for n in names},'rows':rows,'cases':cases,'checks':checks};a.output_dir.mkdir(parents=True,exist_ok=True)
    torch.save(payload,a.output_dir/f'completed_shard_{ctx.rank:02d}.pt');gathered=[None]*ctx.world_size
    if ctx.world_size>1:dist.all_gather_object(gathered,payload)
    else:gathered=[payload]
    for curve in coverage.values():curve.all_reduce_()
    if ctx.primary:
        merged={n:MetricAccumulator() for n in names};rows=[]
        for shard in gathered:
            rows+=shard['rows']
            for n in names:merged[n].merge(MetricAccumulator(shard['metrics'][n]))
        if len(rows)!=len(paths) or len({r['dataset_index'] for r in rows})!=len(paths):raise RuntimeError('not complete unique domain')
        if not a.max_samples and len(rows)!=1294:raise RuntimeError('not the original 1294 endpoints')
        metrics={n:merged[n].finalize() for n in names};curves={n:coverage[n].finalize(contract['validity']['accuracy_coverage_points']) for n in names}
        result={'status':'SMOKE_COMPLETE' if a.max_samples else 'COMPLETE','samples':len(rows),'metrics':metrics,'models':configs,'accuracy_coverage':curves,
          'fixed_failure_points':[c for s in gathered for c in s['cases']],'causal_prefix_checks':[c for s in gathered for c in s['checks']],
          'cache_lineage_sha256':lineage,'v1_checkpoint_sha256':digest(a.v1_checkpoint),'source_sha256':digest(Path(__file__)),'evaluation_contract':contract,
          'fixed_masks':'original immutable A5/history for opportunity and safety; own recursive prediction validity/confidence for temporal warp',
          'acceptance':'report tradeoffs; no five-way intersection requirement; repeated development benchmark','elapsed_s':time.monotonic()-started}
        (a.output_dir/'metrics.json').write_text(json.dumps(result,indent=2)+'\n');(a.output_dir/'per_sample.json').write_text(json.dumps(sorted(rows,key=lambda r:r['dataset_index']),indent=2)+'\n')
        with (a.output_dir/'metrics.csv').open('w') as f:
            writer=csv.writer(f);writer.writerow(['metric',*names])
            for key in metrics['v1']:writer.writerow([key,*[metrics[n][key]['value'] for n in names]])
        print(json.dumps({'status':result['status'],'samples':len(rows)}))
    if ctx.world_size>1:dist.destroy_process_group()

if __name__=='__main__':main()
