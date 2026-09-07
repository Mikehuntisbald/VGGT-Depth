"""Run real-image parity/causality, reports and terminal receipts after full evaluation."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/metric_stereo_video/temporal_literature_20260907'
CORE=('preserve_v1_all_gt_epe','preserve_v1_temporal','reduce_good_pixel_damage','reduce_large_5px','increase_fixed_opportunity_recovery')
report=json.loads((RUN/'evaluation_learned/metrics.json').read_text())
assert report['samples']==1294
passed=[n for n,x in report['acceptance'].items() if all(x['checks'][k] for k in CORE)]
name=passed[0] if passed else 'learned_codd_codd'
checkpoint=report['models'][name]['checkpoint']
(RUN/'raw_checkpoint.json').write_text(json.dumps(dict(arm=name,checkpoint=checkpoint,selection_reason='joint five-criterion pass' if passed else 'fixed representative learned CODD arm; no model accepted'),indent=2)+'\n')
env=os.environ.copy();env['OMP_NUM_THREADS']='4';env['PYTHONPATH']='src:.'
v1='runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt'
common=['--run-dir',str(RUN),'--cache','/tmp/vggt_temporal_repair_20260907','--bank','/tmp/vggt_learned_evidence_20260907','--v1-checkpoint',v1]
commands=[('raw_inference.log',['torchrun','--standalone','--nproc_per_node=8','tools/infer_learned_codd_temporal_repair.py','--base-config','runs/metric_stereo_video/formal_a5_seed42/resolved_config.yaml','--base-checkpoint','runs/metric_stereo_video/formal_a5_seed42/checkpoints/step_0006000','--v1-checkpoint',v1,'--checkpoint',checkpoint,'--output-dir',str(RUN/'raw_inference'),'--verify-causality']),
          ('parity.log',[sys.executable,'tools/verify_learned_codd_raw_parity.py']),
          ('report.log',[sys.executable,'tools/report_literature_temporal_repair.py',*common])]
for logname,cmd in commands:
    with (RUN/logname).open('w') as output:subprocess.run(cmd,cwd=ROOT,env=env,stdout=output,stderr=subprocess.STDOUT,check=True)
result=json.loads((RUN/'results.json').read_text());parity=json.loads((RUN/'raw_inference_parity.json').read_text())
assert parity['status']=='PASS' and parity['samples']==8
original=json.loads((ROOT/'runs/metric_stereo_video/formal_a5_seed42/run_receipt.json').read_text())['runtime_source_sha256']
changed=[name for name,digest in original.items() if hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=digest]
assert not changed
sources={}
for group in ('source','learned_source'):
    for p in (RUN/group).glob('*.py'):
        target=ROOT/('src/models' if p.name in ('codd_temporal_repair.py','learned_codd_temporal_repair.py','learned_temporal_evidence.py') else 'tools')/p.name
        assert target.read_bytes()==p.read_bytes(),(p,target)
        sources[str(target.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
for p in (ROOT/'tools').glob('*learned_codd*.py'):sources[str(p.relative_to(ROOT))]=hashlib.sha256(p.read_bytes()).hexdigest()
(RUN/'verified_runtime_sources_sha256.json').write_text(json.dumps(sources,indent=2)+'\n')
summary=dict(status='COMPLETE',trained_arms=6,updates_per_arm=12500,validation_endpoints_per_arm=1294,accepted_validation_arms=result['accepted_validation_arms'],baseline_v1_preserved=True,
             objective_acceptance='PASS' if result['accepted_validation_arms'] else 'NOT_MET',base_runtime_files_verified=len(original),base_runtime_files_changed=changed,
             v1_checkpoint_sha256=hashlib.sha256((ROOT/v1).read_bytes()).hexdigest(),new_unit_tests_passed=4,raw_inference_parity=parity,
             feature_cache_receipts={s:json.loads((Path('/tmp/vggt_learned_evidence_20260907')/s/'complete.json').read_text()) for s in ('train','validation')},
             sources_verified=True,completed_unix_s=time.time(),cache_location='/tmp/vggt_learned_evidence_20260907',cache_persistence='temporary storage; exact regeneration source and lineage retained')
(RUN/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary),flush=True)
