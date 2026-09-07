#!/usr/bin/env python3
"""Wait for fixed training receipts, then perform full native acceptance work."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/metric_stereo_video/native_temporal_stereo_20260907';CACHE=Path('/tmp/vggt_native_stereo_20260907')

def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()

def main():
    status=RUN/'finalizer_status.json';env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    while True:
        current=json.loads((RUN/'driver_status.json').read_text())
        if current['status']=='FAILED':raise RuntimeError('training driver failed')
        if current['status']=='TRAINING_COMPLETE':break
        status.write_text(json.dumps({'status':'WAITING_FOR_TRAINING','training':current},indent=2)+'\n');time.sleep(30)
    # Keep the complete validation working set in the 512 GiB container memory
    # limit. Dropping unused training file pages does not modify any data.
    for path in (CACHE/'train').glob('*.pt'):
        with path.open('rb') as handle:os.posix_fadvise(handle.fileno(),0,0,os.POSIX_FADV_DONTNEED)
    checkpoints=[str(RUN/name/'final.pt') for name in ('late','early_seed','early_state')]
    launch=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8']
    commands=[('evaluation',launch+['tools/eval_native_temporal_stereo.py','--cache',str(CACHE),'--original-cache','/tmp/vggt_temporal_repair_20260907',
         '--v1-checkpoint','runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt','--checkpoints',*checkpoints,'--diagnostics','--output-dir',str(RUN/'evaluation')]),
      ('raw_final_checks',launch+['tools/check_native_stereo_raw.py','--cache',str(CACHE),'--config','runs/metric_stereo_video/formal_a5_seed42/resolved_config.yaml',
         '--backbone-checkpoint','runs/metric_stereo_video/formal_a5_seed42/checkpoints/step_0006000','--checkpoints',*checkpoints,'--indices','0,2,24,29,300,600,900,1200','--output',str(RUN/'raw_final_checks.json')]),
      ('unit_tests',[sys.executable,'-m','pytest','-q','tests/test_native_temporal_stereo.py']),
      ('report',[sys.executable,'tools/report_native_temporal_stereo.py','--run-dir',str(RUN)])]
    for name,cmd in commands:
        status.write_text(json.dumps({'status':'RUNNING','stage':name,'command':cmd},indent=2)+'\n')
        with (RUN/f'{name}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.write_text(json.dumps({'status':'FAILED','stage':name,'returncode':result.returncode},indent=2)+'\n');raise SystemExit(result.returncode)
    original=json.loads((ROOT/'runs/metric_stereo_video/formal_a5_seed42/run_receipt.json').read_text())['runtime_source_sha256']
    changed=[name for name,value in original.items() if digest(ROOT/name)!=value]
    if changed:raise RuntimeError('original runtime sources changed: '+str(changed))
    launched=json.loads((RUN/'source_at_launch/sha256.json').read_text())
    if any(digest(ROOT/name)!=value for name,value in launched.items()):raise RuntimeError('training source changed during execution')
    v1=digest(ROOT/'runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt')
    if v1!='60fe5a02263c2d39a622e0a0f64db381154c01e01bb33c9dcc21dd295556f269':raise RuntimeError('v1 checkpoint changed')
    archive=RUN/'source_final';sources=[*sorted((ROOT/'src/models/native_temporal_stereo').glob('*.py')),*sorted((ROOT/'tools').glob('*native*stereo.py')),ROOT/'tests/test_native_temporal_stereo.py',ROOT/'docs/native_temporal_stereo_protocol.md']
    hashes={}
    for path in sources:
        relative=path.relative_to(ROOT);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes());hashes[str(relative)]=digest(path)
    (archive/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    cache_receipts=RUN/'cache_receipts';cache_receipts.mkdir(exist_ok=True)
    for name in ('lineage.json','structure.json','native_replay_checks.json','train/plan.json','train/complete.json','validation/plan.json','validation/complete.json','training_labels/complete.json'):
        (cache_receipts/name.replace('/','_')).write_bytes((CACHE/name).read_bytes())
    result=json.loads((RUN/'results.json').read_text())
    summary={'status':'COMPLETE','trained_arms':result['trained_arms'],'updates_per_arm':1000,'validation_endpoints_per_arm':1294,'raw_input_and_causality':'PASS',
      'source_files_checked':len(original),'base_source_files_changed':changed,'v1_checkpoint_sha256':v1,'training_sources_unchanged':True,'cache_temporary':str(CACHE),'completed_unix_s':time.time()}
    (RUN/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n');status.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
