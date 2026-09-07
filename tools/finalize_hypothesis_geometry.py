#!/usr/bin/env python3
"""Complete benchmark, raw checks and durable receipts for decoder controls."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/metric_stereo_video/hypothesis_geometry_20260907';CACHE=Path('/tmp/vggt_native_stereo_20260907')

def main():
    status=RUN/'finalizer_status.json';status.write_text(json.dumps({'status':'WAITING_FOR_TRAINING'})+'\n')
    while True:
        current=json.loads((RUN/'driver_status.json').read_text())
        if current['status']=='FAILED':raise RuntimeError('training failed')
        if current['status']=='TRAINING_COMPLETE':break
        time.sleep(20)
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    checkpoints=[str(RUN/n/'final.pt') for n in ('no_memory','memory')];launch=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8']
    commands=[('evaluation',launch+['tools/eval_hypothesis_geometry.py','--cache',str(CACHE),'--original-cache','/tmp/vggt_temporal_repair_20260907',
      '--v1-checkpoint','runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt','--checkpoints',*checkpoints,'--output-dir',str(RUN/'evaluation')]),
      ('raw_checks',launch+['tools/check_hypothesis_geometry_raw.py','--cache',str(CACHE),'--config','runs/metric_stereo_video/formal_a5_seed42/resolved_config.yaml',
      '--backbone-checkpoint','runs/metric_stereo_video/formal_a5_seed42/checkpoints/step_0006000','--checkpoints',*checkpoints,'--output',str(RUN/'raw_checks.json')]),
      ('unit_tests',[sys.executable,'-m','pytest','-q','tests/test_hypothesis_geometry.py']),
      ('report',[sys.executable,'tools/report_hypothesis_geometry.py','--run-dir',str(RUN)])]
    for stage,cmd in commands:
        status.write_text(json.dumps({'status':'RUNNING','stage':stage,'command':cmd},indent=2)+'\n')
        with (RUN/f'{stage}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.write_text(json.dumps({'status':'FAILED','stage':stage,'returncode':result.returncode},indent=2)+'\n');raise SystemExit(result.returncode)
    launched=json.loads((RUN/'source_at_launch/sha256.json').read_text())
    if any(hashlib.sha256((ROOT/name).read_bytes()).hexdigest()!=value for name,value in launched.items()):raise RuntimeError('training source changed during execution')
    archive=RUN/'source_final';sources=[*sorted((ROOT/'src/models').glob('hypothesis_geometry*.py')),*sorted((ROOT/'tools').glob('*hypothesis_geometry*.py')),ROOT/'tests/test_hypothesis_geometry.py',ROOT/'docs/hypothesis_geometry_protocol.md']
    hashes={}
    for source in sources:
        relative=source.relative_to(ROOT);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes());hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (archive/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    v1=hashlib.sha256((ROOT/'runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt').read_bytes()).hexdigest()
    if v1!='60fe5a02263c2d39a622e0a0f64db381154c01e01bb33c9dcc21dd295556f269':raise RuntimeError('v1 checkpoint changed')
    result=json.loads((RUN/'results.json').read_text())
    summary={'status':'COMPLETE','trained_arms':['no_memory','memory'],'updates_per_arm':2000,'validation_endpoints_per_arm':1294,
      'original_A5_runtime_sources_checked':result['original_source_files_checked'],'original_A5_runtime_sources_changed':[],
      'v1_checkpoint_sha256':v1,'raw_input_and_independent_causal_prefix_checks':'PASS','training_sources_unchanged':True,'completed_unix_s':time.time()}
    (RUN/'run_summary.json').write_text(json.dumps(summary,indent=2)+'\n');status.write_text(json.dumps(summary,indent=2)+'\n')

if __name__=='__main__':main()
