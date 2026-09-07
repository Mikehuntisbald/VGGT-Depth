#!/usr/bin/env python3
"""Durable fixed-budget native architecture experiment and terminal receipts."""
import hashlib,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
RUN=ROOT/'runs/metric_stereo_video/surface_mode_geometry_20260907'
CACHE=Path('/tmp/vggt_native_stereo_20260907')

def main():
    RUN.mkdir(parents=True,exist_ok=True)
    status=RUN/'driver_status.json';started=time.time();env=os.environ.copy()
    env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    sources=[ROOT/'src/models/hypothesis_geometry.py',ROOT/'src/models/surface_mode_geometry.py',ROOT/'src/models/surface_mode_loss.py',ROOT/'tools/train_surface_mode_geometry.py',Path(__file__),ROOT/'docs/surface_mode_geometry_protocol.md']
    archive=RUN/'source_at_launch';archive.mkdir(exist_ok=True)
    hashes={}
    for source in sources:
        relative=source.relative_to(ROOT);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(source.read_bytes());hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (archive/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    for variant in ('wta','mode_pool'):
        directory=RUN/variant
        if directory.exists():raise RuntimeError(f'refusing to mix results in {directory}')
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/train_surface_mode_geometry.py',
          '--cache',str(CACHE),'--output-dir',str(directory),'--variant',variant,'--steps','2000','--initial-checkpoint','runs/metric_stereo_video/hypothesis_geometry_20260907/memory/final.pt']
        status.write_text(json.dumps({'status':'TRAINING','arm':variant,'command':cmd,'started_unix':started},indent=2)+'\n')
        with (RUN/f'{variant}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.write_text(json.dumps({'status':'FAILED','arm':variant,'returncode':result.returncode},indent=2)+'\n');raise SystemExit(result.returncode)
        summary=json.loads((directory/'summary.json').read_text())
        if summary['status']!='COMPLETE' or summary['steps']!=2000:raise RuntimeError('missing terminal training receipt')
    status.write_text(json.dumps({'status':'TRAINING_COMPLETE','elapsed_s':time.time()-started,'next':'full benchmark and raw input checks'},indent=2)+'\n')

if __name__=='__main__':main()
