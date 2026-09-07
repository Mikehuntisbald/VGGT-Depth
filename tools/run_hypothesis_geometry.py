#!/usr/bin/env python3
"""Run equal-budget decoder controls after the native benchmark releases GPUs."""
import hashlib,json,os,struct,subprocess,sys,time,zipfile
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];RUN=ROOT/'runs/metric_stereo_video/hypothesis_geometry_20260907';CACHE=Path('/tmp/vggt_native_stereo_20260907')
PREVIOUS=ROOT/'runs/metric_stereo_video/native_temporal_stereo_20260907'

def main():
    RUN.mkdir(parents=True,exist_ok=True);status=RUN/'driver_status.json'
    status.write_text(json.dumps({'status':'WAITING_FOR_PREVIOUS_EVALUATION'},indent=2)+'\n')
    while not (PREVIOUS/'evaluation/metrics.json').exists():time.sleep(20)
    if json.loads((PREVIOUS/'evaluation/metrics.json').read_text())['status']!='COMPLETE':raise RuntimeError('previous architecture benchmark incomplete')
    # The new decoder uses selected mmap fields, not the giant cost volume.
    # Release ONLY unused volume pages; leave every byte on disk unchanged.
    retained={0,2,24,29,300,600,900,1200};dropped=0
    for split in ('train','validation'):
      for path in (CACHE/split).glob('*.pt'):
        if split=='validation' and int(path.stem) in retained:continue
        with zipfile.ZipFile(path) as archive:
            storage=[z for z in archive.infolist() if z.file_size==132120576]
        if len(storage)!=1 or storage[0].compress_type!=zipfile.ZIP_STORED:raise RuntimeError('native volume storage layout changed')
        z=storage[0]
        with path.open('rb') as f:
            f.seek(z.header_offset+26);name_length,extra_length=struct.unpack('<HH',f.read(4))
            offset=z.header_offset+30+name_length+extra_length
            os.posix_fadvise(f.fileno(),offset,z.file_size,os.POSIX_FADV_DONTNEED)
        dropped+=1
    (RUN/'page_cache_receipt.json').write_text(json.dumps({'files':dropped,'unused_volume_bytes_per_file':132120576,'data_files_modified':False},indent=2)+'\n')
    sources=[ROOT/'src/models/hypothesis_geometry.py',ROOT/'src/models/hypothesis_geometry_loss.py',ROOT/'tools/train_hypothesis_geometry.py',Path(__file__),ROOT/'docs/hypothesis_geometry_protocol.md']
    archive=RUN/'source_at_launch';hashes={}
    for source in sources:
        relative=source.relative_to(ROOT);target=archive/relative;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(source.read_bytes());hashes[str(relative)]=hashlib.sha256(source.read_bytes()).hexdigest()
    (archive/'sha256.json').write_text(json.dumps(hashes,indent=2)+'\n')
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
    started=time.time()
    for variant in ('no_memory','memory'):
        out=RUN/variant
        if out.exists():raise RuntimeError('refusing to mix an existing trained arm')
        cmd=[sys.executable,'-m','torch.distributed.run','--standalone','--nproc_per_node=8','tools/train_hypothesis_geometry.py',
           '--cache',str(CACHE),'--output-dir',str(out),'--variant',variant,'--steps','2000']
        status.write_text(json.dumps({'status':'TRAINING','variant':variant,'command':cmd},indent=2)+'\n')
        with (RUN/f'{variant}.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT)
        if result.returncode:
            status.write_text(json.dumps({'status':'FAILED','variant':variant,'returncode':result.returncode},indent=2)+'\n');raise SystemExit(result.returncode)
        summary=json.loads((out/'summary.json').read_text())
        if summary['status']!='COMPLETE' or summary['steps']!=2000:raise RuntimeError('missing complete training receipt')
    status.write_text(json.dumps({'status':'TRAINING_COMPLETE','elapsed_s':time.time()-started},indent=2)+'\n')

if __name__=='__main__':main()
