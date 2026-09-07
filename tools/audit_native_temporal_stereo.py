#!/usr/bin/env python3
"""Check actual trained parameter ownership and preserved original artifacts."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src'),str(ROOT/'third_party/Fast-FoundationStereo')]
import torch
from tools.eval_native_temporal_stereo import load_package
from tools.train_native_temporal_stereo import digest
from models.native_temporal_stereo.model import NativeTemporalStereo

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    package=load_package(a.cache);rows={}
    for arm in ('late','early_seed','early_state'):
        payload=torch.load(a.run_dir/arm/'final.pt',map_location='cpu',weights_only=False)
        summary=json.loads((a.run_dir/arm/'summary.json').read_text())
        if summary['status']!='COMPLETE' or payload['step']!=1000:raise RuntimeError('incomplete training')
        if digest(a.run_dir/arm/'final.pt')!=summary['checkpoint_sha256']:raise RuntimeError('checkpoint hash changed')
        torch.manual_seed(payload['config']['seed']);initial=NativeTemporalStereo(package,variant=arm).state_dict()
        changes={'refiner':0.,'geometry':0.,'initializer':0.};changed_tensors={k:0 for k in changes}
        for name,value in payload['model'].items():
            if not torch.isfinite(value).all():raise RuntimeError('nonfinite checkpoint tensor: '+name)
            group=name.split('.')[0]
            delta=(value.float()-initial[name].float()).abs()
            changes[group]=max(changes[group],float(delta.max()))
            changed_tensors[group]+=int(bool(delta.any()))
        if changes['refiner']!=0:raise RuntimeError('frozen native stereo solver weights/buffers changed')
        if changes['geometry']==0:raise RuntimeError('joint geometry decoder did not train')
        if arm!='late' and changes['initializer']==0:raise RuntimeError('new temporal matcher did not train')
        if arm=='early_seed':
            if any(not torch.equal(v,initial[n]) for n,v in payload['model'].items() if n.startswith('initializer.state_fusion')):raise RuntimeError('seed-only control trained state fusion')
        rows[arm]={'maximum_parameter_or_buffer_change':changes,'changed_tensor_counts':changed_tensors,'checkpoint_sha256':summary['checkpoint_sha256']}
    original=json.loads((ROOT/'runs/metric_stereo_video/formal_a5_seed42/run_receipt.json').read_text())['runtime_source_sha256']
    changed=[n for n,value in original.items() if digest(ROOT/n)!=value]
    if changed:raise RuntimeError('original A5 sources changed')
    result={'status':'PASS','arms':rows,'A5_runtime_sources_checked':len(original),'A5_runtime_sources_changed':changed,
      'v1_checkpoint_sha256':digest(ROOT/'runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt')}
    (a.run_dir/'parameter_ownership_audit.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))

if __name__=='__main__':main()
