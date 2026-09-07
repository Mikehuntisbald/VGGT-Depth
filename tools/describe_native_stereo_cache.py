#!/usr/bin/env python3
"""Derive native pruned dimensions and sample identities from encoded records."""
import argparse,json
from pathlib import Path
import torch

def main():
    p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);a=p.parse_args()
    paths=sorted((a.cache/'train').glob('*.pt'))
    if not paths:raise RuntimeError('no encoded training clips')
    first=torch.load(paths[0],map_location='cpu',weights_only=False,mmap=True)['inputs']['encoded']
    specification={'hidden_channels':[int(v.shape[1]) for v in first['hidden']],
      'volume_channels':int(first['volume'].shape[1]),'feature_channels':int(first['left'].shape[1]),
      'source':'observed frozen encoder outputs; serialized FFS args.hidden_dims may be stale after pruning'}
    identities={}
    for path in paths:
        record=torch.load(path,map_location='cpu',weights_only=False,mmap=True)
        identities[path.name]=record['identity']
    for path,value in ((a.cache/'structure.json',specification),(a.cache/'train/identities.json',identities)):
        if path.exists():
            if json.loads(path.read_text())!=value:raise RuntimeError('existing cache metadata differs: '+str(path))
        else:path.write_text(json.dumps(value,indent=2)+'\n')
    print(json.dumps({'status':'PASS','structure':specification,'identities':len(identities)}))

if __name__=='__main__':main()
