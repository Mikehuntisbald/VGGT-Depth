"""Verify deployed raw-image path equals fixed-cache prediction for saved endpoints."""
import hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from models.temporal_candidate_repair import TemporalCandidateRepair
from models.learned_codd_temporal_repair import LearnedCoddTemporalRepair
from tools.eval_temporal_candidate_repair import to_device

def main():
    torch.set_num_threads(4)
    run=ROOT/'runs/metric_stereo_video/temporal_literature_20260907'
    cache=Path('/tmp/vggt_temporal_repair_20260907');bank=Path('/tmp/vggt_learned_evidence_20260907')
    vp=torch.load(ROOT/'runs/metric_stereo_video/temporal_repair_20260907/head_v1/final.pt',map_location='cpu',weights_only=False)
    v1=TemporalCandidateRepair();v1.load_state_dict(vp['model'])
    receipt=json.loads((run/'raw_inference/receipt.json').read_text())
    cp=Path(json.loads((run/'raw_checkpoint.json').read_text())['checkpoint']);p=torch.load(cp,map_location='cpu',weights_only=False)
    assert receipt['checkpoint_sha256']==hashlib.sha256(cp.read_bytes()).hexdigest()
    model=LearnedCoddTemporalRepair(v1,v1_shift=p['config']['v1_shift']).cuda().eval();model.load_state_dict(p['model'])
    rows=[]
    with torch.inference_mode():
        for path in sorted((run/'raw_inference').glob('*.pt')):
            raw=to_device(torch.load(path,map_location='cpu',weights_only=False),'cuda')
            record=to_device(torch.load(cache/'validation'/path.name,map_location='cpu',weights_only=False),'cuda')['current']
            evidence=to_device(torch.load(bank/'validation'/path.name,map_location='cpu',weights_only=False),'cuda')['current']
            out=model(record['features'],record['base'],record['history'],record['history_valid'],evidence,p['logit_shift'])
            delta=(out['prediction']-raw['disparity_hr_px']).abs()
            torch.testing.assert_close(out['prediction'],raw['disparity_hr_px'],rtol=0,atol=1e-4)
            assert torch.equal(raw['valid_mask'],record['base_valid'])
            assert raw['causality_checked'] and raw['future_perturbation_past_feature_max_abs']==0
            rows.append(dict(index=int(path.stem),pixels=delta.numel(),maximum_abs_difference_px=float(delta.max()),mean_abs_difference_px=float(delta.mean()),validity_identical=True))
    assert len(rows)==8
    receipt=dict(status='PASS',samples=len(rows),checkpoint_sha256=hashlib.sha256(cp.read_bytes()).hexdigest(),future_perturbation_past_feature_max_abs=0,results=rows)
    (run/'raw_inference_parity.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(receipt))
if __name__=='__main__':main()
