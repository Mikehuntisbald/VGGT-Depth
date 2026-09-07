import torch
from models.learned_temporal_evidence import distinct_peak,learned_evidence
from models.learned_codd_temporal_repair import LearnedCoddTemporalRepair
from models.temporal_candidate_repair import TemporalCandidateRepair


def test_distinct_peak_excludes_adjacent_bin_and_negative_disparity():
    c=torch.zeros(1,1,6,6)
    c[0,0,5]=torch.tensor([.2,.4,.99,1.,.98,.3])
    c[0,0,0,4]=10 # impossible negative disparity
    best,margin,index,valid=distinct_peak(c)
    assert index[0,0,5]==3
    torch.testing.assert_close(margin[0,0,5],torch.tensor(.6))
    assert index[0,0,0]==0 and not valid[0,0,0]


def test_hr_to_feature_matching_units_and_model_fallback():
    torch.manual_seed(42)
    left=torch.randn(1,16,4,8);right=torch.zeros_like(left)
    right[...,:-2]=left[...,2:]
    base=torch.full((1,1,32,64),16.);history=torch.full_like(base,32.)
    valid=torch.ones_like(base,dtype=torch.bool)
    bank=learned_evidence(left,right,dict(base=base,history=history,history_valid=valid))
    assert bank['learned'].shape==(1,24,32,64)
    torch.testing.assert_close(bank['learned'][:,1,:,24:52].float(),torch.ones(1,32,28),atol=1e-3,rtol=0)
    assert torch.all(bank['learned'][:,7,:,24:52]<1e-4)
    features=torch.randn(1,31,32,64)
    v1=TemporalCandidateRepair();model=LearnedCoddTemporalRepair(v1)
    out=model(features,base,history,valid,bank)
    ref,_=v1(features,base,history,valid)
    torch.testing.assert_close(out['prediction'],ref,atol=1e-5,rtol=1e-5)
    valid.fill_(False)
    assert torch.equal(model(features,base,history,valid,bank)['prediction'],base)
