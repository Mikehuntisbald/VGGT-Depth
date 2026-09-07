import torch
from models.temporal_candidate_repair import TemporalCandidateRepair
from models.codd_temporal_repair import CoddTemporalRepair,codd_loss


def test_equal_factorization_preserves_v1_and_fallback():
    torch.manual_seed(42)
    v1=TemporalCandidateRepair();model=CoddTemporalRepair(v1)
    features=torch.randn(20,31);base=torch.rand(20,1)*20+1;history=torch.rand_like(base)*30
    valid=torch.rand_like(base)>.5
    expected,_=v1(features,base,history,valid)
    out=model(features,base,history,valid)
    torch.testing.assert_close(out['prediction'],expected,rtol=1e-5,atol=1e-5)
    assert torch.equal(out['prediction'][~valid],base[~valid])
    assert all(not p.requires_grad for p in model.base.parameters())


def test_codd_gradient_targets_and_empty_groups():
    base=torch.tensor([[10.],[1.],[1.]])
    history=torch.tensor([[1.],[10.],[1.]])
    target=torch.ones(3,1)
    reset=torch.full((3,1),.5,requires_grad=True)
    fusion=torch.full((3,1),.5,requires_grad=True)
    gate=reset*fusion
    out=dict(prediction=base+gate*(history-base),history=history,history_valid=torch.ones_like(base,dtype=torch.bool),reset_weight=reset,fusion_weight=fusion)
    sample=dict(base=base,target=target,target_valid=torch.ones_like(base,dtype=torch.bool),weight=torch.ones_like(base))
    loss,parts=codd_loss(out,sample,'codd')
    loss.backward()
    assert reset.grad[0]<0 and reset.grad[1]>0
    assert fusion.grad[0]<0 and fusion.grad[1]>0
    assert all(x.isfinite() for x in parts.values())
    sample['target_valid'].fill_(False)
    loss,_=codd_loss(out,sample,'codd_regret')
    assert loss.isfinite() and loss.item()==0
