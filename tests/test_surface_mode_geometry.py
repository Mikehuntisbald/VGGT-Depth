import torch
from models.surface_mode_geometry import surface_mode_reduce

def test_duplicate_good_hypotheses_do_not_lose_to_one_bad_label():
    values=torch.tensor([10.,100.,100.5,99.5]).reshape(1,4,1,1)
    probability=torch.tensor([.35,.25,.2,.2]).reshape_as(values)
    result,selected,weights=surface_mode_reduce(values,probability,torch.ones_like(values,dtype=torch.bool))
    assert int(probability.argmax())==0
    assert int(selected)!=0
    torch.testing.assert_close(result,torch.tensor([[[[100.]]]]),atol=1e-5,rtol=0)
    assert weights[0,0,0,0]==0

def test_surface_fusion_cannot_mix_candidates_more_than_four_pixels_apart():
    values=torch.tensor([10.,11.,100.,101.]).reshape(1,4,1,1)
    leaf=torch.tensor([.1,.1,.4,.4],requires_grad=True)
    probability=leaf.reshape_as(values)
    result,_,weights=surface_mode_reduce(values,probability,torch.ones_like(values,dtype=torch.bool))
    assert 100<=float(result)<=101
    assert weights[0,:2].sum()==0
    result.sum().backward()
    assert torch.isfinite(leaf.grad).all()
    assert leaf.grad[:2].abs().sum()==0
    assert leaf.grad[2:].abs().sum()>0
