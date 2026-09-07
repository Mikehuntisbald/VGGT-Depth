import torch
from models.temporal_candidate_repair import TemporalCandidateRepair
from models.controlled_temporal_repair import ControlledTemporalRepair,evidence_bank,shift_integer,supervision_loss


def test_integer_candidates_do_not_mix_two_surfaces():
    disparity=torch.tensor([[[[10.,10.,100.,100.]]]])
    for dx in (-1,0,1):
        shifted=shift_integer(disparity,dx,0)
        assert set(shifted.flatten().tolist())<={0.,10.,100.}
    assert shift_integer(disparity,-1,0)[0,0,0,2]==10


def test_control_initialization_exact_v1():
    torch.manual_seed(42);v1=TemporalCandidateRepair()
    model=ControlledTemporalRepair(v1)
    features=torch.randn(32,31);base=torch.full((32,1),50.);history=torch.full_like(base,10.)
    valid=torch.ones_like(base,dtype=torch.bool)
    expected,_=v1(features,base,history,valid)
    actual=model(features,base,history,valid)['prediction']
    torch.testing.assert_close(actual,expected,rtol=0,atol=0)


def test_recovery_rejection_and_large_regret_gradients():
    def grad(base,history,target,coeff):
        gate=torch.tensor([[.5]],requires_grad=True)
        b=torch.tensor([[base]]);h=torch.tensor([[history]]);t=torch.tensor([[target]])
        out={'gate':gate,'history':h,'history_valid':torch.ones_like(b,dtype=torch.bool),'prediction':b+gate*(h-b)}
        batch={'base':b,'target':t,'weight':torch.ones_like(b),'target_valid':torch.ones_like(b,dtype=torch.bool)}
        total,parts=supervision_loss(out,batch,coeff);total.backward()
        return gate.grad.item(),parts
    recovery,_=grad(20.,5.,5.,{'recovery':1.})
    rejection,parts=grad(5.,20.,5.,{'rejection':1.,'tail':1.})
    assert recovery<0 and rejection>0
    assert parts['tail']>0 and parts['regret']>0


def test_evidence_and_alignment_are_finite_and_inference_only():
    torch.manual_seed(42);h,w=12,14
    rgb=torch.rand(1,3,h,w)
    y,x=torch.meshgrid(torch.arange(h),torch.arange(w),indexing='ij')
    current={'base':torch.full((1,1,h,w),3.),'history':torch.full((1,1,h,w),4.),
        'history_valid':torch.ones(1,1,h,w,dtype=torch.bool),'source_uv':torch.stack((x,y))[None].float(),
        'features':torch.zeros(1,31,h,w)}
    bank=evidence_bank(rgb,rgb,rgb,current)
    assert bank['match'].shape==(1,12,h,w)
    assert bank['align_features'].shape==(1,9,8,h,w)
    assert all(value.isfinite().all() for value in bank.values())
    v1=TemporalCandidateRepair();model=ControlledTemporalRepair(v1,matching=True,alignment=True,ambiguity=True)
    output=model(current['features'],current['base'],current['history'],current['history_valid'],bank)
    expected,_=v1(current['features'],current['base'],current['history'],current['history_valid'])
    torch.testing.assert_close(output['prediction'],expected,rtol=0,atol=0)
    assert (output['history']==4).all()
    output['prediction'].sum().backward()
    assert model.extra.weight.grad is not None


def test_categorical_routing_preserves_small_gap_and_has_training_gradient():
    v1=TemporalCandidateRepair();model=ControlledTemporalRepair(v1,categorical=True)
    features=torch.zeros(2,31);base=torch.tensor([[10.],[10.]])
    history=torch.tensor([[11.],[100.]]);valid=torch.ones(2,1,dtype=torch.bool)
    out=model(features,base,history,valid)
    expected,_=v1(features,base,history,valid)
    torch.testing.assert_close(out['prediction'][0],expected[0],rtol=0,atol=0)
    assert out['prediction'][1].item() in (10.,100.)
    out['prediction'].sum().backward()
    assert model.base.net[-1].bias.grad.abs().item()>0


def test_residual_head_preserves_v1_and_freezes_its_weights():
    torch.manual_seed(42);v1=TemporalCandidateRepair()
    model=ControlledTemporalRepair(v1,residual=True)
    x=torch.randn(16,31);base=torch.full((16,1),10.);history=torch.full((16,1),20.);valid=torch.ones_like(base,dtype=torch.bool)
    actual=model(x,base,history,valid)
    expected,_=v1(x,base,history,valid)
    torch.testing.assert_close(actual['prediction'],expected,rtol=0,atol=0)
    actual['prediction'].sum().backward()
    assert all(not p.requires_grad and p.grad is None for p in model.base.parameters())
    assert model.residual_score[-1].bias.grad.abs().item()>0
