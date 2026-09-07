import torch
from torch import nn
from models.native_temporal_stereo.model import warp_state,TemporalStereoInitializer,StateFusion,sample_volume


def test_warp_uses_its_own_geometry_and_preserves_units():
    K=torch.tensor([[[80.,0,15.5],[0,80.,15.5],[0,0,1.]]])
    state={'image_size':(32,32),'K':K,'baseline':torch.tensor([.1]),'disparity_hr':torch.full((1,1,32,32),16.),
           'valid':torch.ones(1,1,32,32,dtype=torch.bool),'confidence':torch.ones(1,1,32,32),'key':torch.ones(1,16,4,4),'hidden':[torch.randn(1,4,4,4)]}
    result=warp_state(state,K,torch.tensor([.1]),torch.eye(4)[None],(4,4))
    assert result['valid'].all()
    torch.testing.assert_close(result['disparity'],torch.full((1,1,4,4),2.),atol=1e-6,rtol=0)
    # No current-depth estimate is an argument to this transport operation.
    torch.testing.assert_close(result['hidden'][0],state['hidden'][0])


def test_initialization_selects_a_surface_and_falls_back_when_history_missing():
    model=TemporalStereoInitializer(4,2,[4],vggt_channels=2,propagate_hidden=False).eval()
    class PreferVisibleHistory(nn.Module):
        def forward(self,x):return 10*x[:,-3:-2]
    model.cost=PreferVisibleHistory()
    evidence={'seed':torch.ones(1,1,4,4),'left':torch.randn(1,4,4,4),'right':torch.randn(1,4,4,4),'volume':torch.randn(1,2,16,4,4),'hidden':[torch.randn(1,4,4,4)]}
    history={'disparity':torch.full((1,1,4,4),10.),'valid':torch.ones(1,1,4,4,dtype=torch.bool),'confidence':torch.ones(1,1,4,4),'key':torch.randn(1,16,4,4),'hidden':[torch.randn(1,4,4,4)]}
    result=model(evidence,torch.zeros(1,2,4,4),history)
    assert torch.equal(result['seed'],history['disparity'])
    assert torch.all((result['seed']-result['candidates']).abs().amin(1)==0)
    history['valid'].fill_(False)
    result=model(evidence,torch.zeros(1,2,4,4),history)
    assert torch.equal(result['seed'],evidence['seed'])
    assert torch.equal(result['hidden'][0],evidence['hidden'][0])


def test_state_fusion_zero_confidence_cannot_change_current():
    f=StateFusion(4);current=torch.randn(1,4,3,3);past=torch.randn_like(current)*100
    assert torch.equal(f(current,past,torch.zeros(1,1,3,3)),current)
    output=f(current,past,torch.ones(1,1,3,3));output.mean().backward()
    assert f.gates.weight.grad is not None


def test_cost_volume_queries_use_native_disparity_bins():
    volume=torch.arange(4.).view(1,1,4,1,1).expand(1,2,4,3,3)
    value,valid=sample_volume(volume,torch.full((1,1,3,3),1.5))
    torch.testing.assert_close(value,torch.full((1,2,3,3),1.5))
    assert valid.all()
