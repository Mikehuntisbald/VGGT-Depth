import types
import torch
from torch import nn
from models.hypothesis_geometry import HypothesisGeometry,stereo_modes

def test_one_coarse_cell_can_decode_two_distinct_surfaces():
    model=HypothesisGeometry(use_memory=False).eval();h=w=32
    def candidates(self,inputs,t,state):
        values=torch.zeros(1,23,h,w);values[:,0]=10.;values[:,1]=100.
        valid=values>0
        return values,valid,valid.float(),torch.zeros_like(values),[0,1]+[2]*9+[3]*3+[4]*9,torch.ones(1,1,h,w)
    model.build_candidates=types.MethodType(candidates,model)
    class PixelChoice(nn.Module):
        def forward(self,x):
            out=x.new_zeros(23,128,*x.shape[-2:])
            for p in range(64):out[0 if p%8<4 else 1,p]=10.
            return out
    model.pixel_head=PixelChoice()
    inputs={'rgb':torch.zeros(1,1,2,3,h,w,dtype=torch.uint8),
      'encoded':{'left':torch.randn(1,224,4,4),'right':torch.randn(1,224,4,4)},
      'vggt':{'feature':torch.zeros(1,1,192,4,4)},'stereo_lr':torch.ones(1,1,1,16,16)}
    with torch.no_grad():out,_=model.decode(inputs,0,None)
    expected=torch.where(torch.arange(w)%8<4,10.,100.).reshape(1,1,1,w).expand(1,1,h,w)
    torch.testing.assert_close(out.disparity_left_px,expected,rtol=0,atol=0)
    assert not ((out.disparity_left_px>10)&(out.disparity_left_px<100)).any()

def test_correlation_proposals_are_valid_distinct_modes():
    torch.manual_seed(42);left=torch.randn(1,16,4,12);right=torch.randn_like(left)
    proposals,valid=stereo_modes(left,right)
    for i in range(3):
        assert ((proposals[:,i]/8<=torch.arange(12)[None,None,:])|~valid[:,i]).all()
        for j in range(i):
            assert (((proposals[:,i]-proposals[:,j]).abs()>=16)|~valid[:,i]|~valid[:,j]).all()
