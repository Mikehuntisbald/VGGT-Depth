"""Frozen FFS stereo evidence inspired by CODD and TC-Stereo.

No disparity/GT prediction is used to construct features beyond the two fixed
inference candidates. The full-row ambiguity test excludes adjacent peak bins.
Inputs are the actual FFS 1/4-input-grid left/right matching feature tensors.
"""
import math
import torch
import torch.nn.functional as F


def _sample(feature, x, y, target_hw):
    h,w=target_hw
    grid=torch.stack((2*(x+.5)/w-1,2*(y+.5)/h-1),-1)
    return F.grid_sample(feature.float(),grid,align_corners=False,padding_mode='border')


def distinct_peak(correlation):
    """[B,H,left_x,right_x] cosine costs, nonnegative disparity only."""
    b,h,w,wr=correlation.shape
    left=torch.arange(w,device=correlation.device).view(1,1,w,1)
    right=torch.arange(wr,device=correlation.device).view(1,1,1,wr)
    visible=right<=left
    masked=correlation.masked_fill(~visible,-torch.inf)
    best,index=masked.max(-1)
    alternatives=visible&((right-index[...,None]).abs()>1)
    second=masked.masked_fill(~alternatives,-torch.inf).amax(-1)
    has_alternative=alternatives.any(-1)
    margin=torch.where(has_alternative,best-second,torch.zeros_like(best))
    return best,margin,index,has_alternative


def learned_evidence(left_feature,right_feature,current):
    """Return 24 channels on exact HR pixels, without changing candidates."""
    base,history=current['base'].float(),current['history'].float()
    b,_,h,w=base.shape
    _,channels,hf,wf=left_feature.shape
    if h/hf!=w/wf:raise ValueError('anisotropic feature scale')
    scale=w/wf
    if scale!=8:raise ValueError(f'expected FFS feature scale 8 HR px, got {scale}')
    yy,xx=torch.meshgrid(torch.arange(h,device=base.device),torch.arange(w,device=base.device),indexing='ij')
    x=xx.float()[None].expand(b,-1,-1);y=yy.float()[None].expand(b,-1,-1)
    left=F.normalize(_sample(left_feature,x,y,(h,w)),dim=1,eps=1e-6)
    cosine=[];distance=[];validity=[]
    for disparity in (base,history):
        for offset in (-1,0,1):
            xr=x-disparity[:,0]+offset*scale
            right=F.normalize(_sample(right_feature,xr,y,(h,w)),dim=1,eps=1e-6)
            valid=(xr>=0)&(xr<=w-1)
            cosine.append(((left*right).sum(1,keepdim=True)).clamp(-1,1))
            distance.append((left-right).abs().mean(1,keepdim=True)*math.sqrt(channels))
            validity.append(valid[:,None].float())
    l=F.normalize(left_feature.float(),dim=1,eps=1e-6)
    r=F.normalize(right_feature.float(),dim=1,eps=1e-6)
    correlation=torch.einsum('bchw,bchy->bhwy',l,r)
    best,margin,index,has_alternative=distinct_peak(correlation)
    disp=(torch.arange(wf,device=base.device).view(1,1,wf)-index).float()*scale
    def up(v):return F.interpolate(v[:,None].float(),size=(h,w),mode='nearest')
    extra=[up(best),up(margin),((up(disp)-base)/(scale*8)).clamp(-16,16),
           ((up(disp)-history)/(scale*8)).clamp(-16,16),up(has_alternative),current['history_valid'].float()]
    evidence=torch.cat(cosine+distance+validity+extra,1)
    if evidence.shape[1]!=24 or not evidence.isfinite().all():raise RuntimeError('invalid learned evidence')
    return {'learned':evidence.half()}
