"""Select probability mass of a surface, then fuse only inside that surface.

The original hypotheses and all learned parameters are retained. A radius of
2 HR pixels bounds the span of fused members by 4 px: candidates separated by
more than the requested 5 px large-degradation threshold cannot be averaged.
This is an experimentally tested design hypothesis, not an NMRF reproduction.
"""
from dataclasses import replace
import torch
from models.hypothesis_geometry import HypothesisGeometry

def surface_mode_reduce(hypotheses,probabilities,valid,radius_px=2.):
    with torch.no_grad():
        values=hypotheses.detach();prob=probabilities.detach();masses=[]
        for k in range(values.shape[1]):
            members=((values-values[:,k:k+1]).abs()<=radius_px)&valid&valid[:,k:k+1]
            masses.append((prob*members).sum(1,keepdim=True))
        mass=torch.cat(masses,1);selected=mass.argmax(1,keepdim=True)
        center=values.gather(1,selected)
        members=((values-center).abs()<=radius_px)&valid
    weights=probabilities*members
    weights=weights/weights.sum(1,keepdim=True).clamp_min(1e-8)
    prediction=(weights*hypotheses).sum(1,keepdim=True).clamp_min(1e-5)
    return prediction,selected,weights

class SurfaceModeGeometry(HypothesisGeometry):
    def __init__(self,mode_pool=True,radius_px=2.):
        super().__init__(use_memory=True);self.mode_pool=mode_pool;self.radius_px=radius_px
    def decode(self,inputs,t,state):
        output,detail=super().decode(inputs,t,state)
        if self.mode_pool:
            pred,selected,weights=surface_mode_reduce(detail['hypotheses'],detail['probabilities'],detail['candidate_valid'],self.radius_px)
            output=replace(output,disparity_left_px=pred)
            detail.update(selected=selected,mixture_weights=weights)
        else:
            detail['mixture_weights']=torch.zeros_like(detail['probabilities']).scatter_(1,detail['selected'],1.)
        return output,detail
