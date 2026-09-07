"""Native FFS iterative solver with explicit encoded-evidence/state interfaces.

Uses the installed NVIDIA Fast-FoundationStereo modules and geometry/upsampling
interfaces under third_party/Fast-FoundationStereo/LICENSE.txt. No upstream
source is modified. This adapter preserves its recurrence and float32 disparity
state while exposing initialization and final latent states to a causal model.
"""
from __future__ import annotations
import copy
import torch
from torch import nn


class NativeStereoRefiner(nn.Module):
    def __init__(self, ffs, iterations=8):
        super().__init__()
        self.update_block=copy.deepcopy(ffs.update_block)
        self.spx_2_gru=copy.deepcopy(ffs.spx_2_gru)
        self.spx_gru=copy.deepcopy(ffs.spx_gru)
        self.register_buffer('dx',ffs.dx.detach().clone())
        self.dtype=ffs.dtype
        self.low_memory=bool(ffs.args.get('low_memory',False))
        self.levels=int(ffs.args.corr_levels)
        self.iterations=int(iterations)
        self.to(dtype=torch.bfloat16)
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # Frozen solver, differentiable with respect to initialization/latents.
        return super().train(False)

    def forward(self, evidence, seed=None, hidden=None):
        from core.geometry import Combined_Geo_Encoding_Volume
        from core.submodule import context_upsample
        cost=evidence['volume'].to(self.dtype)
        geometry=Combined_Geo_Encoding_Volume(evidence['left'].to(self.dtype),evidence['right'].to(self.dtype),cost,num_levels=self.levels)
        initial=evidence['seed'] if seed is None else seed
        hidden=[x.clone() for x in evidence['hidden']] if hidden is None else hidden
        b,_,h,w=initial.shape
        coords=torch.arange(w,device=initial.device,dtype=torch.float32).view(1,1,w,1).expand(b,h,w,1).contiguous()
        disparity=initial.to(self.dtype);iterations=[]
        for index in range(self.iterations):
            # FFS detaches disparity between optimizer iterations. Seed learning
            # therefore also has explicit GT supervision in the training loss.
            disparity=disparity.detach()
            local=geometry(disparity,coords,dx=self.dx,low_memory=self.low_memory)
            with torch.autocast('cuda',dtype=torch.bfloat16,enabled=disparity.is_cuda):
                hidden,mask_features,delta=self.update_block(hidden,evidence['context'],local.to(self.dtype),disparity,evidence['attention'])
            disparity=disparity+delta.to(self.dtype)
            iterations.append(disparity)
        with torch.autocast('cuda',dtype=torch.bfloat16,enabled=disparity.is_cuda):
            up_feature=self.spx_2_gru(mask_features.to(self.dtype),evidence['stem'].to(self.dtype))
            weights=torch.softmax(self.spx_gru(up_feature),dim=1)
            disparity_lr=context_upsample(disparity.to(self.dtype)*4.,weights).unsqueeze(1).to(self.dtype)
        return dict(disparity_lr=disparity_lr,disparity_feature=disparity,hidden=hidden,iterations=iterations)
