"""Pixel-wise surface-hypothesis decoding of stereo, VGGT and causal history.

NMRF (CVPR 2024) motivates preserving labels until per-label pixel offsets and
winner-takes-all scores are decoded. This is a new geometry decoder, not a
residual applied to A5/v1 geometry outputs; no original geometry decoder runs.
This implementation is not the full NMRF variational/message-passing model.
"""
from dataclasses import dataclass
import torch
from torch import nn
import torch.nn.functional as F
from models.metric_stereo_video_geometry import align_vggt_inverse_depth_to_metric_stereo
from models.metric_stereo_video_system import left_right_stereo_consistency
from models.native_temporal_stereo.model import warp_map,sample_right_feature,gather_winners

SPATIAL=((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1))
LOCAL=((0,0),(-1,0),(1,0),(0,-1),(0,1),(-2,0),(2,0),(0,-2),(0,2))
HISTORY_START=14

def shift(x,dx,dy):
    h,w=x.shape[-2:];p=F.pad(x,(2,2,2,2))
    return p[...,2+dy:2+dy+h,2+dx:2+dx+w]

def up(x,size,mode='bilinear'):
    return F.interpolate(x.float(),size=size,mode=mode,**({'align_corners':False} if mode=='bilinear' else {}))

def fold_pixels(x):
    return F.pixel_unshuffle(x,8)

def unfold_pixels(x):
    return F.pixel_shuffle(x,8)

def stereo_modes(left,right,k=3):
    """Distinct local correlation peaks, not adjacent bins of a single mode."""
    with torch.no_grad(),torch.autocast(device_type=left.device.type,enabled=False):
        left=F.normalize(left.float(),dim=1);right=F.normalize(right.float(),dim=1)
        _,_,h,w=left.shape;d=min(48,w)
        corr=torch.einsum('bchw,bchv->bhwv',left,right)
        x=torch.arange(w,device=left.device)[None,None,:,None]
        disparity=torch.arange(d,device=left.device)[None,None,None,:]
        index=x-disparity
        volume=corr.gather(-1,index.clamp_min(0).expand(left.shape[0],h,-1,-1))
        volume=volume.masked_fill(index<0,-2).permute(0,3,1,2).contiguous()
        peak=F.max_pool3d(volume[:,None],kernel_size=(3,1,1),stride=1,padding=(1,0,0))[:,0]
        available=volume.masked_fill(volume<peak,-1e4);available[:,0]=-1e4
        bins=torch.arange(d,device=left.device)[None,:,None,None];indices=[];supports=[]
        for _ in range(k):
            score,index=available.max(1,keepdim=True);indices.append(index);supports.append((score>-1.01)&(index>0))
            available=available.masked_fill((bins-index).abs()<=1,-1e4)
        return torch.cat(indices,1).float()*8,torch.cat(supports,1)

@dataclass
class HypothesisOutput:
    disparity_left_px:torch.Tensor
    valid_logits:torch.Tensor
    log_variance:torch.Tensor
    valid_probability:torch.Tensor
    valid_mask:torch.Tensor
    uncertainty:torch.Tensor
    confidence:torch.Tensor

class LabelInteraction(nn.Module):
    """Competition between labels and spatial exchange of their embeddings."""
    def __init__(self,channels=64):
        super().__init__();self.norm=nn.LayerNorm(channels)
        self.qkv=nn.Linear(channels,3*channels);self.project=nn.Linear(channels,channels)
        self.spatial=nn.Sequential(nn.Conv2d(channels,channels,3,padding=1),nn.GroupNorm(8,channels),nn.SiLU(),nn.Conv2d(channels,channels,1))
    def forward(self,x,b,k,valid):
        _,c,h,w=x.shape
        tokens=x.reshape(b,k,c,h,w).permute(0,3,4,1,2).reshape(b*h*w,k,c)
        q,ke,v=self.qkv(self.norm(tokens)).chunk(3,dim=-1)
        def heads(z):return z.reshape(-1,k,4,c//4).transpose(1,2)
        keys=valid.permute(0,2,3,1).reshape(b*h*w,k).clone();keys[:,0]=True
        attended=F.scaled_dot_product_attention(heads(q),heads(ke),heads(v),attn_mask=keys[:,None,None,:]).transpose(1,2).reshape(-1,k,c)
        tokens=tokens+self.project(attended)
        x=tokens.reshape(b,h,w,k,c).permute(0,3,4,1,2).reshape(b*k,c,h,w)
        return x+self.spatial(x)

class HypothesisGeometry(nn.Module):
    def __init__(self,use_memory=True):
        super().__init__();self.use_memory=use_memory;self.vggt_schedule='causal_each_frame'
        self.key=nn.Conv2d(224,16,1)
        # Pixel-unshuffled RGB retains all 64 pixel positions within each cell.
        self.context=nn.Sequential(nn.Conv2d(224+192+384,64,1),nn.GroupNorm(8,64),nn.SiLU())
        self.source=nn.Embedding(5,8)
        # Per-pixel photo, LR residual, log disparity, confidence, temporal photo;
        # plus learned feature matching and a source-type embedding.
        self.label=nn.Sequential(nn.Conv2d(64+5*64+1+8,64,1),nn.GroupNorm(8,64),nn.SiLU())
        self.interactions=nn.ModuleList([LabelInteraction(),LabelInteraction()])
        self.pixel_head=nn.Conv2d(64,128,1)
        self.quality=nn.Conv2d(64,128,1)
        nn.init.zeros_(self.pixel_head.weight);nn.init.zeros_(self.pixel_head.bias)
        nn.init.zeros_(self.quality.weight);nn.init.zeros_(self.quality.bias)

    def build_candidates(self,inputs,t,state):
        rgb=inputs['rgb'].float()/255.;size=rgb.shape[-2:];encoded=inputs['encoded']
        stereo=inputs['stereo_lr'][:,t].float()*2
        right=inputs['right_disparity_lr'][:,t].float()*2
        K=inputs['K'][:,t,0].float();baseline=inputs['baseline_m'][:,t].float()
        factor=(K[:,0,0]*baseline).reshape(-1,1,1,1)
        vv=inputs['vggt'];relative=up(vv['inverse_relative'][:,t],stereo.shape[-2:]);vc=up(vv['confidence'][:,t],stereo.shape[-2:])
        with torch.no_grad(),torch.autocast(device_type=stereo.device.type,enabled=False):
            lr=left_right_stereo_consistency(stereo/2,right/2,maximum_error_px=1.,confidence_temperature_px=.5)
            gauge=align_vggt_inverse_depth_to_metric_stereo(relative,stereo/factor,relative_confidence=vc,
               metric_confidence=lr.confidence_left,relative_valid_mask=(relative>0)&(vc>0),metric_valid_mask=lr.valid_left_mask,minimum_overlap=64)
        candidates=[up(stereo,size),up(gauge.inverse_depth_m_inv*factor,size)]
        valid=[torch.isfinite(candidates[0])&(candidates[0]>0),up(((relative>0)&gauge.valid_mask.reshape(-1,1,1,1)).float(),size,'nearest-exact')>.5]
        confidence=[up(lr.confidence_left,size),up(vc,size)];types=[0,1]
        seed=encoded['seed'][t:t+1].float()*8
        for dx,dy in SPATIAL:
            value=up(shift(seed,dx,dy),size,'nearest-exact');candidates.append(value);valid.append(value>0);confidence.append(torch.ones_like(value));types.append(2)
        peaks,pv=stereo_modes(encoded['left'][t:t+1],encoded['right'][t:t+1])
        for k in range(3):
            value=up(peaks[:,k:k+1],size,'nearest-exact');candidates.append(value);valid.append(up(pv[:,k:k+1].float(),size,'nearest-exact')>.5);confidence.append(torch.ones_like(value));types.append(3)
        temporal_photo=[torch.zeros_like(candidates[0]) for _ in candidates]
        if state is not None and self.use_memory:
            transported=warp_map(state['disparity'],state['valid'],state['confidence'],state['K'],K,state['baseline'],baseline,inputs['T_current_from_previous'][:,t])
            past_rgb=gather_winners(state['rgb'],transported.source_uv)
            for dx,dy in LOCAL:
                value=shift(transported.disparity_hr_px,dx,dy);support=shift(transported.valid_mask.float(),dx,dy)>.5
                candidates.append(value);valid.append(support);confidence.append(shift(transported.confidence,dx,dy));types.append(4)
                temporal_photo.append((rgb[:,t,0]-shift(past_rgb,dx,dy)).abs().mean(1,keepdim=True))
        else:
            for _ in LOCAL:
                value=torch.zeros_like(candidates[0]);candidates.append(value);valid.append(value.bool());confidence.append(value);temporal_photo.append(value);types.append(4)
        values=torch.cat(candidates,1).detach();support=torch.cat(valid,1).detach()&torch.isfinite(values)&(values>0)
        values=torch.where(support,values,torch.zeros_like(values))
        return values,support,torch.cat(confidence,1).detach(),torch.cat(temporal_photo,1).detach(),types,up(right,size)

    def decode(self,inputs,t,state):
        rgb=inputs['rgb'].float()/255.;left_rgb,right_rgb=rgb[:,t,0],rgb[:,t,1];b,_,h,w=left_rgb.shape
        encoded=inputs['encoded'];left=encoded['left'][t:t+1].float();right=encoded['right'][t:t+1].float();size=left.shape[-2:]
        candidates,valid,confidence,temporal_photo,types,right_disparity=self.build_candidates(inputs,t,state)
        k=candidates.shape[1]
        context=self.context(torch.cat((left,up(inputs['vggt']['feature'][:,t],size),fold_pixels(torch.cat((left_rgb,right_rgb),1))),1))
        with torch.no_grad():
            right_sample,bounds=sample_right_feature(right_rgb.expand(k,-1,-1,-1),candidates.reshape(k,1,h,w))
            photo=(right_sample-left_rgb.expand(k,-1,-1,-1)).abs().mean(1,keepdim=True)
            rd,_=sample_right_feature(right_disparity.expand(k,-1,-1,-1),candidates.reshape(k,1,h,w))
            lr_error=((rd-candidates.reshape(k,1,h,w)).abs()/20).clamp_max(1)
            photo=torch.where(bounds,photo,torch.ones_like(photo))
        left_key=F.normalize(self.key(left).float(),dim=1);right_key=F.normalize(self.key(right).float(),dim=1)
        coarse=up(candidates,size,'nearest-exact').reshape(k,1,*size)/8
        matched,_=sample_right_feature(right_key.expand(k,-1,-1,-1),coarse)
        similarity=(left_key.expand(k,-1,-1,-1)*F.normalize(matched,dim=1,eps=1e-6)).sum(1,keepdim=True)
        cues=torch.cat((photo,lr_error,torch.log1p(candidates.reshape(k,1,h,w))/6,confidence.reshape(k,1,h,w),temporal_photo.reshape(k,1,h,w)),1)
        source=self.source(torch.tensor(types,device=left.device))[:,:,None,None].expand(-1,-1,*size)
        labels=self.label(torch.cat((context.expand(k,-1,-1,-1),fold_pixels(cues),similarity,source),1))
        label_support=F.max_pool2d(valid.float(),8,8)>.5
        for block in self.interactions:labels=block(labels,b,k,label_support)
        decoded=unfold_pixels(self.pixel_head(labels)).reshape(b,k,2,h,w)
        scores=decoded[:,:,0].float().masked_fill(~valid,-1e4)
        offsets=4*torch.tanh(decoded[:,:,1].float());hypotheses=(candidates+offsets).clamp_min(1e-5)
        probabilities=torch.softmax(scores,dim=1);selected=scores.argmax(1,keepdim=True)
        hard=torch.zeros_like(probabilities).scatter_(1,selected,1.)
        routing=hard+(probabilities-probabilities.detach()) if self.training else hard
        disparity=(routing*hypotheses).sum(1,keepdim=True)
        quality=unfold_pixels(self.quality(context)).float();valid_logits=2+quality[:,:1];log_variance=(-2+quality[:,1:2]).clamp(-8,8)
        probability=torch.sigmoid(valid_logits);uncertainty=log_variance.exp();conf=probability*torch.exp(-.5*uncertainty)
        output=HypothesisOutput(disparity,valid_logits,log_variance,probability,probability>=.5,uncertainty,conf)
        details={'candidates':candidates,'candidate_valid':valid,'hypotheses':hypotheses,'probabilities':probabilities,'selected':selected,
                 'stereo_lr':inputs['stereo_lr'][:,t],'stereo_reference_hr':candidates[:,:1],'source_types':types}
        return output,details

    def forward(self,inputs,gradient_frames=2):
        if inputs['rgb'].shape[0]!=1:raise ValueError('one encoded causal clip per rank')
        count=inputs['rgb'].shape[1];state=None;frames=[];details=[]
        for t in range(count):
            enabled=torch.is_grad_enabled() and self.training and t>=count-gradient_frames
            with torch.set_grad_enabled(enabled),torch.autocast(device_type=inputs['rgb'].device.type,dtype=torch.bfloat16,enabled=inputs['rgb'].is_cuda,cache_enabled=enabled):
                output,detail=self.decode(inputs,t,state);frames.append(output);details.append(detail)
                if self.use_memory:
                    state={'disparity':output.disparity_left_px.detach(),'valid':output.valid_mask.detach(),'confidence':output.confidence.detach(),
                           'rgb':inputs['rgb'][:,t,0].float()/255.,'K':inputs['K'][:,t,0].float(),'baseline':inputs['baseline_m'][:,t].float()}
        return {'frames':frames,'details':details,'endpoint':frames[-1]}
