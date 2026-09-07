"""Controlled recovery/rejection supervision and inference-only matching evidence.

Disparities are HR pixels. Features never consume GT. Position candidates use
integer gathering; no foreground/background depths are averaged to create them.
"""
from __future__ import annotations
import copy
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .temporal_candidate_repair import TemporalCandidateRepair, sample_right

OFFSETS = ((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1))


def shift_integer(value: Tensor, dx: int, dy: int) -> Tensor:
    """Gather value(x+dx,y+dy), zero outside [B,C,H,W], without interpolation."""
    h,w=value.shape[-2:]
    padded=F.pad(value,(1,1,1,1))
    return padded[...,1+dy:1+dy+h,1+dx:1+dx+w]


def _descriptors(rgb: Tensor) -> tuple[Tensor,Tensor,Tensor]:
    gray=rgb.float().mean(1,keepdim=True)
    h,w=gray.shape[-2:]
    patch=F.unfold(gray,3,padding=1).reshape(gray.shape[0],9,h,w)
    centered=patch-patch.mean(1,keepdim=True)
    texture=centered.square().mean(1,keepdim=True).sqrt()
    normalized=F.normalize(centered,dim=1,eps=0.01)
    census=torch.tanh(20*centered)
    return normalized,census,texture


def evidence_bank(left: Tensor, right: Tensor, previous_left: Tensor,
                  current: dict[str,Tensor]) -> dict[str,Tensor]:
    """Build [B,C,H,W] evidence and 9 HR-disparity candidates from past/current inputs."""
    base,history=current['base'].float(),current['history'].float()
    b,_,h,w=base.shape
    uv=current['source_uv'].long()
    indices=(uv[:,1].clamp(0,h-1)*w+uv[:,0].clamp(0,w-1)).flatten(1)[:,None]
    history_rgb=torch.gather(previous_left.float().flatten(2),2,indices.expand(-1,3,-1)).reshape(b,3,h,w)
    candidates=torch.cat([shift_integer(history,dx,dy) for dx,dy in OFFSETS],1)
    validity=torch.cat([shift_integer(current['history_valid'].float(),dx,dy)>0.5 for dx,dy in OFFSETS],1)
    candidates=torch.where(validity,candidates,base.expand_as(candidates))
    alignment=[]
    for k,(dx,dy) in enumerate(OFFSETS):
        transported=shift_integer(history_rgb,dx,dy)
        rgb_difference=(left.float()-transported).abs()
        alignment.append(torch.cat((rgb_difference,
            ((candidates[:,k:k+1]-history)/8).clamp(-16,16),
            ((candidates[:,k:k+1]-base)/8).clamp(-16,16),
            torch.full_like(base,float(dx)),torch.full_like(base,float(dy)),
            validity[:,k:k+1].float()),1))
    align_features=torch.stack(alignment,1) # B,K,8,H,W
    ln,lc,texture=_descriptors(left)
    rn,rc,_=_descriptors(right)
    matching=[]
    for disp in (base,history):
        sampled,ok=sample_right(torch.cat((rn,rc,right.float()),1),disp)
        zncc=(1-(ln*F.normalize(sampled[:,:9],dim=1,eps=0.01)).sum(1,keepdim=True))/2
        census=(lc-sampled[:,9:18]).abs().mean(1,keepdim=True)/2
        photo=(left.float()-sampled[:,18:21]).abs().mean(1,keepdim=True)
        matching.append(torch.cat((zncc,census,photo,ok.float()),1))
    match=torch.cat((matching[0],matching[1],matching[1][:,:3]-matching[0][:,:3],texture),1) # 12
    costs=[]
    for fraction in (0.,0.25,0.5,0.75,1.):
        sampled,ok=sample_right(right,base+fraction*(history-base))
        photo=(left.float()-sampled).abs().mean(1,keepdim=True)
        costs.append(torch.where(ok,photo,torch.ones_like(photo)))
    costs=torch.cat(costs,1)
    ordered=costs.sort(1).values
    probability=torch.softmax(-costs/0.03,1)
    entropy=-(probability*probability.clamp_min(1e-8).log()).sum(1,keepdim=True)/1.6094379124
    spread=(candidates.amax(1,keepdim=True)-candidates.amin(1,keepdim=True))/8
    ambiguity=torch.cat((ordered[:,:1],ordered[:,1:2]-ordered[:,:1],entropy,
        probability[:,1:4].sum(1,keepdim=True),(history-base).abs().clamp_max(128)/8,
        spread.clamp_max(16),texture,current['features'][:,7:8].float()),1) # 8
    return {'match':match.half(),'ambiguity':ambiguity.half(),
            'align_features':align_features.half(),'candidates':candidates,
            'candidate_valid':validity}


class ControlledTemporalRepair(nn.Module):
    """All arms share v1 initialization; disabled new weights have zero effect."""
    def __init__(self, v1: TemporalCandidateRepair, *, matching: bool=False,
                 alignment: bool=False, ambiguity: bool=False, categorical: bool=False, residual: bool=False):
        super().__init__()
        self.base=copy.deepcopy(v1)
        self.matching=matching;self.alignment=alignment;self.ambiguity=ambiguity;self.categorical=categorical
        self.extra=nn.Linear(28,64,bias=False)
        nn.init.zeros_(self.extra.weight)
        self.align_score=nn.Sequential(nn.Linear(8,32),nn.SiLU(),nn.Linear(32,1))
        nn.init.zeros_(self.align_score[-1].weight);nn.init.zeros_(self.align_score[-1].bias)
        self.residual_score=None
        if residual:
            self.base.requires_grad_(False);self.extra.requires_grad_(False);self.align_score.requires_grad_(False)
            self.residual_score=nn.Sequential(nn.Linear(59,64),nn.SiLU(),nn.Linear(64,64),nn.SiLU(),nn.Linear(64,1))
            nn.init.zeros_(self.residual_score[-1].weight);nn.init.zeros_(self.residual_score[-1].bias)

    def forward(self, features: Tensor, base: Tensor, history: Tensor,
                history_valid: Tensor, bank: dict[str,Tensor]|None=None,
                logit_shift: float=0.) -> dict[str,Tensor]:
        dense=features.ndim==4
        if dense:
            b,c,h,w=features.shape
            features=features.permute(0,2,3,1).reshape(-1,c)
            base=base.permute(0,2,3,1).reshape(-1,1)
            history=history.permute(0,2,3,1).reshape(-1,1)
            history_valid=history_valid.permute(0,2,3,1).reshape(-1,1)
            if bank is not None:
                bank={'match':bank['match'].permute(0,2,3,1).reshape(-1,12),
                    'ambiguity':bank['ambiguity'].permute(0,2,3,1).reshape(-1,8),
                    'align_features':bank['align_features'].permute(0,3,4,1,2).reshape(-1,9,8),
                    'candidates':bank['candidates'].permute(0,2,3,1).reshape(-1,9),
                    'candidate_valid':bank['candidate_valid'].permute(0,2,3,1).reshape(-1,9)}
        n=len(features)
        extra=torch.zeros(n,28,device=features.device)
        output={}
        if self.matching:extra[:,:12]=bank['match'].float()
        if self.ambiguity:extra[:,12:20]=bank['ambiguity'].float()
        if self.alignment:
            logits=self.align_score(bank['align_features'].float()).squeeze(-1)
            bias=torch.zeros_like(logits);bias[:,0]=3.
            logits=(logits+bias).masked_fill(~bank['candidate_valid'],-1e4)
            probabilities=torch.softmax(logits,1)
            selected=logits.argmax(1)
            one_hot=F.one_hot(selected,9).float()
            routing=one_hot+(probabilities-probabilities.detach()) if self.training else one_hot
            chosen=(routing*bank['candidates'].float()).sum(1,keepdim=True)
            chosen_valid=bank['candidate_valid'].gather(1,selected[:,None])
            selected_features=(routing[:,:,None]*bank['align_features'].float()).sum(1)
            extra[:,20:23]=selected_features[:,:3]
            extra[:,23:24]=((chosen-base)/8).clamp(-16,16)
            extra[:,24:25]=((chosen-history)/8).clamp(-16,16)
            extra[:,25:26]=-(probabilities*probabilities.clamp_min(1e-8).log()).sum(1,keepdim=True)/2.197224577
            top=probabilities.topk(2,1).values
            extra[:,26:27]=top[:,:1]-top[:,1:2]
            extra[:,27:28]=selected_features[:,5:7].square().sum(1,keepdim=True)
            history,history_valid=chosen,chosen_valid
            output['alignment_logits']=logits
            output['alignment_probabilities']=probabilities
        hidden=self.base.net[1](self.base.net[0](features.float())+self.extra(extra))
        logits=self.base.net[4](self.base.net[3](self.base.net[2](hidden)))
        if self.residual_score is not None:
            reference_gate=torch.sigmoid(logits.float())*history_valid.float()
            output['v1_prediction']=base+reference_gate*(torch.where(history_valid,history,base)-base)
            adjustment=self.residual_score(torch.cat((features.float(),extra),1))
            output['delta_logits']=adjustment
            logits=logits+adjustment
        gate=torch.sigmoid(logits.float()+logit_shift)*history_valid.float()
        output['soft_gate']=gate
        if self.categorical:
            hard=(gate>=.5).float()
            routed=hard+(gate-gate.detach()) if self.training else hard
            gate=torch.where((history-base).abs()>5,routed,gate)*history_valid.float()
        safe=torch.where(history_valid,history,base)
        prediction=base.float()+gate*(safe.float()-base.float())
        output.update(prediction=prediction,gate=gate,history=history,history_valid=history_valid)
        if dense:
            for key in ('prediction','gate','soft_gate','history','history_valid','v1_prediction','delta_logits'):
                if key not in output:continue
                output[key]=output[key].reshape(b,h,w,1).permute(0,3,1,2)
        return output


def weighted_mean(value: Tensor, weights: Tensor) -> Tensor:
    return (value*weights).sum()/weights.sum().clamp_min(1)


def supervision_loss(out: dict[str,Tensor], sample: dict[str,Tensor],
                     coefficients: dict[str,float]) -> tuple[Tensor,dict[str,Tensor]]:
    """Separately normalized recovery/rejection and explicit relative regret."""
    target=sample['target'].float();base=sample['base'].float()
    history=out['history'];gate=out.get('soft_gate',out['gate'])
    w=sample['weight']*sample['target_valid'].float()
    be=(base-target).abs();he=(history.detach()-target).abs()
    error=(out['prediction']-target).abs()
    delta=history.detach()-base
    optimal=((target-base)/torch.where(delta.abs()>1e-3,delta,torch.ones_like(delta))).clamp(0,1)
    risk=delta.abs().clamp_max(10)*out['history_valid'].float()*w
    bce=F.binary_cross_entropy(gate.clamp(1e-6,1-1e-6),optimal,reduction='none')
    primary=weighted_mean(error.clamp_max(30),w)+0.05*(bce*risk).sum()/w.sum().clamp_min(1)
    recovery=(be>1)&(he<1)&out['history_valid']
    rejection=(be<0.1)&((he>be+0.1) if coefficients.get('wide_rejection',False) else (he>1))&out['history_valid']
    recover=weighted_mean(-gate.clamp_min(1e-6).log(),w*recovery.float())
    reject=weighted_mean(-(1-gate).clamp_min(1e-6).log(),w*rejection.float())
    regret=weighted_mean(F.relu(error-be-0.1),w)
    tail=weighted_mean(F.relu(error-be-1)+2*F.relu(error-be-5),w)
    align=primary*0
    if 'alignment_logits' in out:
        candidate_error=(sample['bank']['candidates'].float()-target).abs().masked_fill(~sample['bank']['candidate_valid'],1e6)
        best_error,label=candidate_error.min(1)
        label=torch.where(candidate_error[:,0]-best_error>0.1,label,torch.zeros_like(label))
        losses=F.cross_entropy(out['alignment_logits'],label,reduction='none')[:,None]
        align=weighted_mean(losses,w*sample['bank']['candidate_valid'].any(1,keepdim=True))
    guard=primary*0;sparse=primary*0
    if 'v1_prediction' in out:
        v1_error=(out['v1_prediction'].detach()-target).abs()
        guard=weighted_mean(F.relu(error-v1_error),w)
        sparse=out['delta_logits'].abs().mean()
    good_guard=weighted_mean(F.relu(error-be-0.1),w*(be<0.1))
    total=coefficients.get('v1_regret',0)*guard+coefficients.get('good_guard',0)*good_guard+coefficients.get('sparse',0)*sparse+primary+coefficients.get('recovery',0)*recover+coefficients.get('rejection',0)*reject+coefficients.get('regret',0)*regret+coefficients.get('tail',0)*tail+coefficients.get('alignment',0.05)*align
    return total,{'primary':primary,'recovery':recover,'rejection':reject,'regret':regret,'tail':tail,'alignment':align,'v1_regret':guard,'good_guard':good_guard,'sparse_delta':sparse,'recovery_support':recovery.float().mean(),'rejection_support':rejection.float().mean()}
