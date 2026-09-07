"""Causal history-conditioned stereo initialization and latent-state propagation.

Temporal information enters before FFS cost-volume lookup/refinement. The depth
output is decoded jointly with current VGGT geometry; there is no v1/A5-output
correction head and no fixed historical A5 prediction. Each state is generated
by this model's preceding step. TC-Stereo (ECCV 2024) motivates the placement of
completion/state fusion before iterative matching, rather than post-fusion.
"""
from __future__ import annotations
import copy,contextlib,math
import torch
from torch import nn
import torch.nn.functional as F
from geometry.camera import resize_intrinsics_align_corners_false
from geometry.zbuffer_reproject import zbuffer_reproject
from models.metric_stereo_video_geometry import MetricStereoFrameInput,StereoBackboneFeatures,VGGTCausalGeometryFeatures
from models.metric_stereo_video_system import left_right_stereo_consistency

OFFSETS=((0,0),(-1,0),(1,0),(0,-1),(0,1),(-1,-1),(1,-1),(-1,1),(1,1))


def shifted(x,dx,dy):
    h,w=x.shape[-2:];p=F.pad(x,(1,1,1,1))
    return p[...,1+dy:1+dy+h,1+dx:1+dx+w]


def gather_winners(value,uv):
    b,c,h,w=value.shape
    index=(uv[:,1].long().clamp(0,h-1)*w+uv[:,0].long().clamp(0,w-1)).flatten(1)[:,None].expand(-1,c,-1)
    return value.flatten(2).gather(2,index).reshape(b,c,h,w)


def warp_map(disparity_hr,valid,confidence,K_previous,K_current,baseline_previous,baseline_current,transform):
    with torch.autocast(device_type=disparity_hr.device.type,enabled=False):
        # Map geometry at the given grid, with explicit matching disparity/K units.
        factor=(K_previous[:,0,0]*baseline_previous).reshape(-1,1,1,1)
        disparity=torch.where(valid,disparity_hr.float(),torch.zeros_like(disparity_hr).float())
        depth=torch.where(valid,factor/disparity.clamp_min(1e-8),torch.zeros_like(disparity))
        identity=torch.eye(4,device=disparity.device).expand(len(disparity),-1,-1)
        return zbuffer_reproject(disparity,depth,confidence.float(),K_previous.float(),identity,transform.float(),
          intrinsics_current_hr_3x3=K_current.float(),baseline_previous_m=baseline_previous.float(),baseline_current_m=baseline_current.float())
    

def warp_state(state,K,baseline,transform,size):
    if state is None:return None
    h,w=size;hh,ww=state['image_size']
    kp=resize_intrinsics_align_corners_false(state['K'].float(),w/ww,h/hh)
    kc=resize_intrinsics_align_corners_false(K.float(),w/ww,h/hh)
    previous=F.interpolate(state['disparity_hr'].float(),size=size,mode='nearest-exact')*(w/ww)
    valid=F.interpolate(state['valid'].float(),size=size,mode='nearest-exact')>.5
    confidence=F.interpolate(state['confidence'].float(),size=size,mode='nearest-exact')
    result=warp_map(previous,valid,confidence,kp,kc,state['baseline'],baseline,transform)
    hidden=[F.interpolate(x.float(),size=size,mode='bilinear',align_corners=False) for x in state['hidden']]
    return {'disparity':result.disparity_hr_px,'valid':result.valid_mask,'confidence':result.confidence,'collision':result.collision_mask,
      'key':gather_winners(state['key'],result.source_uv)*result.valid_mask,
      'hidden':[gather_winners(x,result.source_uv)*result.valid_mask for x in hidden]}


def sample_right_feature(feature,disparity):
    b,_,h,w=feature.shape
    y,x=torch.meshgrid(torch.arange(h,device=feature.device),torch.arange(w,device=feature.device),indexing='ij')
    u=x.float()[None]-disparity[:,0].float();v=y.float()[None].expand(b,-1,-1)
    grid=torch.stack((2*(u+.5)/w-1,2*(v+.5)/h-1),-1)
    return F.grid_sample(feature.float(),grid,align_corners=False,padding_mode='border'),((u>=0)&(u<=w-1))[:,None]


def sample_volume(volume,disparity):
    b,c,d,h,w=volume.shape
    position=disparity.float().clamp(0,d-1)
    lo=position.floor().long();hi=(lo+1).clamp_max(d-1)
    lower=torch.gather(volume.float(),2,lo[:,None].expand(-1,c,1,-1,-1)).squeeze(2)
    upper=torch.gather(volume.float(),2,hi[:,None].expand(-1,c,1,-1,-1)).squeeze(2)
    return lower+(upper-lower)*(position-lo),((disparity>=0)&(disparity<=d-1))


class StateFusion(nn.Module):
    """Spatial GRU fusion of current and transported iterative hidden states."""
    def __init__(self,channels):
        super().__init__()
        self.gates=nn.Conv2d(2*channels,2*channels,1)
        self.proposal=nn.Conv2d(2*channels,channels,1)
        # Initially retain the current-image state; learn when past state helps.
        nn.init.zeros_(self.gates.weight);nn.init.constant_(self.gates.bias,2.)
    def forward(self,current,past,availability):
        z,r=self.gates(torch.cat((current,past),1)).chunk(2,1)
        z,r=z.sigmoid(),r.sigmoid()
        proposal=torch.tanh(self.proposal(torch.cat((r*current,past),1)))
        mixed=z*current+(1-z)*proposal
        return current+availability.to(current.dtype)*(mixed-current)


class TemporalStereoInitializer(nn.Module):
    def __init__(self,feature_channels,volume_channels,hidden_channels,vggt_channels=192,propagate_hidden=True):
        super().__init__()
        self.propagate_hidden=propagate_hidden
        self.key=nn.Conv2d(feature_channels,16,1)
        self.context=nn.Sequential(nn.Conv2d(feature_channels+vggt_channels,64,1),nn.GroupNorm(8,64),nn.SiLU())
        # Current context, learned geometry volume, stereo/temporal similarities,
        # disparity difference, validity, visibility confidence and candidate type.
        self.cost=nn.Sequential(nn.Conv2d(64+volume_channels+8,64,3,padding=1),nn.GroupNorm(8,64),nn.SiLU(),
                                nn.Conv2d(64,32,3,padding=1),nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,1,1))
        nn.init.zeros_(self.cost[-1].weight);nn.init.zeros_(self.cost[-1].bias)
        self.state_fusion=nn.ModuleList(StateFusion(c) for c in hidden_channels)

    def forward(self,evidence,vggt_feature,history):
        seed=evidence['seed'].float();b,_,h,w=seed.shape
        left=F.normalize(self.key(evidence['left'].float()).float(),dim=1)
        right=F.normalize(self.key(evidence['right'].float()).float(),dim=1)
        current_context=self.context(torch.cat((evidence['left'].float(),F.interpolate(vggt_feature.float(),size=(h,w),mode='bilinear',align_corners=False)),1))
        candidates=[seed];validity=[torch.ones_like(seed,dtype=torch.bool)];scores=[]
        offsets=[None,*OFFSETS]
        for k,offset in enumerate(offsets):
            if k:
                dx,dy=offset
                candidate=shifted(history['disparity'],dx,dy) if history is not None else torch.zeros_like(seed)
                valid=shifted(history['valid'].float(),dx,dy)>.5 if history is not None else torch.zeros_like(seed,dtype=torch.bool)
                candidates.append(candidate);validity.append(valid)
                temporal_key=shifted(history['key'],dx,dy) if history is not None else torch.zeros_like(left)
                temporal_similarity=(left*F.normalize(temporal_key.float(),dim=1,eps=1e-6)).sum(1,keepdim=True)
                history_conf=shifted(history['confidence'],dx,dy) if history is not None else torch.zeros_like(seed)
            else:
                candidate=seed;valid=validity[0];dx=dy=0
                temporal_similarity=torch.zeros_like(seed);history_conf=torch.zeros_like(seed)
            right_sample,stereo_valid=sample_right_feature(right,candidate)
            stereo_similarity=(left*F.normalize(right_sample,dim=1,eps=1e-6)).sum(1,keepdim=True)
            geometry,geometry_valid=sample_volume(evidence['volume'],candidate)
            geometry=geometry/(geometry.square().mean(1,keepdim=True).sqrt().clamp_min(1))
            cues=torch.cat((stereo_similarity,temporal_similarity,((candidate-seed)/8).clamp(-8,8),
                 stereo_valid.float(),geometry_valid.float(),history_conf,torch.full_like(seed,float(dx)),torch.full_like(seed,float(dy))),1)
            score=self.cost(torch.cat((current_context.float(),geometry,cues),1)).float()
            # equal logits impose no persistent hand-set current-depth preference.
            scores.append(score.masked_fill(~valid,-1e4))
        candidates=torch.cat(candidates,1);validity=torch.cat(validity,1);logits=torch.cat(scores,1)
        probability=torch.softmax(logits,1);selected=logits.argmax(1,keepdim=True)
        hard=torch.zeros_like(probability).scatter_(1,selected,1.)
        routing=hard+(probability-probability.detach()) if self.training else hard
        initialized=(routing*candidates).sum(1,keepdim=True).clamp_min(1e-5)
        hidden=[x.clone() for x in evidence['hidden']]
        if self.propagate_hidden and history is not None:
            # Feature-state attention is allowed; disparity initialization above
            # selects one surface rather than averaging foreground/background.
            hp=probability[:,1:]
            for i,(current,fuser) in enumerate(zip(hidden,self.state_fusion)):
                all_past=torch.stack([shifted(history['hidden'][i],dx,dy) for dx,dy in OFFSETS],1)
                past=(all_past*hp[:,:,None]).sum(1)/hp.sum(1,keepdim=True).clamp_min(1e-6)
                past=F.interpolate(past,size=current.shape[-2:],mode='bilinear',align_corners=False)
                # Shifted valid candidates can fill a center-location warp hole.
                present=F.interpolate((hp*validity[:,1:].float()).sum(1,keepdim=True),size=current.shape[-2:],mode='bilinear',align_corners=False)
                hidden[i]=fuser(current.float(),past,present).to(current.dtype)
        return {'seed':initialized,'hidden':hidden,'key':left,'logits':logits,'candidates':candidates,'candidate_valid':validity,
                'probability':probability,'selected':selected,'right_key':right,'left_key':left,
                'history_support':routing[:,1:].sum(1,keepdim=True)*probability[:,1:].sum(1,keepdim=True)}


class NativeTemporalStereo(nn.Module):
    def __init__(self,initialization,variant='early_state',vggt_schedule='causal_each_frame'):
        super().__init__()
        if variant not in ('late','early_seed','early_state'):raise ValueError(variant)
        self.variant=variant;self.vggt_schedule=vggt_schedule
        self.refiner=copy.deepcopy(initialization['refiner']).requires_grad_(False)
        self.geometry=copy.deepcopy(initialization['geometry']).requires_grad_(True)
        self.geometry.enable_temporal_memory=variant=='late'
        self.initializer=None
        if variant!='late':
            config=initialization['config']
            # All FFS hidden scales share args.hidden_dims (validated from cache).
            self.initializer=TemporalStereoInitializer(config['stereo']['feature_channels'],initialization['volume_channels'],initialization['hidden_channels'],
                  config['vggt']['geometry_channels'],propagate_hidden=variant=='early_state')
            self.geometry.temporal_gate.requires_grad_(False)
        self.refiner.eval()

    def forward(self,inputs,*,prefix_length=None,gradient_frames=2):
        if inputs['rgb'].shape[0]!=1:raise ValueError('native encoded clips require one clip per rank')
        rgb=inputs['rgb'].float()/255.;count=rgb.shape[1] if prefix_length is None else int(prefix_length)
        if not 1<=count<=rgb.shape[1]:raise ValueError('invalid causal prefix')
        geometry_state=None;matching_state=None;outputs=[];details=[]
        for t in range(count):
            enabled=torch.is_grad_enabled() and self.training and t>=count-gradient_frames
            with torch.set_grad_enabled(enabled), torch.autocast(device_type=rgb.device.type,dtype=torch.bfloat16,enabled=rgb.is_cuda,cache_enabled=enabled):
                encoded=inputs['encoded']
                evidence={k:([x[t:t+1] for x in v] if isinstance(v,list) else v[t:t+1]) for k,v in encoded.items()}
                K=inputs['K'][:,t,0].float();baseline=inputs['baseline_m'][:,t].float()
                vggt={k:v[:,t] for k,v in inputs['vggt'].items()}
                if self.vggt_schedule=='endpoint_only' and t<count-1:vggt={k:torch.zeros_like(v) for k,v in vggt.items()}
                transition=None if t==0 else inputs['T_current_from_previous'][:,t].float()
                history=warp_state(matching_state,K,baseline,transition,evidence['seed'].shape[-2:]) if matching_state is not None else None
                init=self.initializer(evidence,vggt['feature'],history) if self.initializer is not None else None
                solved=self.refiner(evidence,init['seed'] if init is not None else None,init['hidden'] if init is not None else None)
                consistency=left_right_stereo_consistency(solved['disparity_lr'],inputs['right_disparity_lr'][:,t],maximum_error_px=1.,confidence_temperature_px=.5)
                stereo_valid=consistency.valid_left_mask;stereo_confidence=consistency.confidence_left
                if init is not None:
                    # A temporally supported solution remains a metric source in
                    # stereo occlusions. Current LR consistency is evidence, not
                    # a veto on a selected, transported historical surface.
                    retention=torch.exp(-(solved['disparity_feature'].float()-init['seed'].float()).abs())
                    support=F.interpolate(init['history_support']*retention,size=solved['disparity_lr'].shape[-2:],mode='bilinear',align_corners=False)
                    stereo_confidence=torch.maximum(stereo_confidence,support)
                    stereo_valid=stereo_valid|(support>0)
                frame=MetricStereoFrameInput(left_rgb=rgb[:,t,0],right_rgb=rgb[:,t,1],intrinsics_left_3x3=K,
                    T_right_from_left_m=inputs['T_right_from_left'][:,t].float(),T_current_from_previous_m=transition,
                    lowres_disparity_left_px=solved['disparity_lr']*2,
                    lowres_disparity_valid_mask=stereo_valid,lowres_disparity_confidence=stereo_confidence,
                    stereo_features=StereoBackboneFeatures(evidence['left'],t),
                    vggt_features=VGGTCausalGeometryFeatures(vggt['feature'],vggt['inverse_relative'],vggt['confidence'],tuple(range(t+1)),t),time_index=t)
                result=self.geometry.forward_step(frame,geometry_state if self.variant=='late' else None)
                outputs.append(result);details.append({'initialization':init,'stereo':solved,'image_seed':evidence['seed']})
                # Explicit truncated BPTT at each causal step. Past values are
                # generated by the current model, not read from a base-output cache.
                geometry_state=result.state.detach()
                if self.initializer is not None:
                    matching_state={'disparity_hr':result.disparity_left_px.detach(),'confidence':result.confidence.detach(),'valid':result.valid_mask.detach(),
                        'K':K,'baseline':baseline,'image_size':tuple(rgb.shape[-2:]),'hidden':[h.detach() for h in solved['hidden']], 'key':init['key'].detach()}
        return {'frames':outputs,'details':details,'endpoint':outputs[-1]}
