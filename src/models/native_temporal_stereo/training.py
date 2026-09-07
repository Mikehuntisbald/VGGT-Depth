"""Supervision for a native temporal stereo solver; labels stay outside forward."""
import torch
import torch.nn.functional as F
from .model import sample_right_feature,warp_map


def average(value,mask):
    mask=mask.float()
    return (value*mask).sum()/mask.sum().clamp_min(1)


def native_loss(output,record):
    labels=record['labels'];inputs=record['inputs'];count=len(output['frames']);parts={}
    total=output['endpoint'].disparity_left_px.sum()*0
    for t in range(max(0,count-2),count):
        end=output['frames'][t];detail=output['details'][t]
        gt=labels['target'][:,t].float();valid=labels['valid'][:,t].bool()&torch.isfinite(gt)&(gt>0)
        pred=end.disparity_left_px.float();error=(pred-gt).abs()
        disparity=average(error.clamp_max(30),valid)
        validity=F.binary_cross_entropy_with_logits(end.valid_logits.float(),valid.float())
        log_error=(pred.clamp_min(1e-6).log()-gt.clamp_min(1e-6).log()).abs()
        uncertainty=average(log_error*torch.exp(-end.log_variance.float())+end.log_variance.float(),valid)
        reference=F.interpolate(record['audit']['ffs_image_only_disparity_lr'][:,t].float()*2,size=gt.shape[-2:],mode='bilinear',align_corners=False)
        regret=average(F.relu(error-(reference-gt).abs()-5),valid)
        seed_loss=pred.sum()*0;choice=seed_loss;recovery=seed_loss;rejection=seed_loss;matching=seed_loss
        init=detail['initialization']
        if init is not None:
            size=init['seed'].shape[-2:];scale=gt.shape[-1]/size[1]
            target=F.interpolate(gt,size=size,mode='nearest-exact')/scale
            supported=F.interpolate(valid.float(),size=size,mode='nearest-exact')>.5
            candidates=init['candidates'];cv=init['candidate_valid']
            errors=(candidates.detach()-target).abs()*scale
            errors=errors.masked_fill(~cv,1e6)
            best_error=errors.amin(1,keepdim=True)
            acceptable=(errors<=best_error+.1)&cv
            prefer_current=errors[:,:1]<=best_error+.1
            only_current=torch.zeros_like(acceptable);only_current[:,0]=True
            acceptable=torch.where(prefer_current,only_current,acceptable)
            success_probability=(init['probability']*acceptable).sum(1,keepdim=True)
            choice=average(-success_probability.clamp_min(1e-6).log(),supported)
            seed_loss=average(((init['seed']-target)*scale).abs().clamp_max(30),supported)
            hist_error=errors[:,1:].amin(1,keepdim=True);current_error=errors[:,:1]
            hp=init['probability'][:,1:].sum(1,keepdim=True);p0=init['probability'][:,:1]
            recover=supported&(current_error>1)&(hist_error<1)
            reject=supported&(current_error<1)&(hist_error>current_error+1)&cv[:,1:].any(1,keepdim=True)
            recovery=average(-hp.clamp_min(1e-6).log(),recover)
            rejection=average(-p0.clamp_min(1e-6).log(),reject)
            # TC-Stereo style GT matching and hard-negative margin supervision.
            # Stereo-occluded pixels are excluded using GT-only label support.
            positive_right,in_bounds=sample_right_feature(init['right_key'],target)
            positive=(init['left_key']*F.normalize(positive_right,dim=1,eps=1e-6)).sum(1,keepdim=True)
            matched=F.interpolate(labels['stereo_matched'][:,t].float(),size=size,mode='nearest-exact')>.5
            matching_support=supported&matched&in_bounds
            negatives=[];negative_valid=[]
            for k in range(candidates.shape[1]):
                right,ok=sample_right_feature(init['right_key'],candidates[:,k:k+1].detach())
                score=(init['left_key']*F.normalize(right,dim=1,eps=1e-6)).sum(1,keepdim=True)
                eligible=cv[:,k:k+1]&ok&((candidates[:,k:k+1].detach()-target).abs()>1.5)
                negatives.append(score.masked_fill(~eligible,-2));negative_valid.append(eligible)
            hardest=torch.cat(negatives,1).amax(1,keepdim=True)
            has_negative=torch.cat(negative_valid,1).any(1,keepdim=True)
            matching=average(1-positive,matching_support)+average(F.relu(hardest+.1-positive.detach()),matching_support&has_negative)
        frame_loss=disparity+.05*validity+.001*uncertainty+.02*regret+.02*seed_loss+.05*choice+.02*(recovery+rejection+matching)
        total=total+frame_loss/2
        for k,v in dict(disparity=disparity,validity=validity,uncertainty=uncertainty,regret=regret,seed=seed_loss,choice=choice,recovery=recovery,rejection=rejection,matching=matching).items():parts[k]=parts.get(k,0)+v.detach()/2
    if count>1:
        t=count-1;previous=output['frames'][t-1];current=output['frames'][t]
        kp,kc=inputs['K'][:,t-1,0].float(),inputs['K'][:,t,0].float();bp,bc=inputs['baseline_m'][:,t-1],inputs['baseline_m'][:,t]
        transform=inputs['T_current_from_previous'][:,t].float()
        gp=labels['target'][:,t-1].float();gc=labels['target'][:,t].float()
        vg=labels['valid'][:,t-1].bool()&torch.isfinite(gp)&(gp>0)
        gw=warp_map(gp,vg,vg.float(),kp,kc,bp,bc,transform)
        pw=warp_map(previous.disparity_left_px.detach(),previous.valid_mask,previous.confidence.detach(),kp,kc,bp,bc,transform)
        mask=gw.valid_mask&labels['valid'][:,t].bool()&pw.valid_mask&torch.isfinite(gc)&(gc>0)
        residual=((current.disparity_left_px.float()-pw.disparity_hr_px)-(gc-gw.disparity_hr_px)).abs().clamp_max(10)
        temporal=average(residual,mask)
        total=total+.1*temporal;parts['temporal']=temporal.detach()
    return total,parts
