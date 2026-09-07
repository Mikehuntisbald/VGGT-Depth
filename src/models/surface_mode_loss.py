"""Shared supervision for the WTA and surface-mass controlled comparison."""
import torch
import torch.nn.functional as F
from models.hypothesis_geometry import HISTORY_START
from models.native_temporal_stereo.model import warp_map

def average(x,mask):
    return (x*mask).sum()/mask.sum().clamp_min(1)

def surface_mode_loss(output,record):
    labels=record['labels'];inputs=record['inputs'];count=len(output['frames']);parts={};total=output['endpoint'].disparity_left_px.sum()*0
    for t in range(count-2,count):
        end=output['frames'][t];detail=output['details'][t];gt=labels['target'][:,t].float();valid=labels['valid'][:,t].bool()&(gt>0)&torch.isfinite(gt)
        candidates=detail['candidates'];hypotheses=detail['hypotheses'];cv=detail['candidate_valid'];prob=detail['probabilities']
        initial_error=(candidates-gt).abs().masked_fill(~cv,1e4)
        error=(hypotheses-gt).abs();best=initial_error.amin(1,keepdim=True)
        # Supervise the total probability of correct surface estimates, not
        # one arbitrary spatial duplicate. Both controlled arms use this loss.
        close=(initial_error<1)&cv
        acceptable=torch.where(close.any(1,keepdim=True),close,(initial_error<=best+.1)&cv)
        choice=average(-(prob*acceptable).sum(1,keepdim=True).clamp_min(1e-6).log(),valid)
        # NMRF-style expected error trains scored per-pixel hypotheses directly.
        expected=average((prob*error.clamp_max(30)).sum(1,keepdim=True),valid)
        supported=valid&cv&(initial_error<4)
        regression=average(F.smooth_l1_loss(hypotheses,gt.expand_as(hypotheses),reduction='none'),supported)
        final_error=(end.disparity_left_px-gt).abs();disparity=average(final_error.clamp_max(30),valid)
        reference_error=(candidates[:,:1]-gt).abs()
        regret=average(F.relu(final_error-reference_error-5),valid)
        he=initial_error[:,HISTORY_START:].amin(1,keepdim=True);hp=prob[:,HISTORY_START:].sum(1,keepdim=True)
        recovery=average(-((prob*((error.detach()<1)&cv)).sum(1,keepdim=True)).clamp_min(1e-6).log(),valid&(reference_error>1)&(he<1))
        rejection=average(-(1-hp).clamp_min(1e-6).log(),valid&(reference_error<.1)&(he>1)&cv[:,HISTORY_START:].any(1,keepdim=True))
        validity=F.binary_cross_entropy_with_logits(end.valid_logits,labels['valid'][:,t].float())
        log_error=(end.disparity_left_px.clamp_min(1e-6).log()-gt.clamp_min(1e-6).log()).abs()
        uncertainty=average(log_error*torch.exp(-end.log_variance)+end.log_variance,valid)
        total=total+(.5*disparity+.5*expected+.2*choice+.2*regression+.05*(recovery+rejection)+.05*regret+.05*validity+.001*uncertainty)/2
        for k,v in dict(disparity=disparity,expected=expected,choice=choice,regression=regression,recovery=recovery,rejection=rejection,regret=regret,validity=validity,uncertainty=uncertainty).items():parts[k]=parts.get(k,0)+v.detach()/2
    t=count-1;previous=output['frames'][t-1];current=output['frames'][t]
    kp,kc=inputs['K'][:,t-1,0].float(),inputs['K'][:,t,0].float();bp,bc=inputs['baseline_m'][:,t-1],inputs['baseline_m'][:,t]
    transform=inputs['T_current_from_previous'][:,t].float();gp=labels['target'][:,t-1].float();gc=labels['target'][:,t].float()
    gw=warp_map(gp,labels['valid'][:,t-1]&(gp>0),labels['valid'][:,t-1].float(),kp,kc,bp,bc,transform)
    pw=warp_map(previous.disparity_left_px.detach(),previous.valid_mask,previous.confidence.detach(),kp,kc,bp,bc,transform)
    mask=gw.valid_mask&labels['valid'][:,t]&pw.valid_mask&(gc>0)
    temporal=average(((current.disparity_left_px-pw.disparity_hr_px)-(gc-gw.disparity_hr_px)).abs().clamp_max(10),mask)
    total=total+.1*temporal;parts['temporal']=temporal.detach()
    return total,parts
