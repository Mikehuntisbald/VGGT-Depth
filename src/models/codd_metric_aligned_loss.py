"""CODD fusion-margin adaptation to the pre-existing 0.1 HR-pixel opportunity contract.
One changed margin, no grid search; reset threshold remains 5 px.
"""
import torch
import torch.nn.functional as F
from .controlled_temporal_repair import weighted_mean,supervision_loss

def codd_loss(out, sample, profile):
    if profile == 'capacity_control':
        return supervision_loss(out, sample, dict(recovery=0.,rejection=0.,regret=0.,tail=0.))
    target = sample['target'].float()
    ec = (sample['base'].float()-target).abs()
    eh = (out['history'].detach().float()-target).abs()
    weight = sample['weight']*sample['target_valid'].float()
    eligible = weight*out['history_valid'].float()
    reset, fusion = out['reset_weight'], out['fusion_weight']
    disparity = weighted_mean(F.smooth_l1_loss(out['prediction'],target,reduction='none'),weight)
    # Each case has its own denominator, as in official FusionLoss.
    reset_reject = weighted_mean(reset, eligible*(eh-ec>5))
    reset_recover = weighted_mean(1-reset, eligible*(ec-eh>5))
    fusion_reject = weighted_mean(fusion, eligible*(eh-ec>.1))
    fusion_recover = weighted_mean(1-fusion, eligible*(ec-eh>.1))
    fusion_tie = weighted_mean((fusion-.5).abs(), eligible*((ec-eh).abs()<=.1))
    error = (out['prediction']-target).abs()
    tail = weighted_mean(F.relu(error-ec-1)+2*F.relu(error-ec-5),weight)
    total = disparity+reset_reject+reset_recover+fusion_reject+fusion_recover+.2*fusion_tie
    if profile == 'codd_regret':
        total = total + .1*tail
    elif profile != 'codd':
        raise ValueError(profile)
    return total, dict(disparity=disparity, reset_reject=reset_reject,
                       reset_recover=reset_recover,fusion_reject=fusion_reject,
                       fusion_recover=fusion_recover,fusion_tie=fusion_tie,tail=tail)
