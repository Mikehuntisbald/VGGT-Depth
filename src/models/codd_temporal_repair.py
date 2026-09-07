"""Isolated CODD reset/fusion supervision on frozen A5/v1 candidates.

Reference: Li et al., WACV 2023, https://github.com/facebookresearch/CODD
(commit dabb908f3643cd37d9fad0eba5429b8fa0359e24), FusionLoss and Eq. 3.
This independently implemented component study is NOT a full CODD reproduction:
no RAFT3D, learned stereo cues, dense convolutional fusion, or recurrent feedback.
All disparity thresholds are measured in this repository's HR pixel units.
"""
from __future__ import annotations
import copy
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .temporal_candidate_repair import TemporalCandidateRepair
from .controlled_temporal_repair import weighted_mean, supervision_loss


class CoddTemporalRepair(nn.Module):
    def __init__(self, v1: TemporalCandidateRepair, *, v1_shift: float = 0.):
        super().__init__()
        self.base = copy.deepcopy(v1).requires_grad_(False)
        self.v1_shift = float(v1_shift)
        self.offsets = nn.Sequential(nn.Linear(31, 64), nn.SiLU(),
                                     nn.Linear(64, 64), nn.SiLU(), nn.Linear(64, 2))
        nn.init.zeros_(self.offsets[-1].weight)
        nn.init.zeros_(self.offsets[-1].bias)

    def forward(self, features: Tensor, base: Tensor, history: Tensor,
                history_valid: Tensor, bank=None, logit_shift: float = 0.):
        dense = features.ndim == 4
        if dense:
            b, c, h, w = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, c)
            base, history, history_valid = [x.permute(0, 2, 3, 1).reshape(-1, 1)
                                           for x in (base, history, history_valid)]
        # Equal factorization preserves the accepted v1 starting prediction.
        # It is an initialization device, not a mechanism from the CODD paper.
        with torch.no_grad():
            gate0 = torch.sigmoid(self.base.net(features.float()) + self.v1_shift)
            root = gate0.sqrt().clamp(1e-6, 1-1e-6)
            prior = torch.logit(root)
        weights = torch.sigmoid(prior + self.offsets(features.float()) + logit_shift)
        reset, fusion = weights[:, :1], weights[:, 1:]
        gate = reset * fusion * history_valid.float()
        safe = torch.where(history_valid, history, base).float()
        result = dict(prediction=base.float()+gate*(safe-base.float()), gate=gate,
                      soft_gate=gate, reset_weight=reset, fusion_weight=fusion,
                      history=history, history_valid=history_valid)
        if dense:
            result = {k:v.reshape(b,h,w,1).permute(0,3,1,2) for k,v in result.items()}
        return result


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
    fusion_reject = weighted_mean(fusion, eligible*(eh-ec>1))
    fusion_recover = weighted_mean(1-fusion, eligible*(ec-eh>1))
    fusion_tie = weighted_mean((fusion-.5).abs(), eligible*((ec-eh).abs()<=1))
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
