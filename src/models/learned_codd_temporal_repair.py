"""Isolated CODD reset/fusion supervision on frozen A5/v1 candidates.

Reference: Li et al., WACV 2023, https://github.com/facebookresearch/CODD
(commit dabb908f3643cd37d9fad0eba5429b8fa0359e24), FusionLoss and Eq. 3.
This independently implemented component study is NOT a full CODD reproduction:
no RAFT3D, dense convolutional fusion, or recurrent feedback.
Adds frozen learned FFS matching cues and TC-Stereo distinct-peak margins.
All disparity thresholds are measured in this repository's HR pixel units.
"""
from __future__ import annotations
import copy
import torch
from torch import Tensor, nn
import torch.nn.functional as F
from .temporal_candidate_repair import TemporalCandidateRepair
from .controlled_temporal_repair import weighted_mean, supervision_loss


class LearnedCoddTemporalRepair(nn.Module):
    def __init__(self, v1: TemporalCandidateRepair, *, v1_shift: float = 0.):
        super().__init__()
        self.base = copy.deepcopy(v1).requires_grad_(False)
        self.v1_shift = float(v1_shift)
        self.offsets = nn.Sequential(nn.Linear(55, 64), nn.SiLU(),
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
        extra=bank['learned']
        if extra.ndim==4:extra=extra.permute(0,2,3,1).reshape(-1,24)
        weights = torch.sigmoid(prior + self.offsets(torch.cat((features.float(),extra.float()),1)) + logit_shift)
        reset, fusion = weights[:, :1], weights[:, 1:]
        gate = reset * fusion * history_valid.float()
        safe = torch.where(history_valid, history, base).float()
        result = dict(prediction=base.float()+gate*(safe-base.float()), gate=gate,
                      soft_gate=gate, reset_weight=reset, fusion_weight=fusion,
                      history=history, history_valid=history_valid)
        if dense:
            result = {k:v.reshape(b,h,w,1).permute(0,3,1,2) for k,v in result.items()}
        return result
