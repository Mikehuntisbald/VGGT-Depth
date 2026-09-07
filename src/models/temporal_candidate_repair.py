"""Trainable causal HR history selection; inputs contain no GT or dynamic labels.

The base A5 estimator is frozen. A separate strictly past prediction supplies
an HR disparity candidate. Current stereo evidence scores that candidate,
without using the erroneous current depth as a hard visibility veto.
"""
from __future__ import annotations
import torch
from torch import Tensor, nn
import torch.nn.functional as F


def sample_right(right: Tensor, disparity_hr_px: Tensor) -> tuple[Tensor, Tensor]:
    """Rectified HR correspondence, half-pixel grid, no disparity rescaling."""
    b, _, h, w = right.shape
    y, x = torch.meshgrid(torch.arange(h, device=right.device), torch.arange(w, device=right.device), indexing="ij")
    u = x[None].float() - disparity_hr_px[:, 0].float()
    valid = torch.isfinite(u) & (u >= 0) & (u <= w-1)
    grid = torch.stack((2*(u+0.5)/w-1, (2*(y.float()+0.5)/h-1)[None].expand(b,-1,-1)), -1)
    result = F.grid_sample(right.float(), torch.nan_to_num(grid), align_corners=False, padding_mode="zeros")
    return result, valid[:,None]


def local_context(value: Tensor) -> list[Tensor]:
    return [value, F.avg_pool2d(value, 3, 1, 1), F.avg_pool2d(value, 9, 1, 4)]


def repair_features(*, left_rgb: Tensor, right_rgb: Tensor,
                    base_disparity_hr_px: Tensor, history_disparity_hr_px: Tensor,
                    history_valid: Tensor, base_confidence: Tensor,
                    history_confidence: Tensor, history_rgb: Tensor,
                    collision: Tensor, old_gate: Tensor, old_valid: Tensor,
                    stereo_disparity_hr_px: Tensor, stereo_confidence: Tensor,
                    internal_history_disparity_hr_px: Tensor) -> Tensor:
    """Return float32 [B,C,H,W] features from inference-available signals only."""
    base = base_disparity_hr_px.float().clamp_min(1e-6)
    hist = history_disparity_hr_px.float().clamp_min(1e-6)
    delta = hist-base
    sampled_base, base_match = sample_right(right_rgb, base)
    sampled_hist, hist_match = sample_right(right_rgb, hist)
    base_photo = (left_rgb.float()-sampled_base).abs().mean(1,keepdim=True)
    hist_photo = (left_rgb.float()-sampled_hist).abs().mean(1,keepdim=True)
    temporal_photo = (left_rgb.float()-history_rgb.float()).abs().mean(1,keepdim=True)
    channels = [base.log()/5, hist.log()/5, (delta/8).clamp(-16,16),
                torch.log(hist/base).clamp(-8,8), history_valid.float(),
                base_confidence.float(), history_confidence.float(),
                collision.float(), old_gate.float(), old_valid.float(),
                base_match.float(), hist_match.float(),
                ((stereo_disparity_hr_px-base)/8).clamp(-16,16), stereo_confidence.float(),
                ((internal_history_disparity_hr_px-base)/8).clamp(-16,16)]
    for value in (base_photo, hist_photo, hist_photo-base_photo, temporal_photo):
        channels.extend(local_context(value))
    for disp in (base, hist):
        mean = F.avg_pool2d(disp, 3, 1, 1, count_include_pad=False)
        channels.append(((disp-mean)/8).clamp(-16,16))
        channels.append((F.max_pool2d(disp,3,1,1)+F.max_pool2d(-disp,3,1,1)).clamp(0,128)/8)
    return torch.nan_to_num(torch.cat(channels,1),nan=0,posinf=16,neginf=-16).clamp(-32,32)


class TemporalCandidateRepair(nn.Module):
    """A small learned candidate selector; no GT is accepted by forward()."""
    def __init__(self, channels: int = 31, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(channels,hidden),nn.SiLU(),
                                 nn.Linear(hidden,hidden),nn.SiLU(),nn.Linear(hidden,1))
        nn.init.zeros_(self.net[-1].weight)
        nn.init.constant_(self.net[-1].bias,-3.0)

    def forward(self, features: Tensor, base_disparity_hr_px: Tensor,
                history_disparity_hr_px: Tensor, history_valid: Tensor,
                logit_shift: float = 0.0) -> tuple[Tensor,Tensor]:
        dense = features.ndim == 4
        inputs = features.permute(0,2,3,1) if dense else features
        logits = self.net(inputs.float()).float()
        if dense:
            logits = logits.permute(0,3,1,2)
        gate = torch.sigmoid(logits + logit_shift) * history_valid.float()
        safe_history = torch.where(history_valid,history_disparity_hr_px,base_disparity_hr_px)
        prediction = base_disparity_hr_px.float() + gate*(safe_history.float()-base_disparity_hr_px.float())
        return prediction,gate
