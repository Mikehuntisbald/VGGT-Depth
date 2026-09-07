import torch
from models.temporal_candidate_repair import TemporalCandidateRepair,repair_features,sample_right


def test_hr_correspondence_direction_and_units():
    right=torch.arange(8).float().reshape(1,1,1,8)
    sampled,valid=sample_right(right,torch.full((1,1,1,8),2.0))
    torch.testing.assert_close(sampled[...,2:],right[...,:-2])
    assert not valid[...,:2].any()
    assert valid[...,2:].all()


def test_invalid_history_is_exact_fallback_even_if_nan():
    model=TemporalCandidateRepair()
    base=torch.full((1,1,3,4),20.)
    result,gate=model(torch.zeros(1,31,3,4),base,torch.full_like(base,float('nan')),torch.zeros_like(base,dtype=torch.bool))
    torch.testing.assert_close(result,base,rtol=0,atol=0)
    assert not gate.any()


def test_history_can_recover_beyond_original_bounded_residual():
    model=TemporalCandidateRepair()
    base=torch.full((1,1,3,4),90.)
    history=torch.full_like(base,10.)
    result,gate=model(torch.zeros(1,31,3,4),base,history,torch.ones_like(base,dtype=torch.bool),logit_shift=20.)
    assert (result>0).all()
    assert (result-10).abs().max()<0.01
    assert (result < base*torch.exp(torch.tensor(-0.5))).all()


def test_features_have_no_gt_argument_and_are_finite():
    shape=(1,1,12,14)
    scalar=torch.ones(shape)
    rgb=torch.rand(1,3,12,14)
    features=repair_features(left_rgb=rgb,right_rgb=rgb,
        base_disparity_hr_px=scalar*3,history_disparity_hr_px=scalar*4,
        history_valid=scalar.bool(),base_confidence=scalar,history_confidence=scalar,
        history_rgb=rgb,collision=scalar*0,old_gate=scalar,old_valid=scalar,
        stereo_disparity_hr_px=scalar*3,stereo_confidence=scalar,
        internal_history_disparity_hr_px=scalar*4)
    assert features.shape==(1,31,12,14)
    assert features.isfinite().all()
    model=TemporalCandidateRepair()
    pred,gate=model(features,scalar*3,scalar*4,scalar.bool())
    pred.mean().backward()
    assert model.net[-1].bias.grad.abs().item()>0
