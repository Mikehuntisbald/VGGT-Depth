import torch
from models.codd_temporal_repair import codd_loss as original
from models.codd_metric_aligned_loss import codd_loss as aligned


def test_only_fusion_tie_band_moves_to_existing_opportunity_threshold():
    base=torch.tensor([[1.5]]);history=torch.tensor([[1.]]);target=history.clone()
    reset=torch.tensor([[.5]]);fusion=torch.tensor([[.8]])
    out=dict(prediction=base+reset*fusion*(history-base),history=history,history_valid=torch.ones_like(base,dtype=torch.bool),reset_weight=reset,fusion_weight=fusion)
    sample=dict(base=base,target=target,target_valid=torch.ones_like(base,dtype=torch.bool),weight=torch.ones_like(base))
    _,old=original(out,sample,'codd');_,new=aligned(out,sample,'codd')
    assert old['fusion_recover']==0 and new['fusion_tie']==0
    torch.testing.assert_close(old['fusion_tie'],torch.tensor(.3))
    torch.testing.assert_close(new['fusion_recover'],torch.tensor(.2))
    for name in ('disparity','reset_recover','reset_reject','tail'):assert torch.equal(old[name],new[name])
