"""Raw-image input adapter for frozen FFS/VGGT evidence, before stereo solving.

Calls installed FFS encoding operators under their upstream license. The old
A5 geometry decoder is never executed. The left branch stops before its first
iterative update; NativeTemporalStereo owns that recurrence and all history.
"""
import torch
from backbones.trainable_stereo import half_resolution_stereo_images
from backbones.ffs_adapter import make_right_reference_pair,restore_right_reference_disparity
from models.metric_stereo_video_system import vggt_unbounded_confidence_to_probability


def encode_stereo(ffs,left,right):
    from core.foundation_stereo import normalize_image,build_gwc_volume_optimized_pytorch1,build_concat_volume_optimized_pytorch1,disparity_regression
    count=left.shape[0]
    left=normalize_image(left);right=normalize_image(right)
    with torch.autocast('cuda',dtype=torch.bfloat16):
        pyramid=ffs.feature(torch.cat((left,right),0))
        left_features=[x[:count] for x in pyramid];right_features=[x[count:] for x in pyramid]
        stem=ffs.stem_2(left)
        gwc=build_gwc_volume_optimized_pytorch1(left_features[0],right_features[0],ffs.args.max_disp//4,ffs.cv_group,normalize=ffs.args.normalize)
        concat=build_concat_volume_optimized_pytorch1(ffs.proj_cmb(left_features[0]),ffs.proj_cmb(right_features[0]),maxdisp=ffs.args.max_disp//4)
        volume=ffs.corr_stem(torch.cat((gwc,concat),1))
        volume=ffs.cost_agg(ffs.corr_feature_att(volume,left_features[0]),left_features)
        seed=disparity_regression(torch.softmax(ffs.classifier(volume).squeeze(1),dim=1),ffs.args.max_disp//4)
        contexts=list(ffs.cnet(*left_features[:3]))
        hidden=[torch.tanh(x[0]) for x in contexts]
        context=[torch.relu(x[1]) for x in contexts];context=[ffs.cam(x)*x for x in context]
        attention=[ffs.sam(x) for x in context]
    return dict(left=left_features[0],right=right_features[0],volume=volume,seed=seed.to(ffs.dtype),hidden=hidden,context=context,attention=attention,stem=stem)


def encode_raw_clip(system,batch):
    """Use only RGB/calibration/motion. Install as the FSDP root forward adapter.

    The root wrapper materializes common frozen encoder parameters; its former
    output decoder is bypassed. This is independent of every cached prediction.
    """
    allowed={'rgb','K','baseline_m','T_current_from_previous','T_right_from_left'}
    if set(batch)!=allowed:raise ValueError('raw encoder accepts only RGB/calibration/motion')
    rgb=batch['rgb'];b,count,_,_,h,w=rgb.shape
    if b!=1 or not 1<=count<=8:raise ValueError('one causal clip of 1–8 frames per encoder rank')
    left,right=half_resolution_stereo_images(rgb[:,:,0],rgb[:,:,1])
    left=left.reshape(count,3,h//2,w//2)*255.;right=right.reshape(count,3,h//2,w//2)*255.
    # A fixed frame-batch shape makes independent short-prefix executions use
    # exactly the same image-only kernels as the training clips. Padding repeats
    # an available past image; FFS has no cross-image attention. VGGT below is
    # never padded and still sees only its actual causal prefix.
    if count<8:
        left=torch.cat((left,left[:1].expand(8-count,-1,-1,-1)),0)
        right=torch.cat((right,right[:1].expand(8-count,-1,-1,-1)),0)
    ffs=system.stereo_backbone.model
    all_encoded=encode_stereo(ffs,left,right)
    encoded={k:([x[:count] for x in v] if isinstance(v,list) else v[:count]) for k,v in all_encoded.items()}
    # Only the independent right-reference image branch is solved here, for
    # left/right evidence after the new model solves the left reference.
    right_ref,left_target=make_right_reference_pair(left,right)
    right_flipped,_,_=system.stereo_backbone._run_once(right_ref,left_target)
    right_disparity=restore_right_reference_disparity(right_flipped)[:count].reshape(b,count,1,h//2,w//2)
    vggt=[None]*count
    for t in range(count-1,-1,-1):
        out=system.vggt_backbone(rgb[:,:t+1,0])
        depth=out.depth_current_arbitrary;valid=torch.isfinite(depth)&(depth>0)
        inverse=torch.where(valid,depth.clamp_min(1e-8).reciprocal(),torch.zeros_like(depth))
        confidence=vggt_unbounded_confidence_to_probability(out.confidence_current_unbounded)*valid.to(depth.dtype)
        vggt[t]={'feature':out.geometry_current,'inverse_relative':inverse,'confidence':confidence}
    return {'encoded':encoded,'rgb':(rgb*255).round().to(torch.uint8),'right_disparity_lr':right_disparity,
       'vggt':{k:torch.stack([v[k] for v in vggt],1) for k in vggt[0]},
       **{k:batch[k] for k in allowed-{'rgb'}}}
