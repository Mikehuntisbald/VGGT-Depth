#!/usr/bin/env python3
"""Build real last-two-frame GT sidecars without recomputing frozen encodings."""
from pathlib import Path
import argparse,json,sys,time,hashlib
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
import torch.distributed as dist
from data.raw_stereo_video_dataset import _resolve_file,_image_size_hw,_load_disparity
from models.metric_stereo_video_system import left_right_stereo_consistency
from models.native_temporal_stereo.model import sample_right_feature
from tools.train_metric_stereo_video import _dataset,_read_config,_distributed_context
from tools.cache_temporal_candidate_repair import atomic_save,cpu


def main():
 p=argparse.ArgumentParser();p.add_argument('--cache',type=Path,required=True);p.add_argument('--config',type=Path,required=True);a=p.parse_args()
 context=_distributed_context();torch.set_num_threads(4);config=_read_config(a.config);dataset=_dataset(config,training=True)
 plan=json.loads((a.cache/'train/plan.json').read_text());indices=plan['indices'];out=a.cache/'training_labels';out.mkdir(exist_ok=True);started=time.monotonic()
 for position in range(context.rank,len(indices),context.world_size):
    index=indices[position];manifest_indices=dataset._indices_for_item(index)
    records=[dataset.records[i] for i in manifest_indices]
    left_path=_resolve_file(records[0].left_path,manifest_directory=dataset.manifest_directory,field_name='left_path')
    hw=_image_size_hw(left_path);crop=dataset._crop_for_item(index,*hw)
    targets=[];valids=[];matches=[];sources=[]
    for record in records[-2:]:
        lp=_resolve_file(record.gt_disparity_path,manifest_directory=dataset.manifest_directory,field_name='gt_disparity_path')
        rp=_resolve_file(record.extras['gt_disparity_right_path'],manifest_directory=dataset.manifest_directory,field_name='gt_disparity_right_path')
        left,vl=_load_disparity(lp,expected_hw=hw,crop=crop);right,vr=_load_disparity(rp,expected_hw=hw,crop=crop)
        l=left[None].to(context.device);r=right[None].to(context.device)
        consistency=left_right_stereo_consistency(l,r,maximum_error_px=1.,confidence_temperature_px=.5)
        rv,bounds=sample_right_feature(vr.float()[None].to(context.device),l)
        matched=consistency.valid_left_mask&vl[None].to(context.device)&bounds&(rv>.999)
        targets.append(left);valids.append(vl);matches.append(matched[0].cpu());sources.append({'left':str(lp),'right':str(rp),'frame_id':record.frame_id})
    target=torch.stack(targets)[None];valid=torch.stack(valids)[None]
    if position in (0,len(indices)//2,len(indices)-1):
        original=torch.load(a.cache/'train'/f'{index:06d}.pt',map_location='cpu',weights_only=False)
        torch.testing.assert_close(target[:,-1],original['labels']['target'][:,-1],atol=0,rtol=0)
        assert torch.equal(valid[:,-1],original['labels']['valid'][:,-1])
    atomic_save({'target':target,'valid':valid,'stereo_matched':torch.stack(matches)[None],'dataset_index':index,'sources':sources},out/f'{index:06d}.pt')
 if context.world_size>1:dist.barrier()
 if context.primary:
    actual=len(list(out.glob('*.pt')));assert actual==len(indices)
    (out/'complete.json').write_text(json.dumps({'status':'COMPLETE','samples':actual,'source':'real Spring left/right GT, exact existing crop; last two frames','input_encodings_changed':False,'endpoint_GT_spot_checks':'first/middle/last exact match','source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),'elapsed_s':time.monotonic()-started},indent=2)+'\n')
 if context.world_size>1:dist.destroy_process_group()
if __name__=='__main__':main()
