#!/usr/bin/env python3
"""Assemble completed literature component evidence without changing acceptance."""
import argparse,hashlib,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import numpy as np
import torch
from models.temporal_candidate_repair import TemporalCandidateRepair
from models.codd_temporal_repair import CoddTemporalRepair
from models.learned_codd_temporal_repair import LearnedCoddTemporalRepair
from tools.eval_temporal_candidate_repair import to_device

CORE=('preserve_v1_all_gt_epe','preserve_v1_temporal','reduce_good_pixel_damage','reduce_large_5px','increase_fixed_opportunity_recovery')
KEYS=('all_gt_penalized_epe_px','temporal_matched_penalized_delta_epe_px','good_current_damage_rate','large_degradation_5px_rate','fixed_opportunity_recovery_rate')


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--bank',type=Path,required=True);p.add_argument('--v1-checkpoint',type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(4)
    reports={k:json.loads((a.run_dir/f'evaluation_{k}/metrics.json').read_text()) for k in ('supervision','learned')}
    baseline={k:v['value'] for k,v in reports['supervision']['metrics']['v1'].items()}
    arms={};models={};intervals={}
    vp=torch.load(a.v1_checkpoint,map_location='cpu',weights_only=False);v1=TemporalCandidateRepair().cuda().eval();v1.load_state_dict(vp['model'])
    for group,report in reports.items():
        assert report['samples']==1294
        rows=json.loads((a.run_dir/f'evaluation_{group}/per_sample.json').read_text())
        assert len({r['dataset_index'] for r in rows})==1294
        for metric,value in report['metrics']['v1'].items():assert baseline[metric]==value['value']
        for name,acceptance in report['acceptance'].items():
            checkpoint=Path(report['models'][name]['checkpoint']);payload=torch.load(checkpoint,map_location='cpu',weights_only=False)
            assert payload['step']==12500
            metrics={k:v['value'] for k,v in report['metrics'][name].items()}
            arms[name]=dict(group=group,metrics=metrics,steps=payload['step'],five_user_conditions_pass=all(acceptance['checks'][k] for k in CORE),checks=acceptance['checks'],checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),development=payload['calibration'][0])
            cls=CoddTemporalRepair if group=='supervision' else LearnedCoddTemporalRepair
            model=cls(v1,v1_shift=payload['config']['v1_shift']).cuda().eval();model.load_state_dict(payload['model']);models[name]=model
            sequence={}
            for row in rows:
                sq=sequence.setdefault(row['sequence_id'],{})
                for key in KEYS:
                    old,new=row['v1/'+key],row[name+'/'+key]
                    assert old[1]==new[1]
                    sums=sq.setdefault(key,[0.,0]);sums[0]+=new[0]-old[0];sums[1]+=old[1]
            rng=np.random.default_rng(42);ci={}
            for key in KEYS:
                x=np.array([s[key] for s in sequence.values()]);draws=rng.integers(len(x),size=(10000,len(x)));z=x[draws].sum(1);delta=z[:,0]/z[:,1].clip(1)
                ci[key]=dict(arm_minus_v1=float(x[:,0].sum()/max(x[:,1].sum(),1)),sequence_bootstrap_95pct=np.quantile(delta,[.025,.975]).tolist())
            intervals[name]=ci
    accepted=[name for name,x in arms.items() if x['five_user_conditions_pass']]
    result=dict(status='COMPLETE',trained_arms=len(arms),updates_per_arm=12500,validation_endpoints_per_arm=1294,baseline_v1=baseline,arms=arms,accepted_validation_arms=accepted,sequence_bootstrap=intervals,
                scope='CODD supervision and frozen FFS matching component studies; not full CODD or TC-Stereo reproduction',validation_role='repeatedly inspected development benchmark, not independent confirmation')
    (a.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    points=[]
    with torch.inference_mode():
        for index,x,y in ((2,253,147),(24,581,200),(29,570,20)):
            record=to_device(torch.load(a.cache/'validation'/f'{index:06d}.pt',map_location='cpu',weights_only=False),'cuda')
            bank=to_device(torch.load(a.bank/'validation'/f'{index:06d}.pt',map_location='cpu',weights_only=False),'cuda')['current']
            c=record['current'];gt=record['target'];old,gate=v1(c['features'],c['base'],c['history'],c['history_valid'],vp['logit_shift'])
            def point(t):return float(t[0,0,y,x])
            row=dict(index=index,x=x,y=y,identity=record['identity'],gt=point(gt),a5=point(c['base']),history=point(c['history']),v1=point(old),v1_error=point((old-gt).abs()),v1_gate=point(gate),learned_cues=bank['learned'][0,:,y,x].float().tolist(),arms={})
            new_outputs={}
            for name,model in models.items():
                out=model(c['features'],c['base'],c['history'],c['history_valid'],bank)
                new_outputs[name]=out
                row['arms'][name]=dict(prediction=point(out['prediction']),error=point((out['prediction']-gt).abs()),gate=point(out['gate']),reset=point(out['reset_weight']),fusion=point(out['fusion_weight']))
            points.append(row)
            if index in (2,24):
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                name='learned_codd_codd';out=new_outputs[name]
                h,w=gt.shape[-2:];x0,x1=max(0,x-64),min(w,x+65);y0,y1=max(0,y-64),min(h,y+65)
                def arr(t):return t.detach().cpu().numpy()
                maps=[('RGB',arr(record['rgb'][0].permute(1,2,0))),('GT',arr(gt[0,0])),('A5',arr(c['base'][0,0])),('Original history',arr(c['history'][0,0])),('v1',arr(old[0,0])),('Learned cues + CODD loss',arr(out['prediction'][0,0])),('v1 error',arr((old-gt).abs()[0,0])),('New error',arr((out['prediction']-gt).abs()[0,0])),('History weight',arr(out['gate'][0,0]))]
                fig,axes=plt.subplots(3,3,figsize=(11,10),constrained_layout=True)
                maxd=float(np.percentile(maps[1][1][y0:y1,x0:x1],98))
                maxerr=max(2,min(100,max(row['v1_error'],row['arms'][name]['error'])))
                for ax,(title,value) in zip(axes.flat,maps):
                    if title=='RGB':ax.imshow(value[y0:y1,x0:x1])
                    else:
                        limit=1 if title=='History weight' else maxerr if 'error' in title else maxd
                        im=ax.imshow(value[y0:y1,x0:x1],vmin=0,vmax=limit,cmap='magma' if 'error' in title else 'viridis');fig.colorbar(im,ax=ax,fraction=.046)
                    ax.plot(x-x0,y-y0,'r+');ax.axis('off');ax.set_title(title)
                fig.suptitle(f"Fixed case {index} ({x},{y}): error {row['v1_error']:.3f} → {row['arms'][name]['error']:.3f} px")
                directory=a.run_dir/'failure_cases';directory.mkdir(exist_ok=True);fig.savefig(directory/f'fixed_case_{index:06d}.png',dpi=130);plt.close(fig)
    (a.run_dir/'fixed_failure_points.json').write_text(json.dumps(points,indent=2)+'\n')
    lines=['# 论文指导下的 temporal 改进：实训验收','',f'6 组实验均完成 12,500 步训练及全部 1,294 个验证端点评估。五项同时通过的版本：{accepted or "无"}。','',
           '参考方法： [CODD, WACV 2023](https://arxiv.org/html/2111.09337v2) 的 reset/fusion 分开监督；[TC-Stereo, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04579.pdf) 的学习特征匹配及排除相邻峰后的歧义判断。两个官方仓库的代码版本和逐项适配差异见 docs/temporal_repair_literature_protocol.md。', '',
           '| 版本 | 全 GT EPE ↓ | temporal ↓ | 好像素受损 % ↓ | 相对 A5 新增 >5px 退化 % ↓ | 固定机会恢复 % ↑ |','|---|---:|---:|---:|---:|---:|']
    for name,metrics in {'v1':baseline,**{n:x['metrics'] for n,x in arms.items()}}.items():
        values=[metrics[k] for k in KEYS]
        lines.append('| '+name+' | '+' | '.join(f'{v*(100 if i>=2 else 1):.6f}' for i,v in enumerate(values))+' |')
    lines += ['', 'capacity_control 使用 v1 原损失；codd 使用官方误差阈值 5/1 px、独立归一化的接受/拒绝监督及 0.2 的近似等优融合正则；codd_regret 额外惩罚相对当前预测的大退化。supervision 组只有原 31 维输入；learned 组加入 24 维实际 A5 FFS 双目特征代价和代价峰差。每组内部结构、数据、初始化与训练预算相同，没有验证集校准。', '',
        '验收条件和分母完全沿用 v1：整体与 temporal 不退步、好像素受损率下降、相对 A5 的 >5 px 退化率下降、固定机会区域恢复率提高。另报相对 v1 新增/消除大退化、1 px 退化、严重度、覆盖率与原生边界/动态/细节分区。完整结果见 results.json 与 evaluation_*/metrics.json。', '',
        '固定失败点对照见 fixed_failure_points.json；其中序列 0005 帧 10 的 (253,147) 是前一轮增强监督错误拒绝可靠历史的案例，帧 32 的 (581,200) 是 v1 过度采纳坏历史的案例。failure_cases/ 使用一致色标展示 GT、原候选、预测和误差。', '',
        '结论边界：这仍是冻结 A5 与历史候选的组件实验，没有复现 CODD 的 RAFT3D 非刚体运动、空间卷积融合和递归记忆，也没有复现 TC-Stereo 的对比代价学习和视差/梯度迭代细化。失败不能被表述为论文方法无效。全量评估集已反复用于开发；按序列 bootstrap 区间见 results.json，不宣称独立确认。', '',
        'Stereo Any Video 的式 (12) 使用 t+1 帧，不能直接放入现有严格因果验收；相关机制若采用，需要单独实现因果版本：[ICCV 2025 论文](https://arxiv.org/html/2503.05549v2)。']
    (a.run_dir/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'status':'COMPLETE','trained_arms':len(arms),'accepted_validation_arms':accepted}),flush=True)
if __name__=='__main__':main()
