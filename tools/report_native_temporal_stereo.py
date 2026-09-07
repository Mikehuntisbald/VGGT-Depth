#!/usr/bin/env python3
"""Summarize completed architecture controls with paired sequence uncertainty."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np

KEYS=('all_gt_penalized_epe_px','temporal_matched_penalized_delta_epe_px','good_current_damage_rate','large_degradation_5px_rate','fixed_opportunity_recovery_rate')


def bootstrap(rows,arm,control):
    sequences=sorted({r['sequence_id'] for r in rows});result={};rng=np.random.default_rng(42)
    draws=rng.integers(len(sequences),size=(10000,len(sequences)))
    for key in KEYS:
        sums=[]
        for seq in sequences:
            group=[r for r in rows if r['sequence_id']==seq]
            a=np.array([r[arm+'/'+key] for r in group]).sum(0);b=np.array([r[control+'/'+key] for r in group]).sum(0)
            if a[1]!=b[1]:raise RuntimeError('fixed metric domain changed: '+key)
            sums.append([a[0]-b[0],a[1]])
        x=np.asarray(sums);samples=x[draws].sum(1);delta=samples[:,0]/samples[:,1].clip(1)
        result[key]={'arm_minus_control':float(x[:,0].sum()/x[:,1].sum()),'sequence_bootstrap_95pct':np.quantile(delta,[.025,.975]).tolist(),'sequences':len(sequences),'draws':10000}
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args()
    evaluation=json.loads((a.run_dir/'evaluation/metrics.json').read_text());rows=json.loads((a.run_dir/'evaluation/per_sample.json').read_text())
    zero=evaluation if 'zero_endpoint_only' in evaluation['metrics'] else json.loads((a.run_dir/'zero_diagnostics/metrics.json').read_text())
    if evaluation['status']!='COMPLETE' or zero['status']!='COMPLETE' or evaluation['samples']!=1294:raise RuntimeError('incomplete full-domain evaluation')
    arms=('late','early_seed','early_state');receipts={}
    for arm in arms:
        receipt=json.loads((a.run_dir/arm/'receipt.json').read_text());summary=json.loads((a.run_dir/arm/'summary.json').read_text())
        if summary['status']!='COMPLETE' or summary['steps']!=1000:raise RuntimeError('not fixed-budget complete')
        if summary['checkpoint_sha256']!=evaluation['models'][arm]['checkpoint_sha256']:raise RuntimeError('evaluated different checkpoint')
        receipts[arm]={'receipt':receipt,'summary':summary}
    for key in ('world_size','seed','training_endpoints','development_endpoints','cache_lineage_sha256','initialization_sha256','structure_sha256','training_labels_receipt_sha256'):
        if len({receipts[arm]['receipt'][key] for arm in arms})!=1:raise RuntimeError('uncontrolled training contract: '+key)
    for key,value in evaluation['metrics']['v1'].items():
        old=zero['metrics']['v1'][key]['value']
        if value['value'] is not None and not np.isclose(value['value'],old,rtol=1e-12,atol=1e-12):raise RuntimeError('baseline replay changed: '+key)
    metrics={n:{k:v['value'] for k,v in values.items()} for n,values in evaluation['metrics'].items()}
    diagnostic={n:{k:v['value'] for k,v in values.items()} for n,values in zero['metrics'].items() if n.startswith('zero_')}
    comparisons={arm:{control:bootstrap(rows,arm,control) for control in (('v1','late','early_seed') if arm=='early_state' else ('v1','late')) if control!=arm} for arm in arms}
    if 'zero_endpoint_only' in evaluation['metrics']:
        comparisons['diagnostics']={
          'sequential_execution_vs_A5':bootstrap(rows,'zero_endpoint_only','A5'),
          'causal_VGGT_schedule':bootstrap(rows,'zero_causal_each_frame','zero_endpoint_only'),
          'training_late':bootstrap(rows,'late','zero_causal_each_frame')}
    raw=json.loads((a.run_dir/'raw_final_checks.json').read_text())
    if raw['status']!='PASS':raise RuntimeError('raw path unverified')
    result={'status':'COMPLETE','trained_arms':list(arms),'updates_per_arm':1000,'validation_endpoints':1294,'metrics':metrics,'zero_update_diagnostics':diagnostic,
      'paired_sequence_comparisons':comparisons,'training':receipts,'raw_verification':raw,'fixed_failure_points':evaluation['fixed_cases'],
      'claim_boundary':'native architecture with frozen image encoders/native iterative weights; not a final-output correction; repeated development benchmark; no claim of full TC-Stereo reproduction',
      'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (a.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 原生时序双目结构：实训结果','',
      '三组分别完成 1,000 次八卡更新、845 个训练端点的相同抽样，以及全部 1,294 个验证端点评估。',
      '历史在新结构中进入 FFS 匹配初始化和迭代隐状态；最终联合 VGGT 几何解码。新模型自行递推历史，前向不读取 A5/v1 最终预测。','',
      '| 版本 | 全 GT EPE ↓ | temporal ↓ | 好像素受损 % ↓ | 相对 A5 >5px 退化 % ↓ | 固定机会恢复 % ↑ |',
      '|---|---:|---:|---:|---:|---:|']
    for name in ('A5','v1',*arms):
        v=[metrics[name][k] for k in KEYS]
        lines.append(f'| {name} | {v[0]:.6f} | {v[1]:.6f} | {100*v[2]:.6f} | {100*v[3]:.6f} | {100*v[4]:.6f} |')
    lines+=['','late 使用相同的每帧因果 VGGT、逐帧求解、完整图像和训练预算，保留后置历史融合；early_seed 将历史移到匹配初始化；early_state 进一步传播迭代隐状态。两个 early 组关闭旧后置历史融合。',
      '新模块另有候选接受/拒绝和 GT 匹配监督，因此 early 与 late 是完整设计方案的比较，并非只移动同一个模块位置的严格单变量实验。early_state 对 early_seed 更直接检验状态传播的增量。','',
      '零更新诊断：', '| 设置 | EPE | temporal |','|---|---:|---:|']
    for name,m in diagnostic.items():lines.append(f"| {name} | {m[KEYS[0]]:.6f} | {m[KEYS[1]]:.6f} |")
    lines+=['','零更新项分离逐帧数值执行和每帧因果 VGGT 输入的影响；相对旧 A5 的全部差异不能单独归因于前置时序设计。',
      '使用组件取舍分析，不要求五项同时变好。results.json 同时报出按 8 个序列 bootstrap 的配对区间、边界/动态/细节分区、有效覆盖率、相对 v1 新增及消除的大退化和严重度。',
      'fixed_failure_points 包含三个事先固定失败点的初始化、迭代后双目解和最终几何结果；由此判断正确解在哪一层丢失。',
      '原始图像入口与缓存入口逐像素一致；扰动最新帧后过去的输入和输出不变。GT 只用于监督与评估。',
      '本轮固定 FFS/VGGT 图像编码器及预训练迭代器权重，训练新的前置网络和联合几何解码器。验证集已反复用于开发，不能当作独立确认。',
      '结构参考 [TC-Stereo, ECCV 2024](https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04579.pdf) 的匹配前补全与状态融合，没有复现其完整梯度空间迭代。','']
    (a.run_dir/'REPORT.md').write_text('\n'.join(lines))
    print(json.dumps({'status':'COMPLETE','arms':list(arms)}))

if __name__=='__main__':main()
