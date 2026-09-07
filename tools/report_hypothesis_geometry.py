#!/usr/bin/env python3
"""Collect replacement decoder evidence, paired effects and fixed failure cases."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT),str(ROOT/'src')]
import torch
from models.hypothesis_geometry import HypothesisGeometry
from tools.train_hypothesis_geometry import digest
from tools.report_native_temporal_stereo import bootstrap,KEYS

def main():
    p=argparse.ArgumentParser();p.add_argument('--run-dir',type=Path,required=True);a=p.parse_args();torch.set_num_threads(1)
    evaluation=json.loads((a.run_dir/'evaluation/metrics.json').read_text());rows=json.loads((a.run_dir/'evaluation/per_sample.json').read_text())
    if evaluation['status']!='COMPLETE' or evaluation['samples']!=1294:raise RuntimeError('incomplete benchmark')
    raw=json.loads((a.run_dir/'raw_checks.json').read_text())
    if raw['status']!='PASS':raise RuntimeError('raw inference unverified')
    receipts={};ownership={}
    for name in ('no_memory','memory'):
        receipt=json.loads((a.run_dir/name/'receipt.json').read_text());summary=json.loads((a.run_dir/name/'summary.json').read_text())
        if summary['status']!='COMPLETE' or summary['steps']!=2000:raise RuntimeError('not completed fixed-budget training')
        if summary['checkpoint_sha256']!=evaluation['models'][name]['checkpoint_sha256']:raise RuntimeError('evaluated a different checkpoint')
        payload=torch.load(a.run_dir/name/'final.pt',map_location='cpu',weights_only=False);torch.manual_seed(42)
        initial=HypothesisGeometry(use_memory=name=='memory').state_dict();changes={}
        for k,v in payload['model'].items():
            if not torch.isfinite(v).all():raise RuntimeError('nonfinite learned tensor')
            changes[k]=float((v-initial[k]).abs().max())
        if not any(changes.values()):raise RuntimeError('decoder parameters unchanged')
        ownership[name]=changes;receipts[name]={'receipt':receipt,'summary':summary}
    for key in ('world_size','seed','steps','training_endpoints','development_endpoints','cache_lineage_sha256','model_source_sha256','loss_source_sha256','trainable_parameters'):
        if receipts['no_memory']['receipt'][key]!=receipts['memory']['receipt'][key]:raise RuntimeError('uncontrolled comparison: '+key)
    metrics={name:{k:v['value'] for k,v in values.items()} for name,values in evaluation['metrics'].items()}
    paired={name:{control:bootstrap(rows,name,control) for control in (('v1','no_memory') if name=='memory' else ('v1',))} for name in ('no_memory','memory')}
    original=json.loads((ROOT/'runs/metric_stereo_video/formal_a5_seed42/run_receipt.json').read_text())['runtime_source_sha256']
    changed=[n for n,value in original.items() if digest(ROOT/n)!=value]
    if changed:raise RuntimeError('original A5 runtime changed')
    result={'status':'COMPLETE','updates_per_arm':2000,'validation_endpoints':1294,'metrics':metrics,'training':receipts,'paired_sequence_comparisons':paired,
      'fixed_failure_points':evaluation['fixed_failure_points'],'raw_verification':raw,'parameter_changes':ownership,'original_source_files_checked':len(original),'original_source_files_changed':changed,
      'claim_boundary':'new pixel-level hypothesis decoder; frozen FFS/VGGT observations; explicit-memory effect isolated within the pair; not full NMRF reproduction; reused development benchmark'}
    (a.run_dir/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# 保留表面候选的几何解码：真实训练与验收','',
      '两组各完成 2,000 次八卡更新，使用相同初始化、845 个训练端点、完整 8 帧图像片段和逐像素真实 GT；全部 1,294 个验证端点均已评估。','',
      '| 模型 | 全 GT EPE ↓ | temporal ↓ | 好像素受损 % ↓ | 相对 A5 >5px 退化 % ↓ | 固定机会恢复 % ↑ |',
      '|---|---:|---:|---:|---:|---:|']
    for name in ('A5','v1','no_memory','memory'):
        v=[metrics[name][key] for key in KEYS];lines.append(f'| {name} | {v[0]:.6f} | {v[1]:.6f} | {100*v[2]:.6f} | {100*v[3]:.6f} | {100*v[4]:.6f} |')
    lines+=['','新结构替换固定加权锚点的几何解码器。23 个候选分别输出 8×8 像素的分数与 ±4 px 偏移，每像素选择一个表面；输入包括原始双目观测、粗匹配模式、VGGT 和本模型自身历史，不读取 A5/v1 最终预测。',
      'no_memory 和 memory 均使用每帧因果 VGGT；差别是是否提供显式视差历史及其局部位置候选。GT 监督保留单元内全部 64 个像素，支持同一粗单元内不同前景/背景表面。',
      '方法参考 [NMRF, CVPR 2024](https://openaccess.thecvf.com/content/CVPR2024/papers/Guan_Neural_Markov_Random_Field_for_Stereo_Matching_CVPR_2024_paper.pdf) 的候选保留和逐像素选择；未复现完整 DPN 或变分推断。','',
      '| 固定失败点 | GT | A5 | v1 | no_memory | memory |','|---|---:|---:|---:|---:|---:|']
    for index in (2,24,29):
        values={r['variant']:r for r in evaluation['fixed_failure_points'] if r['dataset_index']==index};r=values['memory']
        lines.append(f"| {index}: ({r['x']},{r['y']}) | {r['GT']:.4f} | {r['A5']:.4f} | {r['v1']:.4f} | {values['no_memory']['prediction']:.4f} | {r['prediction']:.4f} |")
    lines+=['','固定案例保留每个候选的基础值、预测值、有效性、概率及最终选择；详见 results.json。所有模型继续使用原固定机会与安全分母，历史重投影使用各自真实递归输出、有效性和置信度。',
      'results.json 包含按 8 个序列配对 bootstrap 的区间；evaluation/metrics.json 包含边界、动态、细节、覆盖率、候选可恢复性以及新增/消除大退化的完整指标。允许组件取舍，不要求五项同时变好。',
      '原图/缓存输入和输出、扰动未来图像后的过去输出、物理截断并独立重编码的前缀输出均逐像素一致。原 A5 运行源码与 v1 权重保留。',
      '本设计使用了先前已检查的开发失败点，验证集也已经反复用于开发；这些结果不构成独立确认。','']
    (a.run_dir/'REPORT.md').write_text('\n'.join(lines));print(json.dumps({'status':'COMPLETE','arms':['no_memory','memory']}))

if __name__=='__main__':main()
