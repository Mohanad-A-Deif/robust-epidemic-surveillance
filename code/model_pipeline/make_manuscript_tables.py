#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import pandas as pd

B='\\'
ROW=' '+B+B
SHORT={
'Proposed revised':'Proposed revised','Original proposed ablation':'Original proposed',
'Known-delay proposed oracle':'Known-delay oracle','No-delay ablation':'No-delay ablation',
'No-robustness ablation':'No-robustness ablation','No-graph ablation':'No-graph ablation',
'Fixed geographic graph':'Fixed geographic graph','Arrival interpolation':'Arrival interpolation',
'Oracle timestamp interpolation':'Timestamp oracle','Delay backprojection':'Delay backprojection',
'Kalman/RTS smoother (internal)':'Kalman/RTS (internal)',
'Delay-aware state-space (internal)':'Delay-aware SSM (internal)',
'Robust median smoother':'Robust median','Robust low-rank completion':'Robust low-rank',
'Graph-temporal reconstruction':'Graph-temporal'}

def esc(s):
    s=str(s)
    for a,b in [('&',B+'&'),('%',B+'%'),('_',B+'_'),('#',B+'#')]: s=s.replace(a,b)
    return s

def write(path,text): path.parent.mkdir(parents=True,exist_ok=True);path.write_text(text,encoding='utf-8')

def summary_pivot(path):
    df=pd.read_csv(path);p=df.pivot_table(index='method',columns='metric',values=['mean','std'],aggfunc='first');rows=[]
    for m in p.index:
        def v(kind,metric):
            try:return float(p.loc[m,(kind,metric)])
            except:return np.nan
        rows.append({'method':m,'rmse':v('mean','rmse'),'rmse_sd':v('std','rmse'),'mae':v('mean','mae'),'pearson':v('mean','pearson_r'),'runtime':v('mean','runtime_seconds')})
    return pd.DataFrame(rows).sort_values('rmse')

def bench_table(df,caption,label):
    best=df.rmse.min();lines=[f'{B}begin{{table}}[!htbp]',f'{B}centering',f'{B}caption{{{caption}}}',f'{B}label{{{label}}}',f'{B}small',f'{B}resizebox{{{B}textwidth}}{{!}}{{%',f'{B}begin{{tabular}}{{lrrrr}}',f'{B}toprule',f'Method & RMSE (mean ${B}pm$ SD) & MAE & Pearson $r$ & Runtime (s)'+ROW,f'{B}midrule']
    for _,r in df.iterrows():
        val=f'{r.rmse:.3f} ${B}pm$ {r.rmse_sd:.3f}'
        if np.isclose(r.rmse,best):val=f'{B}textbf{{{val}}}'
        lines.append(f'{esc(SHORT.get(r.method,r.method))} & {val} & {r.mae:.3f} & {r.pearson:.3f} & {r.runtime:.3f}'+ROW)
    lines += [f'{B}bottomrule',f'{B}end{{tabular}}}}',f'{B}begin{{minipage}}{{0.98{B}textwidth}}{B}footnotesize RMSE is on the ${B}log(1+{B}mathrm{{incidence}}/100{{,}}000)$ scale. Oracle methods are reference bounds; methods marked internal are simplified study implementations.{B}end{{minipage}}',f'{B}end{{table}}']
    return '\n'.join(lines)+'\n'

def robustness_table(path):
    df=pd.read_csv(path);means=df.groupby(['scenario','method'],as_index=False).rmse.mean()
    order=['missing_00','missing_10','missing_30','missing_50','delay_light','delay_moderate','delay_severe','outlier_00','outlier_05','outlier_10','outlier_20','outlier_30','prior_mismatch']
    lm={'missing_00':f'Missing 0{B}%','missing_10':f'Missing 10{B}%','missing_30':f'Missing 30{B}%','missing_50':f'Missing 50{B}%','delay_light':'Delay: light','delay_moderate':'Delay: moderate','delay_severe':'Delay: severe','outlier_00':f'Outliers 0{B}%','outlier_05':f'Outliers 5{B}%','outlier_10':f'Outliers 10{B}%','outlier_20':f'Outliers 20{B}%','outlier_30':f'Outliers 30{B}%','prior_mismatch':'Delay-prior mismatch'}
    shown=['Proposed revised','No-delay ablation','No-robustness ablation']
    lines=[f'{B}begin{{table}}[!htbp]',f'{B}centering',f'{B}caption{{Robustness and corruption sensitivity over 20 paired seeds.}}',f'{B}label{{tab:robustness}}',f'{B}small',f'{B}resizebox{{{B}textwidth}}{{!}}{{%',f'{B}begin{{tabular}}{{lrrrl}}',f'{B}toprule','Scenario & Proposed & No delay & No robustness & Lowest-RMSE method'+ROW,f'{B}midrule']
    for sc in order:
        sub=means[means.scenario==sc].sort_values('rmse')
        if sub.empty:continue
        best=sub.iloc[0];cells=[]
        for m in shown:
            q=sub[sub.method==m];x=float(q.rmse.iloc[0]) if len(q) else np.nan;txt='--' if not np.isfinite(x) else f'{x:.3f}'
            if np.isfinite(x) and np.isclose(x,best.rmse):txt=f'{B}textbf{{{txt}}}'
            cells.append(txt)
        besttxt=f'{esc(SHORT.get(best.method,best.method))} ({best.rmse:.3f})'
        lines.append(f'{lm.get(sc,esc(sc))} & {cells[0]} & {cells[1]} & {cells[2]} & {besttxt}'+ROW)
    lines += [f'{B}bottomrule',f'{B}end{{tabular}}}}',f'{B}begin{{minipage}}{{0.98{B}textwidth}}{B}footnotesize One factor changes at a time using nested corruption templates. The lowest value is determined from every method evaluated in that scenario, not only the three displayed numeric columns.{B}end{{minipage}}',f'{B}end{{table}}']
    return '\n'.join(lines)+'\n'

def graph_table(path):
    df=pd.read_csv(path);metrics=['graph_support_f1','graph_auroc','graph_auprc','graph_normalized_frobenius','graph_edge_weight_r','rmse'];p=df[df.metric.isin(metrics)].pivot(index='graph_type',columns='metric',values='mean')
    lines=[f'{B}begin{{table}}[!htbp]',f'{B}centering',f'{B}caption{{Directed graph recovery on separate synthetic test instances.}}',f'{B}label{{tab:graphrecovery}}',f'{B}small',f'{B}resizebox{{{B}textwidth}}{{!}}{{%',f'{B}begin{{tabular}}{{lrrrrrr}}',f'{B}toprule',f'True graph & Support F1 & AUROC & AUPRC & Norm. Frobenius & Edge-weight $r$ & State RMSE'+ROW,f'{B}midrule']
    for gt,r in p.iterrows():lines.append(f"{esc(gt.replace('_',' ').title())} & {r.graph_support_f1:.3f} & {r.graph_auroc:.3f} & {r.graph_auprc:.3f} & {r.graph_normalized_frobenius:.3f} & {r.graph_edge_weight_r:.3f} & {r.rmse:.3f}"+ROW)
    lines += [f'{B}bottomrule',f'{B}end{{tabular}}}}',f'{B}begin{{minipage}}{{0.98{B}textwidth}}{B}footnotesize The true support was not supplied as a candidate mask. Values near 0.5 for AUROC and near zero for edge-weight correlation indicate weak directed-edge identification despite low state RMSE.{B}end{{minipage}}',f'{B}end{{table}}']
    return '\n'.join(lines)+'\n'

def scale_table(path):
    df=pd.read_csv(path);fm={'n_nodes':f'Nodes $n$','n_time':f'Time points $T$','max_delay':f'Maximum delay $D$','graph_density':'True density'}
    lines=[f'{B}begin{{table}}[!htbp]',f'{B}centering',f'{B}caption{{One-factor-at-a-time scalability, runtime, and memory analysis.}}',f'{B}label{{tab:scalability}}',f'{B}small',f'{B}begin{{tabular}}{{lrrrr}}',f'{B}toprule','Factor & Level & Median runtime (s) & Mean peak RSS (MB) & Mean RMSE'+ROW,f'{B}midrule']
    for _,r in df.sort_values(['factor','level']).iterrows():lines.append(f"{fm.get(r.factor,esc(r.factor))} & {r.level:g} & {r.runtime_median:.3f} & {r.memory_mean_mb:.1f} & {r.rmse_mean:.3f}"+ROW)
    lines += [f'{B}bottomrule',f'{B}end{{tabular}}',f'{B}begin{{minipage}}{{0.98{B}textwidth}}{B}footnotesize Peak memory is absolute process resident-set size (RSS). Each row averages three independent seeds; only the named factor changes.{B}end{{minipage}}',f'{B}end{{table}}']
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--final',type=Path,required=True);ap.add_argument('--synthetic',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    write(a.out/'tab_reference.tex',bench_table(summary_pivot(a.final/'reference'/'retrospective_reconstruction_summary_long.csv'),'Retrospective reconstruction on the locked semi-real test interval.','tab:reference'))
    write(a.out/'tab_causal.tex',bench_table(summary_pivot(a.final/'causal'/'causal_nowcasting_summary_long.csv'),'Causal rolling nowcasting on the locked semi-real test interval.','tab:causal'))
    write(a.out/'tab_robustness.tex',robustness_table(a.final/'robustness'/'robustness_seed_metrics.csv'))
    write(a.out/'tab_graph_recovery.tex',graph_table(a.synthetic/'graph_recovery'/'graph_recovery_summary.csv'))
    sp=a.synthetic/'scalability'/'scalability_runtime_memory.csv'
    if sp.exists():write(a.out/'tab_scalability.tex',scale_table(sp))
    print('created',len(list(a.out.glob('*.tex'))),'tables')
if __name__=='__main__':main()
