#!/usr/bin/env python3
"""Create publication tables and grayscale PNG figures from completed results."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

DPI = 600
MARKERS = ['o','s','^','D','v','P','X','<','>','h','*']
LINESTYLES = ['-','--','-.',':']


def setup():
    plt.rcParams.update({
        'font.family':'serif',
        'font.serif':['DejaVu Serif','Times New Roman','Times'],
        'font.size':8,
        'axes.labelsize':8,
        'axes.titlesize':9,
        'legend.fontsize':7,
        'xtick.labelsize':7,
        'ytick.labelsize':7,
        'axes.linewidth':0.7,
        'xtick.direction':'in',
        'ytick.direction':'in',
        'xtick.major.size':3,
        'ytick.major.size':3,
        'savefig.dpi':DPI,
        'figure.dpi':150,
    })


def finish(fig, path: Path, legend=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if legend is not None:
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white', bbox_extra_artists=(legend,))
    else:
        fig.savefig(path, dpi=DPI, bbox_inches='tight', facecolor='white')
    plt.close(fig)


def shorten(name: str) -> str:
    mapping={
        'Baden-Württemberg':'BW','Bayern':'BY','Hessen':'HE','Niedersachsen':'NI',
        'Nordrhein-Westfalen':'NW','Rheinland-Pfalz':'RP',
        'Proposed revised':'Proposed','Original proposed ablation':'Original',
        'Known-delay proposed oracle':'Known-delay oracle','No-delay ablation':'No delay',
        'No-robustness ablation':'No robustness','No-graph ablation':'No graph',
        'Fixed geographic graph':'Fixed geography','Arrival interpolation':'Arrival interpolation',
        'Oracle timestamp interpolation':'Timestamp oracle','Delay backprojection':'Delay backprojection',
        'Kalman/RTS smoother (internal)':'Kalman/RTS','Delay-aware state-space (internal)':'Delay-aware SSM',
        'Robust median smoother':'Robust median','Robust low-rank completion':'Robust low rank',
        'Graph-temporal reconstruction':'Graph-temporal'
    }
    return mapping.get(name,name)


def workflow(out: Path):
    fig,ax=plt.subplots(figsize=(7.2,3.0)); ax.axis('off')
    boxes=[
        (0.02,0.58,0.17,0.25,'Real epidemic\ntrajectories\n(RKI)'),
        (0.22,0.58,0.17,0.25,'Injected transport\ncorruption\n(delay/drop/outlier)'),
        (0.42,0.58,0.17,0.25,'Train/validation\nconfiguration\nand locking'),
        (0.62,0.58,0.17,0.25,'Retrospective\nreconstruction'),
        (0.82,0.58,0.16,0.25,'Causal rolling\nnowcasting'),
        (0.22,0.12,0.17,0.25,'Delay posterior\nand robust\nstate update'),
        (0.42,0.12,0.17,0.25,'Directed graph\nlearning with\nsoft geography'),
        (0.62,0.12,0.17,0.25,'Uncertainty,\nstability and\nstatistical tests'),
    ]
    for x,y,w,h,text in boxes:
        p=FancyBboxPatch((x,y),w,h,boxstyle='round,pad=0.012,rounding_size=0.015',
                         facecolor='0.95',edgecolor='0.15',linewidth=0.8)
        ax.add_patch(p); ax.text(x+w/2,y+h/2,text,ha='center',va='center')
    arrows=[((.19,.705),(.22,.705)),((.39,.705),(.42,.705)),((.59,.705),(.62,.705)),((.79,.705),(.82,.705)),
            ((.305,.58),(.305,.37)),((.39,.245),(.42,.245)),((.59,.245),(.62,.245)),((.705,.37),(.705,.58))]
    for a,b in arrows:
        ax.add_patch(FancyArrowPatch(a,b,arrowstyle='-|>',mutation_scale=9,linewidth=.8,color='0.15'))
    ax.text(.5,.97,'Leakage-safe evaluation and healthcare analytics workflow',ha='center',va='top',fontweight='bold')
    finish(fig,out/'fig_workflow.png')


def rki_trajectories(processed: Path, out: Path):
    df=pd.read_csv(processed/'x_log1p.csv',index_col=0,parse_dates=True)
    fig,ax=plt.subplots(figsize=(7.2,3.2))
    for k,col in enumerate(df.columns):
        ax.plot(df.index,df[col],linestyle=LINESTYLES[k%4],marker=MARKERS[k%len(MARKERS)],
                markevery=28,linewidth=0.9,markersize=2.5,label=shorten(col),color=str(0.1+0.12*k))
    ax.set_xlabel('Date'); ax.set_ylabel('log1p incidence per 100,000')
    ax.grid(True,linestyle=':',linewidth=.45,color='0.75')
    leg=ax.legend(ncol=6,loc='upper center',bbox_to_anchor=(.5,-.18),frameon=False)
    fig.tight_layout()
    finish(fig,out/'fig_rki_trajectories.png',leg)


def benchmark(reference: Path, causal: Path, out: Path):
    for path,name,title in [(reference,'fig_reference_benchmark.png','Retrospective reconstruction on locked test interval'),
                            (causal,'fig_causal_benchmark.png','Causal rolling nowcasting on locked test interval')]:
        df=pd.read_csv(path)
        rm=df[df.metric=='rmse'].copy().sort_values('mean')
        fig,ax=plt.subplots(figsize=(7.2,3.6))
        y=np.arange(len(rm))
        xerr=np.vstack([rm['mean']-rm['ci95_low'],rm['ci95_high']-rm['mean']])
        ax.errorbar(rm['mean'],y,xerr=xerr,fmt='o',capsize=2,linewidth=.8,markersize=3,color='0.15')
        ax.set_yticks(y); ax.set_yticklabels([shorten(v) for v in rm.method])
        ax.invert_yaxis(); ax.set_xlabel('RMSE (mean and bootstrap 95% CI)'); ax.set_title(title)
        ax.grid(True,axis='x',linestyle=':',linewidth=.45,color='0.75')
        fig.tight_layout(); finish(fig,out/name)


def runtime_accuracy(reference: Path,out: Path):
    df=pd.read_csv(reference)
    p=df.pivot_table(index='method',columns='metric',values='mean',aggfunc='first').reset_index()
    fig,ax=plt.subplots(figsize=(6.4,4.0))
    for k,row in p.iterrows():
        ax.scatter(row.runtime_seconds,row.rmse,marker=MARKERS[k%len(MARKERS)],s=25,facecolors='none',edgecolors='0.15')
        ax.annotate(shorten(row.method),(row.runtime_seconds,row.rmse),xytext=(3,3),textcoords='offset points',fontsize=6)
    ax.set_xscale('log'); ax.set_xlabel('Runtime per seed (s, log scale)'); ax.set_ylabel('Retrospective RMSE')
    ax.grid(True,linestyle=':',linewidth=.45,color='0.75'); fig.tight_layout(); finish(fig,out/'fig_runtime_accuracy.png')


def sensitivity(robust_seed: Path,out: Path):
    df=pd.read_csv(robust_seed)
    df=df[df.method=='Proposed revised'].copy()
    specs=[('missingness',['missing_00','missing_10','missing_30','missing_50'],[0,10,30,50],'Injected missingness (%)','fig_missingness_sensitivity.png'),
           ('delay',['delay_light','delay_moderate','delay_severe'],['Light','Moderate','Severe'],'Delay severity','fig_delay_sensitivity.png'),
           ('outlier',['outlier_00','outlier_05','outlier_10','outlier_20','outlier_30'],[0,5,10,20,30],'Injected outliers among reports (%)','fig_outlier_sensitivity.png')]
    for family,order,xlabels,xlabel,file in specs:
        sub=df[df.family==family]
        rows=[]
        for sc in order:
            v=sub[sub.scenario==sc].rmse.to_numpy(float)
            rows.append((v.mean(),v.std(ddof=1)))
        means=np.array([r[0] for r in rows]); sds=np.array([r[1] for r in rows]); x=np.arange(len(order))
        fig,ax=plt.subplots(figsize=(4.8,3.2))
        ax.errorbar(x,means,yerr=sds,fmt='o-',capsize=3,linewidth=.9,markersize=3,color='0.15')
        ax.set_xticks(x); ax.set_xticklabels(xlabels); ax.set_xlabel(xlabel); ax.set_ylabel('RMSE (mean ± SD)')
        ax.grid(True,linestyle=':',linewidth=.45,color='0.75'); fig.tight_layout(); finish(fig,out/file)


def significance_matrix(seed_metrics: Path, tests: Path, out: Path):
    sm=pd.read_csv(seed_metrics); methods=sm.groupby('method').rmse.mean().sort_values().index.tolist()
    t=pd.read_csv(tests)
    n=len(methods); mat=np.full((n,n),np.nan)
    for i,a in enumerate(methods):
        mat[i,i]=0
        for j,b in enumerate(methods):
            if i==j: continue
            r=t[((t.method_a==a)&(t.method_b==b))|((t.method_a==b)&(t.method_b==a))]
            if len(r):
                row=r.iloc[0]; p=float(row.wilcoxon_p_holm)
                mean_a=float(sm[sm.method==a].rmse.mean()); mean_b=float(sm[sm.method==b].rmse.mean())
                mat[i,j]=(1 if mean_a<mean_b else -1) if p<.05 else 0
    fig,ax=plt.subplots(figsize=(7.4,6.7))
    # Use three printable gray levels. A continuous Greys map makes -1 pure
    # white, which can hide the minus symbol and falsely look like missing data.
    cmap=ListedColormap(['0.88','0.55','0.05'])
    norm=BoundaryNorm([-1.5,-0.5,0.5,1.5],cmap.N)
    im=ax.imshow(mat,cmap=cmap,norm=norm,aspect='auto')
    ax.set_xticks(range(n));ax.set_xticklabels([shorten(m) for m in methods],rotation=70,ha='right')
    ax.set_yticks(range(n));ax.set_yticklabels([shorten(m) for m in methods])
    for i in range(n):
        for j in range(n):
            symbol='+' if mat[i,j]==1 else ('-' if mat[i,j]==-1 else ('=' if i!=j else '·'))
            ax.text(j,i,symbol,ha='center',va='center',fontsize=6,
                    color='white' if mat[i,j] == 1 else 'black')
    ax.set_title('Paired Wilcoxon-Holm comparison (row vs column)')
    ax.set_xlabel('+ row lower RMSE; - row higher RMSE; = not significant')
    fig.tight_layout(); finish(fig,out/'fig_significance_matrix.png')


def correlation_heatmap(processed: Path,out: Path):
    df=pd.read_csv(processed/'x_log1p.csv',index_col=0)
    corr=df.corr(method='pearson')
    fig,ax=plt.subplots(figsize=(4.6,4.0))
    im=ax.imshow(corr.to_numpy(),cmap='Greys',vmin=-1,vmax=1)
    labs=[shorten(x) for x in corr.columns]
    ax.set_xticks(range(len(labs)));ax.set_xticklabels(labs,rotation=45,ha='right')
    ax.set_yticks(range(len(labs)));ax.set_yticklabels(labs)
    for i in range(len(labs)):
        for j in range(len(labs)):
            ax.text(j,i,f'{corr.iloc[i,j]:.2f}',ha='center',va='center',fontsize=6,
                    color='white' if corr.iloc[i,j]>.65 else 'black')
    cbar=fig.colorbar(im,ax=ax,shrink=.8);cbar.set_label('Pearson correlation')
    fig.tight_layout();finish(fig,out/'fig_correlation_heatmap.png')


def graph_figures(arrays_dir: Path,nodes_path: Path,threshold_path: Path,out: Path):
    files=sorted(arrays_dir.glob('proposed_seed_*.npz'))
    if not files:return
    ws=np.stack([np.load(f)['w_hat'] for f in files]); w=np.median(ws,axis=0)
    nodes=pd.read_csv(nodes_path); labels=[shorten(s) for s in nodes.state]
    fig,ax=plt.subplots(figsize=(4.7,4.0))
    im=ax.imshow(w,cmap='Greys',vmin=0,vmax=max(float(w.max()),1e-8))
    ax.set_xticks(range(len(labels)));ax.set_xticklabels(labels)
    ax.set_yticks(range(len(labels)));ax.set_yticklabels(labels)
    ax.set_xlabel('Source node j');ax.set_ylabel('Receiving node i')
    cbar=fig.colorbar(im,ax=ax,shrink=.8);cbar.set_label('Median learned weight')
    fig.tight_layout();finish(fig,out/'fig_graph_heatmap.png')

    th=1e-5
    if threshold_path.exists():
        try:
            p=json.loads(threshold_path.read_text());th=float(p['selected_graph_threshold'])
        except Exception:pass
    n=len(labels);angles=np.linspace(0,2*np.pi,n,endpoint=False);pos=np.c_[np.cos(angles),np.sin(angles)]
    fig,ax=plt.subplots(figsize=(5.0,4.6));ax.axis('off')
    ax.scatter(pos[:,0],pos[:,1],s=520,facecolors='white',edgecolors='0.1',linewidths=.9,zorder=3)
    for k,(x,y) in enumerate(pos):ax.text(x,y,labels[k],ha='center',va='center',fontweight='bold',zorder=4)
    maxw=max(float(w.max()),1e-9)
    for i in range(n):
        for j in range(n):
            if i==j or w[i,j]<=th:continue
            start=pos[j]*.88;end=pos[i]*.88
            rad=.16 if (i+j)%2==0 else -.16
            arrow=FancyArrowPatch(start,end,connectionstyle=f'arc3,rad={rad}',arrowstyle='-|>',
                                  mutation_scale=7,linewidth=.35+1.4*w[i,j]/maxw,color=str(.15+.55*(1-w[i,j]/maxw)),alpha=.9)
            ax.add_patch(arrow)
    ax.set_xlim(-1.35,1.35);ax.set_ylim(-1.25,1.25);ax.set_title('Median learned association/propagation graph across seeds')
    fig.tight_layout();finish(fig,out/'fig_learned_network.png')


def convergence(arrays_dir: Path,out: Path):
    histories=[]
    for f in sorted(arrays_dir.glob('proposed_seed_*.npz')):
        h=np.load(f)['normalized_objective'];histories.append(np.asarray(h,float))
    if not histories:return
    m=max(map(len,histories));a=np.full((len(histories),m),np.nan)
    for i,h in enumerate(histories):a[i,:len(h)]=h;a[i,len(h):]=h[-1]
    mean=np.nanmean(a,axis=0);sd=np.nanstd(a,axis=0,ddof=1);x=np.arange(len(mean))
    fig,ax=plt.subplots(figsize=(4.8,3.2));ax.plot(x,mean,'o-',markevery=max(1,len(x)//8),linewidth=.9,markersize=2.5,color='0.15')
    ax.fill_between(x,mean-sd,mean+sd,color='0.8',alpha=.7,linewidth=0)
    ax.set_xlabel('Outer iteration');ax.set_ylabel('Normalized objective');ax.grid(True,linestyle=':',linewidth=.45,color='0.75')
    fig.tight_layout();finish(fig,out/'fig_convergence.png')


def average_ranks(rank_path: Path,out: Path):
    df=pd.read_csv(rank_path).sort_values('average_rank')
    fig,ax=plt.subplots(figsize=(6.0,3.8));y=np.arange(len(df))
    ax.plot(df.average_rank,y,'o',color='0.15',markersize=3)
    for yi,x in zip(y,df.average_rank):ax.hlines(yi,0,x,colors='0.6',linewidth=.6)
    ax.set_yticks(y);ax.set_yticklabels([shorten(x) for x in df.method]);ax.invert_yaxis();ax.set_xlabel('Average RMSE rank (lower is better)')
    ax.grid(True,axis='x',linestyle=':',linewidth=.45,color='0.75');fig.tight_layout();finish(fig,out/'fig_average_ranks.png')


def synthetic_figures(syn: Path,out: Path):
    rec=syn/'graph_recovery'/'graph_recovery_summary.csv'
    if rec.exists():
        df=pd.read_csv(rec); f1=df[df.metric=='graph_support_f1'].copy();
        fig,ax=plt.subplots(figsize=(4.8,3.2));x=np.arange(len(f1));err=np.vstack([f1['mean']-f1.ci95_low,f1.ci95_high-f1['mean']])
        ax.errorbar(x,f1['mean'],yerr=err,fmt='o',capsize=3,color='0.15');ax.set_xticks(x);ax.set_xticklabels(f1.graph_type.str.replace('_',' '))
        ax.set_ylabel('Support F1 (bootstrap 95% CI)');ax.grid(True,axis='y',linestyle=':',linewidth=.45,color='0.75');fig.tight_layout();finish(fig,out/'fig_graph_recovery.png')
    scale=syn/'scalability'/'scalability_runtime_memory.csv'
    if scale.exists():
        df=pd.read_csv(scale)
        for factor,sub in df.groupby('factor'):
            fig,ax=plt.subplots(figsize=(4.6,3.1));sub=sub.sort_values('level')
            ax.plot(sub.level.astype(str),sub.runtime_median,'o-',color='0.15',linewidth=.9,markersize=3)
            ax.set_xlabel(factor.replace('_',' '));ax.set_ylabel('Median runtime (s)');ax.grid(True,linestyle=':',linewidth=.45,color='0.75')
            fig.tight_layout();finish(fig,out/f'fig_scalability_{factor}.png')
    mismatch=syn/'model_mismatch'/'model_mismatch_summary.csv'
    if mismatch.exists():
        df=pd.read_csv(mismatch);df=df[df.metric=='rmse']
        methods=df.groupby('method')['mean'].mean().sort_values().index[:4]
        fig,ax=plt.subplots(figsize=(6.0,3.6));kinds=df.kind.unique();x=np.arange(len(kinds));width=.18
        for k,m in enumerate(methods):
            s=df[df.method==m].set_index('kind').reindex(kinds)
            ax.plot(x,s['mean'],marker=MARKERS[k],linestyle=LINESTYLES[k],linewidth=.8,markersize=3,label=shorten(m),color=str(.15+.18*k))
        ax.set_xticks(x);ax.set_xticklabels([v.replace('_',' ') for v in kinds],rotation=20,ha='right');ax.set_ylabel('RMSE')
        ax.grid(True,linestyle=':',linewidth=.45,color='0.75');leg=ax.legend(loc='upper center',bbox_to_anchor=(.5,-.28),ncol=2,frameon=False)
        fig.tight_layout();finish(fig,out/'fig_model_mismatch.png',leg)


def curated_tables(final: Path,syn: Path,out: Path):
    out.mkdir(parents=True,exist_ok=True)
    ref=pd.read_csv(final/'reference'/'retrospective_reconstruction_summary_long.csv')
    causal=pd.read_csv(final/'causal'/'causal_nowcasting_summary_long.csv')
    def compact(df):
        piv=df.pivot_table(index='method',columns='metric',values=['mean','std','ci95_low','ci95_high'],aggfunc='first')
        rows=[]
        for m in piv.index:
            def val(field,metric):
                try:return float(piv.loc[m,(field,metric)])
                except:return np.nan
            rows.append({'Method':shorten(m),'RMSE_mean':val('mean','rmse'),'RMSE_SD':val('std','rmse'),'RMSE_CI_low':val('ci95_low','rmse'),'RMSE_CI_high':val('ci95_high','rmse'),'MAE_mean':val('mean','mae'),'Pearson_r':val('mean','pearson_r'),'Runtime_s':val('mean','runtime_seconds')})
        return pd.DataFrame(rows).sort_values('RMSE_mean')
    compact(ref).to_csv(out/'final_benchmark_table.csv',index=False)
    compact(causal).to_csv(out/'causal_nowcasting_table.csv',index=False)
    pd.read_csv(final/'robustness'/'sensitivity_table.csv').to_csv(out/'robustness_sensitivity_table.csv',index=False)
    if (syn/'graph_recovery'/'graph_recovery_summary.csv').exists():pd.read_csv(syn/'graph_recovery'/'graph_recovery_summary.csv').to_csv(out/'graph_evaluation_table.csv',index=False)
    if (syn/'model_mismatch'/'model_mismatch_summary.csv').exists():pd.read_csv(syn/'model_mismatch'/'model_mismatch_summary.csv').to_csv(out/'model_mismatch_table.csv',index=False)
    if (syn/'scalability'/'scalability_runtime_memory.csv').exists():pd.read_csv(syn/'scalability'/'scalability_runtime_memory.csv').to_csv(out/'scalability_runtime_table.csv',index=False)
    pd.read_csv(final/'reference'/'retrospective_reconstruction_paired_tests.csv').to_csv(out/'statistical_tests_reference.csv',index=False)
    pd.read_csv(final/'causal'/'causal_nowcasting_paired_tests.csv').to_csv(out/'statistical_tests_causal.csv',index=False)


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--rki-root',type=Path,required=True);ap.add_argument('--final-results',type=Path,required=True);ap.add_argument('--synthetic-results',type=Path,required=True);ap.add_argument('--output',type=Path,required=True);a=ap.parse_args()
    setup();fig=a.output/'figures';tab=a.output/'tables';
    workflow(fig);rki_trajectories(a.rki_root/'data'/'processed',fig)
    benchmark(a.final_results/'reference'/'retrospective_reconstruction_summary_long.csv',a.final_results/'causal'/'causal_nowcasting_summary_long.csv',fig)
    runtime_accuracy(a.final_results/'reference'/'retrospective_reconstruction_summary_long.csv',fig)
    sensitivity(a.final_results/'robustness'/'robustness_seed_metrics.csv',fig)
    significance_matrix(a.final_results/'reference'/'retrospective_reconstruction_seed_metrics.csv',a.final_results/'reference'/'retrospective_reconstruction_paired_tests.csv',fig)
    correlation_heatmap(a.rki_root/'data'/'processed',fig)
    graph_figures(a.final_results/'reference'/'arrays',a.rki_root/'data'/'processed'/'nodes.csv',a.final_results/'protocol'/'locked_test_protocol.json',fig)
    convergence(a.final_results/'reference'/'arrays',fig)
    average_ranks(a.final_results/'reference'/'retrospective_reconstruction_average_ranks.csv',fig)
    synthetic_figures(a.synthetic_results,fig)
    curated_tables(a.final_results,a.synthetic_results,tab)
    print(f'Created {len(list(fig.glob("*.png")))} figures and {len(list(tab.glob("*.csv")))} tables.')

if __name__=='__main__':main()
