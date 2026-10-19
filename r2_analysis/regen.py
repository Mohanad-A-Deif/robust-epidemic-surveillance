from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[1]);HERE=_P(__file__).resolve().parent
import sys;sys.path.insert(0,REPO+'/code/model_pipeline')
import make_publication_artifacts as M, numpy as np, pandas as pd, matplotlib.pyplot as plt
from pathlib import Path
_orig=M.shorten
def sh(n):
    s=_orig(n); return 'Hard-mask variant' if s=='Original' else s
M.shorten=sh
M.setup()
R=Path(REPO+'/results/raw/reference');out=HERE/'figures'
# reference benchmark only
df=pd.read_csv(R/'retrospective_reconstruction_summary_long.csv');rm=df[df.metric=='rmse'].sort_values('mean')
fig,ax=plt.subplots(figsize=(7.2,3.6));y=np.arange(len(rm));xerr=np.vstack([rm['mean']-rm['ci95_low'],rm['ci95_high']-rm['mean']])
ax.errorbar(rm['mean'],y,xerr=xerr,fmt='o',capsize=2,linewidth=.8,markersize=3,color='0.15')
ax.set_yticks(y);ax.set_yticklabels([sh(v) for v in rm.method]);ax.invert_yaxis();ax.set_xlabel('RMSE (mean and bootstrap 95% CI)');ax.set_title('Retrospective reconstruction on locked test interval')
ax.grid(True,axis='x',linestyle=':',linewidth=.45,color='0.75');fig.tight_layout();M.finish(fig,out/'fig_reference_benchmark.png')
M.significance_matrix(R/'retrospective_reconstruction_seed_metrics.csv',R/'retrospective_reconstruction_paired_tests.csv',out)
# average ranks with bootstrap CI
sm=pd.read_csv(R/'retrospective_reconstruction_seed_metrics.csv');P=sm.pivot_table(index='seed',columns='method',values='rmse')
ranks=P.rank(axis=1);avg=ranks.mean();rng=np.random.default_rng(2026);B=np.array([ranks.iloc[rng.integers(0,len(ranks),len(ranks))].mean().values for _ in range(5000)])
lo,hi=np.percentile(B,[2.5,97.5],axis=0);d=pd.DataFrame({'m':avg.index,'a':avg.values,'lo':lo,'hi':hi}).sort_values('a')
print(d.round(2).to_string())
fig,ax=plt.subplots(figsize=(6.0,3.9));y=np.arange(len(d))
ax.errorbar(d.a,y,xerr=np.vstack([d.a-d.lo,d.hi-d.a]),fmt='o',color='0.15',ecolor='0.45',capsize=3,markersize=3.5,linewidth=.9)
ax.set_yticks(y);ax.set_yticklabels([sh(x) for x in d.m]);ax.invert_yaxis();ax.set_xlabel('Average RMSE rank (bootstrap 95% CI; lower is better)')
ax.grid(True,axis='x',linestyle=':',linewidth=.45,color='0.75');ax.tick_params(right=True,top=False);fig.tight_layout();M.finish(fig,out/'fig_average_ranks.png')
# runtime accuracy two panels
p=df.pivot_table(index='method',columns='metric',values='mean',aggfunc='first').reset_index().sort_values('rmse')
grays=['0.1','0.25','0.4','0.55','0.7']
fig,axs=plt.subplots(1,2,figsize=(8.4,3.6));hs=[]
for ax,zoom in zip(axs,[False,True]):
    for k,row in enumerate(p.itertuples()):
        mk=M.MARKERS[k%len(M.MARKERS)];c=grays[k%4];filled=k>=len(M.MARKERS)
        if zoom and row.runtime_seconds<1: continue
        h=ax.scatter(row.runtime_seconds,row.rmse,marker=mk,s=30,facecolors=c if (filled or mk in 'xXP*') else 'none',edgecolors=c,linewidths=.9,label=sh(row.method))
        if not zoom: hs.append(h)
    ax.set_xscale('log');ax.set_xlabel('Runtime per seed (s, log scale)');ax.grid(True,linestyle=':',linewidth=.45,color='0.75')
axs[0].set_ylabel('Retrospective RMSE');axs[0].set_title('All methods');axs[1].set_title('Zoomed iterative-model cluster')
axs[0].text(-0.12,1.04,'(a)',transform=axs[0].transAxes,fontweight='bold');axs[1].text(-0.12,1.04,'(b)',transform=axs[1].transAxes,fontweight='bold')
leg=fig.legend(handles=hs,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,-.02))
fig.tight_layout(rect=[0,.17,1,1]);M.finish(fig,out/'fig_runtime_accuracy.png',leg)
