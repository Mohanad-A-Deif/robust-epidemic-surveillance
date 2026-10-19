from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[1]);HERE=_P(__file__).resolve().parent
import sys,json,glob
sys.path.insert(0,REPO+'/code/model_pipeline')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt,matplotlib.dates as mdates
from matplotlib.colors import LinearSegmentedColormap,ListedColormap,BoundaryNorm
from matplotlib.patches import FancyArrowPatch
from pathlib import Path
import make_publication_artifacts as M
R=Path(REPO);RR=R/'results/raw';OUT=HERE/'figures'
BLACK,RED,BLUE,GREEN,PURPLE,BROWN,TEAL,GRAY='#000000','#8B0000','#00008B','#006400','#4B0082','#7B3F00','#00575E','#4D4D4D'
CYC=[BLUE,RED,GREEN,BLACK,PURPLE,BROWN,TEAL,GRAY]
BLUES=LinearSegmentedColormap.from_list('db',['#FFFFFF','#9FB6D9','#2B4C8C','#00004D'])
plt.rcParams.update({'font.family':'serif','font.serif':['DejaVu Serif'],'font.size':9,'axes.labelsize':9,'axes.titlesize':10,'legend.fontsize':8,
 'xtick.labelsize':8,'ytick.labelsize':8,'axes.linewidth':0.8,'xtick.direction':'in','ytick.direction':'in','savefig.dpi':400})
def sh(n):
    s=M.shorten(n); return {'Original':'Hard-mask variant','Proposed revised':'Proposed'}.get(s,s)
def save(fig,name,leg=None):
    fig.savefig(OUT/name,dpi=400,bbox_inches='tight',facecolor='white',**({'bbox_extra_artists':(leg,)} if leg else {}));plt.close(fig)
def grid(ax,axis='both'): ax.grid(True,axis=axis,linestyle=':',linewidth=.5,color='0.7')
MK=['o','s','^','D','v','P','X','<','>','h','*','p','d','H','8']
def mcolor(name):
    s=sh(name)
    if s=='Proposed': return BLUE
    if s in ('No delay','Hard-mask variant','Known-delay oracle','No graph','No robustness','Fixed geography'): return GREEN
    if s in ('Kalman/RTS','Delay-aware SSM'): return PURPLE
    if s in ('Arrival interpolation','Timestamp oracle','Delay backprojection'): return RED
    return BROWN
# 1 trajectories
df=pd.read_csv(R/'data/processed/x_log1p.csv',index_col=0,parse_dates=True)
fig,ax=plt.subplots(figsize=(7.6,3.3))
for k,c in enumerate(df.columns): ax.plot(df.index,df[c],color=CYC[[0,1,2,3,4,5][k]],ls=['-','--','-.',':','-','--'][k],lw=1.0,marker=MK[k],markevery=28,ms=3,label=sh(c))
ax.set_xlabel('Date');ax.set_ylabel('log1p incidence per 100,000');grid(ax)
leg=ax.legend(ncol=6,loc='upper center',bbox_to_anchor=(.5,-.17),frameon=False);fig.tight_layout();save(fig,'fig_rki_trajectories.png',leg)
# 2 correlation
corr=df.corr();lab=[sh(c) for c in corr.columns]
fig,ax=plt.subplots(figsize=(5.0,4.3));im=ax.imshow(corr.values,cmap=BLUES,vmin=0.965,vmax=1.0)
ax.set_xticks(range(6));ax.set_xticklabels(lab);ax.set_yticks(range(6));ax.set_yticklabels(lab)
for i in range(6):
    for j in range(6): ax.text(j,i,f'{corr.iloc[i,j]:.3f}',ha='center',va='center',fontsize=8,color='white' if corr.iloc[i,j]>0.985 else 'black')
cb=fig.colorbar(im,ax=ax,shrink=.85);cb.set_label('Pearson correlation\n(display range 0.965–1.000)');fig.tight_layout();save(fig,'fig_correlation_heatmap.png')
# 3,7 benchmarks
for f,name,title in [(RR/'reference/retrospective_reconstruction_summary_long.csv','fig_reference_benchmark.png','Retrospective reconstruction on the locked test interval'),(RR/'causal/causal_nowcasting_summary_long.csv','fig_causal_benchmark.png','Causal rolling nowcasting on the locked test interval')]:
    s=pd.read_csv(f);rm=s[s.metric=='rmse'].sort_values('mean');y=np.arange(len(rm))
    fig,ax=plt.subplots(figsize=(7.2,0.27*len(rm)+1.2))
    for yi,(_,r) in zip(y,rm.iterrows()):
        c=mcolor(r.method);ax.errorbar(r['mean'],yi,xerr=[[r['mean']-r.ci95_low],[r.ci95_high-r['mean']]],fmt='o',color=c,ecolor=c,capsize=2.5,ms=4,lw=1)
    ax.set_yticks(y);ax.set_yticklabels([sh(m) for m in rm.method]);ax.invert_yaxis();ax.set_xlabel('RMSE (mean and bootstrap 95% CI)');ax.set_title(title);grid(ax,'x')
    for t in ax.get_yticklabels():
        if t.get_text()=='Proposed': t.set_color(BLUE);t.set_fontweight('bold')
    fig.tight_layout();save(fig,name)
# 4 runtime
s=pd.read_csv(RR/'reference/retrospective_reconstruction_summary_long.csv')
p=s.pivot_table(index='method',columns='metric',values='mean',aggfunc='first').reset_index().sort_values('rmse')
fig,axs=plt.subplots(1,2,figsize=(8.6,3.7));hs=[]
for ax,zoom in zip(axs,[0,1]):
    for k,r in enumerate(p.itertuples()):
        if zoom and r.runtime_seconds<1: continue
        c=mcolor(r.method);h=ax.scatter(r.runtime_seconds,r.rmse,marker=MK[k],s=44 if sh(r.method)=='Proposed' else 32,color=c,edgecolors=c,linewidths=.9,label=sh(r.method))
        if not zoom: hs.append(h)
    ax.set_xscale('log');ax.set_xlabel('Runtime per seed (s, log scale)');grid(ax)
axs[0].set_ylabel('Retrospective RMSE');axs[0].set_title('All methods');axs[1].set_title('Zoomed iterative-model cluster')
for ax,l in zip(axs,'ab'): ax.text(-0.12,1.04,f'({l})',transform=ax.transAxes,fontweight='bold')
leg=fig.legend(handles=hs,loc='lower center',ncol=5,frameon=False,bbox_to_anchor=(.5,-.02));fig.tight_layout(rect=[0,.18,1,1]);save(fig,'fig_runtime_accuracy.png',leg)
# 5 significance
sm=pd.read_csv(RR/'reference/retrospective_reconstruction_seed_metrics.csv');t=pd.read_csv(RR/'reference/retrospective_reconstruction_paired_tests.csv')
t=t[t.metric=='rmse'] if 'metric' in t else t
meth=sm.groupby('method').rmse.mean().sort_values().index.tolist();n=len(meth);mat=np.zeros((n,n))
for i,a in enumerate(meth):
    for j,b in enumerate(meth):
        if i==j: continue
        r=t[((t.method_a==a)&(t.method_b==b))|((t.method_a==b)&(t.method_b==a))]
        if len(r) and r.iloc[0].wilcoxon_p_holm<.05: mat[i,j]=1 if sm[sm.method==a].rmse.mean()<sm[sm.method==b].rmse.mean() else -1
fig,ax=plt.subplots(figsize=(8.2,7.4));cm=ListedColormap(['#C98A8A','#E6E6E6','#5E8F5E']);ax.imshow(mat,cmap=cm,norm=BoundaryNorm([-1.5,-.5,.5,1.5],3))
for i in range(n):
    for j in range(n): ax.text(j,i,'·' if i==j else {1:'+',-1:'−',0:'='}[int(mat[i,j])],ha='center',va='center',fontsize=10,fontweight='bold',color='black')
L=[sh(m) for m in meth];ax.set_xticks(range(n));ax.set_xticklabels(L,rotation=60,ha='right',fontsize=9);ax.set_yticks(range(n));ax.set_yticklabels(L,fontsize=9)
ax.set_title('Paired Wilcoxon signed-rank tests, Holm-adjusted (α = 0.05), row vs column')
ax.set_xlabel('+ row has significantly lower RMSE;  − row has significantly higher RMSE;  = no significant difference')
fig.tight_layout();save(fig,'fig_significance_matrix.png')
# 6 ranks
P=sm.pivot_table(index='seed',columns='method',values='rmse');rk=P.rank(axis=1);avg=rk.mean();rng=np.random.default_rng(2026)
B=np.array([rk.iloc[rng.integers(0,len(rk),len(rk))].mean().values for _ in range(5000)]);lo,hi=np.percentile(B,[2.5,97.5],axis=0)
d=pd.DataFrame({'m':avg.index,'a':avg.values,'lo':lo,'hi':hi}).sort_values('a');y=np.arange(len(d))
fig,ax=plt.subplots(figsize=(6.2,4.2))
for yi,r in zip(y,d.itertuples()): c=mcolor(r.m);ax.errorbar(r.a,yi,xerr=[[r.a-r.lo],[r.hi-r.a]],fmt='o',color=c,ecolor=c,capsize=3,ms=4.5,lw=1)
ax.set_yticks(y);ax.set_yticklabels([sh(m) for m in d.m]);ax.invert_yaxis();ax.set_xlabel('Average RMSE rank (bootstrap 95% CI; lower is better)');grid(ax,'x');fig.tight_layout();save(fig,'fig_average_ranks.png')
# 8-10 sensitivity
rs=pd.read_csv(RR/'robustness/robustness_seed_metrics.csv')
specs=[('missingness',['missing_00','missing_10','missing_30','missing_50'],[0,10,30,50],'Injected missingness (%)','fig_missingness_sensitivity.png'),
 ('delay',['delay_light','delay_moderate','delay_severe'],['Light (D=4)','Moderate (D=6)','Severe (D=8)'],'Delay severity','fig_delay_sensitivity.png'),
 ('outlier',['outlier_00','outlier_05','outlier_10','outlier_20','outlier_30'],[0,5,10,20,30],'Injected outliers among reports (%)','fig_outlier_sensitivity.png')]
for fam,order,xl,xlab,fn in specs:
    fig,ax=plt.subplots(figsize=(5.0,3.4));x=np.arange(len(order))
    for k,(m,c,mk,ls) in enumerate([('Proposed revised',BLUE,'o','-'),('No-delay ablation',GREEN,'s','--'),('No-robustness ablation',RED,'^','-.')]):
        v=[rs[(rs.scenario==sc)&(rs.method==m)].rmse for sc in order]
        if all(len(a)==0 for a in v): continue
        mu=np.array([a.mean() for a in v]);sd=np.array([a.std(ddof=1) for a in v])
        ax.errorbar(x+(k-1)*0.07,mu,yerr=sd,fmt=mk+ls,color=c,capsize=3,ms=4,lw=1.1,label=sh(m))
    ax.set_xticks(x);ax.set_xticklabels(xl);ax.set_xlabel(xlab);ax.set_ylabel('RMSE (mean ± SD)');grid(ax);ax.legend(frameon=False,fontsize=8,loc='upper left',ncol=3 if fam=='delay' else 1);ax.set_ylim(top=ax.get_ylim()[1]*(1.12 if fam=='delay' else 1));fig.tight_layout();save(fig,fn)
# 11-12 graph
files=sorted(glob.glob(str(RR/'reference/arrays/proposed_seed_*.npz')));ws=np.stack([np.load(f)['w_hat'] for f in files]);w=np.median(ws,axis=0)
nodes=[sh(s) for s in pd.read_csv(R/'data/processed/nodes.csv').state];th=json.loads(open(glob.glob(str(RR/'protocol/*.json'))[0]).read())['selected_graph_threshold']
freq=(ws>th).mean(axis=0)
fig,axs=plt.subplots(1,2,figsize=(9.2,4.0))
for ax,mtx,lab,fmt in [(axs[0],w,'Median learned weight','{:.2f}'),(axs[1],freq,'Edge-selection frequency','{:.2f}')]:
    im=ax.imshow(mtx,cmap=BLUES,vmin=0,vmax=max(mtx.max(),1e-8));ax.set_xticks(range(6));ax.set_xticklabels(nodes);ax.set_yticks(range(6));ax.set_yticklabels(nodes)
    ax.set_xlabel('Source node j');ax.set_ylabel('Receiving node i');cb=fig.colorbar(im,ax=ax,shrink=.8);cb.set_label(lab)
    for i in range(6):
        for j in range(6):
            if i!=j: ax.text(j,i,fmt.format(mtx[i,j]),ha='center',va='center',fontsize=7,color='white' if mtx[i,j]>0.6*mtx.max() else 'black')
axs[0].set_title('(a) Median weight across 20 seeds');axs[1].set_title('(b) Selection frequency across 20 seeds');fig.tight_layout();save(fig,'fig_graph_heatmap.png')
ang=np.linspace(0,2*np.pi,6,endpoint=False);pos=np.c_[np.cos(ang),np.sin(ang)]
fig,ax=plt.subplots(figsize=(5.2,4.8));ax.axis('off');mx=max(w.max(),1e-9)
for i in range(6):
    for j in range(6):
        if i==j or w[i,j]<=th: continue
        ax.add_patch(FancyArrowPatch(pos[j]*.86,pos[i]*.86,connectionstyle=f'arc3,rad={.16 if (i+j)%2==0 else -.16}',arrowstyle='-|>',mutation_scale=9,lw=.5+1.8*w[i,j]/mx,color=BLUE if freq[i,j]>=0.5 else '#7A8FBF'))
ax.scatter(pos[:,0],pos[:,1],s=650,facecolors='white',edgecolors=BLACK,linewidths=1.1,zorder=3)
for k,(x,y) in enumerate(pos): ax.text(x,y,nodes[k],ha='center',va='center',fontweight='bold',zorder=4)
ax.set_xlim(-1.35,1.35);ax.set_ylim(-1.3,1.3);ax.set_title('Median exploratory directed association graph across seeds')
ax.text(0,-1.27,'Line width: median weight; dark arrows: selected in ≥50% of seeds',ha='center',fontsize=8);fig.tight_layout();save(fig,'fig_learned_network.png')
# 13 graph recovery
g=pd.read_csv(R/'results/tables/graph_evaluation_table.csv');types=['dense','sparse','time_varying']
fig,axs=plt.subplots(1,3,figsize=(8.6,3.0))
for ax,(met,lab),l in zip(axs,[('rmse','State RMSE\n(lower is better)'),('graph_support_f1','Support F1\n(higher is better)'),('graph_edge_weight_r','Edge-weight correlation\n(higher is better)')],'abc'):
    for k,tp in enumerate(types):
        r=g[(g.metric==met)&(g.graph_type==tp)].iloc[0];ax.errorbar(k,r['mean'],yerr=[[r['mean']-r.ci95_low],[r.ci95_high-r['mean']]],fmt=MK[k],color=[BLUE,RED,GREEN][k],capsize=3,ms=5,lw=1.1)
    if met=='graph_edge_weight_r': ax.axhline(0,color=GRAY,lw=.7,ls=':')
    ax.set_xticks(range(3));ax.set_xticklabels(['Dense','Sparse','Time-varying']);ax.set_ylabel(lab);ax.set_title(f'({l})',loc='left',fontweight='bold');grid(ax,'y')
fig.tight_layout();save(fig,'fig_graph_recovery.png')
# 14 mismatch
mm=pd.read_csv(R/'results/tables/model_mismatch_table.csv');kinds=['abrupt_intervention','nonstationary_delays','nonlinear_dynamics','observation_model_mismatch']
fig,ax=plt.subplots(figsize=(7.0,3.6))
for k,(m,c,mk) in enumerate([('Proposed revised',BLUE,'o'),('No-delay ablation',GREEN,'s'),('Delay-aware state-space (internal)',PURPLE,'^')]):
    for q,kd in enumerate(kinds):
        r=mm[(mm.kind==kd)&(mm.method==m)&(mm.metric=='rmse')].iloc[0]
        ax.errorbar(q+(k-1)*0.18,r['mean'],yerr=[[r['mean']-r.ci95_low],[r.ci95_high-r['mean']]],fmt=mk,color=c,capsize=3,ms=5,lw=1.1,label=sh(m) if q==0 else None)
ax.set_xticks(range(4));ax.set_xticklabels(['Abrupt intervention','Nonstationary delays','Nonlinear dynamics','Observation-model\nmismatch']);ax.set_ylabel('RMSE (mean and bootstrap 95% CI)');grid(ax,'y')
leg=ax.legend(ncol=3,frameon=False,loc='upper center',bbox_to_anchor=(.5,-.16));fig.tight_layout();save(fig,'fig_model_mismatch.png',leg)
# 15 convergence
H=[np.load(f)['normalized_objective'] for f in files];m_=max(map(len,H));A=np.array([np.r_[h,[h[-1]]*(m_-len(h))] for h in H]);mu=A.mean(0);sd=A.std(0,ddof=1);x=np.arange(m_)
fig,ax=plt.subplots(figsize=(5.0,3.3));ax.fill_between(x,mu-sd,mu+sd,color='#9FB6D9',alpha=.6,lw=0,label='±1 SD');ax.plot(x,mu,'o-',color=BLUE,ms=3.5,lw=1.2,label='Mean')
ax.set_xlabel('Outer iteration');ax.set_ylabel('Normalized objective');grid(ax);ax.legend(frameon=False);fig.tight_layout();save(fig,'fig_convergence.png')
print('done')
