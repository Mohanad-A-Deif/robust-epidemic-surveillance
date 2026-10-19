from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[1]);HERE=_P(__file__).resolve().parent
import numpy as np,pandas as pd,glob,matplotlib
matplotlib.use('Agg');import matplotlib.pyplot as plt, matplotlib.dates as mdates
exec(open('decision.py').read().split("rows=[]")[0])
val=sp.time_index[sp.split=='validation'].to_numpy();Tv=inc7(truth)[:,val]
def cal(E): return np.median(Tv[None]/np.maximum(E[:,:,val],1e-9),axis=2)[:,:,None]
order=[('Arrival interpolation',0,'Arrival-aligned interpolation'),('Delay backprojection',0,'Delay backprojection'),('Kalman/RTS smoother (internal)',0,'Kalman/RTS smoother'),('No-robustness ablation',0,'No-robustness ablation'),('Robust median smoother',0,'Robust median smoother'),('No-delay ablation',0,'No-delay ablation'),('Proposed',0,'Proposed method'),('Robust median smoother',1,'Robust median smoother + calibration'),('No-delay ablation',1,'No-delay ablation + calibration'),('Proposed',1,'Proposed method + calibration')]
rows=[]
for key,c,lab in order:
    E=inc7(B[key]);f=cal(E) if c else 1.0
    E7=E[:,:,test]*f;E7p=E[:,:,test-7]*f;ape=np.abs(E7-T7)/T7
    seedv={'MdAPE':[100*np.median(ape[k]) for k in range(ape.shape[0])],'Gross50':[100*(ape[k]>0.5).mean() for k in range(ape.shape[0])],
      'Trend':[100*(np.sign(E7[k]-E7p[k])==trend_true).mean() for k in range(ape.shape[0])],'Peak':[np.abs(np.argmax(E7[k],axis=1)-np.argmax(T7,axis=1)).mean() for k in range(ape.shape[0])],
      'Mis100':[100*((E7[k]>=100)!=(T7>=100)).mean() for k in range(ape.shape[0])]}
    sdrow={k+'_sd':np.std(v,ddof=1) for k,v in seedv.items()}
    rows.append({**sdrow,'Method':lab,'MdAPE':100*np.median(ape),'Bias':100*np.median((E7-T7)/T7),'P95':100*np.percentile(ape,95),'Gross50':100*(ape>0.5).mean(),
      'Trend':100*(np.sign(E7-E7p)==trend_true).mean(),'Peak':np.abs(np.argmax(E7,axis=2)-np.argmax(T7,axis=1)[None]).mean(),'Mis100':100*((E7>=100)!=(T7>=100)).mean()})
df=pd.DataFrame(rows);pd.set_option('display.width',250);print(df.round(1).to_string());df.to_csv('decision_table.csv',index=False)
P=B['Proposed'];fP=cal(inc7(P))
rm=np.sqrt(((P[:,:,test]-truth[None,:,test])**2).mean(axis=(1,2)));s=int(np.argsort(rm)[len(rm)//2])
names=list(X.columns);d=dates[test]
plt.rcParams.update({'font.family':'serif','font.size':10})
fig,axs=plt.subplots(1,2,figsize=(11,5.2))
for ax,st in zip(axs,['Bayern','Nordrhein-Westfalen']):
    i=names.index(st)
    ax.plot(d,T7[i],color='black',lw=2.2,label='Consolidated reference')
    YL=650
    for key,col,ls,lab in [('Arrival interpolation','#8B0000','--','Arrival-aligned interpolation')]:
        yv=inc7(B[key][s])[i,test];ax.plot(d,np.minimum(yv,YL),color=col,lw=1.1,ls=ls,label=lab)
        over=yv>YL
        if False:
            idx=np.where(over)[0];groups=np.split(idx,np.where(np.diff(idx)>1)[0]+1)
            for gp in groups:
                kmax=gp[np.argmax(yv[gp])];ax.plot(d[kmax],YL,marker='^',color=col,ms=7,clip_on=False,zorder=5)
                ax.annotate(f'peak {yv[kmax]:.0f}',(d[kmax],YL),xytext=(6,-12),textcoords='offset points',fontsize=8,color=col)
    ax.plot(d,inc7(B['Robust median smoother'][s])[i,test],color='#006400',lw=1.2,ls='-.',label='Robust median smoother')
    ax.plot(d,inc7(P[s])[i,test],color='#00008B',lw=1.0,ls=':',label='Proposed method (uncalibrated)')
    ax.plot(d,fP[s,i,0]*inc7(P[s])[i,test],color='#00008B',lw=1.8,label='Proposed method + calibration')
    ax.axhline(100,color='gray',lw=0.8,ls=(0,(1,1)))
    ax.set_title(st);ax.set_ylabel('7-day incidence per 100,000');ax.xaxis.set_major_formatter(mdates.DateFormatter('%d %b'))
    ax.grid(alpha=0.25);ax.set_ylim(0,YL);ax.set_yticks(range(0,651,50))
h,l=axs[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,fontsize=9)
fig.tight_layout(rect=[0,0.13,1,1]);fig.savefig('fig_decision_case.png',dpi=300)
print('seed',1101+s)
