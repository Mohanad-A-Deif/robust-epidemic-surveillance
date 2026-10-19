from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[1]);HERE=_P(__file__).resolve().parent
import numpy as np,pandas as pd,glob
R=REPO
X=pd.read_csv(R+'/data/processed/x_log1p.csv',index_col=0);dates=pd.to_datetime(X.index);truth=X.to_numpy(float).T
sp=pd.read_csv(R+'/data/processed/splits.csv');test=sp.time_index[sp.split=='test'].to_numpy()
print('test',dates[test[0]].date(),dates[test[-1]].date())
B=dict(np.load('baseline_xhat.npz'))
B['Proposed']=np.stack([np.load(f)['x_hat'] for f in sorted(glob.glob(R+'/results/raw/reference/arrays/proposed_seed_*.npz'))])
def inc7(x):  # x: (...,6,T) log1p daily incidence -> 7-day incidence per 100k ending at t
    d=np.expm1(np.clip(x,0,None)); c=np.cumsum(d,axis=-1); out=c.copy(); out[...,7:]=c[...,7:]-c[...,:-7]; return out
T7=inc7(truth)[:,test]
print('true 7d inc range on test', T7.min().round(1), np.percentile(T7,[25,50,75]).round(1), T7.max().round(1))
T7prev=inc7(truth)[:,test-7]; trend_true=np.sign(T7-T7prev)
rows=[]
for m,A in B.items():
    rm=np.sqrt(((A[:,:,test]-truth[None,:,test])**2).mean(axis=(1,2)))
    E=inc7(A); E7=E[:,:,test]; E7p=E[:,:,test-7]
    ape=np.abs(E7-T7)/T7
    tr=np.sign(E7-E7p)
    r={'method':m,'rmse_log':rm.mean(),'MdAPE_7d_%':100*np.median(ape),'MAPE_7d_%':100*ape.mean(),
       'within10%_%':100*(ape<=0.10).mean(),'trend_agree_%':100*(tr==trend_true).mean()}
    for th in [50,100,200]:
        r[f'miscls_{th}_%']=100*((E7>=th)!=(T7>=th)).mean()
    # peak date error (days) over test interval per state
    pk=np.abs(np.argmax(E7,axis=2)-np.argmax(T7,axis=1)[None]); r['peak_err_days']=pk.mean()
    # bias of 7-day burden
    r['bias_7d_%']=100*np.median((E7-T7)/T7)
    rows.append(r)
df=pd.DataFrame(rows).set_index('method');pd.set_option('display.width',250);print(df.round(2).to_string())
df.to_csv('decision_metrics.csv')
for th in [50,100,200]: print(th,'true share above',(T7>=th).mean().round(3))
print('---- tails & calibration')
val=sp.time_index[sp.split=='validation'].to_numpy()
Tv=inc7(truth)[:,val]
rows=[]
for m,A in B.items():
    E=inc7(A);E7=E[:,:,test];ape=np.abs(E7-T7)/T7
    ratio=np.median(Tv[None]/np.maximum(E[:,:,val],1e-9))
    Ec=E7*ratio; apec=np.abs(Ec-T7)/T7
    rows.append({'method':m,'P90_APE':100*np.percentile(ape,90),'P95_APE':100*np.percentile(ape,95),'gross>50%':100*(ape>0.5).mean(),'gross>100%':100*(ape>1).mean(),
      'val_ratio':ratio,'cal_MdAPE':100*np.median(apec),'cal_bias':100*np.median((Ec-T7)/T7),'cal_mis100':100*((Ec>=100)!=(T7>=100)).mean(),'cal_mis50':100*((Ec>=50)!=(T7>=50)).mean(),'cal_P95':100*np.percentile(apec,95)})
print(pd.DataFrame(rows).set_index('method').round(2).to_string())
