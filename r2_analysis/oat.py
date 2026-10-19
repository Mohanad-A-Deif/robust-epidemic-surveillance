from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[1]);HERE=_P(__file__).resolve().parent
import sys,json,time
from pathlib import Path
sys.path.insert(0,REPO+'/code/model_pipeline')
import numpy as np,pandas as pd
import final_study as fs
from concurrent.futures import ProcessPoolExecutor
root=_P(REPO)
b=fs.load_processed_bundle(root/'data/processed',root/'data/metadata');conf=json.loads((_P(REPO)/'configs'/'data_config.json').read_text())
sp=fs.split_indices(b.splits,b.x_log1p.index);ve=int(sp['validation'].max());truth=b.x_log1p.to_numpy(float).T[:,:ve+1]
sc=fs.scenario_lookup(conf)['reference_moderate'];pm=fs.delay_pmfs(sc,conf)
files=sorted((root/'data/scenarios/main/reference_moderate').glob('messages_seed_*.csv.gz'))[:5]
cands=dict(fs.candidate_configs(np.asarray(b.adjacency,float),False))
import dataclasses
base=dataclasses.replace(cands['C01_default'],sigma_w=0.25)
specs=[('base',{})]
for k,lo,hi in [('lambda_g',0.005,0.03),('lambda_w',0.005,0.03),('lambda_row',0.003,0.02),('geo_prior_weight',0.003,0.02)]:
    specs+= [(f'{k}_low',{k:lo}),(f'{k}_high',{k:hi})]
specs+=[('bundle_low',{'lambda_g':0.005,'lambda_w':0.005,'lambda_row':0.003,'geo_prior_weight':0.003}),('bundle_high',{'lambda_g':0.03,'lambda_w':0.03,'lambda_row':0.02,'geo_prior_weight':0.02})]
tasks=[]
for name,upd in specs:
    cfg=dataclasses.replace(base,**upd)
    for f in files:
        seed=int(f.stem.split('_')[-1].split('.')[0])
        tasks.append({'candidate':name,'config':fs.config_to_jsonable(cfg),'message_file':str(f),'seed':seed,'truth':truth,'validation_indices':sp['validation'],'validation_end':ve,'max_delay':int(sc['max_delay']),'delay_pmfs':{k:v.tolist() for k,v in pm.items()}})
t=time.time()
with ProcessPoolExecutor() as ex: res=list(ex.map(fs._fit_tuning_task,tasks))
rows=[];graphs={}
for r in res:
    w=r.pop('w_hat');graphs.setdefault(r['candidate'],[]).append(w);rows.append(r)
df=pd.DataFrame(rows);df.to_csv('oat_seed_metrics.csv',index=False)
np.savez('oat_graphs.npz',**{k:np.stack(v) for k,v in graphs.items()})
print(df.groupby('candidate',sort=False)[['rmse','mae']].agg(['mean','std']).round(4))
print('elapsed',time.time()-t)
