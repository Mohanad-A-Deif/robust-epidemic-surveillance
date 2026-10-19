from pathlib import Path as _P
REPO=str(_P(__file__).resolve().parents[1]);HERE=_P(__file__).resolve().parent
import sys,json,time
from pathlib import Path
sys.path.insert(0,REPO+'/code/model_pipeline')
import numpy as np,pandas as pd,final_study as fs
from final_study import build_method, read_messages, config_from_jsonable
from concurrent.futures import ProcessPoolExecutor
root=_P(REPO);R=Path(REPO)
b=fs.load_processed_bundle(root/'data/processed',root/'data/metadata');conf=json.loads((_P(REPO)/'configs'/'data_config.json').read_text())
sel=json.loads(sorted((R/'results/raw/protocol').glob('*'))[0].read_text())['selected_config']
sc=fs.scenario_lookup(conf)['reference_moderate'];pm=fs.delay_pmfs(sc,conf)
files=sorted((root/'data/scenarios/main/reference_moderate').glob('messages_seed_*.csv.gz'))
methods=["No-delay ablation","Arrival interpolation","Delay backprojection","Kalman/RTS smoother (internal)","Robust median smoother","No-robustness ablation"]
truth=b.x_log1p.to_numpy(float).T
def run(t):
    f,m=t
    cfg=config_from_jsonable(sel); est=build_method(m,cfg,np.asarray(b.adjacency,float),quick=False)
    r=est.fit(messages=read_messages(f),n_nodes=6,n_time=truth.shape[1],max_delay=int(sc['max_delay']),delay_pmf_by_slice=pm)
    return m,int(Path(f).stem.split('_')[-1].split('.')[0]),r.x_hat
tasks=[(str(f),m) for f in files for m in methods]
t=time.time()
with ProcessPoolExecutor() as ex: res=list(ex.map(run,tasks))
out={}
for m,s,x in res: out.setdefault(m,{})[s]=x
np.savez('baseline_xhat.npz',**{m:np.stack([out[m][s] for s in sorted(out[m])]) for m in out})
print('elapsed',time.time()-t)
