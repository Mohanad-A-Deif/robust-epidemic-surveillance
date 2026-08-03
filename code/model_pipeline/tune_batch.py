#!/usr/bin/env python3
import argparse,json,time
from pathlib import Path
import numpy as np,pandas as pd
import final_study as fs
p=argparse.ArgumentParser();p.add_argument('--rki-root',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--start',type=int,required=True);p.add_argument('--end',type=int,required=True);a=p.parse_args()
a.output.mkdir(parents=True,exist_ok=True);(a.output/'graphs').mkdir(exist_ok=True)
b=fs.load_processed_bundle(a.rki_root/'data/processed',a.rki_root/'data/metadata');conf=json.loads((a.rki_root/'config.json').read_text());sp=fs.split_indices(b.splits,b.x_log1p.index);ve=int(sp['validation'].max());truth=b.x_log1p.to_numpy(float).T[:,:ve+1];sc=fs.scenario_lookup(conf)['reference_moderate'];pm=fs.delay_pmfs(sc,conf);files=sorted((a.rki_root/'data/scenarios/main/reference_moderate').glob('messages_seed_*.csv.gz'))[:5];cands=fs.candidate_configs(np.asarray(b.adjacency,float),False)[a.start:a.end]
rows=[];started=time.perf_counter()
for name,cfg in cands:
 for file in files:
  seed=int(file.stem.split('_')[-1].split('.')[0]); task={'candidate':name,'config':fs.config_to_jsonable(cfg),'message_file':str(file),'seed':seed,'truth':truth,'validation_indices':sp['validation'],'validation_end':ve,'max_delay':int(sc['max_delay']),'delay_pmfs':{k:v.tolist() for k,v in pm.items()}}
  r=fs._fit_tuning_task(task);np.save(a.output/'graphs'/f'{name}_{seed}.npy',r.pop('w_hat'));rows.append(r)
 print(name,flush=True)
pd.DataFrame(rows).to_csv(a.output/f'tuning_batch_{a.start}_{a.end}.csv',index=False)
print('elapsed',time.perf_counter()-started)
