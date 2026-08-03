#!/usr/bin/env python3
"""Run one causal-nowcasting seed for debugging without machine-specific paths."""
from __future__ import annotations
import argparse, json, time
from pathlib import Path
import numpy as np
import final_study as fs

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--repo-root',type=Path,default=REPO)
    p.add_argument('--seed',type=int,default=1101)
    p.add_argument('--method',default='Proposed revised')
    args=p.parse_args(); root=args.repo_root
    b=fs.load_processed_bundle(root/'data/processed',root/'data/metadata')
    conf=json.loads((root/'configs/data_config.json').read_text(encoding='utf-8'))
    sc=fs.scenario_lookup(conf)['reference_moderate']; pm=fs.delay_pmfs(sc,conf)
    selected,_,protocol=fs.load_selected(root/'results/raw')
    split=fs.split_indices(b.splits,b.x_log1p.index); truth=b.x_log1p.to_numpy(float).T
    file=root/'data/scenarios/main/reference_moderate'/f'messages_seed_{args.seed}.csv.gz'
    stop=protocol['causal_rolling_stopping_protocol']
    task={'message_file':str(file),'seed':args.seed,'method':args.method,
      'selected_config':fs.config_to_jsonable(selected),'adjacency':np.asarray(b.adjacency).tolist(),
      'truth':truth,'test_indices':split['test'],'max_delay':int(sc['max_delay']),
      'delay_pmfs':{k:v.tolist() for k,v in pm.items()},
      'first_outer':stop['first_test_day_outer_iterations'],
      'subsequent_outer':stop['subsequent_day_outer_iterations'],'quick':False}
    start=time.perf_counter(); result=fs._causal_task(task)
    print('elapsed',time.perf_counter()-start); print(result['row'])

if __name__=='__main__': main()
