#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np
import pandas as pd
import final_study as fs
from epidemic_results.io_utils import load_processed_bundle
from epidemic_results.study_protocol import split_indices, config_to_jsonable

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def parse_args():
    p=argparse.ArgumentParser()
    p.add_argument('--rki-root',type=Path,default=REPO)
    p.add_argument('--reference-results',type=Path,default=REPO/'results/raw')
    p.add_argument('--output',type=Path,default=REPO/'outputs/recomputed_robustness')
    p.add_argument('--workers',type=int,default=4)
    return p.parse_args()

def main():
    args=parse_args(); rki=args.rki_root; reference=args.reference_results; out=args.output
    out.mkdir(parents=True,exist_ok=True)
    bundle=load_processed_bundle(rki/'data/processed',rki/'data/metadata')
    conf=json.loads((rki/'configs/data_config.json').read_text(encoding='utf-8'))
    lookup=fs.scenario_lookup(conf)
    selected,threshold,_=fs.load_selected(reference)
    splits=split_indices(bundle.splits,bundle.x_log1p.index)
    truth=bundle.x_log1p.to_numpy(float).T
    files=sorted((rki/'data/scenarios').glob('**/messages_seed_*.csv.gz'))
    methods=['No-delay ablation','No-robustness ablation']
    tasks=[]
    for file in files:
        head=pd.read_csv(file,nrows=1); sc=str(head.loc[0,'scenario']); fam=str(head.loc[0,'family']); seed=int(head.loc[0,'seed'])
        spec=lookup[sc]; pmfs=fs.delay_pmfs(spec,conf)
        for method in methods:
            tasks.append({'message_file':str(file),'scenario':sc,'family':fam,'seed':seed,'method':method,
              'selected_config':config_to_jsonable(selected),'adjacency':np.asarray(bundle.adjacency).tolist(),
              'truth':truth,'test_indices':splits['test'],'max_delay':int(spec['max_delay']),
              'delay_pmfs':{k:v.tolist() for k,v in pmfs.items()},'graph_threshold':threshold,'quick':False})
    res=fs.run_parallel(tasks,fs._fit_retro_task,args.workers,'robustness-ablations')
    rows=pd.DataFrame([x['row'] for x in res]).sort_values(['family','scenario','method','seed'])
    fs.save_table(rows,out/'robustness_seed_metrics','Robustness ablations on fixed nested templates.','tab:robustness_seed')
    fs.summarize_and_stats(rows,out,'robustness')

if __name__=='__main__': main()
