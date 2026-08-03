#!/usr/bin/env python3
from __future__ import annotations
import argparse, json
from dataclasses import replace
from pathlib import Path
import pandas as pd
import final_study as fs
import synthetic_study as ss
from epidemic_results.study_protocol import config_to_jsonable

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[1]

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--reference-results',type=Path,default=REPO/'results/raw')
    p.add_argument('--synthetic-results',type=Path,default=REPO/'results/raw/synthetic')
    p.add_argument('--output',type=Path,default=REPO/'outputs/recomputed_scalability')
    p.add_argument('--workers',type=int,default=4)
    args=p.parse_args()
    base,_,_=fs.load_selected(args.reference_results)
    proto=json.loads((args.synthetic_results/'calibration/synthetic_protocol.json').read_text(encoding='utf-8'))
    cfg=ss.synthetic_config(base,proto['selected_lambda_w'],proto['selected_lambda_g'],False)
    scfg=replace(cfg,max_outer_iter=5,max_x_iter=2,max_w_iter=1,warmup_outer_iter=1,graph_stage_outer_iter=1)
    basevals={'n':8,'t':160,'d':6,'density':.18}
    levels={'n_nodes':[6,12,24],'n_time':[80,160,320],'max_delay':[2,6,10],'graph_density':[.10,.30,.50]}
    tasks=[]
    for factor,vals in levels.items():
        for level in vals:
            params=basevals.copy(); key={'n_nodes':'n','n_time':'t','max_delay':'d','graph_density':'density'}[factor]; params[key]=level
            for seed in range(7001,7004): tasks.append({'factor':factor,'level':level,'seed':seed,**params,'config':config_to_jsonable(scfg)})
    rows=pd.DataFrame(ss.parallel(tasks,ss.scalability_task,args.workers,'scalability'))
    args.output.mkdir(parents=True,exist_ok=True)
    fs.save_table(rows,args.output/'scalability_seed_metrics','One-factor-at-a-time scalability experiments.','tab:scale_seed')
    summary=rows.groupby(['factor','level'],as_index=False).agg(runtime_median=('runtime_seconds','median'),runtime_mean=('runtime_seconds','mean'),memory_mean_mb=('reported_peak_memory_mb','mean'),rmse_mean=('rmse','mean'))
    fs.save_table(summary,args.output/'scalability_runtime_memory','Runtime and memory scaling.','tab:scale')

if __name__=='__main__': main()
