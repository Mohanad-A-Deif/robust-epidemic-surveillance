#!/usr/bin/env python3
"""Synthetic graph-recovery, model-mismatch, and scalability experiments."""
from __future__ import annotations

import argparse
import json
import math
import os
import time
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import final_study as fs
from epidemic_results.metrics import graph_metrics, state_metrics, graph_stability
from epidemic_results.proposed_model import DelayAwareRobustGraphInference, InferenceConfig
from epidemic_results.statistics import summarize_seed_results, pairwise_method_tests, average_ranks, friedman_test
from epidemic_results.study_protocol import build_method, config_to_jsonable, config_from_jsonable


def parse_args():
    p=argparse.ArgumentParser();p.add_argument('--reference-output',type=Path,required=True);p.add_argument('--output',type=Path,required=True);p.add_argument('--workers',type=int,default=6);p.add_argument('--quick',action='store_true');return p.parse_args()


def stable_graph(n:int,density:float,seed:int)->np.ndarray:
    rng=np.random.default_rng(seed);mask=rng.random((n,n))<density;np.fill_diagonal(mask,False)
    for i in range(n):
        if not mask[i].any(): mask[i,rng.choice([j for j in range(n) if j!=i])]=True
    w=np.zeros((n,n));w[mask]=rng.uniform(.05,.35,mask.sum())
    sums=w.sum(1)
    for i,s in enumerate(sums):
        if s>2.2:w[i]*=2.2/s
    return w


def simulate(w1:np.ndarray,t_count:int,seed:int,kind:str='linear',w2:np.ndarray|None=None)->np.ndarray:
    rng=np.random.default_rng(seed);n=w1.shape[0];x=np.zeros((n,t_count));x[:,0]=rng.uniform(.2,.8,n);beta=.12;gamma=.08
    for t in range(t_count-1):
        w=w2 if (w2 is not None and t>=t_count//2) else w1
        degree=np.diag(w.sum(1));a=(1-gamma)*np.eye(n)-beta*(degree-w)
        drift=.025*(1+np.sin(2*np.pi*t/21+np.arange(n)))
        nxt=a@x[:,t]+drift+rng.normal(0,.02,n)
        if kind=='nonlinear': nxt+=.06*x[:,t]*(1-x[:,t]/3.0)
        if kind=='abrupt_intervention' and t>=t_count//2: nxt*=.72
        x[:,t+1]=np.maximum(nxt,.005)
    for node,start,amp in [(1,t_count//4,.8),(n-2,2*t_count//3,.6)]:
        width=min(14,t_count-start);pulse=amp*np.sin(np.linspace(0,np.pi,width));x[node,start:start+width]+=pulse
    return x


def make_messages(x:np.ndarray,seed:int,max_delay:int=6,drop:float=.1,outlier:float=.05,kind:str='standard')->pd.DataFrame:
    ss=np.random.SeedSequence(seed);r_delay,r_drop,r_out,r_mag,r_obs=[np.random.default_rng(s) for s in ss.spawn(5)]
    n,t_count=x.shape
    base=np.array([.30,.25,.18,.12,.08,.05,.02],dtype=float)
    if max_delay+1 > len(base):
        extra=max_delay+1-len(base)
        base=np.r_[base, base[-1]*(0.60**np.arange(1,extra+1))]
    pmf1=base[:max_delay+1];pmf1/=pmf1.sum()
    pmf2=np.arange(1,max_delay+2,dtype=float);pmf2/=pmf2.sum()
    scale=max(float(np.std(x)),.05);rows=[];counter=0
    for i in range(n):
        for t in range(t_count):
            counter+=1;pmf=pmf2 if kind=='nonstationary_delay' and t>=t_count//2 else pmf1
            d=int(r_delay.choice(np.arange(max_delay+1),p=pmf));dropped=bool(r_drop.random()<drop);arrival=t+d;right=bool((not dropped) and arrival>=t_count);received=not dropped and not right
            is_out=bool(received and r_out.random()<outlier);addition=float(r_mag.standard_t(3)*5*scale) if is_out else 0.0
            value=None
            if received:
                if kind=='observation_mismatch': value=float(max(x[i,t]*np.exp(r_obs.normal(0,.25))+addition,0.0))
                else:value=float(x[i,t]+addition)
            rows.append({'message_id':f's{seed}_m{counter}','scenario':kind,'family':'synthetic','seed':seed,'node_id':i,'slice_id':'default','generation_index':t,'clean_value':float(x[i,t]),'delay':d,'arrival_index':arrival if received else '', 'received':received,'dropped':dropped,'right_censored':right,'is_outlier':is_out,'observed_value':value if received else ''})
    return pd.DataFrame(rows)


def pmfs(max_delay:int=6)->dict[str,np.ndarray]:
    base=np.array([.30,.25,.18,.12,.08,.05,.02],dtype=float)
    if max_delay+1 > len(base):
        extra=max_delay+1-len(base)
        base=np.r_[base, base[-1]*(0.60**np.arange(1,extra+1))]
    p=base[:max_delay+1];p/=p.sum();return {'default':p,'standard':p,'critical':p}


def synthetic_config(base:InferenceConfig,lambda_w:float|None=None,lambda_g:float|None=None,quick:bool=False)->InferenceConfig:
    return replace(base,geographic_reference=None,candidate_mask=None,geo_prior_weight=0.0,lambda_w=base.lambda_w if lambda_w is None else lambda_w,lambda_g=base.lambda_g if lambda_g is None else lambda_g,max_outer_iter=6 if quick else 10,max_x_iter=2 if quick else 3,max_w_iter=1 if quick else 2,warmup_outer_iter=2,graph_stage_outer_iter=1)


def calibrate(base:InferenceConfig,out:Path,quick:bool)->tuple[InferenceConfig,float,pd.DataFrame]:
    n=8;t_count=120;w=stable_graph(n,.18,9001);lambdas=[(.003,.005),(.008,.01),(.015,.015),(.03,.02),(.06,.03)]
    if quick:lambdas=lambdas[:3]
    rows=[];graphs={}
    for lw,lg in lambdas:
        key=f'lw{lw}_lg{lg}';graphs[key]=[]
        for seed in range(5001,5004 if quick else 5006):
            x=simulate(w,t_count,seed);m=make_messages(x,seed);cfg=synthetic_config(base,lw,lg,quick);r=DelayAwareRobustGraphInference(cfg).fit(m,n,t_count,6,pmfs());graphs[key].append(r.w_hat)
            row={'candidate':key,'seed':seed,'lambda_w':lw,'lambda_g':lg,**state_metrics(x[:,72:96],r.x_hat[:,72:96])}
            # threshold-free graph criterion
            scores=graph_metrics(w,r.w_hat,threshold=1e-8);row['graph_auprc']=scores['graph_auprc'];row['graph_auroc']=scores['graph_auroc'];rows.append(row)
    frame=pd.DataFrame(rows);summary=frame.groupby(['candidate','lambda_w','lambda_g'],as_index=False).agg(auprc_mean=('graph_auprc','mean'),rmse_mean=('rmse','mean')).sort_values(['auprc_mean','rmse_mean'],ascending=[False,True]);best=str(summary.iloc[0].candidate);lw=float(summary.iloc[0].lambda_w);lg=float(summary.iloc[0].lambda_g)
    th_rows=[]
    for th in np.unique(np.r_[1e-5,np.logspace(-4,-.5,24)]):
        f1=[]
        for g in graphs[best]:f1.append(graph_metrics(w,g,float(th))['graph_support_f1'])
        th_rows.append({'threshold':th,'mean_f1':np.mean(f1),'sd_f1':np.std(f1,ddof=1)})
    thf=pd.DataFrame(th_rows);threshold=float(thf.sort_values(['mean_f1','threshold'],ascending=[False,True]).iloc[0].threshold)
    fs.save_table(frame,out/'calibration'/'synthetic_validation_seed_metrics','Synthetic validation calibration; test seeds are disjoint.','tab:syn_cal_seed');fs.save_table(summary,out/'calibration'/'synthetic_validation_summary','Synthetic validation selection of graph sparsity weights.','tab:syn_cal');fs.save_table(thf,out/'calibration'/'synthetic_graph_threshold','Graph support threshold selected only on synthetic validation seeds.','tab:syn_threshold')
    cfg=synthetic_config(base,lw,lg,quick);fs.write_json({'selected_lambda_w':lw,'selected_lambda_g':lg,'selected_threshold':threshold,'validation_seeds':list(range(5001,5004 if quick else 5006)),'test_seeds':list(range(6001,6004 if quick else 6021))},out/'calibration'/'synthetic_protocol.json');return cfg,threshold,frame


def recovery_task(task:dict[str,Any])->dict[str,Any]:
    cfg=config_from_jsonable(task['config']);n=task['n'];t_count=task['t'];kind=task['graph_type'];seed=task['seed'];w1=np.asarray(task['w1']);w2=np.asarray(task['w2']) if task.get('w2') is not None else None
    x=simulate(w1,t_count,seed,w2=w2);m=make_messages(x,seed)
    r=DelayAwareRobustGraphInference(cfg).fit(m,n,t_count,6,pmfs())
    target=w1 if w2 is None else .5*(w1+w2)
    return {'row':{'graph_type':kind,'seed':seed,**state_metrics(x[:,int(.8*t_count):],r.x_hat[:,int(.8*t_count):]),**graph_metrics(target,r.w_hat,task['threshold']),'runtime_seconds':r.runtime_seconds,'maximum_row_sum':float(r.w_hat.sum(1).max())},'w':r.w_hat,'target':target}


def mismatch_task(task:dict[str,Any])->dict[str,Any]:
    cfg=config_from_jsonable(task['config']);n=8;t_count=120;w=stable_graph(n,.18,9001);kind=task['kind'];seed=task['seed'];dyn='nonlinear' if kind=='nonlinear_dynamics' else ('abrupt_intervention' if kind=='abrupt_intervention' else 'linear');x=simulate(w,t_count,seed,dyn);message_kind={'nonstationary_delays':'nonstationary_delay','observation_model_mismatch':'observation_mismatch'}.get(kind,'standard');m=make_messages(x,seed,kind=message_kind)
    method=task['method'];est=build_method(method,cfg,np.zeros((n,n)),quick=task.get('quick',False));r=est.fit(messages=m,n_nodes=n,n_time=t_count,max_delay=6,delay_pmf_by_slice=pmfs());return {'kind':kind,'seed':seed,'method':method,**state_metrics(x[:,96:],r.x_hat[:,96:]),'runtime_seconds':r.runtime_seconds}


def scalability_task(task:dict[str,Any])->dict[str,Any]:
    cfg=config_from_jsonable(task['config']);n=task['n'];t=task['t'];d=task['d'];density=task['density'];seed=task['seed'];w=stable_graph(n,density,10000+seed+n+t+d);x=simulate(w,t,seed);m=make_messages(x,seed,max_delay=d);start=time.perf_counter();r=DelayAwareRobustGraphInference(cfg).fit(m,n,t,d,pmfs(d));return {'factor':task['factor'],'level':task['level'],'seed':seed,'n_nodes':n,'n_time':t,'max_delay':d,'graph_density_true':density,'runtime_seconds':time.perf_counter()-start,'reported_peak_memory_mb':r.peak_memory_mb,'iterations':r.iterations,'rmse':state_metrics(x[:,int(.8*t):],r.x_hat[:,int(.8*t):])['rmse']}


def parallel(tasks,fn,workers,label):
    out=[];done=0;started=time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers,mp_context=mp.get_context('spawn')) as ex:
        fut=[ex.submit(fn,t) for t in tasks]
        for f in as_completed(fut):out.append(f.result());done+=1
        
    print(label,len(out),'elapsed',time.perf_counter()-started,flush=True);return out


def main():
    a=parse_args();a.output.mkdir(parents=True,exist_ok=True);base,_,_=fs.load_selected(a.reference_output);cfg,threshold,_=calibrate(base,a.output,a.quick)
    seeds=list(range(6001,6004 if a.quick else 6021));n=8;t=120;graphs={'sparse':(stable_graph(n,.18,9101),None),'dense':(stable_graph(n,.48,9102),None),'time_varying':(stable_graph(n,.18,9103),stable_graph(n,.18,9104))}
    tasks=[]
    for gt,(w1,w2) in graphs.items():
        for seed in seeds:tasks.append({'graph_type':gt,'seed':seed,'n':n,'t':t,'w1':w1,'w2':w2,'config':config_to_jsonable(cfg),'threshold':threshold})
    res=parallel(tasks,recovery_task,a.workers,'recovery');rows=pd.DataFrame([r['row'] for r in res]);fs.save_table(rows,a.output/'graph_recovery'/'graph_recovery_seed_metrics','Directed graph recovery without a true-support candidate mask.','tab:graph_recovery_seed');summary=summarize_seed_results(rows,['graph_support_precision','graph_support_recall','graph_support_f1','graph_auroc','graph_auprc','graph_normalized_frobenius','graph_structural_hamming','graph_edge_weight_r','rmse','runtime_seconds'],group_cols=['graph_type']);fs.save_table(summary,a.output/'graph_recovery'/'graph_recovery_summary','Graph recovery over independent seeds.','tab:graph_recovery')
    edge=[]
    for gt in graphs:
        subset=[r for r in res if r['row']['graph_type']==gt];stack=np.stack([r['w'] for r in subset]);target=subset[0]['target']
        for i in range(n):
            for j in range(n):
                if i!=j:edge.append({'graph_type':gt,'source':i,'target':j,'true_weight':target[i,j],'selection_frequency':float(np.mean(stack[:,i,j]>threshold)),'mean_weight':float(stack[:,i,j].mean()),'sd_weight':float(stack[:,i,j].std(ddof=1))})
        fs.write_json(graph_stability([r['w'] for r in subset],threshold),a.output/'graph_recovery'/f'{gt}_stability.json')
    fs.save_table(pd.DataFrame(edge),a.output/'graph_recovery'/'edge_selection_frequency','Edge selection frequency across seeds for fixed true graph structures.','tab:edge_frequency')

    kinds=['nonlinear_dynamics','abrupt_intervention','nonstationary_delays','observation_model_mismatch'];methods=['Proposed revised','No-delay ablation','No-robustness ablation','Arrival interpolation','Kalman/RTS smoother (internal)','Delay-aware state-space (internal)'];mt=[]
    for kind in kinds:
        for seed in seeds:
            for method in methods:mt.append({'kind':kind,'seed':seed,'method':method,'config':config_to_jsonable(cfg),'quick':a.quick})
    mr=pd.DataFrame(parallel(mt,mismatch_task,a.workers,'mismatch'));fs.save_table(mr,a.output/'model_mismatch'/'model_mismatch_seed_metrics','Model-mismatch experiments on nonlinear dynamics, interventions, delays, and observation laws.','tab:mismatch_seed');ms=summarize_seed_results(mr,['rmse','mae','r2','runtime_seconds'],group_cols=['kind','method']);fs.save_table(ms,a.output/'model_mismatch'/'model_mismatch_summary','Model-mismatch summary.','tab:mismatch')
    tests=[]
    for kind,g in mr.groupby('kind'):
        p=pairwise_method_tests(g,'rmse',group_cols=('seed',));p.insert(0,'kind',kind);tests.append(p)
    fs.save_table(pd.concat(tests,ignore_index=True),a.output/'model_mismatch'/'model_mismatch_paired_tests','Paired within-scenario tests with Holm correction.','tab:mismatch_tests')

    scfg=replace(cfg,max_outer_iter=5,max_x_iter=2,max_w_iter=1,warmup_outer_iter=1,graph_stage_outer_iter=1)
    basevals={'n':8,'t':160,'d':6,'density':.18};levels={'n_nodes':[6,12,24],'n_time':[80,160,320],'max_delay':[2,6,10],'graph_density':[.10,.30,.50]};st=[]
    for factor,vals in levels.items():
        for level in vals:
            params=basevals.copy();key={'n_nodes':'n','n_time':'t','max_delay':'d','graph_density':'density'}[factor];params[key]=level
            for seed in range(7001,7002 if a.quick else 7004):st.append({'factor':factor,'level':level,'seed':seed,**params,'config':config_to_jsonable(scfg)})
    sr=pd.DataFrame(parallel(st,scalability_task,a.workers,'scalability'));fs.save_table(sr,a.output/'scalability'/'scalability_seed_metrics','One-factor-at-a-time scalability experiments.','tab:scale_seed');ss=sr.groupby(['factor','level'],as_index=False).agg(runtime_median=('runtime_seconds','median'),runtime_mean=('runtime_seconds','mean'),memory_mean_mb=('reported_peak_memory_mb','mean'),rmse_mean=('rmse','mean'));fs.save_table(ss,a.output/'scalability'/'scalability_runtime_memory','Runtime and memory scaling.','tab:scale')
    fs.write_json({'complete':True,'test_seeds':seeds,'threshold':threshold},a.output/'synthetic_manifest.json')

if __name__=='__main__':main()
