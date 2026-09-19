"""Subject-level summaries and paired exploratory uncertainty.
Exact empirical bootstrap enumerates 5**5 ordered draws (no Monte Carlo seed).
These are descriptive intervals, not independent-subject external validation.
"""
from pathlib import Path
import itertools,json
import numpy as np
import pandas as pd
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
MEASURES=['false_clusters_per_hour','clusters_per_hour','authorization_fraction','positive_authorization_rate','authorized_support_fraction','center_coverage','end_coverage','center_pre_fraction','end_pre_fraction','center_median_lead','end_median_lead']
def boot(x):
 x=np.asarray(x,dtype=float);x=x[np.isfinite(x)];n=len(x)
 if not n:return dict(mean=np.nan,sd=np.nan,ci_low=np.nan,ci_high=np.nan,n=0)
 draws=np.array(list(itertools.product(range(n),repeat=n)))
 v=x[draws].mean(axis=1)
 lo,hi=np.quantile(v,[.025,.975])
 return dict(mean=float(x.mean()),sd=float(x.std(ddof=1)) if n>1 else np.nan,ci_low=float(lo),ci_high=float(hi),n=n)
def grouped(df,keys,metrics):
 rows=[]
 for key,g in df.groupby(keys,dropna=False,sort=False):
  if not isinstance(key,tuple):key=(key,)
  row=dict(zip(keys,key));row['n_subjects']=g.subject.nunique()
  for col in metrics:
   for stat,v in boot(g[col]).items():row[col+'_'+stat]=v
  for col in ['hours','edfs','windows','positive_windows','authorized_windows','true_authorized_windows','false_authorized_windows','false_clusters','clusters','center_events','center_pre','center_post','center_missed','end_events','end_pre','end_post','end_missed']:
   if col in g:row['pooled_'+col]=g[col].sum()
  if 'hours' in g:
   row['pooled_false_clusters_per_hour']=g.false_clusters.sum()/g.hours.sum() if 'false_clusters' in g else np.nan
  rows.append(row)
 return pd.DataFrame(rows)
def paired(df,keycols,a,b):
 rows=[]
 for key,g in df.groupby(keycols,dropna=False):
  if not isinstance(key,tuple):key=(key,)
  q=g.pivot(index='subject',columns='policy',values='false_clusters_per_hour')
  if a not in q or b not in q:continue
  delta=q[b]-q[a];st=boot(delta)
  v=delta.to_numpy();sg=np.array(list(itertools.product([-1,1],repeat=len(v))))
  p=float(np.mean(np.abs((sg*v).mean(axis=1))>=abs(v.mean())-1e-12))
  rows.append(dict(zip(keycols,key),baseline=a,comparison=b,**st,signflip_p=p,subjects_lower=int((delta<0).sum()),subjects_equal=int((delta==0).sum()),subjects_higher=int((delta>0).sum())))
 return rows
def main():
 p=pd.read_csv(R/'policies_by_fold.csv');assert p.fold.nunique()==5
 ps=grouped(p,['exposure','policy','cooldown_s'],MEASURES);ps.to_csv(R/'policy_summary.csv',index=False)
 b=pd.read_csv(R/'temporal_comparators_by_fold.csv')
 bs=grouped(b,['exposure','family','refractory_s'],MEASURES);bs.to_csv(R/'comparator_summary.csv',index=False)
 e=pd.read_csv(R/'event_timing_by_event.csv');er=[]
 for (ex,pol,clock),g in e.groupby(['exposure','policy','clock']):
  cov=g.latency_seconds.dropna();pre=g[g.category=='pre_onset'].lead_seconds
  er.append(dict(exposure=ex,policy=pol,clock=clock,events=len(g),pre=int((g.category=='pre_onset').sum()),post=int((g.category=='at_or_after_onset_only').sum()),missed=int((g.category=='missed').sum()),pooled_median_latency=cov.median() if len(cov) else np.nan,latency_q25=cov.quantile(.25) if len(cov) else np.nan,latency_q75=cov.quantile(.75) if len(cov) else np.nan,pooled_median_lead=-cov.median() if len(cov) else np.nan,pre_only_median_lead=pre.median() if len(pre) else np.nan))
 pd.DataFrame(er).to_csv(R/'event_timing_pooled.csv',index=False)
 comparisons=[]
 for a,c in [('Calibrated threshold only','+ persistence'),('+ persistence','+ persistence + overlap'),('Calibrated threshold only','+ attribution overlap'),('Calibrated threshold only','+ persistence + overlap')]:comparisons+=paired(p[p.cooldown_s==0],['exposure'],a,c)
 pd.DataFrame(comparisons).to_csv(R/'factorial_paired_effects.csv',index=False)
 bb=b.copy();bb['policy']=bb.family;comparisons=[]
 for c in ['threshold','persistence','majority3','firing_power']:comparisons+=paired(bb,['exposure','refractory_s'],'full',c)
 pd.DataFrame(comparisons).to_csv(R/'comparator_paired_effects.csv',index=False)
 sw=pd.read_csv(R/'fixed_gamma_sweep_by_fold.csv');ss=grouped(sw,['exposure','policy','theta','gamma'],MEASURES);ss.to_csv(R/'fixed_gamma_sweep_summary.csv',index=False)
 lag=pd.read_csv(R/'attribution_lag_complete_by_fold.csv');ls=grouped(lag,['exposure','lag_s'],['mean_true','mean_false','difference','score_stratified_difference']);ls.to_csv(R/'attribution_lag_complete_summary.csv',index=False)
 cal=pd.read_csv(R/'complete_calibration_by_fold.csv');cs=[]
 for (ex,method),g in cal.groupby(['exposure','method']):
  row=dict(exposure=ex,method=method)
  for col in ['auroc','auprc','brier','ece','prevalence']:
   for k,v in boot(g[col]).items():row[col+'_'+k]=v
  cs.append(row)
 pd.DataFrame(cs).to_csv(R/'calibration_summary.csv',index=False)
 # Original model reconstruction from archived metrics, not just aggregate means.
 reference=pd.read_csv(ROOT/'inputs/reference/repro_nested_lightgbm.csv')
 observed=pd.read_csv(R/'reconstructed_discrimination_calibration.csv');observed=observed[(observed.collection=='original')&observed.fold.str.startswith('loso')]
 q=observed.merge(reference,on=['fold','method'],suffixes=('','_reference'))
 for col in ['auroc','auprc','brier']:
  assert (q[col]-q[col+'_reference']).abs().max()<1e-10
 q.to_csv(R/'score_reproduction_comparison.csv',index=False)
 # Record count/hours distributions and case-level source sampling.
 m=pd.read_csv(ROOT/'inputs/derived_inputs/collection_manifest.csv');counts=[]
 for sub,g in m.groupby('subject_id'):
  for ex,gg in [('original',g[g.in_original_analysis]),('additional',g[~g.in_original_analysis]),('combined',g)]:
   counts.append(dict(subject=sub,exposure=ex,edfs=len(gg),hours=gg.duration_seconds.sum()/3600,seizure_edfs=int(gg.public_seizure_bearing.sum())))
 pd.DataFrame(counts).to_csv(R/'collection_account.csv',index=False)
 # Key results readily inspectable, not rounded prior to calculations.
 select=ps[(ps.policy.isin(['Raw threshold 0.5','Calibrated threshold only','+ persistence','+ persistence + overlap']))&(ps.cooldown_s==0)]
 print(select[['exposure','policy','false_clusters_per_hour_mean','false_clusters_per_hour_sd','pooled_false_clusters','pooled_hours','end_coverage_mean','pooled_end_pre','pooled_end_post','pooled_end_missed']].to_string(index=False))
 print('\nCOMPARATORS\n',bs[bs.exposure=='combined'][['family','refractory_s','false_clusters_per_hour_mean','false_clusters_per_hour_sd','pooled_end_pre','pooled_end_post','pooled_end_missed']].to_string(index=False))
 print('\nTIMING\n',pd.DataFrame(er).query("exposure=='combined' and policy=='+ persistence + overlap'").to_string(index=False))
if __name__=='__main__':main()
