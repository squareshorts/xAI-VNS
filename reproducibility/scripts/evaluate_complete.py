"""Frozen exposure evaluation and train-only temporal comparator benchmark.
Outputs are never used to choose thresholds on test/additional recordings.
"""
from pathlib import Path
import sys, json, time
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from policy_core import Stream,policy_mask
from fit_frozen_models import metrics
ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'results';PRED=ROOT/'predictions'

def main():
 events=pd.read_csv(ROOT/'inputs/derived_inputs/event_manifest.csv')
 reference=pd.read_csv(ROOT/'inputs/reference/trainonly_policy_selection_by_fold.csv')
 original_ref=pd.read_csv(ROOT/'inputs/reference/factorial_policy_by_fold.csv')
 policyrows=[];eventrows=[];clusterrows=[];sweeprows=[];lagrows=[];selectionrows=[];benchmark=[];bench_events=[];calmetrics=[];checks=[];selgrids=[]
 names={'threshold':'Calibrated threshold only','persistence':'+ persistence','overlap':'+ attribution overlap','full':'+ persistence + overlap'}
 for subject_id in ['chb01','chb02','chb03','chb05','chb08']:
  predfile=PRED/f'loso_{subject_id}.csv.gz'
  deadline=time.time()+900
  while not (RES/f'loso_{subject_id}_repro.csv').exists():
   if time.time()>deadline:raise TimeoutError('Waiting for completed fit '+subject_id)
   time.sleep(2)
  t0=time.time();fold=predfile.name.split('.')[0];subject=fold[-5:]
  df=pd.read_csv(predfile,float_precision='round_trip')
  row=reference[reference.fold==fold].iloc[0];theta=float(row.theta)
  cal=Stream(df[df.partition=='calibration'].copy(),events)
  streams={ex:Stream(df[df.partition==ex].copy(),events) for ex in ['original','additional']}
  streams['combined']=Stream(df[df.partition!='calibration'].copy(),events)
  # Verify archived cross-fitted Brier and threshold-only operating-point rule.
  cb={method:brier_score_loss(cal.y,cal.rows[f'p_{method}_oof']) for method in ['sigmoid','isotonic']}
  method=min(cb,key=cb.get);assert method==row.selected_calibrator
  pcal=cal.rows[f'p_{method}_oof'].to_numpy()
  grid=[]
  for th in np.arange(1,100)/100:
   mask=pcal>th
   q=cal.quick_selection_metrics(mask,'center');grid.append(dict(fold=fold,theta=th,**q))
  g=pd.DataFrame(grid);feasible=g[g.false_clusters_per_hour<=2]
  chosen=feasible.sort_values(['positive_window_sensitivity','false_clusters_per_hour','theta'],ascending=[False,True,True]).iloc[0]
  checks.append(dict(fold=fold,test='train_only_theta_reconstruction',observed=chosen.theta,expected=theta,error=chosen.theta-theta))
  if not np.isclose(chosen.theta,theta):raise AssertionError((fold,'theta',chosen.theta,theta))
  for key in cb:
   checks.append(dict(fold=fold,test='calibration_oof_brier_'+key,observed=cb[key],expected=row['oof_brier_'+key],error=cb[key]-row['oof_brier_'+key]))
  selectionrows.append(dict(fold=fold,subject=subject,selected_calibrator=method,theta=theta,**cb,calibration_hours=cal.hours,calibration_events=len(cal.ev),selection_rule='threshold_only; maximize positive-window sensitivity under <=2 false support-clusters/h',calibration_false_clusters_per_hour=chosen.false_clusters_per_hour))
  for ex,s in streams.items():
   p=s.rows['p_isotonic'].to_numpy();raw=s.rows.p_raw.to_numpy()
   for meth in ['raw','sigmoid','isotonic']:
    calmetrics.append(dict(fold=fold,subject=subject,exposure=ex,method=meth,hours=s.hours,**metrics(s.y,s.rows['p_'+meth].to_numpy())))
   policies={'Raw threshold 0.5':raw>.5,'Raw persistence 0.5':s.persistence(raw>.5),'Raw overlap 0.5':(raw>.5)&(s.lag_overlap()>.4),'Nominal raw-plus-isotonic':(raw>.5)&(p>.5),'Nominal full 0.5':policy_mask(s,p,.5,'full')}
   policies.update({name:policy_mask(s,p,theta,family) for family,name in names.items()})
   # Lag-gate sensitivity holds the original score/persistence/threshold fixed.
   for lag in [2,3,5,10]:policies[f'Full overlap lag {2*lag}s']=policy_mask(s,p,theta,'full',lag=lag)
   for policy,mask in policies.items():
    for cool in ([0,30,60,120] if policy in ['Raw threshold 0.5','Nominal full 0.5','+ persistence + overlap'] else [0]):
     context=dict(fold=fold,subject=subject,exposure=ex,policy=policy,cooldown_s=cool,theta=theta if policy in names.values() or policy.startswith('Full overlap') else .5)
     q=s.metrics(mask,cool);policyrows.append(dict(**context,**q))
     if cool==0 and ex=='original' and policy in names.values():
      rr=original_ref[(original_ref.fold==fold)&(original_ref.configuration==policy.replace('+ persistence + overlap','+ persistence + attribution overlap'))].iloc[0]
      for key,expected in [('false_clusters_per_hour',rr.false_clusters_per_hour),('authorization_fraction',rr.authorization_rate),('center_coverage',rr.event_coverage)]:
       err=q[key]-expected;checks.append(dict(fold=fold,test=policy+' '+key,observed=q[key],expected=expected,error=err))
       assert abs(err)<1e-9,(fold,policy,key,q[key],expected)
     if cool==0:
      for clock in ['center','end']:
       er=s.event_records(mask,clock)
       if len(er):er=er.assign(**context);eventrows.append(er)
     if ex=='combined' and policy in ['Raw threshold 0.5','Nominal full 0.5','Calibrated threshold only','+ persistence','+ persistence + overlap']:
      cr=s.cluster_records(mask,cool).assign(**context);clusterrows.append(cr)
   # All sweep points have gamma=.4, never average across gate settings.
   for th in np.arange(1,10)/10:
    for policy,mask in [('Raw threshold',raw>th),('Raw persistence',s.persistence(raw>th)),('Isotonic full',policy_mask(s,p,th,'full'))]:
     sweeprows.append(dict(fold=fold,subject=subject,exposure=ex,policy=policy,theta=th,gamma=.4 if policy=='Isotonic full' else np.nan,**s.metrics(mask)))
   for lag in [1,2,3,5,10]:
    ss=s.lag_overlap(lag);valid=np.isfinite(ss)&(raw>.5);true=valid&(s.y==1);false=valid&(s.y==0)
    # Score-stratified descriptive difference; weights are positive-window counts.
    diff=[];weights=[]
    for lo,hi in zip(np.arange(.5,1.,.1),np.arange(.6,1.1,.1)):
     v=valid&(raw>=lo)&(raw<hi+1e-12);vt=v&(s.y==1);vf=v&(s.y==0)
     if vt.sum()>=1 and vf.sum()>=1:diff.append(ss[vt].mean()-ss[vf].mean());weights.append(vt.sum())
    lagrows.append(dict(fold=fold,subject=subject,exposure=ex,lag_s=2*lag,n_true=int(true.sum()),n_false=int(false.sum()),mean_true=float(ss[true].mean()) if true.any() else np.nan,mean_false=float(ss[false].mean()) if false.any() else np.nan,difference=float(ss[true].mean()-ss[false].mean()) if true.any() and false.any() else np.nan,score_stratified_difference=float(np.average(diff,weights=weights)) if weights else np.nan,n_strata=len(diff)))
  # Adapted temporal families selected on the ORIGINAL OOF calibration block.
  # Shared objective across ALL families: max end-time event coverage under budget,
  # then lower burden, then more pre-onset coverage, lower theta, shorter history.
  for refractory in [0,300]:
   for family in ['threshold','persistence','full','majority3','firing_power']:
    cg=[]
    for m in ([5,15,30,60,150] if family=='firing_power' else [3 if family=='majority3' else 1]):
     for th in np.arange(1,100)/100:
      mask=policy_mask(cal,pcal,th,family,m=m,refractory=refractory)
      q=cal.quick_selection_metrics(mask,'end')
      cg.append(dict(fold=fold,family=family,refractory_s=refractory,m=m,theta=th,**q))
    g=pd.DataFrame(cg);selgrids.append(g)
    feasible=g[g.false_clusters_per_hour<=2]
    if feasible.empty:raise AssertionError('No feasible rule')
    choice=feasible.sort_values(['covered','false_clusters_per_hour','pre','theta','m'],ascending=[False,True,False,True,True]).iloc[0]
    for ex,s in streams.items():
     mask=policy_mask(s,s.rows.p_isotonic.to_numpy(),float(choice.theta),family,m=int(choice.m),refractory=refractory)
     context=dict(fold=fold,subject=subject,exposure=ex,family=family,refractory_s=refractory,m=int(choice.m),theta=float(choice.theta),calibration_covered=int(choice.covered),calibration_events=int(choice.events),calibration_false_clusters_per_hour=choice.false_clusters_per_hour)
     benchmark.append(dict(**context,**s.metrics(mask)))
     if ex=='combined':
      er=s.event_records(mask,'end').assign(**context);bench_events.append(er)
  # Per-fold result snapshot; evaluation can be repeated from cached predictions.
  pd.DataFrame(policyrows).to_csv(RES/'policies_by_fold.csv',index=False)
  pd.concat(eventrows,ignore_index=True).to_csv(RES/'event_timing_by_event.csv',index=False)
  pd.concat(clusterrows,ignore_index=True).to_csv(RES/'cluster_ledger.csv.gz',index=False,compression='gzip')
  pd.DataFrame(sweeprows).to_csv(RES/'fixed_gamma_sweep_by_fold.csv',index=False)
  pd.DataFrame(lagrows).to_csv(RES/'attribution_lag_complete_by_fold.csv',index=False)
  pd.DataFrame(selectionrows).to_csv(RES/'frozen_selection_reconstruction.csv',index=False)
  pd.DataFrame(benchmark).to_csv(RES/'temporal_comparators_by_fold.csv',index=False)
  pd.concat(bench_events,ignore_index=True).to_csv(RES/'temporal_comparator_events.csv',index=False)
  pd.DataFrame(calmetrics).to_csv(RES/'complete_calibration_by_fold.csv',index=False)
  pd.DataFrame(checks).to_csv(RES/'reproduction_checks.csv',index=False)
  pd.concat(selgrids,ignore_index=True).to_csv(RES/'comparator_trainonly_grid.csv.gz',index=False,compression='gzip')
  print('Evaluated and checkpointed',fold,'seconds',round(time.time()-t0,1),flush=True)
 if len(selectionrows)!=5:raise AssertionError('Not all 5 LOSO prediction files available')
 print('ALL FIVE FOLDS COMPLETE',flush=True)
if __name__=='__main__':main()
