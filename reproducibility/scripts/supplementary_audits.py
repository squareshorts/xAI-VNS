from pathlib import Path
import sys,json,itertools
import numpy as np
import pandas as pd
from policy_core import Stream,policy_mask
ROOT=Path(__file__).resolve().parents[1];R=ROOT/'results'
def main():
 events=pd.read_csv(ROOT/'inputs/derived_inputs/event_manifest.csv')
 rows=[];evrows=[];bins=[];edfrows=[]
 for f in sorted((ROOT/'predictions').glob('within_chb??.csv.gz')):
  df=pd.read_csv(f,float_precision='round_trip');s=Stream(df[df.partition=='original'],events);fold=f.name.split('.')[0];p=s.rows.p_isotonic.to_numpy();r=s.rows.p_raw.to_numpy()
  for name,mask in [('Raw threshold 0.5',r>.5),('Raw persistence 0.5',s.persistence(r>.5)),('Raw overlap 0.5',(r>.5)&(s.lag_overlap()>.4)),('Nominal raw-plus-isotonic',(r>.5)&(p>.5)),('Nominal full 0.5',policy_mask(s,p,.5,'full'))]:
   rows.append(dict(fold=fold,policy=name,**s.metrics(mask)))
   for clock in ['center','end']:evrows.append(s.event_records(mask,clock).assign(fold=fold,policy=name))
 pd.DataFrame(rows).to_csv(R/'within_subject_timing_by_fold.csv',index=False)
 pd.concat(evrows,ignore_index=True).to_csv(R/'within_subject_timing_by_event.csv',index=False)
 # Bin data, equal-count evidence + full prediction references.
 ref=pd.read_csv(ROOT/'inputs/reference/trainonly_policy_selection_by_fold.csv')
 for f in sorted((ROOT/'predictions').glob('loso_chb??.csv.gz')):
  fold=f.name.split('.')[0];df=pd.read_csv(f,float_precision='round_trip')
  for ex in ['original','additional','combined']:
   d=df[df.partition==ex] if ex!='combined' else df[df.partition!='calibration']
   for method in ['raw','sigmoid','isotonic']:
    p=d['p_'+method].to_numpy();y=d.binary_label.to_numpy();ids=np.digitize(p,np.arange(.1,1.,.1))
    for b in range(10):
     m=ids==b;bins.append(dict(fold=fold,exposure=ex,method=method,bin=b+1,n=int(m.sum()),mean_probability=float(p[m].mean()) if m.any() else np.nan,observed_fraction=float(y[m].mean()) if m.any() else np.nan))
  theta=ref[ref.fold==fold].iloc[0].theta
  for (ex,sid,edf),g in df[df.partition!='calibration'].groupby(['partition','subject_id','edf_file']):
   s=Stream(g,events);p=s.rows.p_isotonic.to_numpy();raw=s.rows.p_raw.to_numpy()
   for pol,mask in [('raw_0.5',raw>.5),('calibrated_threshold',p>theta),('complete_train_selected',policy_mask(s,p,theta,'full'))]:
    edfrows.append(dict(fold=fold,exposure=ex,subject=sid,edf=edf,policy=pol,**s.metrics(mask)))
 pd.DataFrame(bins).to_csv(R/'calibration_bins_complete.csv',index=False)
 pd.DataFrame(edfrows).to_csv(R/'per_edf_policy_metrics.csv',index=False)
 # Exact provenance for original partitions, including EDF identities.
 m=pd.read_csv(ROOT/'inputs/reference/reconstructed_edf_partition_membership.csv')
 m.to_csv(R/'original_partition_membership.csv',index=False)
 summary=m.groupby(['evaluation_mode','fold','partition'],as_index=False).agg(edfs=('edf_file','count'),seconds=('duration_seconds','sum'),windows=('windows','sum'),positive_windows=('positive_windows','sum'),seizures=('seizures','sum'))
 summary['hours']=summary.seconds/3600;summary.to_csv(R/'original_partition_counts.csv',index=False)
 print('Within timing, full bin counts, per-EDF endpoints and partition audit saved.')
if __name__=='__main__':main()
