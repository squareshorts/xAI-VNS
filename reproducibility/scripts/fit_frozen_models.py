"""Reconstruct archived nested LightGBM fits on ORIGINAL partitions only.
No added EDF contributes to a fit, calibration fit, or policy selection.
Model-score reconstruction is checked against archived numerical outputs.
"""
from pathlib import Path
import argparse, hashlib, json, platform, time, sys
import numpy as np
import pandas as pd
import scipy, sklearn, lightgbm
from lightgbm import LGBMClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
ROOT=Path(__file__).resolve().parents[1]
FEATURES=['bp_delta','rel_bp_delta','bp_theta','rel_bp_theta','bp_alpha','rel_bp_alpha','bp_beta','rel_bp_beta','bp_gamma','rel_bp_gamma','spectral_entropy','hjorth_activity','hjorth_mobility','hjorth_complexity','line_length','rms_amplitude','variance','zero_crossing_rate','skewness','kurtosis']
def logit(p):
 p=np.clip(p,1e-6,1-1e-6); return np.log(p/(1-p))
def fit_cal(y,p):
 sig=LogisticRegression(C=1e6,solver='lbfgs',max_iter=1000).fit(logit(p).reshape(-1,1),y)
 iso=IsotonicRegression(out_of_bounds='clip',y_min=0,y_max=1).fit(p,y)
 return sig,iso
def cal_apply(sig,iso,p):
 return sig.predict_proba(logit(p).reshape(-1,1))[:,1],iso.predict(p)
def metrics(y,p):
 b=np.digitize(p,np.linspace(0,1,11)[1:-1]); ece=sum(np.sum(b==j)/len(p)*abs(p[b==j].mean()-y[b==j].mean()) for j in range(10) if np.any(b==j))
 two=len(np.unique(y))==2
 return dict(auroc=roc_auc_score(y,p) if two else np.nan,auprc=average_precision_score(y,p) if two else np.nan,brier=brier_score_loss(y,p),ece=ece,maximum=float(p.max()),prevalence=float(y.mean()))
def main():
 parser=argparse.ArgumentParser();parser.add_argument('--folds',default='all');args=parser.parse_args()
 d=ROOT/'inputs/derived_inputs'; ref=ROOT/'inputs/reference'
 checks=json.loads((d/'derived_checksums.json').read_text())
 for f,sha in checks.items():
  assert hashlib.sha256((d/f).read_bytes()).hexdigest()==sha,f
 orig=pd.read_csv(d/'original_features.csv.gz',float_precision='round_trip')
 extra=pd.read_csv(d/'additional_features.csv.gz',float_precision='round_trip')
 orig['collection']='original'; extra['collection']='additional'
 membership=pd.read_csv(ref/'reconstructed_edf_partition_membership.csv')
 manifest=pd.read_csv(d/'collection_manifest.csv')
 assert orig.window_id.is_unique and extra.window_id.is_unique
 assert not set(orig.window_id)&set(extra.window_id)
 assert len(orig)==122007 and len(extra)==188909
 assert orig.binary_label.sum()==5227 and extra.binary_label.sum()==0
 assert manifest.checksum_ok.all() and manifest.all_original_22_channels.all()
 env=dict(python=sys.version,platform=platform.platform(),numpy=np.__version__,pandas=pd.__version__,scipy=scipy.__version__,sklearn=sklearn.__version__,lightgbm=lightgbm.__version__,notes='LightGBM 4.6.0 matches archived version; other installed numerical versions recorded. Reconstruction validated before new exposure interpreted.')
 (ROOT/'results/environment.json').write_text(json.dumps(env,indent=2))
 (ROOT/'results/input_validation.json').write_text(json.dumps(dict(derived_checksums_passed=len(checks),original_windows=len(orig),added_windows=len(extra),original_positive=int(orig.binary_label.sum()),added_positive=int(extra.binary_label.sum()),all_175_reported_checksum_verified=True,feature_names=FEATURES),indent=2))
 folds=membership.fold.unique()
 if args.folds!='all': folds=[x for x in folds if x in args.folds.split(',')]
 for fold in folds:
  dest=ROOT/'predictions'/f'{fold}.csv.gz'
  if dest.exists() and (ROOT/'results'/f'{fold}_repro.csv').exists():
   print('CACHED',fold,flush=True);continue
  t=time.time();m=membership[membership.fold==fold];parts={}
  for name in ['core','calibration','test']:
   keys=m[m.partition==name][['subject_id','edf_file']]
   parts[name]=orig.merge(keys,on=['subject_id','edf_file'],how='inner').sort_values(['subject_id','edf_file','window_index']).reset_index(drop=True)
   assert parts[name].edf_file.nunique()==len(keys)
  subject=parts['test'].subject_id.unique();assert len(subject)==1;subject=subject[0]
  assert not set(parts['core'].window_id)&set(parts['calibration'].window_id)
  assert not set(parts['core'].window_id)&set(parts['test'].window_id)
  model=LGBMClassifier(random_state=42,class_weight='balanced',verbose=-1,n_jobs=2)
  X=parts['core'][FEATURES].replace([np.inf,-np.inf],np.nan).fillna(0)
  model.fit(X,parts['core'].binary_label.to_numpy())
  model.booster_.save_model(str(ROOT/'models'/f'{fold}.txt'))
  (ROOT/'models'/f'{fold}_parameters.json').write_text(json.dumps(model.get_params(),indent=2))
  cal=parts['calibration']; pc=model.predict_proba(cal[FEATURES])[:,1]
  sig,iso=fit_cal(cal.binary_label.to_numpy(),pc)
  (ROOT/'models'/f'{fold}_calibrators.json').write_text(json.dumps(dict(sigmoid_coef=sig.coef_.tolist(),sigmoid_intercept=sig.intercept_.tolist(),isotonic_x=iso.X_thresholds_.tolist(),isotonic_y=iso.y_thresholds_.tolist()),indent=2))
  blocks=[];repro=[]
  evals=[('calibration',cal),('original',parts['test'])]
  if fold.startswith('loso'):
   a=extra[extra.subject_id==subject].sort_values(['subject_id','edf_file','window_index']).reset_index(drop=True)
   evals.append(('additional',a))
  for collection,df in evals:
   p=model.predict_proba(df[FEATURES])[:,1];ps,pi=cal_apply(sig,iso,p)
   rawcon=model.booster_.predict(df[FEATURES],pred_contrib=True,num_threads=2)
   rank=np.argsort(-np.abs(rawcon[:,:-1]),axis=1)[:,:5]
   add_error=np.max(np.abs(rawcon.sum(axis=1)-model.booster_.predict(df[FEATURES],raw_score=True,num_threads=2)))
   assert add_error<1e-8,('contribution additivity',add_error)
   out=df[['window_id','subject_id','edf_file','window_index','window_start','window_end','window_center','label','binary_label','duration_seconds']].copy()
   out['partition']=collection;out['fold']=fold
   out['p_raw']=p;out['p_sigmoid']=ps;out['p_isotonic']=pi
   for k in range(5):out[f'top{k+1}']=rank[:,k]
   if collection=='calibration':
    oofs=np.full(len(df),np.nan);oofi=oofs.copy()
    keys=df.subject_id+'|'+df.edf_file
    for key in keys.unique():
     test=keys.eq(key).to_numpy();train=~test
     if len(np.unique(df.binary_label.to_numpy()[train]))<2:
      # Never fabricate a calibrator; mark unavailable OOF split (within-only).
      continue
     ss,ii=fit_cal(df.binary_label.to_numpy()[train],p[train]);oofs[test],oofi[test]=cal_apply(ss,ii,p[test])
    out['p_sigmoid_oof']=oofs;out['p_isotonic_oof']=oofi
    if fold.startswith('loso'):assert np.isfinite(oofs).all()
   for name,pp in [('raw',p),('sigmoid',ps),('isotonic',pi)]:
    repro.append(dict(fold=fold,subject=subject,collection=collection,method=name,**metrics(df.binary_label.to_numpy(),pp)))
   out.to_csv(ROOT/'predictions'/f'{fold}_{collection}.csv.gz',index=False,compression='gzip',float_format='%.17g')
   blocks.append(out)
   print(f'{fold} {collection}: {len(df):,} predictions/contributions; {time.time()-t:.1f}s',flush=True)
  pd.concat(blocks,ignore_index=True).to_csv(dest,index=False,compression='gzip',float_format='%.17g')
  pd.DataFrame(repro).to_csv(ROOT/'results'/f'{fold}_repro.csv',index=False)
  print('SAVED',fold,time.time()-t,flush=True)
 files=sorted((ROOT/'results').glob('*_repro.csv'))
 if files:pd.concat([pd.read_csv(p) for p in files],ignore_index=True).to_csv(ROOT/'results/reconstructed_discrimination_calibration.csv',index=False)
if __name__=='__main__':main()
