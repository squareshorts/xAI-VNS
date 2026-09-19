"""Auditable, record-local policy and endpoint definitions.
Legacy clusters use unions of selected WINDOW SUPPORTS, not notification times.
True clusters require >=1 selected positive-center window. Cooldown is merging,
whereas refractory suppression changes the decision mask. Never conflate these.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd

@dataclass
class Stream:
 rows: pd.DataFrame
 events: pd.DataFrame
 def __post_init__(self):
  self.rows=self.rows.sort_values(['subject_id','edf_file','window_index']).reset_index(drop=True)
  self.n=len(self.rows); self.y=self.rows.binary_label.to_numpy(dtype=int, copy=True)
  self.start=self.rows.window_start.to_numpy(float);self.end=self.rows.window_end.to_numpy(float);self.center=self.rows.window_center.to_numpy(float)
  self.keys=(self.rows.subject_id+'|'+self.rows.edf_file).to_numpy()
  self.new=np.r_[True,self.keys[1:]!=self.keys[:-1]] if self.n else np.array([],bool)
  self.gid=np.cumsum(self.new)-1
  begins=np.flatnonzero(self.new);ends=np.r_[begins[1:],self.n]
  self.groups=[np.arange(a,b) for a,b in zip(begins,ends)]
  self.hours=float(self.rows.drop_duplicates(['subject_id','edf_file']).duration_seconds.sum()/3600)
  selected=set(self.keys)
  self.events=self.events[(self.events.subject_id+'|'+self.events.edf_file).isin(selected)].copy()
  self.ev=[];self.pos_seconds=0.
  for ev in self.events.itertuples():
   key=ev.subject_id+'|'+ev.edf_file;lo=max(0.,ev.seizure_onset-300);hi=ev.seizure_offset
   self.ev.append((ev, np.flatnonzero((self.keys==key)&(self.center>=lo)&(self.center<=hi)), np.flatnonzero((self.keys==key)&(self.end>=lo)&(self.end<=hi))))
  # Length of union of label-positive intervals, clipped to each EDF.
  for key,g in zip([self.keys[x[0]] for x in self.groups],self.groups):
   e=self.events[(self.events.subject_id+'|'+self.events.edf_file)==key]
   intervals=sorted((max(0.,r.seizure_onset-300),min(float(self.rows.iloc[g[0]].duration_seconds),r.seizure_offset)) for r in e.itertuples())
   left=right=None
   for a,b in intervals:
    if left is None:left,right=a,b
    elif a<=right:right=max(right,b)
    else:self.pos_seconds+=right-left;left,right=a,b
   if left is not None:self.pos_seconds+=right-left
  self.at_risk_hours=self.hours-self.pos_seconds/3600
  self.top=self.rows[[f'top{i}' for i in range(1,6)]].to_numpy(dtype=int, copy=True)
  self._lag={}
 def lag_overlap(self,lag=1):
  if lag not in self._lag:
   s=np.full(self.n,np.nan)
   if self.n>lag:
    valid=(self.gid[lag:]==self.gid[:-lag]) & np.isclose(self.center[lag:]-self.center[:-lag],2*lag)
    a=self.top[lag:];b=self.top[:-lag]
    overlap=(a[:,:,None]==b[:,None,:]).any(axis=2).sum(axis=1)
    idx=np.flatnonzero(valid)+lag;s[idx]=overlap[valid]/(10-overlap[valid])
   self._lag[lag]=s
  return self._lag[lag]
 def persistence(self,binary,lag=1):
  out=np.zeros(self.n,bool)
  out[lag:]=binary[lag:] & binary[:-lag] & (self.gid[lag:]==self.gid[:-lag]) & np.isclose(self.center[lag:]-self.center[:-lag],2*lag)
  return out
 def moving_fraction(self,binary,m):
  out=np.full(self.n,np.nan)
  for g in self.groups:
   if len(g)<m:continue
   cs=np.r_[0,np.cumsum(binary[g],dtype=float)]
   out[g[m-1:]]=(cs[m:]-cs[:-m])/m
  return out
 def refractory(self,mask,seconds):
  if seconds<=0:return mask.copy()
  out=np.zeros(self.n,bool)
  for g in self.groups:
   next_allowed=-np.inf
   for i in g[mask[g]]:
    if self.end[i]>=next_allowed:
     out[i]=True;next_allowed=self.end[i]+seconds
  return out
 def cluster_records(self,mask,cooldown=0):
  idx=np.flatnonzero(mask)
  if not len(idx):return pd.DataFrame(columns=['subject_id','edf_file','support_start','support_end','decision_first','decision_last','n_windows','n_positive_windows','false_cluster','duration_seconds'])
  new=np.r_[True,(self.gid[idx[1:]]!=self.gid[idx[:-1]]) | (self.start[idx[1:]]>self.end[idx[:-1]]+cooldown)]
  a=np.flatnonzero(new);b=np.r_[a[1:],len(idx)]
  counts=b-a; positives=np.add.reduceat(self.y[idx],a)
  first=idx[a];last=idx[b-1]
  return pd.DataFrame(dict(subject_id=self.rows.subject_id.to_numpy()[first],edf_file=self.rows.edf_file.to_numpy()[first],support_start=self.start[first],support_end=self.end[last],decision_first=self.end[first],decision_last=self.end[last],n_windows=counts,n_positive_windows=positives,false_cluster=positives==0,duration_seconds=self.end[last]-self.start[first]))
 def counts(self,mask,cooldown=0):
  idx=np.flatnonzero(mask)
  if not len(idx):return 0,0,0.
  new=np.r_[True,(self.gid[idx[1:]]!=self.gid[idx[:-1]]) | (self.start[idx[1:]]>self.end[idx[:-1]]+cooldown)]
  a=np.flatnonzero(new);b=np.r_[a[1:],len(idx)]
  positives=np.add.reduceat(self.y[idx],a)
  # Support union seconds excludes artificial cooldown gaps. Original windows fixed 4 s.
  durations=np.minimum(4.,np.r_[4.,np.where(self.gid[idx[1:]]==self.gid[idx[:-1]],self.end[idx[1:]]-self.end[idx[:-1]],4.)]).sum()
  return int(len(a)),int(np.sum(positives==0)),float(durations)
 def event_records(self,mask,clock='center'):
  records=[];t=self.center if clock=='center' else self.end
  for ev,ic,ie in self.ev:
   ix=ic if clock=='center' else ie;idx=ix[mask[ix]]
   first=t[idx[0]] if len(idx) else np.nan;lat=first-ev.seizure_onset
   category='missed' if not len(idx) else ('pre_onset' if lat<0 else 'at_or_after_onset_only')
   records.append(dict(subject_id=ev.subject_id,edf_file=ev.edf_file,event_index=ev.event_index,onset=ev.seizure_onset,offset=ev.seizure_offset,clock=clock,first_authorization=first,latency_seconds=lat,lead_seconds=-lat,category=category,n_authorizations=len(idx)))
  return pd.DataFrame(records)
 def metrics(self,mask,cooldown=0):
  total,false,support=self.counts(mask,cooldown)
  n=int(mask.sum());tp=int(self.y[mask].sum());out=dict(hours=self.hours,edfs=len(self.groups),windows=self.n,positive_windows=int(self.y.sum()),authorized_windows=n,true_authorized_windows=tp,false_authorized_windows=n-tp,authorization_fraction=n/self.n if self.n else np.nan,positive_authorization_rate=tp/n if n else np.nan,positive_window_sensitivity=tp/self.y.sum() if self.y.sum() else np.nan,clusters=total,false_clusters=false,clusters_per_hour=total/self.hours,false_clusters_per_hour=false/self.hours,false_windows_per_hour=(n-tp)/self.hours,authorized_support_seconds=support,authorized_support_fraction=support/(self.hours*3600),at_risk_hours=self.at_risk_hours,false_clusters_per_nonrisk_hour=false/self.at_risk_hours if self.at_risk_hours else np.nan)
  for clock in ['center','end']:
   r=self.event_records(mask,clock);ne=len(r)
   if not ne:
    out.update({f'{clock}_{k}':np.nan for k in ['coverage','pre_fraction','post_fraction','missed_fraction','median_lead','median_latency','pre_among_covered']})
    for k in ['events','pre','post','missed']:out[f'{clock}_{k}']=0
    continue
   pre=int((r.category=='pre_onset').sum());post=int((r.category=='at_or_after_onset_only').sum());mis=ne-pre-post
   out.update({f'{clock}_events':ne,f'{clock}_pre':pre,f'{clock}_post':post,f'{clock}_missed':mis,f'{clock}_coverage':(pre+post)/ne,f'{clock}_pre_fraction':pre/ne,f'{clock}_post_fraction':post/ne,f'{clock}_missed_fraction':mis/ne,f'{clock}_median_lead':r.lead_seconds.median() if pre+post else np.nan,f'{clock}_median_latency':r.latency_seconds.median() if pre+post else np.nan,f'{clock}_pre_among_covered':pre/(pre+post) if pre+post else np.nan})
  return out
 def quick_selection_metrics(self,mask,clock='end'):
  n,false,_=self.counts(mask)
  covered=pre=0
  for ev,ic,ie in self.ev:
   ix=ie if clock=='end' else ic;idx=ix[mask[ix]]
   if len(idx):
    covered+=1;pre+=int((self.end if clock=='end' else self.center)[idx[0]]<ev.seizure_onset)
  return dict(false_clusters_per_hour=false/self.hours,covered=covered,pre=pre,events=len(self.ev),authorized_windows=int(mask.sum()),positive_window_sensitivity=float(self.y[mask].sum()/self.y.sum()) if self.y.sum() else np.nan)

def policy_mask(s,p,theta,family,m=3,refractory=0,lag=1):
 b=p>theta
 if family=='threshold':mask=b
 elif family=='persistence':mask=s.persistence(b)
 elif family=='overlap':mask=b&(s.lag_overlap(lag)>.4)
 elif family=='full':mask=s.persistence(b)&(s.lag_overlap(lag)>.4)
 elif family=='majority3':mask=s.moving_fraction(b,3)>.5
 elif family=='firing_power':mask=s.moving_fraction(b,m)>.7
 else:raise ValueError(family)
 return s.refractory(mask,refractory)
