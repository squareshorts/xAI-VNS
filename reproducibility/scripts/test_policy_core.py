import unittest
import numpy as np
import pandas as pd
from policy_core import Stream,policy_mask

def stream(n=12,event=(10.,16.)):
 d=pd.DataFrame(dict(window_id=[str(i) for i in range(n)],subject_id='test',edf_file='a',window_index=np.arange(n),window_start=np.arange(n)*2.,window_end=np.arange(n)*2.+4.,window_center=np.arange(n)*2.+2.,binary_label=np.zeros(n,int),duration_seconds=n*2+2))
 for i in range(1,6):d['top'+str(i)]=i-1
 ev=pd.DataFrame([dict(subject_id='test',edf_file='a',event_index=1,seizure_onset=event[0],seizure_offset=event[1])]) if event else pd.DataFrame(columns=['subject_id','edf_file','event_index','seizure_onset','seizure_offset'])
 return Stream(d,ev)
class TestPolicies(unittest.TestCase):
 def test_empty(self):
  s=stream();self.assertEqual(s.counts(np.zeros(s.n,bool))[:2],(0,0))
 def test_skip_zero_still_merge_touching_supports(self):
  s=stream();m=np.zeros(s.n,bool);m[[0,2]]=True
  self.assertEqual(s.counts(m)[:2],(1,1))
 def test_gap_splits(self):
  s=stream();m=np.zeros(s.n,bool);m[[0,3]]=True
  self.assertEqual(s.counts(m)[:2],(2,2))
  self.assertEqual(s.counts(m,2)[:2],(1,1))
 def test_any_selected_positive(self):
  s=stream();s.y[3]=1;m=np.zeros(s.n,bool);m[[0,3]]=True
  self.assertEqual(s.counts(m)[:2],(2,1));self.assertEqual(s.counts(m,2)[:2],(1,0))
 def test_unselected_positive_does_not_reclassify(self):
  s=stream();s.y[1]=1;m=np.zeros(s.n,bool);m[[0,2]]=True
  self.assertEqual(s.counts(m)[:2],(1,1))
 def test_no_edf_crossing(self):
  s=stream();s.rows.loc[6:,'edf_file']='b';s=Stream(s.rows,s.events);m=np.ones(s.n,bool)
  self.assertEqual(s.counts(m,300)[:2],(2,2))
 def test_center_pre_end_at_onset(self):
  s=stream();m=np.zeros(s.n,bool);m[3]=True
  self.assertEqual(s.event_records(m,'center').iloc[0].category,'pre_onset')
  self.assertEqual(s.event_records(m,'end').iloc[0].category,'at_or_after_onset_only')
 def test_after_offset_not_covered(self):
  s=stream();m=np.zeros(s.n,bool);m[7]=True
  self.assertEqual(s.event_records(m,'center').iloc[0].category,'at_or_after_onset_only')
  self.assertEqual(s.event_records(m,'end').iloc[0].category,'missed')
 def test_strict_threshold(self):
  s=stream();p=np.full(s.n,.5)
  self.assertEqual(policy_mask(s,p,.5,'threshold').sum(),0)
 def test_persistence(self):
  s=stream();b=np.ones(s.n,bool);r=s.persistence(b)
  self.assertFalse(r[0]);self.assertTrue(r[1:].all())
 def test_overlap_jaccard(self):
  s=stream();s.top[1]=[0,1,2,5,6]
  self.assertAlmostEqual(s.lag_overlap()[1],3/7)
 def test_lag_warmup(self):
  s=stream();self.assertTrue(np.isnan(s.lag_overlap(2)[:2]).all());self.assertTrue((s.lag_overlap(2)[2:]==1).all())
 def test_firing_power_causal(self):
  s=stream();b=np.zeros(s.n,bool);b[5:]=True
  x=s.moving_fraction(b,5);self.assertTrue(np.isnan(x[:4]).all());self.assertEqual(x[4],0);self.assertEqual(x[8],.8)
 def test_refractory_changes_mask_not_merge(self):
  s=stream();b=np.ones(s.n,bool);m=s.refractory(b,6)
  np.testing.assert_array_equal(np.flatnonzero(m),[0,3,6,9])
 def test_support_not_sum_window_lengths(self):
  s=stream();m=np.zeros(s.n,bool);m[[0,1]]=True
  self.assertEqual(s.counts(m)[2],6)
if __name__=='__main__':unittest.main()
