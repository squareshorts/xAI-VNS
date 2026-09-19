"""Software unit tests only. Numerical fixtures are NOT experimental EEG data."""
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import urllib.error
import numpy as np
from scipy import signal, stats
import complete_collection as c
from audit_local_collection import CHANNELS, inspect_edf
from feature_core import FEATURE_NAMES, extract_features
from prepare_features import parse_summary


class Response:
    def __init__(self, data, status=200, headers=None):
        self.data=io.BytesIO(data)
        self.status=status
        self.headers=headers or {'Content-Length':str(len(data))}
    def __enter__(self):return self
    def __exit__(self,*args):self.data.close()
    def read(self,n=-1):return self.data.read(n)


class Downloads(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory()
        self.path=Path(self.tmp.name)/'data.edf'
        self.payload=b'unit-test-transfer-contents'*20
        self.digest=hashlib.sha256(self.payload).hexdigest()
    def tearDown(self):self.tmp.cleanup()
    def test_complete_download(self):
        with patch.object(c.urllib.request,'urlopen',return_value=Response(self.payload)):
            result=c.verified_download('test/data.edf',self.path,self.digest,retries=1)
        self.assertEqual(self.path.read_bytes(),self.payload)
        self.assertEqual(result['download_status'],'downloaded_verified')
    def test_existing_verified_never_requested(self):
        self.path.write_bytes(self.payload)
        with patch.object(c.urllib.request,'urlopen') as fetch:
            result=c.verified_download('test/data.edf',self.path,self.digest,retries=1)
            fetch.assert_not_called()
        self.assertEqual(result['download_status'],'already_verified')
    def test_existing_corrupt_not_overwritten(self):
        self.path.write_bytes(b'keep-me')
        with self.assertRaises(ValueError):c.verified_download('test/data.edf',self.path,self.digest,retries=1)
        self.assertEqual(self.path.read_bytes(),b'keep-me')
    def test_resume_partial(self):
        part=self.path.with_name(self.path.name+'.part')
        part.write_bytes(self.payload[:23])
        response=Response(self.payload[23:],206,{'Content-Range':f'bytes 23-{len(self.payload)-1}/{len(self.payload)}','Content-Length':str(len(self.payload)-23)})
        with patch.object(c.urllib.request,'urlopen',return_value=response) as fetch:
            c.verified_download('test/data.edf',self.path,self.digest,retries=1)
            self.assertEqual(fetch.call_args.args[0].get_header('Range'),'bytes=23-')
        self.assertEqual(self.path.read_bytes(),self.payload)
    def test_range_ignored_restarts_partial_only(self):
        self.path.with_name(self.path.name+'.part').write_bytes(self.payload[:23])
        with patch.object(c.urllib.request,'urlopen',return_value=Response(self.payload)):
            c.verified_download('test/data.edf',self.path,self.digest,retries=1)
        self.assertEqual(self.path.read_bytes(),self.payload)
    def test_bad_checksum_quarantined(self):
        with patch.object(c.urllib.request,'urlopen',return_value=Response(b'wrong')):
            with self.assertRaises(OSError):c.verified_download('test/data.edf',self.path,self.digest,retries=1)
        self.assertFalse(self.path.exists())
        self.assertEqual(len(list(self.path.parent.glob('*.mismatch.*'))),1)
    def test_wrong_range_rejected(self):
        self.path.with_name(self.path.name+'.part').write_bytes(self.payload[:23])
        response=Response(self.payload[23:],206,{'Content-Range':f'bytes 0-{len(self.payload)-24}/{len(self.payload)}','Content-Length':str(len(self.payload)-23)})
        with patch.object(c.urllib.request,'urlopen',return_value=response):
            with self.assertRaises(OSError):c.verified_download('test/data.edf',self.path,self.digest,retries=1)
        self.assertFalse(self.path.exists())
    def test_checksum_registry(self):
        text=(self.digest+'  ./chb01/a.edf\n'+self.digest+' *chb02/b.edf\n').encode()
        parsed=c.parse_checksums(text)
        self.assertEqual(parsed['chb01/a.edf'],self.digest)
        self.assertEqual(parsed['chb02/b.edf'],self.digest)


def make_edf(path):
    n=22
    field=lambda value,size:str(value).encode().ljust(size,b' ')
    header=b''.join([field('0',8),field('unit-test',80),field('numerical-fixture',80),field('01.01.01',8),field('00.00.00',8),field(256*(n+1),8),field('',44),field(1,8),field(4,8),field(n,4)])
    fields=[(CHANNELS,16),(['']*n,80),(['uV']*n,8),([-100]*n,8),([100]*n,8),([-32768]*n,8),([32767]*n,8),(['']*n,80),([1024]*n,8),(['']*n,32)]
    data=header+b''.join(b''.join(field(v,width) for v in values) for values,width in fields)+b'\x00'*(22*1024*2)
    path.write_bytes(data)


class Metadata(unittest.TestCase):
    def test_edf_header_length_and_channels(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'fixture.edf';make_edf(p)
            info=inspect_edf(p)
            self.assertEqual(info['header_status'],'size_consistent')
            self.assertTrue(info['all_original_22_channels'])
            self.assertTrue(info['original_channels_256_hz'])
            p.write_bytes(p.read_bytes()[:-2])
            self.assertEqual(inspect_edf(p)['header_status'],'size_mismatch')
    def test_annotation_parser_refuses_missing_count(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.txt';p.write_text('File Name: a.edf\nNo count\n')
            with self.assertRaises(ValueError):parse_summary(p)
    def test_annotation_zero_and_numbered_seizures(self):
        with tempfile.TemporaryDirectory() as td:
            p=Path(td)/'s.txt'
            p.write_text('File Name: a.edf\nNumber of Seizures in File: 0\nFile Name: b.edf\nNumber of Seizures in File: 2\nSeizure 1 Start Time: 10 seconds\nSeizure 1 End Time: 20 seconds\nSeizure 2 Start Time: 25 seconds\nSeizure 2 End Time: 30 seconds\n')
            annotations,order=parse_summary(p)
            self.assertEqual(annotations['a.edf'],[])
            self.assertEqual(annotations['b.edf'],[(10.,20.),(25.,30.)])
            self.assertEqual(order['b.edf'],2)
    def test_expected_inventory_counts(self):
        rows=json.loads((c.HERE/'expected_inventory.json').read_text())
        self.assertEqual(len(rows),175)
        self.assertEqual(sum(r['in_original_analysis'] for r in rows),70)
        self.assertEqual(sum(r['public_seizure_bearing'] for r in rows),27)
        self.assertFalse(any(r['public_seizure_bearing'] for r in rows if not r['in_original_analysis']))


def scalar_original(window):
    """Independent single-window version of the archived feature formulas."""
    f,p=signal.welch(window,256,nperseg=256)
    bands=[(.5,4),(4,8),(8,13),(13,30),(30,50)]
    bp=[np.trapezoid(p[:,(f>=lo)&(f<=hi)],f[(f>=lo)&(f<=hi)],axis=-1) for lo,hi in bands]
    total=np.sum(bp,axis=0)
    values=[]
    for x in bp:values.extend([np.mean(x),np.mean(x/(total+1e-9))])
    norm=p/(np.sum(p,axis=-1,keepdims=True)+1e-9)
    values.append(np.mean(-np.sum(norm*np.log2(norm+1e-9),axis=-1)))
    v=np.var(window,axis=-1);d=np.diff(window,axis=-1);d2=np.diff(d,axis=-1)
    v1=np.var(d,axis=-1);v2=np.var(d2,axis=-1)
    mob=np.sqrt(v1/(v+1e-9));comp=np.sqrt(v2/(v1+1e-9))/(mob+1e-9)
    values.extend([np.mean(v),np.mean(mob),np.mean(comp),np.mean(np.sum(np.abs(d),axis=-1)),np.mean(np.sqrt(np.mean(window**2,axis=-1))),np.mean(v),np.mean(np.sum(np.diff(np.sign(window),axis=-1)!=0,axis=-1)/1024),np.mean(stats.skew(window,axis=-1)),np.mean(stats.kurtosis(window,axis=-1))])
    return np.nan_to_num(values,nan=0,posinf=0,neginf=0)


class Features(unittest.TestCase):
    def test_feature_formulas_match_scalar_reference(self):
        x=np.random.default_rng(17).normal(0,1e-4,(5,22,1024)).astype(np.float32)
        actual=extract_features(x)
        self.assertEqual(list(actual.columns),FEATURE_NAMES)
        expected=np.vstack([scalar_original(w) for w in x])
        np.testing.assert_allclose(actual,expected,rtol=2e-6,atol=1e-14)
    def test_batch_size_invariance(self):
        x=np.random.default_rng(42).normal(0,1e-4,(5,22,1024)).astype(np.float32)
        a=extract_features(x).to_numpy()
        b=np.vstack([extract_features(x[i:i+2]).to_numpy() for i in range(0,len(x),2)])
        np.testing.assert_array_equal(a,b)
    def test_original_duplicate_variance_preserved(self):
        x=np.random.default_rng(2).normal(0,1e-4,(2,22,1024)).astype(np.float32)
        f=extract_features(x)
        np.testing.assert_array_equal(f.hjorth_activity,f.variance)


if __name__=='__main__':unittest.main(verbosity=2)
