"""Verify the original feature implementation and export only derived data.

Requires NumPy, SciPy, pandas, MNE and pyarrow. No fitting, no label-dependent
selection of added EDFs. Original EEG, feature and split files remain read-only.
"""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import re
import shutil
import sys
import traceback
import warnings
import zipfile
import numpy as np
import pandas as pd
import mne
from audit_local_collection import CHANNELS
from complete_collection import HERE, log, sha256, json_write, csv_write
from feature_core import FEATURE_NAMES, extract_features


def canonical_channel(name):
    return re.sub(r'-\d+$', '', str(name).strip().upper().replace(' ', ''))


def open_recording(path):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', RuntimeWarning)
        raw = mne.io.read_raw_edf(path, preload=False, verbose='ERROR')
    first = {}
    for i, name in enumerate(raw.ch_names):
        first.setdefault(canonical_channel(name), i)
    missing = [name for name in CHANNELS if name not in first]
    if missing:
        raw.close()
        raise ValueError('Original channel set unavailable: ' + ';'.join(missing))
    if raw.info['sfreq'] != 256:
        raw.close()
        raise ValueError(f'Unexpected MNE sampling rate: {raw.info["sfreq"]}')
    return raw, [first[name] for name in CHANNELS]


def read_windows(raw, picks, indices):
    # Use the original volts-to-float32 conversion, before feature computation.
    return np.stack([raw.get_data(picks=picks, start=int(i)*512, stop=int(i)*512+1024).astype(np.float32)
                     for i in indices])


def parse_summary(path):
    output = {}
    order = {}
    text = path.read_text(encoding='utf-8', errors='strict')
    for rank, block in enumerate(text.split('File Name: ')[1:], 1):
        filename = block.splitlines()[0].strip()
        match = re.search(r'Number of Seizures in File:\s*(\d+)', block)
        if not match:
            raise ValueError(f'No explicit seizure count for {filename}')
        count = int(match.group(1))
        starts = [float(x) for x in re.findall(r'Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds?', block)]
        ends = [float(x) for x in re.findall(r'Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds?', block)]
        if len(starts) != count or len(ends) != count or any(b < a for a,b in zip(starts,ends)):
            raise ValueError(f'Invalid seizure annotations for {filename}')
        if filename in output:
            raise ValueError('Duplicate summary filename: ' + filename)
        output[filename] = list(zip(starts, ends))
        order[filename] = rank
    return output, order


def atomic_csv_gz(frame, path):
    temp = path.with_name(path.name + '.tmp')
    frame.to_csv(temp, index=False, compression={'method':'gzip', 'mtime':0}, float_format='%.17g')
    os.replace(temp, path)


def copy_gzip(source, destination):
    temp = destination.with_name(destination.name + '.tmp')
    with source.open('rb') as src, temp.open('wb') as fout, gzip.GzipFile(filename='', mode='wb', fileobj=fout, mtime=0) as dst:
        shutil.copyfileobj(src, dst, length=4*1024*1024)
    os.replace(temp, destination)


def run(args):
    work = args.workspace.resolve()
    plan = json.loads((work / 'protocol.json').read_text())
    frozen = work / 'frozen_original'
    out = work / 'derived_inputs'
    out.mkdir(parents=True, exist_ok=True)
    summaries = json.loads((work / 'collection_summary.json').read_text())
    if not summaries['complete'] or summaries['sha256_verified'] != 175:
        raise RuntimeError('Collection is incomplete or not checksum-verified; rerun acquisition first.')
    ledger = pd.read_csv(work / 'collection_manifest.csv', keep_default_na=False)
    for name, expected in plan['original_uploaded_input_sha256'].items():
        if sha256(frozen / name) != expected:
            raise RuntimeError('Original frozen input changed: ' + name)
    features = pd.read_parquet(frozen / 'chbmit_features.parquet')
    if features['window_id'].duplicated().any():
        raise ValueError('Original feature table contains duplicate window IDs')
    if len(features) != 122007 or int(features.binary_label.sum()) != 5227:
        raise ValueError('Original feature counts do not match the audited data')
    if any(c not in features for c in FEATURE_NAMES):
        raise ValueError('Original feature schema does not match the frozen extractor')
    expected_keys = set(ledger.loc[ledger.in_original_analysis.astype(str).str.lower().eq('true'), 'relative_path'])
    actual_keys = set(features.subject_id + '/' + features.edf_file)
    if actual_keys != expected_keys:
        raise ValueError('Original feature EDF identities do not match the 70-file manifest')
    by_path = ledger.set_index('relative_path')
    version_names = ['numpy','scipy','pandas','mne','pyarrow','lightgbm','scikit-learn']
    versions = {}
    for name in version_names:
        try: versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError: versions[name] = 'not_installed'
    json_write(out / 'execution_environment.json', {'python':sys.version, 'executable':sys.executable, 'packages':versions})
    source_digest = sha256(HERE / 'feature_core.py')
    checks = []
    rtol, atol = plan['feature_validation']['rtol'], plan['feature_validation']['atol']
    # All original EDFs contribute fixed first/middle/last validation windows.
    # Their indices are chosen independently of labels, feature values and model scores.
    for count, ((subject, edf), group) in enumerate(features.groupby(['subject_id','edf_file'], sort=True), 1):
        group = group.sort_values('window_index')
        sample = group.iloc[sorted(set([0,len(group)//2,len(group)-1]))]
        path = Path(by_path.loc[subject + '/' + edf, 'path'])
        raw, picks = open_recording(path)
        try:
            windows = read_windows(raw, picks, sample.window_index.to_numpy())
        finally:
            raw.close()
        actual = extract_features(windows).to_numpy(dtype=float)
        target = sample[FEATURE_NAMES].to_numpy(dtype=float)
        close = np.isclose(actual, target, rtol=rtol, atol=atol, equal_nan=False)
        for i, item in enumerate(sample.itertuples(index=False)):
            for j, feature in enumerate(FEATURE_NAMES):
                checks.append({'subject_id':subject, 'edf_file':edf, 'window_id':item.window_id,
                               'feature':feature, 'cached':target[i,j], 'recomputed':actual[i,j],
                               'absolute_error':abs(actual[i,j]-target[i,j]), 'pass':bool(close[i,j])})
        log(f'Original feature check [{count:2d}/70]: {edf}: {"PASS" if close.all() else "FAIL"}')
    check_frame = pd.DataFrame(checks)
    check_frame.to_csv(out / 'original_feature_reproduction_checks.csv', index=False, float_format='%.17g')
    if not check_frame['pass'].all():
        with zipfile.ZipFile(work / 'feature_reproduction_failure.zip','w',zipfile.ZIP_DEFLATED) as z:
            for name in ['original_feature_reproduction_checks.csv','execution_environment.json']:
                z.write(out / name, name)
        raise RuntimeError('Feature reproduction failed. STOP: no added features were accepted. Send feature_reproduction_failure.zip; do not relax tolerances.')
    log('All 210 reference windows passed the fixed feature tolerances.')
    annotations, full_order = {}, {}
    events = []
    for subject in plan['subjects']:
        annotations[subject], full_order[subject] = parse_summary(work / 'public_metadata' / subject / f'{subject}-summary.txt')
    file_status = []
    additions = []
    cache = work / 'feature_cache_additional'
    cache.mkdir(exist_ok=True)
    extra_rows = ledger[~ledger.in_original_analysis.astype(str).str.lower().eq('true')]
    if len(extra_rows) != 105:
        raise ValueError('Additional EDF count is not 105')
    # Audit annotation membership for every file, including originals.
    for entry in ledger.itertuples(index=False):
        if entry.edf_file not in annotations[entry.subject_id]:
            raise ValueError('EDF is not explicitly annotated in the official summary: ' + entry.relative_path)
        intervals = annotations[entry.subject_id][entry.edf_file]
        if bool(intervals) != (str(entry.public_seizure_bearing).lower()=='true'):
            raise ValueError('Seizure-file list/summary disagreement: ' + entry.relative_path)
        for i,(start,end) in enumerate(intervals,1):
            events.append({'subject_id':entry.subject_id,'edf_file':entry.edf_file,'event_index':i,'seizure_onset':start,'seizure_offset':end})
    if len(events) != 27:
        raise ValueError('Expected 27 annotated seizures across these cases, found ' + str(len(events)))
    for count, entry in enumerate(extra_rows.itertuples(index=False), 1):
        status = {'subject_id':entry.subject_id,'edf_file':entry.edf_file,'duration_seconds':entry.duration_seconds,
                  'sha256':entry.sha256,'status':'pending','reason':'','windows':0}
        if not (entry.header_status=='size_consistent' and str(entry.all_original_22_channels).lower()=='true' and str(entry.original_channels_256_hz).lower()=='true'):
            status.update(status='technically_ineligible_original_montage', reason='Header, length, or original 22 channels at 256 Hz unavailable: ' + str(entry.missing_original_channels))
            file_status.append(status)
            log(f'[{count:3d}/105] {entry.edf_file}: EXPLICIT TECHNICAL EXCLUSION; see manifest.')
            continue
        filename = entry.subject_id + '__' + entry.edf_file + '.csv.gz'
        cached_file, cached_meta = cache / filename, cache / (filename + '.json')
        try:
            if cached_file.exists() and cached_meta.exists():
                meta = json.loads(cached_meta.read_text())
                if meta['edf_sha256']==entry.sha256 and meta['extractor_sha256']==source_digest and meta['environment']==versions and meta['csv_sha256']==sha256(cached_file):
                    table = pd.read_csv(cached_file)
                    if len(table) != meta['windows']:
                        raise ValueError('Cached row count mismatch')
                    status.update(status='cached_verified',windows=len(table))
                    additions.append(table)
                    file_status.append(status)
                    log(f'[{count:3d}/105] {entry.edf_file}: cached, {len(table):,} windows')
                    continue
                raise ValueError('Feature cache provenance differs. Use a fresh workspace; do not overwrite silently.')
            path = Path(entry.path)
            if sha256(path) != entry.sha256:
                raise ValueError('EDF changed after acquisition')
            raw, picks = open_recording(path)
            try:
                duration = raw.n_times / 256.0
                n = (raw.n_times - 1024)//512 + 1
                if n <= 0:
                    raise ValueError('EDF shorter than one four-second window')
                chunks = []
                for start in range(0,n,args.batch_windows):
                    stop = min(n,start+args.batch_windows)
                    # One contiguous read per batch; reproduces original per-window float32 samples.
                    block = raw.get_data(picks=picks,start=start*512,stop=(stop-1)*512+1024).astype(np.float32)
                    windows = np.stack([block[:,i*512:i*512+1024] for i in range(stop-start)])
                    chunks.append(extract_features(windows))
                values = pd.concat(chunks,ignore_index=True)
            finally:
                raw.close()
            indices = np.arange(n)
            centers = indices*2.0+2.0
            labels = np.zeros(n,dtype=np.int8)
            for a,b in annotations[entry.subject_id][entry.edf_file]:
                labels[(centers>=a-300)&(centers<a)&(labels!=2)] = 1
                labels[(centers>=a)&(centers<=b)] = 2
            table = pd.DataFrame({
                'window_id':[f'{entry.subject_id}|{entry.edf_file}|{i:06d}' for i in indices],
                'subject_id':entry.subject_id,'edf_file':entry.edf_file,
                'recording_order':full_order[entry.subject_id][entry.edf_file],
                'window_index':indices,'window_start':indices*2.0,'window_end':indices*2.0+4.0,
                'window_center':centers,'label':labels,'binary_label':(labels>0).astype(int),
                'split':'extended_test_only','evaluation_mode':'held_out_subject_recording_extension',
                'sampling_rate':256.0,'duration_seconds':duration,'channel_count':22,
                'collection':'additional',
            })
            table = pd.concat([table,values],axis=1)
            atomic_csv_gz(table,cached_file)
            json_write(cached_meta,{'edf_sha256':entry.sha256,'extractor_sha256':source_digest,'environment':versions,
                                    'windows':len(table),'csv_sha256':sha256(cached_file)})
            status.update(status='extracted_verified',duration_seconds=duration,windows=n)
            additions.append(table)
            log(f'[{count:3d}/105] {entry.edf_file}: {n:,} windows extracted')
        except Exception as exc:
            status.update(status='FAILED',reason=str(exc))
            log(f'[{count:3d}/105] {entry.edf_file}: FAILED: {exc}')
        file_status.append(status)
        csv_write(out/'additional_feature_manifest.csv',file_status)
    csv_write(out/'additional_feature_manifest.csv',file_status)
    failures = [r for r in file_status if r['status']=='FAILED']
    if failures:
        raise RuntimeError(f'{len(failures)} files failed feature extraction. Check additional_feature_manifest.csv and resume; no final success bundle generated.')
    if not additions:
        raise RuntimeError('No additional recordings compatible with original feature definition')
    additional = pd.concat(additions,ignore_index=True)
    if additional.window_id.duplicated().any() or set(additional.window_id)&set(features.window_id):
        raise ValueError('Overlapping or duplicate original/additional window IDs')
    atomic_csv_gz(additional,out/'additional_features.csv.gz')
    # Portable exports allow reanalysis without rereading EDFs or altering original caches.
    atomic_csv_gz(features,out/'original_features.csv.gz')
    copy_gzip(frozen/'chbmit_window_labels.csv',out/'original_window_labels.csv.gz')
    pd.DataFrame(events).to_csv(out/'event_manifest.csv',index=False)
    shutil.copy2(work/'collection_manifest.csv',out/'collection_manifest.csv')
    shutil.copy2(work/'protocol.json',out/'protocol.json')
    shutil.copy2(frozen/'snapshot_manifest.json',out/'snapshot_manifest.json')
    original_hours = features.drop_duplicates(['subject_id','edf_file']).duration_seconds.sum()/3600
    additional_hours = additional.drop_duplicates(['subject_id','edf_file']).duration_seconds.sum()/3600
    summary = {'time_utc':datetime.now(timezone.utc).isoformat(),
               'stage':'Verified feature inputs only; no added-recording model results yet.',
               'original_edfs':70,'original_hours':float(original_hours),'original_windows':len(features),
               'added_expected_edfs':105,'added_usable_edfs':len(additions),'added_hours':float(additional_hours),
               'added_windows':len(additional),'complete_usable_edfs':70+len(additions),
               'technical_exclusions':[r for r in file_status if r['status']=='technically_ineligible_original_montage'],
               'all_175_usable':70+len(additions)==175,
               'reference_windows_checked':int(check_frame.window_id.nunique()),'feature_comparisons':len(check_frame),
               'reference_pass':bool(check_frame['pass'].all()),
               'record_local_annotation_caveat':'Same original record-local five-minute labeling convention; no preictal labels propagated across EDF boundaries. Seizure-free added files are not proof of physiologically interictal EEG.',
               'sources':{'feature_core_sha256':source_digest,'pinned_commit':plan['source_commit']}}
    json_write(out/'preparation_summary.json',summary)
    checksums = {p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()}
    json_write(out/'derived_checksums.json',checksums)
    archive = work/'xai_vns_complete_recording_inputs.zip'
    tmp = archive.with_name(archive.name+'.tmp')
    with zipfile.ZipFile(tmp,'w',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for p in sorted(out.iterdir()):
            if p.is_file():z.write(p,'derived_inputs/'+p.name)
        for p in sorted((work/'public_metadata').rglob('*')):
            if p.is_file():z.write(p,'public_metadata/'+str(p.relative_to(work/'public_metadata')).replace('\\','/'))
        for p in sorted(HERE.glob('*.py')):z.write(p,'preparation_code/'+p.name)
        for p in sorted((HERE/'reference').glob('*')):
            if p.is_file():z.write(p,'reference/'+p.name)
    os.replace(tmp,archive)
    log(json.dumps(summary,indent=2))
    log(f'\nUPLOAD THIS DERIVED-DATA BUNDLE (no EDFs):\n{archive}')
    if summary['technical_exclusions']:
        log('Some EDFs are technically incompatible. Report them; do not claim all 175 were evaluated.')


if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--workspace',type=Path,required=True)
    ap.add_argument('--batch-windows',type=int,default=256)
    args=ap.parse_args()
    if not 1 <= args.batch_windows <= 2048:ap.error('batch-windows must be in 1..2048')
    try:
        run(args)
    except Exception as exc:
        log(f'ERROR: {exc}')
        traceback.print_exc()
        sys.exit(1)
