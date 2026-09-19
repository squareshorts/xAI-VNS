"""Read-only inventory of the five original xAI-VNS CHB-MIT cases.

Python 3.9+; standard library only. Reads EDF headers, not EEG samples.
Does not download, move, delete, or modify EEG files. Header/length consistency
is not a checksum or signal-quality validation.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from zipfile import ZipFile, ZIP_DEFLATED

CHANNELS = 'FP1-F7;F7-T7;T7-P7;P7-O1;FP1-F3;F3-C3;C3-P3;P3-O1;FP2-F4;F4-C4;C4-P4;P4-O2;FP2-F8;F8-T8;T8-P8;P8-O2;FZ-CZ;CZ-PZ;P7-T7;T7-FT9;FT9-FT10;FT10-T8'.split(';')
PATTERN = re.compile(r'^chb(?:01|02|03|05|08)_.+\.edf$', re.I)
SKIP_DIRS = {'.git', '.venv', 'venv', 'node_modules', '__pycache__'}


def inspect_edf(path: Path) -> dict:
    row = {'edf_file': path.name.lower(), 'path': str(path.resolve()),
           'file_bytes': '', 'header_status': 'unreadable', 'duration_seconds': '',
           'all_original_22_channels': False, 'missing_original_channels': '',
           'original_channels_256_hz': False, 'header_error': ''}
    try:
        row['file_bytes'] = path.stat().st_size
        with path.open('rb') as stream:
            head = stream.read(256)
            if len(head) != 256 or head[:8].decode('ascii').strip() != '0':
                raise ValueError('Not a complete standard EDF fixed header')
            header_bytes = int(head[184:192])
            n_records = int(head[236:244])
            record_seconds = float(head[244:252])
            ns = int(head[252:256])
            if not (1 <= ns <= 1024) or header_bytes != 256 * (ns + 1) or record_seconds <= 0:
                raise ValueError('Invalid EDF header dimensions')
            signals = stream.read(header_bytes - 256)
            if len(signals) != header_bytes - 256:
                raise ValueError('Incomplete signal headers')
        labels = [re.sub(r'-\d+$', '', signals[i*16:(i+1)*16].decode('ascii').strip().upper().replace(' ', '')) for i in range(ns)]
        samples = [int(signals[216*ns+i*8:216*ns+(i+1)*8]) for i in range(ns)]
        if any(n <= 0 for n in samples):
            raise ValueError('Invalid sample count')
        missing = [c for c in CHANNELS if c not in labels]
        row['all_original_22_channels'] = not missing
        row['missing_original_channels'] = ';'.join(missing)
        row['original_channels_256_hz'] = not missing and all(abs(samples[labels.index(c)]/record_seconds - 256.0) < 1e-9 for c in CHANNELS)
        if n_records < 0:
            row['header_status'] = 'record_count_unknown'
        else:
            expected_bytes = header_bytes + n_records * 2 * sum(samples)
            row['duration_seconds'] = n_records * record_seconds
            row['header_status'] = 'size_consistent' if expected_bytes == row['file_bytes'] else 'size_mismatch'
            if row['header_status'] == 'size_mismatch':
                row['header_error'] = f'Header implies {expected_bytes} bytes; actual {row["file_bytes"]}'
    except (OSError, ValueError, UnicodeError, OverflowError) as exc:
        row['header_error'] = str(exc)
    return row


def csv_write(path: Path, rows: list[dict], fields: list[str] | None = None) -> None:
    fields = fields or list(rows[0])
    with path.open('w', newline='', encoding='utf-8-sig') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', action='append', type=Path, help='Folder to search recursively; repeat for other drives. Default C:\\work.')
    ap.add_argument('--output', type=Path, default=Path('local_collection_report'))
    args = ap.parse_args()
    roots = args.root or [Path(r'C:\work')]
    for root in roots:
        if not root.is_dir():
            ap.error(f'Search folder does not exist: {root}')
    expected = json.loads((Path(__file__).parent/'expected_inventory.json').read_text(encoding='utf-8'))
    found, scan_errors, seen_paths = [], [], set()
    for root in roots:
        print(f'Scanning {root} ...', flush=True)
        for current, dirs, files in os.walk(root, onerror=lambda err: scan_errors.append(str(err)), followlinks=False):
            dirs[:] = [d for d in dirs if d not in SKIP_DIRS]
            for filename in files:
                if not PATTERN.match(filename):
                    continue
                path = (Path(current)/filename).resolve()
                if str(path).lower() in seen_paths:
                    continue
                seen_paths.add(str(path).lower())
                found.append(inspect_edf(path))
    by_name = defaultdict(list)
    for row in found:
        by_name[row['edf_file']].append(row)
    ledger = []
    for entry in expected:
        copies = by_name[entry['edf_file']]
        complete = [r for r in copies if r['header_status']=='size_consistent']
        compatible = [r for r in complete if r['all_original_22_channels'] and r['original_channels_256_hz']]
        ledger.append({'subject_id':entry['subject_id'], 'edf_file':entry['edf_file'],
          'originally_analyzed':entry['in_original_analysis'], 'seizure_bearing':entry['public_seizure_bearing'],
          'copies_found':len(copies), 'size_consistent_copies':len(complete), 'compatible_copies':len(compatible),
          'local_status':'compatible_header_and_length' if compatible else ('present_needs_check' if copies else 'not_found_in_searched_roots'),
          'compatible_path':compatible[0]['path'] if compatible else '', 'all_paths':' | '.join(r['path'] for r in copies)})
    summary = []
    for subject in sorted({r['subject_id'] for r in ledger}):
        group = [r for r in ledger if r['subject_id']==subject]
        summary.append({'subject_id':subject, 'expected_edfs':len(group),
                        'unique_present':sum(r['copies_found']>0 for r in group),
                        'compatible_header_and_length':sum(r['compatible_copies']>0 for r in group),
                        'not_found':sum(r['copies_found']==0 for r in group),
                        'present_needs_check':sum(r['copies_found']>0 and r['compatible_copies']==0 for r in group)})
    out=args.output.resolve(); out.mkdir(parents=True,exist_ok=True)
    csv_write(out/'expected_file_status.csv',ledger)
    csv_write(out/'subject_summary.csv',summary)
    csv_write(out/'all_local_copies.csv',found, list(inspect_edf(Path('__nonexistent__'))))
    csv_write(out/'files_needing_resolution.csv',[r for r in ledger if r['compatible_copies']==0],list(ledger[0]))
    (out/'scan_metadata.json').write_text(json.dumps({'time_utc':datetime.now(timezone.utc).isoformat(),
      'roots':[str(r.resolve()) for r in roots], 'scan_errors':scan_errors,
      'notes':['No raw EEG files modified or copied.', 'Header/size consistency is not checksum verification or EEG signal-quality screening.',
               'Original status is from the May 2026 analysis manifest; present status refers only to this scan.',
               'No independent copy check: multiple size-consistent files may still differ in content.']},indent=2),encoding='utf-8')
    for row in summary:
        print('{subject_id}: expected {expected_edfs}; found {unique_present}; header-compatible {compatible_header_and_length}; missing {not_found}; check {present_needs_check}'.format(**row))
    archive=out.with_suffix('.zip')
    with ZipFile(archive,'w',ZIP_DEFLATED) as z:
        for name in ['expected_file_status.csv','subject_summary.csv','all_local_copies.csv','files_needing_resolution.csv','scan_metadata.json']:
            z.write(out/name,arcname=name)
    print(f'\nUpload this metadata-only report: {archive}')
    if scan_errors:
        print(f'WARNING: {len(scan_errors)} folders/files could not be scanned; see scan_metadata.json.')


if __name__=='__main__':
    main()
