"""Complete only the missing xAI-VNS recordings in a separate workspace.

Standard library only. Existing project files are read-only. Downloads use
.part files, retries and SHA-256 verification against PhysioNet's registry.
No model fitting and no change to original train/calibration/test partitions.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from audit_local_collection import inspect_edf

HERE = Path(__file__).resolve().parent
BASE = 'https://physionet.org/files/chbmit/1.0.0/'
PRINT_LOCK = threading.Lock()


def log(text: str) -> None:
    with PRINT_LOCK:
        print(text, flush=True)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4 * 1024 * 1024), b''):
            h.update(data)
    return h.hexdigest()


def json_write(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=True) + '\n', encoding='utf-8')
    os.replace(tmp, path)


def csv_write(path: Path, rows: list[dict], fields=None) -> None:
    if not rows and fields is None:
        raise ValueError('Cannot determine fields for an empty CSV')
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    with tmp.open('w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    os.replace(tmp, path)


def fetch_bytes(relative: str, retries: int = 6) -> bytes:
    url = BASE + urllib.parse.quote(relative, safe='/+.')
    for attempt in range(retries):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'xAI-VNS-reproducibility/1.1', 'Accept-Encoding': 'identity'})
            with urllib.request.urlopen(request, timeout=60) as response:
                data = response.read()
            if not data:
                raise OSError('Empty response: ' + url)
            return data
        except (OSError, urllib.error.URLError) as exc:
            if attempt + 1 == retries:
                raise OSError(f'Failed to fetch {url}: {exc}') from exc
            log(f'Retrying metadata {relative}: {exc}')
            time.sleep(min(30, 2 ** (attempt + 1)))
    raise AssertionError('unreachable')


def parse_checksums(data: bytes) -> dict[str, str]:
    records = {}
    for line in data.decode('utf-8-sig').splitlines():
        match = re.match(r'^([0-9a-fA-F]{64})\s+\*?(.+?)\s*$', line)
        if match:
            name = match.group(2)
            if name.startswith('./'):
                name = name[2:]
            records[name] = match.group(1).lower()
    if not records:
        raise ValueError('No SHA-256 entries in the official checksum registry')
    return records


def snapshot(project: Path, output: Path, plan: dict) -> dict:
    """Freeze the exact uploaded inputs before any additional EEG is processed."""
    frozen = output / 'frozen_original'
    frozen.mkdir(parents=True, exist_ok=True)
    copied = {}
    for name, expected in plan['original_uploaded_input_sha256'].items():
        source = project / 'data' / 'processed' / name
        if not source.is_file():
            raise FileNotFoundError(f'Original processed input missing: {source}')
        digest = sha256(source)
        if digest != expected:
            raise ValueError(f'Original input has changed: {source}\nExpected {expected}\nFound    {digest}\nSTOP: do not regenerate or overwrite it. Send the log for reconciliation.')
        destination = frozen / name
        if destination.exists():
            if sha256(destination) != expected:
                raise ValueError(f'Frozen snapshot differs from reference: {destination}')
        else:
            shutil.copy2(source, destination)
            if sha256(destination) != expected:
                raise OSError('Snapshot verification failed: ' + str(destination))
        copied[name] = {'source': str(source), 'snapshot': str(destination), 'sha256': digest}
    for folder in ['src', 'scripts', 'configs']:
        source_folder = project / folder
        if source_folder.is_dir():
            for source in sorted(source_folder.rglob('*')):
                if not source.is_file() or source.suffix not in {'.py', '.yaml', '.yml', '.toml'}:
                    continue
                relative = source.relative_to(project)
                destination = frozen / 'project_sources' / relative
                destination.parent.mkdir(parents=True, exist_ok=True)
                if not destination.exists():
                    shutil.copy2(source, destination)
    for name in ['table1_dataset_summary.csv', 'table10_nested_calibration_split_audit.csv']:
        source = project / 'results' / 'tables' / name
        if source.is_file() and not (frozen / name).exists():
            shutil.copy2(source, frozen / name)
    plan_path = output / 'protocol.json'
    if plan_path.exists() and json.loads(plan_path.read_text()) != plan:
        raise ValueError('Workspace uses a different protocol; select a new workspace.')
    json_write(plan_path, plan)
    json_write(frozen / 'snapshot_manifest.json', copied)
    return copied


def verified_download(relative: str, destination: Path, expected: str, retries: int = 6) -> dict:
    """Resume partial content only when Content-Range is valid; never overwrite originals."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if sha256(destination) == expected:
            return {'download_status': 'already_verified', 'sha256': expected}
        raise ValueError(f'Existing completed download has a checksum mismatch: {destination}. It was not overwritten.')
    part = destination.with_name(destination.name + '.part')
    url = BASE + urllib.parse.quote(relative, safe='/+.')
    for attempt in range(retries):
        try:
            if part.exists() and sha256(part) == expected:
                os.replace(part, destination)
                return {'download_status': 'resumed_verified', 'sha256': expected}
            offset = part.stat().st_size if part.exists() else 0
            headers = {'User-Agent': 'xAI-VNS-reproducibility/1.1', 'Accept-Encoding': 'identity'}
            if offset:
                headers['Range'] = f'bytes={offset}-'
            request = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(request, timeout=90) as response:
                status = response.status
                length = int(response.headers.get('Content-Length', '0'))
                if status == 206:
                    content_range = response.headers.get('Content-Range', '')
                    match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+|\*)', content_range)
                    if not match or int(match.group(1)) != offset:
                        raise OSError(f'Invalid Content-Range: {content_range}')
                    mode = 'ab' if offset else 'wb'
                elif status == 200:
                    # The server ignored Range: safely restart the partial file, not a completed file.
                    mode = 'wb'
                    offset = 0
                else:
                    raise OSError(f'Unexpected HTTP status {status}')
                available = shutil.disk_usage(destination.parent).free
                if available < max(length + 128 * 1024**2, 256 * 1024**2):
                    raise OSError('Insufficient disk space for ' + str(destination))
                received = 0
                with part.open(mode) as f:
                    while True:
                        block = response.read(1024 * 1024)
                        if not block:
                            break
                        f.write(block)
                        received += len(block)
                if length and received != length:
                    raise OSError(f'Incomplete HTTP response: {received}/{length} bytes')
            actual = sha256(part)
            if actual != expected:
                # Preserve bad data for audit; do not append to a known-corrupt partial file.
                quarantine = part.with_name(part.name + '.mismatch.' + str(time.time_ns()))
                os.replace(part, quarantine)
                raise OSError(f'SHA-256 mismatch for {relative}; preserved at {quarantine}')
            os.replace(part, destination)
            return {'download_status': 'downloaded_verified', 'sha256': actual}
        except urllib.error.HTTPError as exc:
            if exc.code == 416 and part.exists():
                if sha256(part) == expected:
                    os.replace(part, destination)
                    return {'download_status': 'resumed_verified', 'sha256': expected}
                os.replace(part, part.with_name(part.name + '.range_error.' + str(time.time_ns())))
            if attempt + 1 == retries:
                raise
            log(f'Retry {attempt + 1}/{retries}: {relative}: {exc}')
            time.sleep(min(30, 2 ** (attempt + 1)))
        except (OSError, urllib.error.URLError, ValueError) as exc:
            if attempt + 1 == retries:
                raise
            log(f'Retry {attempt + 1}/{retries}: {relative}: {exc}')
            time.sleep(min(30, 2 ** (attempt + 1)))
    raise AssertionError('unreachable')


def read_public_metadata(output: Path, expected: list[dict]) -> dict[str, str]:
    meta = output / 'public_metadata'
    meta.mkdir(parents=True, exist_ok=True)
    # Always verify current version-1.0.0 registry against the pinned expected filenames.
    for name in ['SHA256SUMS.txt', 'RECORDS', 'RECORDS-WITH-SEIZURES']:
        data = fetch_bytes(name)
        path = meta / name
        if path.exists() and path.read_bytes() != data:
            raise ValueError(f'Official metadata changed during this run: {name}; do not silently combine versions.')
        path.write_bytes(data)
    checks = parse_checksums((meta / 'SHA256SUMS.txt').read_bytes())
    all_records = set((meta / 'RECORDS').read_text().splitlines())
    seizure_records = set((meta / 'RECORDS-WITH-SEIZURES').read_text().splitlines())
    subjects = {r['subject_id'] for r in expected}
    selected = {r for r in all_records if r.split('/')[0] in subjects and r.lower().endswith('.edf')}
    if selected != {r['relative_path'] for r in expected}:
        raise ValueError('Official recording list differs from pinned 175-file inventory.')
    for row in expected:
        rel = row['relative_path']
        if rel not in checks:
            raise ValueError('Missing official EDF checksum: ' + rel)
        if (rel in seizure_records) != bool(row['public_seizure_bearing']):
            raise ValueError('Official seizure-file list differs from inventory: ' + rel)
    for subject in sorted(subjects):
        rel = f'{subject}/{subject}-summary.txt'
        data = fetch_bytes(rel)
        if checks.get(rel) != hashlib.sha256(data).hexdigest():
            raise ValueError('Official summary checksum mismatch: ' + rel)
        path = meta / subject / f'{subject}-summary.txt'
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != data:
            raise ValueError('Summary changed during this run: ' + rel)
        path.write_bytes(data)
    return checks


def metadata_archive(output: Path) -> Path:
    dest = output / 'collection_status.zip'
    with zipfile.ZipFile(dest, 'w', zipfile.ZIP_DEFLATED) as z:
        for name in ['collection_manifest.csv', 'collection_summary.json', 'protocol.json']:
            p = output / name
            if p.exists():
                z.write(p, name)
    return dest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--workspace', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--snapshot-only', action='store_true')
    args = parser.parse_args()
    project, output = args.project.resolve(), args.workspace.resolve()
    if project == output or project in output.parents:
        parser.error('Workspace must be outside the original repository.')
    if not 1 <= args.workers <= 4:
        parser.error('Use one to four workers to avoid overloading PhysioNet.')
    output.mkdir(parents=True, exist_ok=True)
    expected = json.loads((HERE / 'expected_inventory.json').read_text())
    plan = json.loads((HERE / 'protocol.json').read_text())
    if len(expected) != 175 or sum(r['in_original_analysis'] for r in expected) != 70:
        raise ValueError('Invalid expected inventory')
    snapshot(project, output, plan)
    log('Original feature/label files verified and frozen. Original repository remains unchanged.')
    if args.snapshot_only:
        return
    checks = read_public_metadata(output, expected)
    rows, errors = [], []
    # Verify all original EDFs without re-downloading or replacing any of them.
    for entry in expected:
        if not entry['in_original_analysis']:
            continue
        rel = entry['relative_path']
        path = project / 'data' / 'external' / 'chbmit' / rel
        row = dict(entry, path=str(path), download_status='original_read_only')
        if not path.is_file():
            row.update(checksum_ok=False, error='Original EDF missing')
            errors.append(rel + ': original EDF missing')
        else:
            actual = sha256(path)
            row.update(sha256=actual, checksum_ok=(actual == checks[rel]), **{k:v for k,v in inspect_edf(path).items() if k not in {'path','edf_file'}})
            if not row['checksum_ok']:
                errors.append(rel + ': ORIGINAL checksum mismatch; not overwritten')
        rows.append(row)
    if errors:
        csv_write(output / 'collection_manifest.csv', rows)
        json_write(output / 'collection_summary.json', {'complete':False, 'errors':errors})
        log('STOP: ' + str(metadata_archive(output)))
        raise RuntimeError('\n'.join(errors))
    log('All 70 original EDFs passed official SHA-256 verification.')
    additions = [r for r in expected if not r['in_original_analysis']]
    log(f'Downloading/verifying {len(additions)} added EDFs to {output / "additional_edfs"}')
    def work(entry):
        rel = entry['relative_path']
        path = output / 'additional_edfs' / rel
        row = dict(entry, path=str(path))
        try:
            row.update(verified_download(rel, path, checks[rel]))
            row['checksum_ok'] = True
            row.update({k:v for k,v in inspect_edf(path).items() if k not in {'path','edf_file'}})
        except Exception as exc:
            row.update(download_status='FAILED', checksum_ok=False, error=str(exc))
        return row
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(work, e) for e in additions]
        for completed, future in enumerate(as_completed(futures), start=1):
            row = future.result()
            rows.append(row)
            log(f'[{completed:3d}/105] {row["edf_file"]}: {row["download_status"]}')
            csv_write(output / 'collection_manifest.csv', sorted(rows, key=lambda r:r['relative_path']))
    rows.sort(key=lambda r:r['relative_path'])
    failed = [r['relative_path'] for r in rows if not r.get('checksum_ok')]
    incompatible = [r['relative_path'] for r in rows if r.get('checksum_ok') and not (r.get('header_status')=='size_consistent' and r.get('all_original_22_channels') and r.get('original_channels_256_hz'))]
    summary = {'time_utc':datetime.now(timezone.utc).isoformat(), 'complete':len(rows)==175 and not failed,
               'expected':175, 'original':70, 'added':105, 'sha256_verified':sum(bool(r.get('checksum_ok')) for r in rows),
               'download_or_checksum_failures':failed, 'header_or_channel_incompatible':incompatible,
               'note':'Compatibility failures are explicit; no channel set is silently changed. No model has been fit or evaluated in this stage.'}
    json_write(output / 'collection_summary.json', summary)
    metadata_archive(output)
    log(json.dumps(summary, indent=2))
    if failed:
        raise RuntimeError('Collection incomplete. Rerun the SAME command to resume; do not delete partial downloads.')
    log('COLLECTION COMPLETE. Next stage: verify feature extraction against original cached windows.')


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        log('Interrupted. Completed and partial downloads retained. Rerun to resume.')
        sys.exit(130)
    except Exception as exc:
        log(f'ERROR: {exc}')
        sys.exit(1)
