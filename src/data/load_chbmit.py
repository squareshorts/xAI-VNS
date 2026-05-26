from __future__ import annotations

import re
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import mne
import pandas as pd

from src.utils.logging_utils import setup_logger
from src.utils.paths import REPORTS_DIR, TABLES_DIR

logger = setup_logger("load_chbmit")


def normalize_subject_id(subject: str | int) -> str:
    """Return CHB-MIT subject IDs in chbXX form."""
    value = str(subject).strip().lower()
    digits = "".join(re.findall(r"\d+", value))
    if digits:
        return f"chb{int(digits):02d}"
    if value.startswith("chb"):
        return value
    return f"chb{value}"


def natural_edf_key(filename: str) -> Tuple:
    """Sort CHB-MIT EDF names in recording order, including names like chb02_16+."""
    stem = Path(filename).stem.lower()
    parts = re.split(r"(\d+)", stem)
    return tuple(int(part) if part.isdigit() else part for part in parts)


def canonical_channel_name(name: str) -> str:
    """Normalize MNE/EDF channel labels and collapse MNE duplicate suffixes."""
    channel = str(name).strip().upper().replace(" ", "")
    # MNE makes duplicate EDF channel names unique by appending -0, -1, ...
    return re.sub(r"-\d+$", "", channel)


def parse_chbmit_summary(summary_path: Path) -> Dict[str, List[Tuple[int, int]]]:
    """Parse a CHB-MIT summary file into seizure intervals by EDF filename."""
    seizures_by_file: Dict[str, List[Tuple[int, int]]] = {}

    if not summary_path.exists():
        logger.warning("Summary file not found: %s", summary_path)
        return seizures_by_file

    content = summary_path.read_text(encoding="utf-8", errors="ignore")
    blocks = content.split("File Name: ")

    for block in blocks[1:]:
        lines = [line.strip() for line in block.splitlines() if line.strip()]
        if not lines:
            continue

        filename = lines[0]
        seizures_by_file[filename] = []

        count_match = re.search(r"Number of Seizures in File:\s*(\d+)", block)
        if not count_match:
            logger.warning("No seizure count found for %s in %s", filename, summary_path)
            continue

        seizure_count = int(count_match.group(1))
        if seizure_count == 0:
            continue

        starts = [
            int(value)
            for value in re.findall(
                r"Seizure(?:\s+\d+)?\s+Start Time:\s*(\d+)\s*seconds?", block
            )
        ]
        ends = [
            int(value)
            for value in re.findall(
                r"Seizure(?:\s+\d+)?\s+End Time:\s*(\d+)\s*seconds?", block
            )
        ]

        if len(starts) != seizure_count or len(ends) != seizure_count:
            logger.warning(
                "Summary annotation mismatch for %s: expected %s seizures, parsed %s starts and %s ends",
                filename,
                seizure_count,
                len(starts),
                len(ends),
            )

        seizures_by_file[filename] = list(zip(starts, ends))

    return seizures_by_file


def _subjects_from_config(config: Dict | None, fallback_subject: str | None) -> List[str]:
    if config and config.get("subjects"):
        subjects = config["subjects"]
        if isinstance(subjects, (str, int)):
            subjects = [subjects]
        return [normalize_subject_id(subject) for subject in subjects]

    if fallback_subject:
        return [normalize_subject_id(fallback_subject)]

    return ["chb01"]


def _ordered_present_edfs(subject_dir: Path, summary_order: Iterable[str]) -> List[str]:
    present = {path.name for path in subject_dir.glob("*.edf")}
    ordered = [filename for filename in summary_order if filename in present]
    extras = sorted(present.difference(ordered), key=natural_edf_key)
    if extras:
        logger.warning(
            "%s EDF files are present but not listed in the summary and will be treated as unannotated: %s",
            len(extras),
            ", ".join(extras),
        )
    return ordered + extras


def _channel_inventory(raw_channel_names: List[str]) -> Dict:
    canonical = [canonical_channel_name(channel) for channel in raw_channel_names]
    counts = Counter(canonical)
    seen: set[str] = set()

    kept_channels: List[str] = []
    duplicate_channels: List[str] = []
    original_to_canonical: Dict[str, str] = {}

    for original, channel in zip(raw_channel_names, canonical):
        original_to_canonical[original] = channel
        if channel in seen:
            duplicate_channels.append(f"{original}->{channel}")
            continue
        seen.add(channel)
        kept_channels.append(channel)

    return {
        "kept_channels": kept_channels,
        "duplicate_channels": duplicate_channels,
        "duplicate_bases": sorted(name for name, count in counts.items() if count > 1),
        "original_to_canonical": original_to_canonical,
    }


def _read_edf_metadata(edf_path: Path) -> Dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose=False)

    inventory = _channel_inventory(raw.info["ch_names"])
    sfreq = float(raw.info["sfreq"])
    duration_sec = float(raw.n_times / sfreq)

    return {
        "sfreq": sfreq,
        "duration_seconds": duration_sec,
        "raw_channel_names": list(raw.info["ch_names"]),
        **inventory,
    }


def _select_common_channels(recordings: List[Dict], configured_channels: Iterable[str] | None) -> List[str]:
    channel_sets = [set(recording["available_channels"]) for recording in recordings]
    if not channel_sets:
        return []

    if configured_channels:
        requested = [canonical_channel_name(channel) for channel in configured_channels]
        missing = [
            channel for channel in requested if any(channel not in channel_set for channel_set in channel_sets)
        ]
        if missing:
            logger.warning(
                "Configured channels not present in every EDF and excluded: %s",
                ", ".join(sorted(set(missing))),
            )
        common = [channel for channel in requested if channel not in missing]
    else:
        first_order = recordings[0]["available_channels"]
        common = [channel for channel in first_order if all(channel in channel_set for channel_set in channel_sets)]

    if not common:
        raise ValueError("No common EEG channel set could be found across the selected EDF files.")

    return common


def _write_channel_report(recordings: List[Dict], common_channels: List[str], subjects: List[str]) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    report_path = REPORTS_DIR / "channel_selection_real_eeg.md"

    duplicate_counter: Counter[str] = Counter()
    excluded_counter: Counter[str] = Counter()
    edf_counts: Counter[str] = Counter(recording["subject_id"] for recording in recordings)

    for recording in recordings:
        duplicate_counter.update(recording["duplicate_bases"])
        excluded_counter.update(recording["excluded_channels"])

    lines = [
        "# Channel Selection for Real CHB-MIT EEG",
        "",
        "Local EDF files only were used. No synthetic EEG and no downloaded files were included.",
        "",
        "## Subjects",
    ]
    for subject in subjects:
        lines.append(f"- {subject}: {edf_counts[subject]} EDF files")

    lines.extend(
        [
            "",
            "## Harmonization Rule",
            "- Channel labels were normalized by trimming whitespace, uppercasing, and removing MNE duplicate suffixes such as `-0` and `-1`.",
            "- When an EDF contained duplicate channel labels, the first occurrence was retained and later duplicate occurrences were excluded.",
            "- The final feature matrix uses the common channel set present in every selected EDF file.",
            "",
            f"## Final Common Channel Set ({len(common_channels)} channels)",
        ]
    )
    lines.extend(f"{idx}. `{channel}`" for idx, channel in enumerate(common_channels, start=1))

    lines.append("")
    lines.append("## Duplicate Channels Removed")
    if duplicate_counter:
        for channel, count in sorted(duplicate_counter.items()):
            lines.append(f"- `{channel}`: duplicate occurrence removed in {count} EDF files")
    else:
        lines.append("- None")

    lines.append("")
    lines.append("## Channels Excluded From Final Common Set")
    if excluded_counter:
        for channel, count in sorted(excluded_counter.items()):
            lines.append(f"- `{channel}`: excluded in {count} EDF files")
    else:
        lines.append("- None beyond duplicate-channel removal; all unique channels were common.")

    lines.append("")
    lines.append("## Per-Recording Notes")
    for recording in recordings:
        duplicates = ", ".join(recording["duplicate_channels"]) or "none"
        excluded = ", ".join(recording["excluded_channels"]) or "none"
        lines.append(
            f"- {recording['subject_id']} `{recording['edf_file']}`: duplicates removed = {duplicates}; excluded = {excluded}"
        )

    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved channel selection report to %s", report_path)


def load_chbmit_data(
    data_dir: Path,
    subject: str | None = None,
    train_files: List[str] | None = None,
    test_files: List[str] | None = None,
    config: Dict | None = None,
) -> Dict:
    """
    Load local CHB-MIT metadata for one or more subjects.

    Only EDF files physically present under data/external/chbmit/chbXX are analyzed.
    Summary entries for absent EDF files are ignored rather than fabricated.
    """
    del train_files, test_files

    subjects = _subjects_from_config(config, subject)
    data_dir = Path(data_dir)

    recordings: List[Dict] = []
    summary_records: List[Dict] = []

    for subject_id in subjects:
        subject_dir = data_dir / subject_id
        summary_path = subject_dir / f"{subject_id}-summary.txt"

        if not subject_dir.exists():
            raise FileNotFoundError(f"Missing local subject folder: {subject_dir}")
        if not summary_path.exists():
            raise FileNotFoundError(f"Missing local CHB-MIT summary file: {summary_path}")

        seizure_annotations = parse_chbmit_summary(summary_path)
        ordered_edfs = _ordered_present_edfs(subject_dir, seizure_annotations.keys())

        if not ordered_edfs:
            logger.warning("No local EDF files found for %s; subject skipped.", subject_id)
            continue

        missing_from_local = sorted(set(seizure_annotations).difference(ordered_edfs), key=natural_edf_key)
        if missing_from_local:
            logger.info(
                "%s summary-listed EDF files for %s are absent locally and were skipped.",
                len(missing_from_local),
                subject_id,
            )

        for recording_order, filename in enumerate(ordered_edfs, start=1):
            edf_path = subject_dir / filename
            seizures = seizure_annotations.get(filename, [])

            try:
                metadata = _read_edf_metadata(edf_path)
            except Exception as exc:
                logger.error("Error reading EDF metadata for %s: %s", edf_path, exc)
                continue

            recording = {
                "subject_id": subject_id,
                "edf_file": filename,
                "edf_path": str(edf_path),
                "recording_order": recording_order,
                "seizures": seizures,
                "seizure_count": len(seizures),
                "seizure_onsets": [start for start, _ in seizures],
                "seizure_offsets": [end for _, end in seizures],
                "summary_path": str(summary_path),
                "available_channels": metadata["kept_channels"],
                **metadata,
            }
            recordings.append(recording)

    if not recordings:
        raise ValueError("No local EDF metadata could be loaded. Cannot run real EEG analysis.")

    common_channels = _select_common_channels(recordings, (config or {}).get("channels"))

    for recording in recordings:
        recording["common_channels"] = common_channels
        recording["channels_used"] = len(common_channels)
        recording["excluded_channels"] = [
            channel for channel in recording["available_channels"] if channel not in common_channels
        ]

        seizure_onsets = ";".join(str(value) for value in recording["seizure_onsets"])
        seizure_offsets = ";".join(str(value) for value in recording["seizure_offsets"])

        summary_records.append(
            {
                "subject_id": recording["subject_id"],
                "edf_file": recording["edf_file"],
                "recording_order": recording["recording_order"],
                "duration_seconds": recording["duration_seconds"],
                "eeg_hours": recording["duration_seconds"] / 3600.0,
                "sampling_rate": recording["sfreq"],
                "original_channel_count": len(recording["raw_channel_names"]),
                "unique_channel_count": len(recording["available_channels"]),
                "channels_used": recording["channels_used"],
                "channel_names": ";".join(common_channels),
                "available_channel_names": ";".join(recording["available_channels"]),
                "duplicate_channels_removed": ";".join(recording["duplicate_channels"]),
                "excluded_channels": ";".join(recording["excluded_channels"]),
                "seizure_count": recording["seizure_count"],
                "seizure_onsets": seizure_onsets,
                "seizure_offsets": seizure_offsets,
                "present_locally": True,
                "annotation_source": Path(recording["summary_path"]).name,
            }
        )

    df_summary = pd.DataFrame(summary_records)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    table_path = TABLES_DIR / "table1_dataset_summary.csv"
    df_summary.to_csv(table_path, index=False)
    logger.info("Saved dataset summary to %s", table_path)

    _write_channel_report(recordings, common_channels, subjects)

    return {
        "subjects": subjects,
        "recordings": recordings,
        "common_channels": common_channels,
        "dataset_summary": df_summary,
    }
