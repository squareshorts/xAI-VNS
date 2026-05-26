from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Tuple

import mne
import numpy as np
import pandas as pd

from src.data.load_chbmit import canonical_channel_name
from src.utils.logging_utils import setup_logger

logger = setup_logger("make_windows")


def get_windowing_config(config: Dict) -> Dict[str, float]:
    """Read windowing values from the nested CHB-MIT config with legacy fallbacks."""
    windowing = config.get("windowing", {}) if config else {}
    window_len_sec = float(
        windowing.get("window_length_sec", config.get("window_length_sec", 4.0))
    )
    overlap = float(windowing.get("overlap", config.get("overlap_pct", 0.5)))
    preictal_sec = float(
        windowing.get(
            "preictal_sec",
            config.get("preictal_sec", config.get("preictal_duration_min", 5.0) * 60.0),
        )
    )

    if not 0 <= overlap < 1:
        raise ValueError(f"Window overlap must be in [0, 1); got {overlap}")

    return {
        "window_length_sec": window_len_sec,
        "overlap": overlap,
        "preictal_sec": preictal_sec,
    }


def _select_channel_indices(raw_channel_names: List[str], common_channels: List[str]) -> List[int]:
    first_index_by_channel: Dict[str, int] = {}
    for idx, original_name in enumerate(raw_channel_names):
        channel = canonical_channel_name(original_name)
        if channel in common_channels and channel not in first_index_by_channel:
            first_index_by_channel[channel] = idx

    missing = [channel for channel in common_channels if channel not in first_index_by_channel]
    if missing:
        raise ValueError(f"Recording is missing common channels: {', '.join(missing)}")

    return [first_index_by_channel[channel] for channel in common_channels]


def _label_window(center_sec: float, seizures: List[Tuple[int, int]], preictal_sec: float) -> int:
    label = 0
    for seizure_start, seizure_end in seizures:
        if seizure_start <= center_sec <= seizure_end:
            return 2
        if seizure_start - preictal_sec <= center_sec < seizure_start:
            label = 1
    return label


def _load_recording_data(recording: Dict, common_channels: List[str]) -> Tuple[np.ndarray, float]:
    if "data" in recording:
        return np.asarray(recording["data"], dtype=np.float32), float(recording["sfreq"])

    edf_path = Path(recording["edf_path"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)

    picks = _select_channel_indices(raw.info["ch_names"], common_channels)
    data = raw.get_data(picks=picks).astype(np.float32, copy=False)
    return data, float(raw.info["sfreq"])


def create_sliding_windows_for_recording(
    recording: Dict,
    window_len_sec: float = 4.0,
    overlap_pct: float = 0.5,
    preictal_sec: float = 300.0,
    common_channels: List[str] | None = None,
    split: str = "unassigned",
    evaluation_mode: str = "feature_extraction",
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    """
    Slice one EDF into overlapping windows and return labels plus per-window metadata.

    Labels:
    0 = interictal, 1 = preictal, 2 = ictal.
    Binary label is 1 for preictal or ictal windows.
    """
    common_channels = common_channels or recording.get("common_channels") or recording.get("ch_names")
    if not common_channels:
        raise ValueError("No channel list supplied for window creation.")

    data, sfreq = _load_recording_data(recording, common_channels)
    seizures = recording.get("seizures", [])

    n_channels, n_samples = data.shape
    window_len_samples = int(round(window_len_sec * sfreq))
    step_samples = int(round(window_len_samples * (1.0 - overlap_pct)))
    step_samples = max(step_samples, 1)

    if window_len_samples <= 0 or n_samples < window_len_samples:
        empty_metadata = pd.DataFrame(
            columns=[
                "window_id",
                "subject_id",
                "edf_file",
                "window_index",
                "window_start",
                "window_end",
                "window_center",
                "label",
                "binary_label",
                "split",
                "evaluation_mode",
            ]
        )
        return (
            np.empty((0, n_channels, max(window_len_samples, 0)), dtype=np.float32),
            np.array([], dtype=int),
            empty_metadata,
            sfreq,
        )

    n_windows = (n_samples - window_len_samples) // step_samples + 1
    windows = np.empty((n_windows, n_channels, window_len_samples), dtype=np.float32)
    labels = np.zeros(n_windows, dtype=int)
    metadata_records: List[Dict] = []

    subject_id = recording.get("subject_id", recording.get("subject", "unknown_subject"))
    edf_file = recording.get("edf_file", recording.get("filename", "unknown.edf"))
    duration_seconds = float(recording.get("duration_seconds", n_samples / sfreq))
    recording_order = int(recording.get("recording_order", 0))

    logger.info(
        "Creating %s windows of %.3fs for %s %s",
        n_windows,
        window_len_sec,
        subject_id,
        edf_file,
    )

    for idx in range(n_windows):
        start_idx = idx * step_samples
        end_idx = start_idx + window_len_samples
        windows[idx] = data[:, start_idx:end_idx]

        window_start = start_idx / sfreq
        window_end = end_idx / sfreq
        window_center = (start_idx + window_len_samples / 2.0) / sfreq
        label = _label_window(window_center, seizures, preictal_sec)
        labels[idx] = label

        metadata_records.append(
            {
                "window_id": f"{subject_id}|{edf_file}|{idx:06d}",
                "subject_id": subject_id,
                "edf_file": edf_file,
                "recording_order": recording_order,
                "window_index": idx,
                "window_start": window_start,
                "window_end": window_end,
                "window_center": window_center,
                "label": label,
                "binary_label": int(label > 0),
                "split": split,
                "evaluation_mode": evaluation_mode,
                "sampling_rate": sfreq,
                "duration_seconds": duration_seconds,
                "channel_count": len(common_channels),
            }
        )

    return windows, labels, pd.DataFrame(metadata_records), sfreq


def process_all_recordings(
    data_dict: Dict,
    config: Dict,
) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame, float]:
    """
    Legacy helper that processes all recordings and concatenates windows.

    The multi-subject pipeline uses this per recording to keep memory bounded, but
    this function remains available for small tests and backwards compatibility.
    """
    cfg = get_windowing_config(config)
    common_channels = data_dict.get("common_channels")

    all_windows = []
    all_labels = []
    all_metadata = []
    sfreq = None

    for recording in data_dict["recordings"]:
        windows, labels, metadata, rec_sfreq = create_sliding_windows_for_recording(
            recording,
            window_len_sec=cfg["window_length_sec"],
            overlap_pct=cfg["overlap"],
            preictal_sec=cfg["preictal_sec"],
            common_channels=common_channels,
        )
        if len(windows) == 0:
            continue
        all_windows.append(windows)
        all_labels.append(labels)
        all_metadata.append(metadata)
        sfreq = rec_sfreq if sfreq is None else sfreq

    if not all_windows:
        raise ValueError("No windows could be extracted.")

    return (
        np.concatenate(all_windows, axis=0),
        np.concatenate(all_labels, axis=0),
        pd.concat(all_metadata, ignore_index=True),
        float(sfreq),
    )
