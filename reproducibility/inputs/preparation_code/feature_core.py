"""Original feature calculations, with only logging removed.

Reference: squareshorts/xAI-VNS commit e586c0998f0f4962ff0a4597084824e317fb9e86,
src/features/eeg_features.py::extract_features (vectorized implementation).
Fixed feature flags: all features enabled, as in configs/chbmit.yaml.
No changes to the original dimensional constants, units, or epsilons.
"""
import numpy as np
import pandas as pd
from scipy import signal, stats

FEATURE_NAMES = [
    'bp_delta', 'rel_bp_delta', 'bp_theta', 'rel_bp_theta',
    'bp_alpha', 'rel_bp_alpha', 'bp_beta', 'rel_bp_beta',
    'bp_gamma', 'rel_bp_gamma', 'spectral_entropy',
    'hjorth_activity', 'hjorth_mobility', 'hjorth_complexity',
    'line_length', 'rms_amplitude', 'variance', 'zero_crossing_rate',
    'skewness', 'kurtosis',
]


def extract_features(windows, sfreq=256.0):
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError('Expected (windows, channels, samples)')
    if windows.shape[1:] != (22, 1024) or sfreq != 256:
        raise ValueError('This frozen extractor requires 22 channels, 1024 samples, 256 Hz')
    n_samples = windows.shape[-1]
    columns = {}
    freqs, psd = signal.welch(windows, sfreq, nperseg=min(int(round(sfreq)), n_samples), axis=-1)
    bandpowers = {}
    bands = {'delta':(0.5,4), 'theta':(4,8), 'alpha':(8,13), 'beta':(13,30), 'gamma':(30,50)}
    integrate = np.trapezoid if hasattr(np, 'trapezoid') else np.trapz
    for name, (low, high) in bands.items():
        index = np.logical_and(freqs >= low, freqs <= high)
        bandpowers[name] = integrate(psd[:, :, index], freqs[index], axis=-1)
    total_power = np.sum(list(bandpowers.values()), axis=0)
    for name, power in bandpowers.items():
        columns['bp_' + name] = np.mean(power, axis=1)
        columns['rel_bp_' + name] = np.mean(power / (total_power + 1e-9), axis=1)
    norm = psd / (np.sum(psd, axis=-1, keepdims=True) + 1e-9)
    entropy = -np.sum(norm * np.log2(norm + 1e-9), axis=-1)
    columns['spectral_entropy'] = np.mean(entropy, axis=1)
    d1 = np.diff(windows, axis=-1)
    d2 = np.diff(d1, axis=-1)
    vy = np.var(windows, axis=-1)
    v1 = np.var(d1, axis=-1)
    v2 = np.var(d2, axis=-1)
    mobility = np.sqrt(v1 / (vy + 1e-9))
    complexity = np.sqrt(v2 / (v1 + 1e-9)) / (mobility + 1e-9)
    columns['hjorth_activity'] = np.mean(vy, axis=1)
    columns['hjorth_mobility'] = np.mean(mobility, axis=1)
    columns['hjorth_complexity'] = np.mean(complexity, axis=1)
    columns['line_length'] = np.mean(np.sum(np.abs(np.diff(windows, axis=-1)), axis=-1), axis=1)
    columns['rms_amplitude'] = np.mean(np.sqrt(np.mean(windows**2, axis=-1)), axis=1)
    columns['variance'] = np.mean(np.var(windows, axis=-1), axis=1)
    columns['zero_crossing_rate'] = np.mean(np.sum(np.diff(np.sign(windows), axis=-1) != 0, axis=-1) / n_samples, axis=1)
    columns['skewness'] = np.mean(stats.skew(windows, axis=-1), axis=1)
    columns['kurtosis'] = np.mean(stats.kurtosis(windows, axis=-1), axis=1)
    result = pd.DataFrame({name:columns[name] for name in FEATURE_NAMES})
    return result.replace([np.inf, -np.inf], np.nan).fillna(0.0)
