import numpy as np
import scipy.signal as signal
import scipy.stats as stats
import pandas as pd
from typing import Dict, Tuple
from src.utils.logging_utils import setup_logger

logger = setup_logger("eeg_features")


def _integrate_power(values: np.ndarray, freqs: np.ndarray, axis: int = -1) -> np.ndarray:
    integrator = getattr(np, "trapezoid", np.trapz)
    return integrator(values, freqs, axis=axis)

def compute_bandpower(data: np.ndarray, sfreq: float, bands: Dict[str, Tuple[float, float]]) -> Dict[str, np.ndarray]:
    """Computes absolute bandpower for given frequency bands."""
    freqs, psd = signal.welch(data, sfreq, nperseg=int(sfreq)) # shape: (channels, freqs)
    
    bandpowers = {}
    for band_name, (low, high) in bands.items():
        idx_band = np.logical_and(freqs >= low, freqs <= high)
        bp = _integrate_power(psd[:, idx_band], freqs[idx_band], axis=-1)
        bandpowers[band_name] = bp
    return bandpowers


def get_feature_names(feature_flags: Dict) -> list[str]:
    feature_names = []
    if feature_flags.get('bandpower', True):
        for band in ['delta', 'theta', 'alpha', 'beta', 'gamma']:
            feature_names.append(f'bp_{band}')
            if feature_flags.get('relative_bandpower', True):
                feature_names.append(f'rel_bp_{band}')
    if feature_flags.get('spectral_entropy', True):
        feature_names.append('spectral_entropy')
    if feature_flags.get('hjorth', True):
        feature_names.extend(['hjorth_activity', 'hjorth_mobility', 'hjorth_complexity'])
    if feature_flags.get('line_length', True):
        feature_names.append('line_length')
    if feature_flags.get('rms_amplitude', True):
        feature_names.append('rms_amplitude')
    if feature_flags.get('variance', True):
        feature_names.append('variance')
    if feature_flags.get('zero_crossing_rate', True):
        feature_names.append('zero_crossing_rate')
    if feature_flags.get('skew_kurtosis', True):
        feature_names.extend(['skewness', 'kurtosis'])
    return feature_names


def extract_window_features(window: np.ndarray, sfreq: float, feature_flags: Dict) -> np.ndarray:
    """
    Extracts features for a single window across all channels and flattens them,
    or aggregates them across channels.
    For this POC, we will extract features per channel and take the mean across channels 
    to reduce dimensionality, as spatial location isn't our primary focus.
    """
    n_channels, n_samples = window.shape
    features = []
    
    # 1. Bandpower
    if feature_flags.get('bandpower', True):
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50)
        }
        bps = compute_bandpower(window, sfreq, bands)
        
        total_power = np.sum(list(bps.values()), axis=0)
        
        for name, bp in bps.items():
            features.append(np.mean(bp)) # Mean across channels
            if feature_flags.get('relative_bandpower', True):
                features.append(np.mean(bp / (total_power + 1e-9)))
                
    # 1.5 Spectral Entropy
    if feature_flags.get('spectral_entropy', True):
        # Calculate power spectral density
        freqs, psd = signal.welch(window, sfreq, nperseg=int(sfreq))
        # Normalize PSD to get PDF
        psd_norm = psd / (np.sum(psd, axis=-1, keepdims=True) + 1e-9)
        # Calculate entropy
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-9), axis=-1)
        features.append(np.mean(entropy))
                
    # 2. Hjorth Parameters
    if feature_flags.get('hjorth', True):
        diff1 = np.diff(window, axis=-1)
        diff2 = np.diff(diff1, axis=-1)
        
        var_y = np.var(window, axis=-1)
        var_d1 = np.var(diff1, axis=-1)
        var_d2 = np.var(diff2, axis=-1)
        
        activity = var_y
        mobility = np.sqrt(var_d1 / (var_y + 1e-9))
        complexity = np.sqrt(var_d2 / (var_d1 + 1e-9)) / (mobility + 1e-9)
        
        features.append(np.mean(activity))
        features.append(np.mean(mobility))
        features.append(np.mean(complexity))
        
    # 3. Line Length
    if feature_flags.get('line_length', True):
        ll = np.sum(np.abs(np.diff(window, axis=-1)), axis=-1)
        features.append(np.mean(ll))
        
    # 4. RMS Amplitude
    if feature_flags.get('rms_amplitude', True):
        rms = np.sqrt(np.mean(window**2, axis=-1))
        features.append(np.mean(rms))
        
    # 5. Variance
    if feature_flags.get('variance', True):
        features.append(np.mean(np.var(window, axis=-1)))
        
    # 6. Zero-Crossing Rate
    if feature_flags.get('zero_crossing_rate', True):
        zcr = np.sum(np.diff(np.sign(window), axis=-1) != 0, axis=-1) / n_samples
        features.append(np.mean(zcr))
        
    # 7. Skewness and Kurtosis
    if feature_flags.get('skew_kurtosis', True):
        skew = stats.skew(window, axis=-1)
        kurt = stats.kurtosis(window, axis=-1)
        features.append(np.mean(skew))
        features.append(np.mean(kurt))
        
    return np.nan_to_num(np.array(features), nan=0.0, posinf=0.0, neginf=0.0)

def extract_features(windows: np.ndarray, sfreq: float, feature_flags: Dict) -> pd.DataFrame:
    """Extract interpretable EEG features for all windows and return a DataFrame."""
    windows = np.asarray(windows)
    if windows.ndim != 3:
        raise ValueError("Expected windows with shape (n_windows, n_channels, n_samples).")

    n_windows, _, n_samples = windows.shape
    logger.info("Extracting features for %s windows", n_windows)

    feature_columns: dict[str, np.ndarray] = {}

    psd = None
    freqs = None
    if (
        feature_flags.get('bandpower', True)
        or feature_flags.get('spectral_entropy', True)
    ):
        nperseg = min(int(round(sfreq)), n_samples)
        freqs, psd = signal.welch(windows, sfreq, nperseg=nperseg, axis=-1)

    if feature_flags.get('bandpower', True):
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 50),
        }
        bandpowers: dict[str, np.ndarray] = {}
        for band_name, (low, high) in bands.items():
            idx_band = np.logical_and(freqs >= low, freqs <= high)
            bp = _integrate_power(psd[:, :, idx_band], freqs[idx_band], axis=-1)
            bandpowers[band_name] = bp

        total_power = np.sum(list(bandpowers.values()), axis=0)
        for band_name, bp in bandpowers.items():
            feature_columns[f'bp_{band_name}'] = np.mean(bp, axis=1)
            if feature_flags.get('relative_bandpower', True):
                feature_columns[f'rel_bp_{band_name}'] = np.mean(
                    bp / (total_power + 1e-9),
                    axis=1,
                )

    if feature_flags.get('spectral_entropy', True):
        psd_norm = psd / (np.sum(psd, axis=-1, keepdims=True) + 1e-9)
        entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-9), axis=-1)
        feature_columns['spectral_entropy'] = np.mean(entropy, axis=1)

    if feature_flags.get('hjorth', True):
        diff1 = np.diff(windows, axis=-1)
        diff2 = np.diff(diff1, axis=-1)
        var_y = np.var(windows, axis=-1)
        var_d1 = np.var(diff1, axis=-1)
        var_d2 = np.var(diff2, axis=-1)
        mobility = np.sqrt(var_d1 / (var_y + 1e-9))
        complexity = np.sqrt(var_d2 / (var_d1 + 1e-9)) / (mobility + 1e-9)
        feature_columns['hjorth_activity'] = np.mean(var_y, axis=1)
        feature_columns['hjorth_mobility'] = np.mean(mobility, axis=1)
        feature_columns['hjorth_complexity'] = np.mean(complexity, axis=1)

    if feature_flags.get('line_length', True):
        line_length = np.sum(np.abs(np.diff(windows, axis=-1)), axis=-1)
        feature_columns['line_length'] = np.mean(line_length, axis=1)

    if feature_flags.get('rms_amplitude', True):
        rms = np.sqrt(np.mean(windows**2, axis=-1))
        feature_columns['rms_amplitude'] = np.mean(rms, axis=1)

    if feature_flags.get('variance', True):
        feature_columns['variance'] = np.mean(np.var(windows, axis=-1), axis=1)

    if feature_flags.get('zero_crossing_rate', True):
        zcr = np.sum(np.diff(np.sign(windows), axis=-1) != 0, axis=-1) / n_samples
        feature_columns['zero_crossing_rate'] = np.mean(zcr, axis=1)

    if feature_flags.get('skew_kurtosis', True):
        skew = stats.skew(windows, axis=-1)
        kurt = stats.kurtosis(windows, axis=-1)
        feature_columns['skewness'] = np.mean(skew, axis=1)
        feature_columns['kurtosis'] = np.mean(kurt, axis=1)

    feature_names = get_feature_names(feature_flags)
    df = pd.DataFrame({name: feature_columns[name] for name in feature_names})
    return df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
