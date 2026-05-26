import numpy as np
from src.features.eeg_features import extract_window_features

def test_extract_window_features():
    sfreq = 256.0
    n_channels = 2
    n_samples = int(2.0 * sfreq) # 2 seconds
    
    # Create dummy data
    np.random.seed(42)
    window = np.random.randn(n_channels, n_samples)
    
    feature_flags = {
        'bandpower': True,
        'relative_bandpower': True,
        'spectral_entropy': True,
        'hjorth': True,
        'line_length': True,
        'rms_amplitude': True,
        'variance': True,
        'zero_crossing_rate': True,
        'skew_kurtosis': True
    }
    
    features = extract_window_features(window, sfreq, feature_flags)
    
    # 5 bp + 5 rel bp + 1 spectral entropy + 3 hjorth + 1 ll + 1 rms + 1 var + 1 zcr + 2 skew/kurt = 20
    assert len(features) == 20
    assert not np.isnan(features).any()
    assert not np.isinf(features).any()
