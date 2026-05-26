import numpy as np
from src.simulation.threshold_policies import FixedThresholdPolicy, PatientSpecificThresholdPolicy, ExplanationConstrainedPolicy

def test_fixed_threshold():
    policy = FixedThresholdPolicy(threshold=0.5)
    probs = np.array([0.1, 0.4, 0.6, 0.9])
    triggers = policy.get_triggers(probs)
    np.testing.assert_array_equal(triggers, [0, 0, 1, 1])

def test_explanation_constrained_policy():
    base = FixedThresholdPolicy(threshold=0.5)
    policy = ExplanationConstrainedPolicy(base, min_consecutive=3)
    
    probs = np.array([0.1, 0.6, 0.7, 0.4, 0.6, 0.7, 0.8, 0.9])
    # triggers should be 0 unless 3 consecutive > 0.5
    # base triggers: [0, 1, 1, 0, 1, 1, 1, 1]
    # final triggers:[0, 0, 0, 0, 0, 0, 1, 1]
    triggers = policy.get_triggers(probs)
    np.testing.assert_array_equal(triggers, [0, 0, 0, 0, 0, 0, 1, 1])

def test_patient_specific_threshold():
    policy = PatientSpecificThresholdPolicy(target_fa_per_hour=2.0, window_len_sec=3600.0) # 1 hour per sample for easy math
    # 10 samples (10 hours)
    val_probs = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95])
    # None are actual seizures (all 0)
    val_labels = np.zeros(10)
    
    # We want 2 FA per hour -> over 10 hours, we want 20 FAs? 
    # Wait, 1 sample = 1 hour, so 10 samples = 10 hours.
    # Target FA/hr = 2.0. So 20 FAs in 10 hours. 
    # But we only have 10 samples. This means we want FA rate to be as close to 2.0 as possible.
    # Actually, let's just make it simpler.
    # If threshold is 0.75, probs >= 0.75 are 0.8, 0.9, 0.95 -> 3 FAs in 10 hours = 0.3 FA/hr.
    
    # Let's set target FA/hr to 0.3. The best threshold should pick 3 FAs -> threshold around 0.71-0.8.
    policy = PatientSpecificThresholdPolicy(target_fa_per_hour=0.3, window_len_sec=3600.0)
    policy.fit(val_probs, val_labels)
    assert 0.7 < policy.threshold <= 0.8
