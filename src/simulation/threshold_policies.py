import numpy as np
from typing import List, Tuple
from sklearn.metrics import confusion_matrix
import pandas as pd

class SimulationPolicy:
    def get_triggers(self, probabilities: np.ndarray, timestamps: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class FixedThresholdPolicy(SimulationPolicy):
    def __init__(self, threshold: float = 0.5):
        self.threshold = threshold
        
    def get_triggers(self, probabilities: np.ndarray, timestamps: np.ndarray = None) -> np.ndarray:
        return (probabilities >= self.threshold).astype(int)

class PatientSpecificThresholdPolicy(SimulationPolicy):
    def __init__(self, target_fa_per_hour: float = 2.0, window_len_sec: float = 4.0):
        self.target_fa_per_hour = target_fa_per_hour
        self.window_len_sec = window_len_sec
        self.threshold = 0.5
        
    def fit(self, val_probabilities: np.ndarray, val_labels: np.ndarray, val_hours: float = None):
        """Finds threshold that achieves target FA rate on validation set."""
        # This is a simplified optimization for the POC
        thresholds = np.linspace(0.1, 0.9, 81)
        best_thresh = 0.5
        min_diff = float('inf')
        
        if val_hours is None:
            val_hours = len(val_labels) * self.window_len_sec / 3600.0
        
        for th in thresholds:
            preds = (val_probabilities >= th).astype(int)
            if np.sum(preds) == 0:
                continue
            
            # Using sklearn confusion matrix handling edge cases
            cm = confusion_matrix(val_labels, preds)
            if cm.shape == (2, 2):
                fp = cm[0, 1]
            else:
                fp = 0
                
            fa_rate = fp / (val_hours + 1e-9)
            diff = abs(fa_rate - self.target_fa_per_hour)
            
            if diff < min_diff:
                min_diff = diff
                best_thresh = th
                
        self.threshold = best_thresh
        return self

    def get_triggers(self, probabilities: np.ndarray, timestamps: np.ndarray = None) -> np.ndarray:
        return (probabilities >= self.threshold).astype(int)

class ExplanationConstrainedPolicy(SimulationPolicy):
    """
    Triggers only if prob > threshold AND model explanation is consistent.
    For this POC, we approximate this by requiring N consecutive windows to exceed threshold.
    """
    def __init__(self, base_policy: SimulationPolicy, min_consecutive: int = 3):
        self.base_policy = base_policy
        self.min_consecutive = min_consecutive
        
    def get_triggers(self, probabilities: np.ndarray, timestamps: np.ndarray = None) -> np.ndarray:
        base_triggers = self.base_policy.get_triggers(probabilities, timestamps)
        
        # Require 'min_consecutive' 1s in a row
        final_triggers = np.zeros_like(base_triggers)
        
        count = 0
        for i in range(len(base_triggers)):
            if base_triggers[i] == 1:
                count += 1
            else:
                count = 0
                
            if count >= self.min_consecutive:
                final_triggers[i] = 1
                
        return final_triggers
