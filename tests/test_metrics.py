import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score

def test_metrics_computation():
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.1, 0.4, 0.35, 0.8])
    
    auroc = roc_auc_score(y_true, y_prob)
    auprc = average_precision_score(y_true, y_prob)
    
    assert auroc >= 0.0 and auroc <= 1.0
    assert auprc >= 0.0 and auprc <= 1.0
