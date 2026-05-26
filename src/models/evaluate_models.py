from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
)

from src.utils.logging_utils import setup_logger
from src.utils.paths import TABLES_DIR

logger = setup_logger("evaluate_models")


def _safe_auc(metric_fn, y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(metric_fn(y_true, y_prob))


def evaluate_models(
    train_results: Dict[str, Any],
    context: Dict[str, Any] | None = None,
    write_table: bool = False,
) -> pd.DataFrame:
    """Evaluate trained models on a fold test set."""
    models = train_results['models']
    X_test = train_results['X_test']
    y_test = np.asarray(train_results['y_test']).astype(int)
    context = context or {}

    logger.info("Evaluating %s models on %s test windows", len(models), len(y_test))

    test_hours = float(context.get("test_hours", train_results.get("test_hours", 0.0)))
    if test_hours <= 0:
        window_step_sec = float(context.get("window_step_sec", train_results.get("window_step_sec", 4.0)))
        test_hours = len(y_test) * window_step_sec / 3600.0

    results = []

    for name, model in models.items():
        if hasattr(model, 'predict_proba'):
            y_prob = model.predict_proba(X_test)[:, 1]
        else:
            y_prob = model.predict(X_test)

        y_prob = np.nan_to_num(y_prob, nan=0.0, posinf=1.0, neginf=0.0)
        y_pred = (y_prob >= 0.5).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
        sensitivity = tp / (tp + fn + 1e-9)
        specificity = tn / (tn + fp + 1e-9)
        _, _, f1, _ = precision_recall_fscore_support(
            y_test,
            y_pred,
            average='binary',
            zero_division=0,
        )

        row = {
            **context,
            "model": name,
            "auroc": _safe_auc(roc_auc_score, y_test, y_prob),
            "auprc": _safe_auc(average_precision_score, y_test, y_prob),
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
            "f1": float(f1),
            "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
            "brier_score": float(brier_score_loss(y_test, y_prob)),
            "false_alarms_per_hour": float(fp / (test_hours + 1e-9)),
            "true_positives": int(tp),
            "false_positives": int(fp),
            "true_negatives": int(tn),
            "false_negatives": int(fn),
            "test_windows": int(len(y_test)),
            "test_positive_windows": int(y_test.sum()),
            "test_hours": test_hours,
        }
        results.append(row)

    df_results = pd.DataFrame(results)

    if write_table:
        table_path = TABLES_DIR / "table2_model_performance_real_eeg.csv"
        df_results.to_csv(table_path, index=False)
        logger.info("Saved Table 2 to %s", table_path)

    return df_results
