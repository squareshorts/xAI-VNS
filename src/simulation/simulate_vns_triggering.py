from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

from src.simulation.threshold_policies import (
    ExplanationConstrainedPolicy,
    FixedThresholdPolicy,
    PatientSpecificThresholdPolicy,
)
from src.utils.logging_utils import setup_logger
from src.utils.paths import TABLES_DIR

logger = setup_logger("simulate_vns_triggering")


def _predict_probability(model: Any, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probabilities = model.predict_proba(X)[:, 1]
    else:
        probabilities = model.predict(X)
    return np.nan_to_num(probabilities, nan=0.0, posinf=1.0, neginf=0.0)


def simulate_vns(
    train_results: Dict[str, Any],
    config: Dict,
    context: Dict[str, Any] | None = None,
    write_table: bool = False,
) -> pd.DataFrame:
    """Simulate triggering policies on real-EEG model predictions for one fold."""
    logger.info("Simulating VNS triggering policies...")

    context = context or {}
    y_test = np.asarray(train_results['y_test']).astype(int)
    y_train = np.asarray(train_results['y_train']).astype(int)
    X_test = train_results['X_test']
    X_train = train_results['X_train']

    windowing = config.get("windowing", {}) if config else {}
    window_len_sec = float(windowing.get("window_length_sec", config.get("window_length_sec", 4.0)))
    overlap = float(windowing.get("overlap", config.get("overlap_pct", 0.5)))
    window_step_sec = float(context.get("window_step_sec", window_len_sec * (1.0 - overlap)))
    train_hours = float(context.get("train_hours", len(y_train) * window_step_sec / 3600.0))
    test_hours = float(context.get("test_hours", len(y_test) * window_step_sec / 3600.0))

    policy_config = config.get("policies", {}) if config else {}
    fixed_threshold = float(policy_config.get("fixed_threshold", config.get("fixed_threshold", 0.5)))
    target_fa_per_hour = float(policy_config.get("target_fa_per_hour", config.get("target_fa_per_hour", 2.0)))
    min_consistency_windows = int(
        policy_config.get("min_consistency_windows", config.get("min_consistency_windows", 3))
    )

    results = []

    for model_name, model in train_results['models'].items():
        y_prob_test = _predict_probability(model, X_test)
        y_prob_train = _predict_probability(model, X_train)

        subject_specific_policy = PatientSpecificThresholdPolicy(
            target_fa_per_hour=target_fa_per_hour,
            window_len_sec=window_step_sec,
        )
        subject_specific_policy.fit(y_prob_train, y_train, val_hours=train_hours)

        policies = {
            "Fixed threshold = 0.5": FixedThresholdPolicy(threshold=fixed_threshold),
            "Subject-specific threshold": subject_specific_policy,
            "Explanation-constrained threshold": ExplanationConstrainedPolicy(
                FixedThresholdPolicy(threshold=fixed_threshold),
                min_consecutive=min_consistency_windows,
            ),
        }

        for policy_name, policy in policies.items():
            triggers = policy.get_triggers(y_prob_test)
            total_stimulations = int(np.sum(triggers))
            true_stimulations = int(np.sum((triggers == 1) & (y_test == 1)))
            false_stimulations = int(np.sum((triggers == 1) & (y_test == 0)))
            missed_positive_windows = int(np.sum((triggers == 0) & (y_test == 1)))
            sensitivity = true_stimulations / (
                true_stimulations + missed_positive_windows + 1e-9
            )

            threshold = getattr(policy, "threshold", None)
            if threshold is None and hasattr(policy, "base_policy"):
                threshold = getattr(policy.base_policy, "threshold", None)

            results.append(
                {
                    **context,
                    "model": model_name,
                    "policy": policy_name,
                    "threshold": threshold,
                    "total_stimulations": total_stimulations,
                    "stimulations_per_hour": total_stimulations / (test_hours + 1e-9),
                    "false_stimulations_per_hour": false_stimulations / (test_hours + 1e-9),
                    "window_level_sensitivity": sensitivity,
                    "true_stimulations": true_stimulations,
                    "false_stimulations": false_stimulations,
                    "test_hours": test_hours,
                    "test_windows": int(len(y_test)),
                    "test_positive_windows": int(y_test.sum()),
                }
            )

    df_results = pd.DataFrame(results)

    if write_table:
        table_path = TABLES_DIR / "table3_policy_comparison_real_eeg.csv"
        df_results.to_csv(table_path, index=False)
        logger.info("Saved Table 3 to %s", table_path)

    return df_results
