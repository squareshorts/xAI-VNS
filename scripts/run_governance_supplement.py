from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FEATURES_PATH = PROJECT_ROOT / "data" / "processed" / "chbmit_features.parquet"
LABELS_PATH = PROJECT_ROOT / "data" / "processed" / "chbmit_window_labels.csv"
DATASET_SUMMARY_PATH = PROJECT_ROOT / "results" / "tables" / "table1_dataset_summary.csv"
CONFIG_PATH = PROJECT_ROOT / "configs" / "chbmit.yaml"
TABLES_DIR = PROJECT_ROOT / "results" / "tables"
FIGURES_DIR = PROJECT_ROOT / "results" / "figures"

METADATA_COLUMNS = {
    "window_id",
    "subject_id",
    "edf_file",
    "recording_order",
    "window_index",
    "window_start",
    "window_end",
    "window_center",
    "label",
    "binary_label",
    "split",
    "evaluation_mode",
    "sampling_rate",
    "duration_seconds",
    "channel_count",
}

MODEL_LABELS = {
    "lightgbm": "LightGBM",
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
}


@dataclass(frozen=True)
class GovernanceConfig:
    theta: float = 0.5
    top_k: int = 5
    gamma: float = 0.4
    calibration_bins: int = 10
    preictal_seconds: float = 300.0


def _load_governance_config() -> GovernanceConfig:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}

    policies = config.get("policies", {})
    windowing = config.get("windowing", {})
    return GovernanceConfig(
        theta=float(policies.get("fixed_threshold", 0.5)),
        top_k=int(policies.get("top_k_attribution_features", 5)),
        gamma=float(policies.get("attribution_stability_gamma", 0.4)),
        calibration_bins=int(config.get("calibration_bins", 10)),
        preictal_seconds=float(windowing.get("preictal_sec", 300.0)),
    )


def _format_mean_std(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def _safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def _fit_models(X_train: pd.DataFrame, y_train: np.ndarray, seed: int = 42) -> Dict[str, object]:
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        random_state=seed,
                        max_iter=2000,
                        class_weight="balanced",
                    ),
                ),
            ]
        ).fit(X_train, y_train),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ).fit(X_train, y_train),
        "lightgbm": LGBMClassifier(
            random_state=seed,
            class_weight="balanced",
            verbose=-1,
        ).fit(X_train, y_train),
    }


def _predict_probability(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[:, 1]
    else:
        probability = model.predict(X)
    return np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)


def _calibration_summary(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int,
) -> tuple[float, pd.DataFrame]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(y_prob, edges[1:-1], right=False)

    records: List[Dict] = []
    ece = 0.0
    n = len(y_true)
    for bin_id in range(n_bins):
        mask = bin_ids == bin_id
        count = int(mask.sum())
        lower = float(edges[bin_id])
        upper = float(edges[bin_id + 1])
        if count == 0:
            records.append(
                {
                    "bin": bin_id + 1,
                    "bin_lower": lower,
                    "bin_upper": upper,
                    "count": 0,
                    "mean_predicted_probability": np.nan,
                    "observed_positive_rate": np.nan,
                    "absolute_calibration_error": np.nan,
                }
            )
            continue

        mean_probability = float(y_prob[mask].mean())
        observed_rate = float(y_true[mask].mean())
        abs_error = abs(observed_rate - mean_probability)
        ece += (count / n) * abs_error
        records.append(
            {
                "bin": bin_id + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "absolute_calibration_error": abs_error,
            }
        )
    return float(ece), pd.DataFrame(records)


def _topk_indices_from_lightgbm(model: object, X: pd.DataFrame, top_k: int) -> np.ndarray:
    contributions = model.booster_.predict(X, pred_contrib=True)
    attribution = np.abs(np.asarray(contributions[:, :-1]))
    return np.argsort(-attribution, axis=1)[:, :top_k]


def _explanation_stability(
    rows: pd.DataFrame,
    topk_indices: np.ndarray,
    top_k: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_rows = len(rows)
    stability = np.full(n_rows, np.nan, dtype=float)
    previous_probability_index = np.full(n_rows, -1, dtype=int)
    has_previous = np.zeros(n_rows, dtype=bool)

    rows = rows.reset_index(drop=True)
    for _, group in rows.groupby(["subject_id", "edf_file"], sort=False):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        current = indices[1:]
        previous = indices[:-1]
        previous_probability_index[current] = previous
        has_previous[current] = True

        for cur, prev in zip(current, previous):
            overlap = len(set(topk_indices[cur]).intersection(topk_indices[prev]))
            stability[cur] = overlap / float((2 * top_k) - overlap)

    return stability, previous_probability_index, has_previous


def _policy_metrics(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    triggers: np.ndarray,
    stability: np.ndarray,
    policy_name: str,
    context: Dict,
) -> Dict:
    test_hours = float(context["test_hours"])
    total = int(triggers.sum())
    true_stimulations = int(((triggers == 1) & (y_true == 1)).sum())
    false_stimulations = int(((triggers == 1) & (y_true == 0)).sum())
    positives = int(y_true.sum())
    triggered_stability = stability[triggers == 1]
    triggered_stability = triggered_stability[~np.isnan(triggered_stability)]

    return {
        **context,
        "model": "lightgbm",
        "policy": policy_name,
        "total_stimulations": total,
        "stimulations_per_hour": _safe_divide(total, test_hours),
        "false_stimulations_per_hour": _safe_divide(false_stimulations, test_hours),
        "window_level_sensitivity": _safe_divide(true_stimulations, positives),
        "mean_s_t_triggered": float(np.mean(triggered_stability)) if len(triggered_stability) else np.nan,
        "median_s_t_triggered": float(np.median(triggered_stability)) if len(triggered_stability) else np.nan,
        "triggered_windows_with_s_t": int(len(triggered_stability)),
        "test_positive_windows": positives,
        "test_windows": int(len(y_true)),
        "test_hours": test_hours,
    }


def _parse_event_seconds(value: object) -> List[float]:
    if pd.isna(value):
        return []
    if isinstance(value, (int, float, np.integer, np.floating)):
        return [float(value)]
    text = str(value).strip()
    if not text:
        return []
    return [float(part) for part in text.split(";") if part.strip()]


def _events_from_dataset_summary(summary: pd.DataFrame) -> pd.DataFrame:
    records: List[Dict] = []
    for row in summary.itertuples(index=False):
        starts = _parse_event_seconds(getattr(row, "seizure_onsets"))
        offsets = _parse_event_seconds(getattr(row, "seizure_offsets"))
        for event_index, (start, offset) in enumerate(zip(starts, offsets), start=1):
            records.append(
                {
                    "subject_id": row.subject_id,
                    "edf_file": row.edf_file,
                    "event_index": event_index,
                    "seizure_onset": float(start),
                    "seizure_offset": float(offset),
                }
            )
    return pd.DataFrame(records)


def _event_level_metrics(
    rows: pd.DataFrame,
    triggers: np.ndarray,
    events: pd.DataFrame,
    policy_name: str,
    context: Dict,
    preictal_seconds: float,
) -> Dict:
    test_recordings = rows[["subject_id", "edf_file"]].drop_duplicates()
    fold_events = events.merge(test_recordings, on=["subject_id", "edf_file"], how="inner")

    latencies: List[float] = []
    covered = 0
    pre_onset = 0
    rows = rows.reset_index(drop=True).copy()
    rows["trigger"] = triggers.astype(int)

    for event in fold_events.itertuples(index=False):
        event_start = max(0.0, float(event.seizure_onset) - preictal_seconds)
        event_end = float(event.seizure_offset)
        event_rows = rows[
            (rows["subject_id"] == event.subject_id)
            & (rows["edf_file"] == event.edf_file)
            & (rows["window_center"] >= event_start)
            & (rows["window_center"] <= event_end)
            & (rows["trigger"] == 1)
        ].sort_values("window_center")

        if event_rows.empty:
            continue

        first_trigger = float(event_rows.iloc[0]["window_center"])
        latency = first_trigger - float(event.seizure_onset)
        latencies.append(latency)
        covered += 1
        if latency < 0:
            pre_onset += 1

    return {
        **context,
        "model": "lightgbm",
        "policy": policy_name,
        "seizure_events": int(len(fold_events)),
        "events_with_authorization": int(covered),
        "pre_onset_authorizations": int(pre_onset),
        "event_level_sensitivity": _safe_divide(covered, len(fold_events)),
        "median_latency_seconds": float(np.median(latencies)) if latencies else np.nan,
        "iqr_latency_seconds": (
            float(np.percentile(latencies, 75) - np.percentile(latencies, 25))
            if len(latencies) >= 2
            else np.nan
        ),
        "false_stimulations_per_hour": _safe_divide(
            int(((triggers == 1) & (rows["binary_label"].to_numpy().astype(int) == 0)).sum()),
            float(context["test_hours"]),
        ),
    }


def _event_detail_records(
    rows: pd.DataFrame,
    triggers: np.ndarray,
    events: pd.DataFrame,
    policy_name: str,
    context: Dict,
    preictal_seconds: float,
) -> List[Dict]:
    test_recordings = rows[["subject_id", "edf_file"]].drop_duplicates()
    fold_events = events.merge(test_recordings, on=["subject_id", "edf_file"], how="inner")

    rows = rows.reset_index(drop=True).copy()
    rows["trigger"] = triggers.astype(int)
    records: List[Dict] = []

    for event in fold_events.itertuples(index=False):
        event_start = max(0.0, float(event.seizure_onset) - preictal_seconds)
        event_end = float(event.seizure_offset)
        event_rows = rows[
            (rows["subject_id"] == event.subject_id)
            & (rows["edf_file"] == event.edf_file)
            & (rows["window_center"] >= event_start)
            & (rows["window_center"] <= event_end)
            & (rows["trigger"] == 1)
        ].sort_values("window_center")

        first_trigger = np.nan
        latency = np.nan
        authorized = not event_rows.empty
        if authorized:
            first_trigger = float(event_rows.iloc[0]["window_center"])
            latency = first_trigger - float(event.seizure_onset)

        records.append(
            {
                **context,
                "model": "lightgbm",
                "policy": policy_name,
                "subject_id": event.subject_id,
                "edf_file": event.edf_file,
                "event_index": int(event.event_index),
                "seizure_onset": float(event.seizure_onset),
                "seizure_offset": float(event.seizure_offset),
                "authorized": bool(authorized),
                "first_authorization_time": first_trigger,
                "latency_seconds": latency,
                "pre_onset_authorization": bool(authorized and latency < 0),
            }
        )

    return records


def _sum_unique_recording_hours(rows: pd.DataFrame) -> float:
    return float(rows.drop_duplicates(["subject_id", "edf_file"])["duration_seconds"].sum() / 3600.0)


def _aggregate_mean_std(
    df: pd.DataFrame,
    group_cols: Iterable[str],
    value_cols: Iterable[str],
) -> pd.DataFrame:
    aggregations = {"n_folds": ("fold_id", "nunique")}
    for column in value_cols:
        aggregations[f"{column}_mean"] = (column, "mean")
        aggregations[f"{column}_std"] = (column, "std")
    result = df.groupby(list(group_cols), dropna=False).agg(**aggregations).reset_index()
    for column in result.columns:
        if column.endswith("_std"):
            result[column] = result[column].fillna(0.0)
    return result


def _aggregate_event_metrics(event_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (evaluation_mode, policy), group in event_df.groupby(["evaluation_mode", "policy"], sort=False):
        total_events = int(group["seizure_events"].sum())
        covered = int(group["events_with_authorization"].sum())
        pre_onset = int(group["pre_onset_authorizations"].sum())
        latencies = []
        for value in group["median_latency_seconds"].dropna().to_numpy():
            latencies.append(float(value))
        records.append(
            {
                "evaluation_mode": evaluation_mode,
                "policy": policy,
                "n_folds": int(group["fold_id"].nunique()),
                "seizure_events": total_events,
                "events_with_authorization": covered,
                "pre_onset_authorizations": pre_onset,
                "event_level_sensitivity": _safe_divide(covered, total_events),
                "median_fold_latency_seconds": float(np.median(latencies)) if latencies else np.nan,
                "false_stimulations_per_hour_mean": float(group["false_stimulations_per_hour"].mean()),
                "false_stimulations_per_hour_std": float(group["false_stimulations_per_hour"].std(ddof=1)),
            }
        )
    result = pd.DataFrame(records)
    result["false_stimulations_per_hour_std"] = result["false_stimulations_per_hour_std"].fillna(0.0)
    return result


def _aggregate_event_details(event_details_df: pd.DataFrame, events_by_fold_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (evaluation_mode, policy), group in event_details_df.groupby(["evaluation_mode", "policy"], sort=False):
        authorized = group[group["authorized"] == True]
        fold_group = events_by_fold_df[
            (events_by_fold_df["evaluation_mode"] == evaluation_mode)
            & (events_by_fold_df["policy"] == policy)
        ]
        latencies = authorized["latency_seconds"].dropna().to_numpy(dtype=float)
        records.append(
            {
                "evaluation_mode": evaluation_mode,
                "policy": policy,
                "n_folds": int(group["fold_id"].nunique()),
                "seizure_events": int(len(group)),
                "events_with_authorization": int(len(authorized)),
                "pre_onset_authorizations": int(authorized["pre_onset_authorization"].sum()),
                "event_level_sensitivity": _safe_divide(len(authorized), len(group)),
                "median_latency_seconds": float(np.median(latencies)) if len(latencies) else np.nan,
                "iqr_latency_seconds": (
                    float(np.percentile(latencies, 75) - np.percentile(latencies, 25))
                    if len(latencies) >= 2
                    else np.nan
                ),
                "false_stimulations_per_hour_mean": float(fold_group["false_stimulations_per_hour"].mean()),
                "false_stimulations_per_hour_std": float(
                    fold_group["false_stimulations_per_hour"].std(ddof=1)
                ),
            }
        )

    result = pd.DataFrame(records)
    result["false_stimulations_per_hour_std"] = result["false_stimulations_per_hour_std"].fillna(0.0)
    return result


def _plot_lightgbm_calibration(calibration_bins: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    modes = [
        "held_out_subject",
        "within_subject_chronological",
    ]
    labels = {
        "held_out_subject": "Held-out subject",
        "within_subject_chronological": "Within-subject chronological",
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharex=True, sharey=True)
    for axis, mode in zip(axes, modes):
        rows = calibration_bins[
            (calibration_bins["model"] == "lightgbm")
            & (calibration_bins["evaluation_mode"] == mode)
        ].copy()
        rows = rows.dropna(subset=["mean_predicted_probability", "observed_positive_rate"])
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.45", linewidth=1.0)
        axis.plot(
            rows["mean_predicted_probability"],
            rows["observed_positive_rate"],
            marker="o",
            color="#1f77b4",
            linewidth=1.8,
        )
        axis.set_title(labels[mode])
        axis.set_xlabel("Mean predicted probability")
        axis.grid(True, alpha=0.25)
    axes[0].set_ylabel("Observed positive fraction")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure6_lightgbm_calibration_real_eeg.png", dpi=300)
    plt.close(fig)


def _make_manuscript_ready_tables(
    governance_agg: pd.DataFrame,
    calibration_agg: pd.DataFrame,
    event_agg: pd.DataFrame,
) -> None:
    table4 = governance_agg.copy()
    table4["stimulations_per_hour"] = [
        _format_mean_std(row.stimulations_per_hour_mean, row.stimulations_per_hour_std, 2)
        for row in table4.itertuples()
    ]
    table4["false_stimulations_per_hour"] = [
        _format_mean_std(row.false_stimulations_per_hour_mean, row.false_stimulations_per_hour_std, 2)
        for row in table4.itertuples()
    ]
    table4["window_level_sensitivity"] = [
        _format_mean_std(row.window_level_sensitivity_mean, row.window_level_sensitivity_std, 3)
        for row in table4.itertuples()
    ]
    table4["mean_s_t_triggered"] = [
        _format_mean_std(row.mean_s_t_triggered_mean, row.mean_s_t_triggered_std, 3)
        for row in table4.itertuples()
    ]
    table4[
        [
            "evaluation_mode",
            "policy",
            "stimulations_per_hour",
            "false_stimulations_per_hour",
            "window_level_sensitivity",
            "mean_s_t_triggered",
        ]
    ].to_csv(TABLES_DIR / "table4_explanation_stability_governance_latex.csv", index=False)

    table5 = calibration_agg.copy()
    table5["brier_score"] = [
        _format_mean_std(row.brier_score_mean, row.brier_score_std, 3)
        for row in table5.itertuples()
    ]
    table5["expected_calibration_error"] = [
        _format_mean_std(row.expected_calibration_error_mean, row.expected_calibration_error_std, 3)
        for row in table5.itertuples()
    ]
    table5[["evaluation_mode", "model", "brier_score", "expected_calibration_error"]].to_csv(
        TABLES_DIR / "table5_calibration_latex.csv",
        index=False,
    )

    table_s1 = event_agg.copy()
    table_s1["event_count"] = [
        f"{int(row.events_with_authorization)}/{int(row.seizure_events)}"
        for row in table_s1.itertuples()
    ]
    table_s1["false_stimulations_per_hour"] = [
        _format_mean_std(row.false_stimulations_per_hour_mean, row.false_stimulations_per_hour_std, 2)
        for row in table_s1.itertuples()
    ]
    table_s1[
        [
            "evaluation_mode",
            "policy",
            "event_count",
            "pre_onset_authorizations",
            "event_level_sensitivity",
            "median_latency_seconds",
            "iqr_latency_seconds",
            "false_stimulations_per_hour",
        ]
    ].to_csv(TABLES_DIR / "tableS1_event_level_policy_latex.csv", index=False)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    governance_config = _load_governance_config()

    features_df = pd.read_parquet(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    dataset_summary = pd.read_csv(DATASET_SUMMARY_PATH)
    event_df = _events_from_dataset_summary(dataset_summary)

    feature_columns = [column for column in features_df.columns if column not in METADATA_COLUMNS]
    feature_matrix = features_df.set_index("window_id")[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    binary_labels = features_df.set_index("window_id")["binary_label"].astype(int)

    calibration_records: List[Dict] = []
    calibration_bin_records: List[pd.DataFrame] = []
    governance_records: List[Dict] = []
    event_records: List[Dict] = []
    event_detail_records: List[Dict] = []

    grouped = labels_df.groupby(["evaluation_mode", "fold_id", "fold_subject_id"], sort=False)
    for (evaluation_mode, fold_id, fold_subject_id), fold_rows in grouped:
        fold_rows = fold_rows.sort_values(["subject_id", "edf_file", "window_index"]).reset_index(drop=True)
        train_rows = fold_rows[fold_rows["split"] == "train"].copy()
        test_rows = fold_rows[fold_rows["split"] == "test"].copy()
        test_rows = test_rows.sort_values(["subject_id", "edf_file", "window_index"]).reset_index(drop=True)

        X_train = feature_matrix.loc[train_rows["window_id"]].copy()
        y_train = binary_labels.loc[train_rows["window_id"]].to_numpy()
        X_test = feature_matrix.loc[test_rows["window_id"]].copy()
        y_test = binary_labels.loc[test_rows["window_id"]].to_numpy()

        models = _fit_models(X_train, y_train)
        test_hours = _sum_unique_recording_hours(test_rows)
        context = {
            "evaluation_mode": evaluation_mode,
            "fold_id": fold_id,
            "fold_subject_id": fold_subject_id,
            "test_hours": test_hours,
        }

        for model_name, model in models.items():
            probability = _predict_probability(model, X_test)
            ece, bins = _calibration_summary(y_test, probability, governance_config.calibration_bins)
            brier = float(np.mean((probability - y_test) ** 2))
            calibration_records.append(
                {
                    **context,
                    "model": model_name,
                    "model_label": MODEL_LABELS[model_name],
                    "brier_score": brier,
                    "expected_calibration_error": ece,
                }
            )
            bins.insert(0, "model", model_name)
            bins.insert(0, "fold_subject_id", fold_subject_id)
            bins.insert(0, "fold_id", fold_id)
            bins.insert(0, "evaluation_mode", evaluation_mode)
            calibration_bin_records.append(bins)

        lightgbm_model = models["lightgbm"]
        probability = _predict_probability(lightgbm_model, X_test)
        topk = _topk_indices_from_lightgbm(lightgbm_model, X_test, governance_config.top_k)
        stability, previous_index, has_previous = _explanation_stability(test_rows, topk, governance_config.top_k)

        previous_probability = np.full(len(probability), np.nan, dtype=float)
        valid_previous = previous_index >= 0
        previous_probability[valid_previous] = probability[previous_index[valid_previous]]

        fixed = probability > governance_config.theta
        temporal = fixed & has_previous & (previous_probability > governance_config.theta)
        attribution_stable = temporal & (stability > governance_config.gamma)

        policy_triggers = {
            "Fixed threshold": fixed.astype(int),
            "Temporal persistence": temporal.astype(int),
            "Attribution-stability gating": attribution_stable.astype(int),
        }

        for policy_name, triggers in policy_triggers.items():
            governance_records.append(
                _policy_metrics(test_rows, y_test, triggers, stability, policy_name, context)
            )
            event_records.append(
                _event_level_metrics(
                    test_rows,
                    triggers,
                    event_df,
                    policy_name,
                    context,
                    governance_config.preictal_seconds,
                )
            )
            event_detail_records.extend(
                _event_detail_records(
                    test_rows,
                    triggers,
                    event_df,
                    policy_name,
                    context,
                    governance_config.preictal_seconds,
                )
            )

    calibration_df = pd.DataFrame(calibration_records)
    calibration_bins_df = pd.concat(calibration_bin_records, ignore_index=True)
    governance_df = pd.DataFrame(governance_records)
    events_by_fold_df = pd.DataFrame(event_records)
    event_details_df = pd.DataFrame(event_detail_records)

    calibration_agg = _aggregate_mean_std(
        calibration_df,
        ["evaluation_mode", "model", "model_label"],
        ["brier_score", "expected_calibration_error"],
    )
    governance_agg = _aggregate_mean_std(
        governance_df,
        ["evaluation_mode", "model", "policy"],
        [
            "total_stimulations",
            "stimulations_per_hour",
            "false_stimulations_per_hour",
            "window_level_sensitivity",
            "mean_s_t_triggered",
            "median_s_t_triggered",
        ],
    )
    event_agg = _aggregate_event_details(event_details_df, events_by_fold_df)

    calibration_df.to_csv(TABLES_DIR / "table5_calibration_by_fold_real_eeg.csv", index=False)
    calibration_bins_df.to_csv(TABLES_DIR / "table5_calibration_bins_real_eeg.csv", index=False)
    calibration_agg.to_csv(TABLES_DIR / "table5_calibration_real_eeg.csv", index=False)
    governance_df.to_csv(TABLES_DIR / "table4_explanation_stability_governance_by_fold_real_eeg.csv", index=False)
    governance_agg.to_csv(TABLES_DIR / "table4_explanation_stability_governance_real_eeg.csv", index=False)
    events_by_fold_df.to_csv(TABLES_DIR / "tableS1_event_level_policy_by_fold_real_eeg.csv", index=False)
    event_details_df.to_csv(TABLES_DIR / "tableS1_event_level_policy_by_event_real_eeg.csv", index=False)
    event_agg.to_csv(TABLES_DIR / "tableS1_event_level_policy_real_eeg.csv", index=False)

    _plot_lightgbm_calibration(calibration_bins_df)
    _make_manuscript_ready_tables(governance_agg, calibration_agg, event_agg)

    print("Saved governance supplement tables and calibration figure.")
    print(f"top_k={governance_config.top_k}, gamma={governance_config.gamma}, theta={governance_config.theta}")


if __name__ == "__main__":
    main()
