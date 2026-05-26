from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import yaml


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
    "logistic_regression": "Logistic regression",
    "random_forest": "Random forest",
    "lightgbm": "LightGBM",
}


@dataclass(frozen=True)
class RevisionConfig:
    theta: float = 0.5
    top_k: int = 5
    gamma: float = 0.4
    n_bins: int = 10
    preictal_seconds: float = 300.0
    calibration_fraction: float = 0.3
    bootstrap_iterations: int = 2000
    random_seed: int = 42
    cooldown_seconds: Tuple[int, ...] = (0, 30, 60, 120)


def load_revision_config() -> RevisionConfig:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    policies = config.get("policies", {})
    windowing = config.get("windowing", {})
    return RevisionConfig(
        theta=float(policies.get("fixed_threshold", 0.5)),
        top_k=int(policies.get("top_k_attribution_features", 5)),
        gamma=float(policies.get("attribution_stability_gamma", 0.4)),
        n_bins=int(config.get("calibration_bins", 10)),
        preictal_seconds=float(windowing.get("preictal_sec", 300.0)),
        random_seed=int(config.get("random_seed", 42)),
    )


def safe_auc(metric_fn, y_true: np.ndarray, y_score: np.ndarray) -> float:
    if len(np.unique(y_true)) < 2:
        return np.nan
    return float(metric_fn(y_true, y_score))


def safe_divide(numerator: float, denominator: float) -> float:
    return float(numerator / denominator) if denominator > 0 else np.nan


def logit(probability: np.ndarray) -> np.ndarray:
    clipped = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(clipped / (1.0 - clipped))


def fit_models(X_train: pd.DataFrame, y_train: np.ndarray, seed: int) -> Dict[str, object]:
    models = {
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
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=100,
            random_state=seed,
            class_weight="balanced",
            n_jobs=-1,
        ),
        "lightgbm": LGBMClassifier(
            random_state=seed,
            class_weight="balanced",
            verbose=-1,
        ),
    }
    for model in models.values():
        model.fit(X_train, y_train)
    return models


def predict_probability(model: object, X: pd.DataFrame) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probability = model.predict_proba(X)[:, 1]
    else:
        probability = model.predict(X)
    return np.nan_to_num(probability, nan=0.0, posinf=1.0, neginf=0.0)


def split_core_calibration(train_rows: pd.DataFrame, calibration_fraction: float) -> Tuple[pd.DataFrame, pd.DataFrame, Dict]:
    train_rows = train_rows.copy()
    train_rows["recording_key"] = train_rows["subject_id"] + "|" + train_rows["edf_file"]

    recording_info = (
        train_rows.groupby(["subject_id", "edf_file", "recording_order"], sort=False)["binary_label"]
        .max()
        .reset_index()
        .sort_values(["subject_id", "recording_order"])
    )
    calibration_keys: set[str] = set()
    for _, group in recording_info.groupby("subject_id", sort=False):
        n_cal = max(1, int(round(len(group) * calibration_fraction)))
        selected = group.tail(n_cal)
        calibration_keys.update(selected["subject_id"] + "|" + selected["edf_file"])

    cal_mask = train_rows["recording_key"].isin(calibration_keys)
    core_rows = train_rows[~cal_mask].copy()
    cal_rows = train_rows[cal_mask].copy()

    def has_two_classes(rows: pd.DataFrame) -> bool:
        return rows["binary_label"].nunique() == 2

    fallback_used = False
    if len(core_rows) == 0 or len(cal_rows) == 0 or not has_two_classes(core_rows) or not has_two_classes(cal_rows):
        fallback_used = True
        positive_recordings = set(
            recording_info[recording_info["binary_label"] == 1]
            .tail(max(1, int(round(recording_info["binary_label"].sum() * calibration_fraction))))
            .assign(recording_key=lambda df: df["subject_id"] + "|" + df["edf_file"])["recording_key"]
        )
        negative_recordings = set(
            recording_info[recording_info["binary_label"] == 0]
            .tail(max(1, int(round((len(recording_info) - recording_info["binary_label"].sum()) * calibration_fraction)))) 
            .assign(recording_key=lambda df: df["subject_id"] + "|" + df["edf_file"])["recording_key"]
        )
        calibration_keys = positive_recordings.union(negative_recordings)
        cal_mask = train_rows["recording_key"].isin(calibration_keys)
        core_rows = train_rows[~cal_mask].copy()
        cal_rows = train_rows[cal_mask].copy()

    audit = {
        "core_windows": int(len(core_rows)),
        "calibration_windows": int(len(cal_rows)),
        "core_positive_windows": int(core_rows["binary_label"].sum()),
        "calibration_positive_windows": int(cal_rows["binary_label"].sum()),
        "core_recordings": int(core_rows["recording_key"].nunique()),
        "calibration_recordings": int(cal_rows["recording_key"].nunique()),
        "calibration_fallback_used": bool(fallback_used),
        "core_has_both_classes": bool(has_two_classes(core_rows)),
        "calibration_has_both_classes": bool(has_two_classes(cal_rows)),
        "core_calibration_recording_overlap": int(
            len(
                set(core_rows["recording_key"].unique()).intersection(
                    set(cal_rows["recording_key"].unique())
                )
            )
        ),
    }
    if not audit["core_has_both_classes"] or not audit["calibration_has_both_classes"]:
        raise RuntimeError("Nested calibration split does not contain both classes.")
    return core_rows, cal_rows, audit


def fit_calibrators(y_cal: np.ndarray, p_cal: np.ndarray) -> Dict[str, object]:
    sigmoid = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    sigmoid.fit(logit(p_cal).reshape(-1, 1), y_cal)

    isotonic = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    isotonic.fit(p_cal, y_cal)
    return {"raw": None, "sigmoid": sigmoid, "isotonic": isotonic}


def apply_calibrator(calibrator: object, p_test: np.ndarray, method: str) -> np.ndarray:
    if method == "raw":
        return p_test
    if method == "sigmoid":
        return calibrator.predict_proba(logit(p_test).reshape(-1, 1))[:, 1]
    if method == "isotonic":
        return calibrator.predict(p_test)
    raise ValueError(f"Unknown calibration method: {method}")


def calibration_slope_intercept(y_true: np.ndarray, probability: np.ndarray) -> Tuple[float, float]:
    if len(np.unique(y_true)) < 2:
        return np.nan, np.nan
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
    model.fit(logit(probability).reshape(-1, 1), y_true)
    return float(model.coef_[0, 0]), float(model.intercept_[0])


def calibration_bins(y_true: np.ndarray, probability: np.ndarray, n_bins: int) -> Tuple[float, pd.DataFrame]:
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bin_ids = np.digitize(probability, edges[1:-1], right=False)
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
        mean_probability = float(probability[mask].mean())
        observed_rate = float(y_true[mask].mean())
        absolute_error = abs(mean_probability - observed_rate)
        ece += (count / n) * absolute_error
        records.append(
            {
                "bin": bin_id + 1,
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_predicted_probability": mean_probability,
                "observed_positive_rate": observed_rate,
                "absolute_calibration_error": absolute_error,
            }
        )
    return float(ece), pd.DataFrame(records)


def topk_contributions(model_name: str, model: object, X: pd.DataFrame, top_k: int) -> np.ndarray:
    if model_name == "lightgbm":
        contributions = model.booster_.predict(X, pred_contrib=True)[:, :-1]
        attribution = np.abs(contributions)
    elif model_name == "logistic_regression":
        scaler = model.named_steps["scaler"]
        classifier = model.named_steps["classifier"]
        transformed = scaler.transform(X)
        attribution = np.abs(transformed * classifier.coef_[0])
    else:
        importances = np.asarray(model.feature_importances_, dtype=float)
        attribution = np.tile(importances.reshape(1, -1), (len(X), 1))
    return np.argsort(-attribution, axis=1)[:, :top_k]


def explanation_stability(rows: pd.DataFrame, topk: np.ndarray, top_k: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = rows.reset_index(drop=True)
    stability = np.full(len(rows), np.nan)
    previous_index = np.full(len(rows), -1, dtype=int)
    has_previous = np.zeros(len(rows), dtype=bool)
    for _, group in rows.groupby(["subject_id", "edf_file"], sort=False):
        indices = group.index.to_numpy()
        if len(indices) < 2:
            continue
        for previous, current in zip(indices[:-1], indices[1:]):
            overlap = len(set(topk[current]).intersection(topk[previous]))
            stability[current] = overlap / float((2 * top_k) - overlap)
            previous_index[current] = previous
            has_previous[current] = True
    return stability, previous_index, has_previous


def parse_event_seconds(value: object) -> List[float]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text:
        return []
    return [float(part) for part in text.split(";") if part.strip()]


def events_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    records = []
    for row in summary.itertuples(index=False):
        starts = parse_event_seconds(row.seizure_onsets)
        offsets = parse_event_seconds(row.seizure_offsets)
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


def authorization_clusters(rows: pd.DataFrame, triggers: np.ndarray, cooldown_seconds: float = 0.0) -> Dict:
    rows = rows.reset_index(drop=True).copy()
    rows["trigger"] = triggers.astype(int)
    rows["binary_label_array"] = rows["binary_label"].astype(int)
    clusters: List[Dict] = []
    for _, group in rows.groupby(["subject_id", "edf_file"], sort=False):
        group = group.sort_values("window_start")
        active = False
        cluster_has_positive = False
        cluster_start = np.nan
        cluster_end = np.nan

        def close_cluster() -> None:
            clusters.append(
                {
                    "has_positive": bool(cluster_has_positive),
                    "duration_seconds": float(max(0.0, cluster_end - cluster_start)),
                }
            )

        for row in group.itertuples(index=False):
            if row.trigger != 1:
                continue
            window_start = float(row.window_start)
            window_end = float(row.window_end)
            is_positive = bool(row.binary_label_array == 1)
            if not active:
                active = True
                cluster_start = window_start
                cluster_end = window_end
                cluster_has_positive = is_positive
                continue
            if window_start <= cluster_end + cooldown_seconds:
                cluster_end = max(cluster_end, window_end)
                cluster_has_positive = cluster_has_positive or is_positive
            else:
                close_cluster()
                active = True
                cluster_start = window_start
                cluster_end = window_end
                cluster_has_positive = is_positive
        if active:
            close_cluster()

    durations = [cluster["duration_seconds"] for cluster in clusters]
    false_clusters = [cluster for cluster in clusters if not cluster["has_positive"]]
    return {
        "authorization_clusters": int(len(clusters)),
        "false_authorization_clusters": int(len(false_clusters)),
        "median_authorization_cluster_duration_seconds": float(np.median(durations)) if durations else np.nan,
    }


def event_metrics(rows: pd.DataFrame, triggers: np.ndarray, events: pd.DataFrame, preictal_seconds: float) -> Dict:
    test_recordings = rows[["subject_id", "edf_file"]].drop_duplicates()
    fold_events = events.merge(test_recordings, on=["subject_id", "edf_file"], how="inner")
    rows = rows.reset_index(drop=True).copy()
    rows["trigger"] = triggers.astype(int)
    latencies = []
    authorized = 0
    pre_onset = 0
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
        latency = float(event_rows.iloc[0]["window_center"]) - float(event.seizure_onset)
        latencies.append(latency)
        authorized += 1
        pre_onset += int(latency < 0)
    return {
        "seizure_events": int(len(fold_events)),
        "events_with_authorization": int(authorized),
        "pre_onset_authorizations": int(pre_onset),
        "event_level_coverage": safe_divide(authorized, len(fold_events)),
        "median_lead_time_seconds": float(-np.median(latencies)) if latencies else np.nan,
        "median_latency_seconds": float(np.median(latencies)) if latencies else np.nan,
    }


def policy_metrics(
    rows: pd.DataFrame,
    y_true: np.ndarray,
    triggers: np.ndarray,
    stability: np.ndarray | None,
    events: pd.DataFrame,
    cfg: RevisionConfig,
    context: Dict,
    model_name: str,
    policy: str,
    policy_settings: Dict | None = None,
    cluster_cooldown_seconds: float = 0.0,
) -> Dict:
    test_hours = float(context["test_hours"])
    triggers = triggers.astype(int)
    total = int(triggers.sum())
    true_authorizations = int(((triggers == 1) & (y_true == 1)).sum())
    false_authorizations = int(((triggers == 1) & (y_true == 0)).sum())
    cluster_summary = authorization_clusters(rows, triggers, cooldown_seconds=cluster_cooldown_seconds)
    event_summary = event_metrics(rows, triggers, events, cfg.preictal_seconds)
    if stability is None:
        mean_stability = np.nan
    else:
        selected = stability[triggers == 1]
        selected = selected[~np.isnan(selected)]
        mean_stability = float(selected.mean()) if len(selected) else np.nan
    return {
        **context,
        "model": model_name,
        "model_label": MODEL_LABELS[model_name],
        "policy": policy,
        **(policy_settings or {}),
        "cluster_cooldown_seconds": float(cluster_cooldown_seconds),
        "authorized_windows": total,
        "authorization_rate": safe_divide(total, len(y_true)),
        "authorization_windows_per_hour": safe_divide(total, test_hours),
        "false_authorization_windows": false_authorizations,
        "false_authorization_windows_per_hour": safe_divide(false_authorizations, test_hours),
        "positive_authorization_rate": safe_divide(true_authorizations, total),
        "window_level_sensitivity": safe_divide(true_authorizations, int(y_true.sum())),
        "mean_s_t_authorized": mean_stability,
        **cluster_summary,
        "authorization_clusters_per_hour": safe_divide(
            cluster_summary["authorization_clusters"], test_hours
        ),
        "false_authorization_clusters_per_hour": safe_divide(
            cluster_summary["false_authorization_clusters"], test_hours
        ),
        **event_summary,
    }


def recording_hours(rows: pd.DataFrame) -> float:
    return float(rows.drop_duplicates(["subject_id", "edf_file"])["duration_seconds"].sum() / 3600.0)


def bootstrap_ci(values: Iterable[float], iterations: int, seed: int) -> Tuple[float, float]:
    values = np.asarray([v for v in values if not pd.isna(v)], dtype=float)
    if len(values) == 0:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    means = [rng.choice(values, size=len(values), replace=True).mean() for _ in range(iterations)]
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def aggregate_with_ci(df: pd.DataFrame, group_cols: List[str], metric_cols: List[str], cfg: RevisionConfig) -> pd.DataFrame:
    records = []
    for keys, group in df.groupby(group_cols, sort=False, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        record = dict(zip(group_cols, keys))
        record["n_folds"] = int(group["fold_id"].nunique()) if "fold_id" in group else len(group)
        for metric in metric_cols:
            values = group[metric].dropna().to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(values.mean()) if len(values) else np.nan
            record[f"{metric}_std"] = float(values.std(ddof=1)) if len(values) > 1 else 0.0
            ci_low, ci_high = bootstrap_ci(values, cfg.bootstrap_iterations, cfg.random_seed)
            record[f"{metric}_ci95_low"] = ci_low
            record[f"{metric}_ci95_high"] = ci_high
        records.append(record)
    return pd.DataFrame(records)


def leakage_audit(labels_df: pd.DataFrame) -> pd.DataFrame:
    records = []
    for (evaluation_mode, fold_id, fold_subject_id), fold_rows in labels_df.groupby(
        ["evaluation_mode", "fold_id", "fold_subject_id"], sort=False
    ):
        train = fold_rows[fold_rows["split"] == "train"].copy()
        test = fold_rows[fold_rows["split"] == "test"].copy()
        train_keys = set((train["subject_id"] + "|" + train["edf_file"]).unique())
        test_keys = set((test["subject_id"] + "|" + test["edf_file"]).unique())
        train_subjects = set(train["subject_id"].unique())
        test_subjects = set(test["subject_id"].unique())
        chronology_ok = True
        if evaluation_mode == "within_subject_chronological":
            for subject_id, subject_rows in fold_rows.groupby("subject_id"):
                train_orders = subject_rows[subject_rows["split"] == "train"]["recording_order"]
                test_orders = subject_rows[subject_rows["split"] == "test"]["recording_order"]
                if len(train_orders) and len(test_orders) and train_orders.max() >= test_orders.min():
                    chronology_ok = False
        records.append(
            {
                "evaluation_mode": evaluation_mode,
                "fold_id": fold_id,
                "fold_subject_id": fold_subject_id,
                "train_windows": int(len(train)),
                "test_windows": int(len(test)),
                "train_positive_windows": int(train["binary_label"].sum()),
                "test_positive_windows": int(test["binary_label"].sum()),
                "edf_overlap_train_test": int(len(train_keys.intersection(test_keys))),
                "subject_overlap_train_test": int(len(train_subjects.intersection(test_subjects))),
                "held_out_subject_absent_from_train": bool(
                    fold_subject_id not in train_subjects if evaluation_mode == "held_out_subject" else True
                ),
                "within_subject_chronology_ok": bool(chronology_ok),
                "split_is_file_level": bool(len(train_keys.intersection(test_keys)) == 0),
                "overlapping_windows_cross_split_possible": bool(len(train_keys.intersection(test_keys)) > 0),
            }
        )
    return pd.DataFrame(records)


def threshold_baseline_score(X: pd.DataFrame) -> np.ndarray:
    values = X["line_length"].to_numpy(dtype=float)
    min_value = np.nanmin(values)
    max_value = np.nanmax(values)
    if max_value <= min_value:
        return np.zeros_like(values)
    return (values - min_value) / (max_value - min_value)


def display_mode(mode: str) -> str:
    return {
        "held_out_subject": "Held-out subject",
        "within_subject_chronological": "Within-subject chronological",
    }.get(mode, mode)


def plot_model_performance(performance_summary: pd.DataFrame, baseline_summary: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    modes = ["held_out_subject", "within_subject_chronological"]
    model_order = ["logistic_regression", "random_forest", "lightgbm", "line_length_score"]
    labels = {
        "logistic_regression": "Logistic\nregression",
        "random_forest": "Random\nforest",
        "lightgbm": "LightGBM",
        "line_length_score": "Line\nlength",
    }
    colors = {
        "logistic_regression": "#4c78a8",
        "random_forest": "#72b7b2",
        "lightgbm": "#f58518",
        "line_length_score": "#8f8f8f",
    }

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.2), sharey="col")
    for row_idx, mode in enumerate(modes):
        model_rows = performance_summary[performance_summary["evaluation_mode"] == mode].copy()
        baseline_rows = baseline_summary[baseline_summary["evaluation_mode"] == mode].copy()
        line_row = baseline_rows[baseline_rows["baseline"] == "line_length_score"].iloc[0]
        prevalence_row = baseline_rows[baseline_rows["baseline"] == "test_prevalence"].iloc[0]
        records = []
        for model in ["logistic_regression", "random_forest", "lightgbm"]:
            row = model_rows[model_rows["model"] == model].iloc[0]
            records.append(
                {
                    "model": model,
                    "auroc_mean": row.auroc_mean,
                    "auroc_std": row.auroc_std,
                    "auprc_mean": row.auprc_mean,
                    "auprc_std": row.auprc_std,
                }
            )
        records.append(
            {
                "model": "line_length_score",
                "auroc_mean": line_row.auroc_mean,
                "auroc_std": line_row.auroc_std,
                "auprc_mean": line_row.auprc_mean,
                "auprc_std": line_row.auprc_std,
            }
        )
        plot_df = pd.DataFrame(records).set_index("model").loc[model_order].reset_index()
        x = np.arange(len(plot_df))
        for col_idx, metric in enumerate(["auroc", "auprc"]):
            axis = axes[row_idx, col_idx]
            axis.bar(
                x,
                plot_df[f"{metric}_mean"],
                yerr=plot_df[f"{metric}_std"],
                capsize=3,
                color=[colors[model] for model in plot_df["model"]],
                edgecolor="0.2",
                linewidth=0.5,
            )
            reference = 0.5 if metric == "auroc" else float(prevalence_row.auprc_mean)
            axis.axhline(reference, color="0.25", linestyle="--", linewidth=1.1)
            axis.set_xticks(x)
            axis.set_xticklabels([labels[model] for model in plot_df["model"]], fontsize=8)
            axis.set_ylim(0, 1.0 if metric == "auroc" else max(0.42, plot_df[f"{metric}_mean"].max() + 0.25))
            axis.grid(True, axis="y", alpha=0.22)
            if row_idx == 0:
                axis.set_title("AUROC" if metric == "auroc" else "AUPRC")
            if col_idx == 0:
                axis.set_ylabel(display_mode(mode))
            if metric == "auprc":
                axis.text(
                    0.02,
                    reference + 0.01,
                    "prevalence",
                    transform=axis.get_yaxis_transform(),
                    fontsize=8,
                    color="0.25",
                    va="bottom",
                )
            else:
                axis.text(
                    0.02,
                    reference + 0.02,
                    "chance",
                    transform=axis.get_yaxis_transform(),
                    fontsize=8,
                    color="0.25",
                    va="bottom",
                )
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure2_model_performance_real_eeg.png", dpi=300)
    plt.close(fig)


def plot_nested_calibration(bins: pd.DataFrame) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    methods = [("raw", "Raw"), ("sigmoid", "Platt"), ("isotonic", "Isotonic")]
    modes = [
        ("held_out_subject", "Held-out subject"),
        ("within_subject_chronological", "Within-subject chronological"),
    ]
    colors = {"raw": "#4c78a8", "sigmoid": "#f58518", "isotonic": "#54a24b"}

    fig = plt.figure(figsize=(11.8, 6.2))
    grid = gridspec.GridSpec(2, 2, height_ratios=[3.0, 1.0], hspace=0.12, wspace=0.22)
    reliability_axes = [fig.add_subplot(grid[0, i]) for i in range(2)]
    histogram_axes = [fig.add_subplot(grid[1, i], sharex=reliability_axes[i]) for i in range(2)]

    for axis, hist_axis, (mode, label) in zip(reliability_axes, histogram_axes, modes):
        axis.plot([0, 1], [0, 1], linestyle="--", color="0.35", linewidth=1.0)
        mode_rows = bins[(bins["evaluation_mode"] == mode) & (bins["model"] == "lightgbm")]
        for method, method_label in methods:
            rows = mode_rows[mode_rows["calibration_method"] == method].dropna(
                subset=["mean_predicted_probability", "observed_positive_rate"]
            )
            for _, fold_rows in rows.groupby("fold_id", sort=False):
                sizes = 10 + 90 * (fold_rows["count"] / max(1, rows["count"].max()))
                axis.scatter(
                    fold_rows["mean_predicted_probability"],
                    fold_rows["observed_positive_rate"],
                    s=sizes,
                    color=colors[method],
                    alpha=0.22,
                    edgecolors="none",
                )
            mean_rows = (
                rows.groupby("bin", as_index=False)
                .agg(
                    mean_predicted_probability=("mean_predicted_probability", "mean"),
                    observed_positive_rate=("observed_positive_rate", "mean"),
                    count=("count", "sum"),
                )
                .dropna()
            )
            axis.plot(
                mean_rows["mean_predicted_probability"],
                mean_rows["observed_positive_rate"],
                marker="o",
                linewidth=2.2,
                color=colors[method],
                label=method_label,
            )
            hist_rows = (
                rows.groupby(["bin", "bin_lower", "bin_upper"], as_index=False)
                .agg(count=("count", "sum"))
                .sort_values("bin")
            )
            centers = (hist_rows["bin_lower"] + hist_rows["bin_upper"]) / 2.0
            hist_axis.step(
                centers,
                hist_rows["count"],
                where="mid",
                color=colors[method],
                linewidth=1.6,
                label=method_label,
            )
        axis.set_title(label)
        axis.set_ylabel("Observed positive fraction")
        axis.grid(True, alpha=0.25)
        hist_axis.set_xlabel("Predicted probability bin")
        hist_axis.set_ylabel("Samples")
        hist_axis.set_yscale("log")
        hist_axis.grid(True, axis="y", alpha=0.22)
    reliability_axes[1].legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure3_calibration_diagnostics_real_eeg.png", dpi=300)
    fig.savefig(FIGURES_DIR / "figure7_nested_calibration_comparison_real_eeg.png", dpi=300)
    plt.close(fig)


def plot_attribution_stability(policy_df: pd.DataFrame, diagnostic_df: pd.DataFrame, stability_windows: pd.DataFrame) -> None:
    lightgbm = policy_df[policy_df["model"] == "lightgbm"].copy()
    policies = ["Fixed threshold", "Threshold + attribution-stability gate"]
    modes = ["held_out_subject", "within_subject_chronological"]
    fig, axes = plt.subplots(2, 2, figsize=(11.2, 7.4))

    burden = (
        lightgbm[lightgbm["policy"].isin(policies)]
        .groupby(["evaluation_mode", "policy"], as_index=False)
        .agg(
            false_authorization_clusters_per_hour=("false_authorization_clusters_per_hour", "mean"),
            mean_s_t_authorized=("mean_s_t_authorized", "mean"),
        )
    )
    x = np.arange(len(modes))
    width = 0.34
    for offset, policy in [(-width / 2, policies[0]), (width / 2, policies[1])]:
        rows = burden[burden["policy"] == policy].set_index("evaluation_mode").loc[modes]
        axes[0, 0].bar(
            x + offset,
            rows["false_authorization_clusters_per_hour"],
            width=width,
            label=policy,
        )
        axes[0, 1].bar(
            x + offset,
            rows["mean_s_t_authorized"],
            width=width,
            label=policy,
        )
    axes[0, 0].set_ylabel("False authorization clusters/h")
    axes[0, 1].set_ylabel("Mean $S_t$ among authorizations")
    for axis in axes[0]:
        axis.set_xticks(x)
        axis.set_xticklabels([display_mode(mode) for mode in modes], rotation=12, ha="right")
        axis.grid(True, axis="y", alpha=0.22)
    axes[0, 1].legend(frameon=False, fontsize=8, loc="best")

    box_data = []
    box_labels = []
    window_rows = stability_windows.dropna(subset=["s_t"]).copy()
    for label, value in [("True", 1), ("False", 0)]:
        values = window_rows[window_rows["binary_label"] == value]["s_t"].to_numpy(dtype=float)
        box_data.append(values)
        box_labels.append(label)
    axes[1, 0].boxplot(box_data, labels=box_labels, showfliers=False, patch_artist=True)
    jitter = np.random.default_rng(42)
    for idx, values in enumerate(box_data, start=1):
        sample = values if len(values) <= 900 else jitter.choice(values, size=900, replace=False)
        axes[1, 0].scatter(
            jitter.normal(idx, 0.045, size=len(sample)),
            sample,
            s=4,
            alpha=0.10,
            color="#4c78a8" if idx == 1 else "#f58518",
            edgecolors="none",
        )
    axes[1, 0].set_ylabel("$S_t$ among fixed-threshold authorizations")
    axes[1, 0].grid(True, axis="y", alpha=0.22)

    diagnostic = diagnostic_df[diagnostic_df["model"] == "lightgbm"].copy()
    positions = np.arange(len(modes))
    spearman_data = [
        diagnostic[diagnostic["evaluation_mode"] == mode]["probability_s_t_spearman"].dropna().to_numpy(dtype=float)
        for mode in modes
    ]
    axes[1, 1].boxplot(spearman_data, labels=[display_mode(mode) for mode in modes], showfliers=True)
    axes[1, 1].axhline(0, linestyle="--", color="0.35", linewidth=1.0)
    axes[1, 1].set_ylabel("Spearman($p_t$, $S_t$)")
    axes[1, 1].tick_params(axis="x", labelrotation=12)
    axes[1, 1].grid(True, axis="y", alpha=0.22)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure4_attribution_stability_governance_real_eeg.png", dpi=300)
    plt.close(fig)


def plot_pareto(sweep: pd.DataFrame) -> None:
    lightgbm = sweep[(sweep["model"] == "lightgbm") & (sweep["evaluation_mode"] == "held_out_subject")].copy()
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.1))
    for policy, rows in lightgbm.groupby("policy", sort=False):
        grouped = rows.groupby("theta", as_index=False).agg(
            event_level_coverage=("event_level_coverage", "mean"),
            false_authorization_clusters_per_hour=("false_authorization_clusters_per_hour", "mean"),
            authorization_clusters_per_hour=("authorization_clusters_per_hour", "mean"),
            mean_s_t_authorized=("mean_s_t_authorized", "mean"),
            authorization_rate=("authorization_rate", "mean"),
        )
        x_false = grouped["false_authorization_clusters_per_hour"].clip(lower=0.01)
        x_total = grouped["authorization_clusters_per_hour"].clip(lower=0.01)
        axes[0].plot(x_false, grouped["event_level_coverage"], marker="o", label=policy)
        axes[1].plot(grouped["authorization_rate"], grouped["mean_s_t_authorized"], marker="o", label=policy)
        axes[2].plot(x_total, grouped["event_level_coverage"], marker="o", label=policy)
    selected = lightgbm[
        (lightgbm["policy"] == "Full governed policy")
        & (np.isclose(lightgbm["theta"].astype(float), 0.5))
        & (np.isclose(lightgbm["gamma"].astype(float), 0.4))
    ]
    if not selected.empty:
        selected_mean = selected.agg(
            {
                "false_authorization_clusters_per_hour": "mean",
                "authorization_clusters_per_hour": "mean",
                "event_level_coverage": "mean",
                "authorization_rate": "mean",
                "mean_s_t_authorized": "mean",
            }
        )
        axes[0].scatter(
            max(float(selected_mean["false_authorization_clusters_per_hour"]), 0.01),
            selected_mean["event_level_coverage"],
            s=95,
            marker="*",
            color="black",
            zorder=5,
        )
        axes[1].scatter(
            selected_mean["authorization_rate"],
            selected_mean["mean_s_t_authorized"],
            s=95,
            marker="*",
            color="black",
            zorder=5,
        )
        axes[2].scatter(
            max(float(selected_mean["authorization_clusters_per_hour"]), 0.01),
            selected_mean["event_level_coverage"],
            s=95,
            marker="*",
            color="black",
            zorder=5,
        )
    axes[0].set_xscale("log")
    axes[2].set_xscale("log")
    axes[0].set_xlabel("False authorization clusters/h")
    axes[0].set_ylabel("Event-level authorization coverage")
    axes[1].set_xlabel("Positive authorization rate")
    axes[1].set_ylabel("Mean $S_t$ among authorizations")
    axes[2].set_xlabel("Authorization clusters/h")
    axes[2].set_ylabel("Event-level authorization coverage")
    for axis in axes:
        axis.grid(True, alpha=0.25)
    axes[2].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "figure5_burden_constrained_operating_curves_real_eeg.png", dpi=300)
    fig.savefig(FIGURES_DIR / "figure8_policy_burden_pareto_real_eeg.png", dpi=300)
    plt.close(fig)


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    cfg = load_revision_config()

    features_df = pd.read_parquet(FEATURES_PATH)
    labels_df = pd.read_csv(LABELS_PATH)
    summary_df = pd.read_csv(DATASET_SUMMARY_PATH)
    events_df = events_from_summary(summary_df)

    feature_columns = [column for column in features_df.columns if column not in METADATA_COLUMNS]
    X_all = features_df.set_index("window_id")[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    y_all = features_df.set_index("window_id")["binary_label"].astype(int)

    audit_df = leakage_audit(labels_df)
    audit_df.to_csv(TABLES_DIR / "table6_leakage_control_audit.csv", index=False)

    calibration_records: List[Dict] = []
    bin_records: List[pd.DataFrame] = []
    policy_records: List[Dict] = []
    cooldown_records: List[Dict] = []
    sweep_records: List[Dict] = []
    split_audit_records: List[Dict] = []
    baseline_records: List[Dict] = []
    stability_diagnostic_records: List[Dict] = []
    stability_window_records: List[Dict] = []

    for (evaluation_mode, fold_id, fold_subject_id), fold_rows in labels_df.groupby(
        ["evaluation_mode", "fold_id", "fold_subject_id"], sort=False
    ):
        fold_rows = fold_rows.sort_values(["subject_id", "edf_file", "window_index"]).reset_index(drop=True)
        train_rows = fold_rows[fold_rows["split"] == "train"].copy()
        test_rows = fold_rows[fold_rows["split"] == "test"].copy()
        test_rows = test_rows.sort_values(["subject_id", "edf_file", "window_index"]).reset_index(drop=True)

        core_rows, cal_rows, split_audit = split_core_calibration(train_rows, cfg.calibration_fraction)
        context = {
            "evaluation_mode": evaluation_mode,
            "fold_id": fold_id,
            "fold_subject_id": fold_subject_id,
            "test_hours": recording_hours(test_rows),
        }
        split_audit_records.append({**context, **split_audit})

        X_core = X_all.loc[core_rows["window_id"]]
        y_core = y_all.loc[core_rows["window_id"]].to_numpy()
        X_cal = X_all.loc[cal_rows["window_id"]]
        y_cal = y_all.loc[cal_rows["window_id"]].to_numpy()
        X_test = X_all.loc[test_rows["window_id"]]
        y_test = y_all.loc[test_rows["window_id"]].to_numpy()

        models = fit_models(X_core, y_core, cfg.random_seed)

        prevalence = float(y_test.mean())
        line_score = threshold_baseline_score(X_test)
        baseline_records.append(
            {
                **context,
                "baseline": "test_prevalence",
                "auroc": 0.5,
                "auprc": prevalence,
                "prevalence": prevalence,
            }
        )
        baseline_records.append(
            {
                **context,
                "baseline": "line_length_score",
                "auroc": safe_auc(roc_auc_score, y_test, line_score),
                "auprc": safe_auc(average_precision_score, y_test, line_score),
                "prevalence": prevalence,
            }
        )

        for model_name, model in models.items():
            p_cal_raw = predict_probability(model, X_cal)
            p_test_raw = predict_probability(model, X_test)
            calibrators = fit_calibrators(y_cal, p_cal_raw)
            topk = topk_contributions(model_name, model, X_test, cfg.top_k)
            stability, previous_index, has_previous = explanation_stability(test_rows, topk, cfg.top_k)
            previous_raw = np.full(len(p_test_raw), np.nan)
            previous_valid = previous_index >= 0
            previous_raw[previous_valid] = p_test_raw[previous_index[previous_valid]]

            calibrated_probabilities: Dict[str, np.ndarray] = {}
            for method, calibrator in calibrators.items():
                probability = apply_calibrator(calibrator, p_test_raw, method)
                calibrated_probabilities[method] = probability
                ece, bins = calibration_bins(y_test, probability, cfg.n_bins)
                slope, intercept = calibration_slope_intercept(y_test, probability)
                calibration_records.append(
                    {
                        **context,
                        "model": model_name,
                        "model_label": MODEL_LABELS[model_name],
                        "calibration_method": method,
                        "auroc": safe_auc(roc_auc_score, y_test, probability),
                        "auprc": safe_auc(average_precision_score, y_test, probability),
                        "brier_score": float(brier_score_loss(y_test, probability)),
                        "expected_calibration_error": ece,
                        "calibration_slope": slope,
                        "calibration_intercept": intercept,
                        "max_predicted_probability": float(np.max(probability)),
                    }
                )
                bins.insert(0, "calibration_method", method)
                bins.insert(0, "model", model_name)
                bins.insert(0, "fold_subject_id", fold_subject_id)
                bins.insert(0, "fold_id", fold_id)
                bins.insert(0, "evaluation_mode", evaluation_mode)
                bin_records.append(bins)

            p_iso = calibrated_probabilities["isotonic"]
            previous_iso = np.full(len(p_iso), np.nan)
            previous_iso[previous_valid] = p_iso[previous_index[previous_valid]]

            fixed = p_test_raw > cfg.theta
            temporal = fixed & has_previous & (previous_raw > cfg.theta)
            attribution = fixed & (stability > cfg.gamma)
            calibration_gate = fixed & (p_iso > cfg.theta)
            full = (p_iso > cfg.theta) & has_previous & (previous_iso > cfg.theta) & (stability > cfg.gamma)
            policies = {
                "Fixed threshold": fixed,
                "Threshold + temporal persistence": temporal,
                "Threshold + attribution-stability gate": attribution,
                "Threshold + calibration-aware gate": calibration_gate,
                "Full governed policy": full,
            }
            policy_settings = {
                "Fixed threshold": {
                    "threshold": cfg.theta,
                    "calibration_method": "raw",
                    "persistence_requirement_windows": 1,
                    "stability_threshold": np.nan,
                },
                "Threshold + temporal persistence": {
                    "threshold": cfg.theta,
                    "calibration_method": "raw",
                    "persistence_requirement_windows": 2,
                    "stability_threshold": np.nan,
                },
                "Threshold + attribution-stability gate": {
                    "threshold": cfg.theta,
                    "calibration_method": "raw",
                    "persistence_requirement_windows": 1,
                    "stability_threshold": cfg.gamma,
                },
                "Threshold + calibration-aware gate": {
                    "threshold": cfg.theta,
                    "calibration_method": "raw + isotonic",
                    "persistence_requirement_windows": 1,
                    "stability_threshold": np.nan,
                },
                "Full governed policy": {
                    "threshold": cfg.theta,
                    "calibration_method": "isotonic",
                    "persistence_requirement_windows": 2,
                    "stability_threshold": cfg.gamma,
                },
            }
            for policy_name, trigger_mask in policies.items():
                policy_records.append(
                    policy_metrics(
                        test_rows,
                        y_test,
                        trigger_mask.astype(int),
                        stability,
                        events_df,
                        cfg,
                        context,
                        model_name,
                        policy_name,
                        policy_settings[policy_name],
                        0.0,
                    )
                )
                if model_name == "lightgbm":
                    for cooldown in cfg.cooldown_seconds:
                        cooldown_records.append(
                            policy_metrics(
                                test_rows,
                                y_test,
                                trigger_mask.astype(int),
                                stability,
                                events_df,
                                cfg,
                                context,
                                model_name,
                                policy_name,
                                policy_settings[policy_name],
                                float(cooldown),
                            )
                        )

            if model_name == "lightgbm":
                valid = (~np.isnan(stability)) & (has_previous == True)
                fixed_mask = fixed & valid
                fixed_true = fixed_mask & (y_test == 1)
                fixed_false = fixed_mask & (y_test == 0)
                for idx in np.where(fixed_mask)[0]:
                    stability_window_records.append(
                        {
                            **context,
                            "model": model_name,
                            "window_id": test_rows.iloc[idx]["window_id"],
                            "binary_label": int(y_test[idx]),
                            "probability": float(p_test_raw[idx]),
                            "s_t": float(stability[idx]),
                        }
                    )
                stability_diagnostic_records.append(
                    {
                        **context,
                        "model": model_name,
                        "probability_s_t_spearman": float(
                            pd.Series(p_test_raw[valid]).corr(
                                pd.Series(stability[valid]), method="spearman"
                            )
                        )
                        if int(valid.sum()) > 2
                        else np.nan,
                        "mean_s_t_fixed_true_authorizations": float(stability[fixed_true].mean())
                        if int(fixed_true.sum())
                        else np.nan,
                        "mean_s_t_fixed_false_authorizations": float(stability[fixed_false].mean())
                        if int(fixed_false.sum())
                        else np.nan,
                        "mean_probability_fixed_true_authorizations": float(p_test_raw[fixed_true].mean())
                        if int(fixed_true.sum())
                        else np.nan,
                        "mean_probability_fixed_false_authorizations": float(p_test_raw[fixed_false].mean())
                        if int(fixed_false.sum())
                        else np.nan,
                        "fixed_true_authorizations": int(fixed_true.sum()),
                        "fixed_false_authorizations": int(fixed_false.sum()),
                    }
                )
                for theta in np.round(np.arange(0.1, 0.91, 0.1), 2):
                    for gamma in [0.0, 0.25, 0.4, 0.67]:
                        raw_fixed = p_test_raw > theta
                        raw_temporal = raw_fixed & has_previous & (previous_raw > theta)
                        full_sweep = (p_iso > theta) & has_previous & (previous_iso > theta) & (stability > gamma)
                        for policy_name, trigger_mask in {
                            "Fixed threshold": raw_fixed,
                            "Temporal persistence": raw_temporal,
                            "Full governed policy": full_sweep,
                        }.items():
                            if policy_name == "Fixed threshold":
                                sweep_settings = {
                                    "threshold": theta,
                                    "calibration_method": "raw",
                                    "persistence_requirement_windows": 1,
                                    "stability_threshold": np.nan,
                                }
                            elif policy_name == "Temporal persistence":
                                sweep_settings = {
                                    "threshold": theta,
                                    "calibration_method": "raw",
                                    "persistence_requirement_windows": 2,
                                    "stability_threshold": np.nan,
                                }
                            else:
                                sweep_settings = {
                                    "threshold": theta,
                                    "calibration_method": "isotonic",
                                    "persistence_requirement_windows": 2,
                                    "stability_threshold": gamma,
                                }
                            sweep_records.append(
                                {
                                    "theta": theta,
                                    "gamma": gamma,
                                    **policy_metrics(
                                        test_rows,
                                        y_test,
                                        trigger_mask.astype(int),
                                        stability,
                                        events_df,
                                        cfg,
                                        context,
                                        model_name,
                                        policy_name,
                                        sweep_settings,
                                        0.0,
                                    ),
                                }
                            )

    nested_calibration_df = pd.DataFrame(calibration_records)
    nested_bins_df = pd.concat(bin_records, ignore_index=True)
    policy_df = pd.DataFrame(policy_records)
    cooldown_df = pd.DataFrame(cooldown_records)
    sweep_df = pd.DataFrame(sweep_records)
    nested_split_audit_df = pd.DataFrame(split_audit_records)
    baseline_df = pd.DataFrame(baseline_records)
    stability_diagnostic_df = pd.DataFrame(stability_diagnostic_records)
    stability_window_df = pd.DataFrame(stability_window_records)

    calibration_agg = aggregate_with_ci(
        nested_calibration_df,
        ["evaluation_mode", "model", "model_label", "calibration_method"],
        [
            "auroc",
            "auprc",
            "brier_score",
            "expected_calibration_error",
            "calibration_slope",
            "calibration_intercept",
            "max_predicted_probability",
        ],
        cfg,
    )
    policy_agg = aggregate_with_ci(
        policy_df,
        [
            "evaluation_mode",
            "model",
            "model_label",
            "policy",
            "threshold",
            "calibration_method",
            "persistence_requirement_windows",
            "stability_threshold",
            "cluster_cooldown_seconds",
        ],
        [
            "authorized_windows",
            "authorization_rate",
            "authorization_windows_per_hour",
            "false_authorization_windows_per_hour",
            "positive_authorization_rate",
            "window_level_sensitivity",
            "mean_s_t_authorized",
            "authorization_clusters_per_hour",
            "false_authorization_clusters_per_hour",
            "median_authorization_cluster_duration_seconds",
            "event_level_coverage",
            "pre_onset_authorizations",
            "median_lead_time_seconds",
        ],
        cfg,
    )
    cooldown_agg = aggregate_with_ci(
        cooldown_df,
        [
            "evaluation_mode",
            "model",
            "model_label",
            "policy",
            "threshold",
            "calibration_method",
            "persistence_requirement_windows",
            "stability_threshold",
            "cluster_cooldown_seconds",
        ],
        [
            "authorization_windows_per_hour",
            "authorization_clusters_per_hour",
            "false_authorization_clusters_per_hour",
            "median_authorization_cluster_duration_seconds",
            "event_level_coverage",
            "pre_onset_authorizations",
            "median_lead_time_seconds",
            "positive_authorization_rate",
            "mean_s_t_authorized",
        ],
        cfg,
    )
    baseline_agg = aggregate_with_ci(
        baseline_df,
        ["evaluation_mode", "baseline"],
        ["auroc", "auprc", "prevalence"],
        cfg,
    )
    stability_diagnostic_agg = aggregate_with_ci(
        stability_diagnostic_df,
        ["evaluation_mode", "model"],
        [
            "probability_s_t_spearman",
            "mean_s_t_fixed_true_authorizations",
            "mean_s_t_fixed_false_authorizations",
            "mean_probability_fixed_true_authorizations",
            "mean_probability_fixed_false_authorizations",
        ],
        cfg,
    )
    selected_operating_points = cooldown_agg[
        (cooldown_agg["model"] == "lightgbm")
        & (np.isclose(cooldown_agg["cluster_cooldown_seconds"].astype(float), 0.0))
    ].copy()
    selected_operating_points = selected_operating_points[
        [
            "evaluation_mode",
            "policy",
            "threshold",
            "calibration_method",
            "persistence_requirement_windows",
            "stability_threshold",
            "cluster_cooldown_seconds",
            "authorization_windows_per_hour_mean",
            "authorization_windows_per_hour_std",
            "authorization_clusters_per_hour_mean",
            "authorization_clusters_per_hour_std",
            "false_authorization_clusters_per_hour_mean",
            "false_authorization_clusters_per_hour_std",
            "event_level_coverage_mean",
            "event_level_coverage_std",
            "median_lead_time_seconds_mean",
            "median_lead_time_seconds_std",
            "mean_s_t_authorized_mean",
            "mean_s_t_authorized_std",
        ]
    ]
    event_level_summary_records = []
    for (evaluation_mode, policy), group in policy_df[policy_df["model"] == "lightgbm"].groupby(
        ["evaluation_mode", "policy"], sort=False
    ):
        event_level_summary_records.append(
            {
                "evaluation_mode": evaluation_mode,
                "policy": policy,
                "events": int(group["seizure_events"].sum()),
                "authorized": int(group["events_with_authorization"].sum()),
                "pre_onset_authorizations": int(group["pre_onset_authorizations"].sum()),
                "pooled_event_coverage": safe_divide(
                    int(group["events_with_authorization"].sum()), int(group["seizure_events"].sum())
                ),
                "fold_mean_event_coverage": float(group["event_level_coverage"].mean()),
                "fold_sd_event_coverage": float(group["event_level_coverage"].std(ddof=1)),
                "median_lead_time_seconds_mean": float(group["median_lead_time_seconds"].mean()),
                "median_lead_time_seconds_std": float(group["median_lead_time_seconds"].std(ddof=1)),
                "false_authorization_clusters_per_hour_mean": float(
                    group["false_authorization_clusters_per_hour"].mean()
                ),
                "false_authorization_clusters_per_hour_std": float(
                    group["false_authorization_clusters_per_hour"].std(ddof=1)
                ),
            }
        )
    event_level_summary = pd.DataFrame(event_level_summary_records)

    nested_calibration_df.to_csv(TABLES_DIR / "table7_nested_calibration_by_fold.csv", index=False)
    nested_bins_df.to_csv(TABLES_DIR / "table7_nested_calibration_bins.csv", index=False)
    calibration_agg.to_csv(TABLES_DIR / "table7_nested_calibration_summary.csv", index=False)
    policy_df.to_csv(TABLES_DIR / "table8_policy_ablation_by_fold.csv", index=False)
    policy_agg.to_csv(TABLES_DIR / "table8_policy_ablation_summary.csv", index=False)
    sweep_df.to_csv(TABLES_DIR / "table9_burden_constrained_policy_sweep.csv", index=False)
    nested_split_audit_df.to_csv(TABLES_DIR / "table10_nested_calibration_split_audit.csv", index=False)
    baseline_df.to_csv(TABLES_DIR / "table11_baselines_by_fold.csv", index=False)
    baseline_agg.to_csv(TABLES_DIR / "table11_baselines_summary.csv", index=False)
    stability_diagnostic_df.to_csv(TABLES_DIR / "table12_stability_probability_diagnostic_by_fold.csv", index=False)
    stability_diagnostic_agg.to_csv(TABLES_DIR / "table12_stability_probability_diagnostic_summary.csv", index=False)
    stability_window_df.to_csv(TABLES_DIR / "table12_stability_authorization_windows.csv", index=False)
    cooldown_df.to_csv(TABLES_DIR / "table13_cooldown_cluster_sensitivity_by_fold.csv", index=False)
    cooldown_agg.to_csv(TABLES_DIR / "table13_cooldown_cluster_sensitivity_summary.csv", index=False)
    selected_operating_points.to_csv(TABLES_DIR / "table14_selected_operating_points.csv", index=False)
    event_level_summary.to_csv(TABLES_DIR / "table15_event_level_authorization_summary.csv", index=False)

    performance_summary = pd.read_csv(TABLES_DIR / "table2_model_performance_real_eeg.csv")
    plot_model_performance(performance_summary, baseline_agg)
    plot_nested_calibration(nested_bins_df)
    plot_attribution_stability(policy_df, stability_diagnostic_df, stability_window_df)
    plot_pareto(sweep_df)

    print("Saved acceptance-revision analysis outputs.")
    print(f"Nested calibration rows: {len(nested_calibration_df)}")
    print(f"Policy ablation rows: {len(policy_df)}")
    print(f"Policy sweep rows: {len(sweep_df)}")


if __name__ == "__main__":
    main()
