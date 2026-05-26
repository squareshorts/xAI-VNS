from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List

import numpy as np
import pandas as pd
import yaml

from src.data.load_chbmit import load_chbmit_data
from src.data.make_windows import create_sliding_windows_for_recording, get_windowing_config
from src.features.eeg_features import extract_features
from src.models.evaluate_models import evaluate_models
from src.models.explain_models import get_feature_importances
from src.models.train_models import train_models
from src.simulation.simulate_vns_triggering import simulate_vns
from src.utils.logging_utils import log_environment_info, setup_logger
from src.utils.paths import (
    CONFIG_DIR,
    PROCESSED_DATA_DIR,
    PROJECT_ROOT,
    REPORTS_DIR,
    TABLES_DIR,
    ensure_directories,
)
from src.utils.seed import set_global_seed
from src.visualization.plot_results import generate_all_plots


METRIC_COLUMNS = [
    "auroc",
    "auprc",
    "sensitivity",
    "specificity",
    "f1",
    "balanced_accuracy",
    "brier_score",
    "false_alarms_per_hour",
]

POLICY_COLUMNS = [
    "total_stimulations",
    "stimulations_per_hour",
    "false_stimulations_per_hour",
    "window_level_sensitivity",
]


def resolve_data_dir(config: dict) -> Path:
    data_dir_value = config.get("data_dir", "data/external/chbmit")
    data_dir = Path(data_dir_value)
    if not data_dir.is_absolute():
        data_dir = PROJECT_ROOT / data_dir
    return data_dir


def _read_config(config_path: str | None) -> Dict:
    if config_path:
        path = Path(config_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
    else:
        path = CONFIG_DIR / "chbmit.yaml"

    with open(path, "r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    return config


def _recording_key(subject_id: str, edf_file: str) -> str:
    return f"{subject_id}|{edf_file}"


def _choose_chronological_split(subject_recordings: pd.DataFrame, test_fraction: float) -> int:
    """Return the number of early recordings assigned to train."""
    n_recordings = len(subject_recordings)
    if n_recordings < 2:
        raise ValueError("Within-subject evaluation requires at least two EDF files.")

    target_train_count = int(round(n_recordings * (1.0 - test_fraction)))
    target_train_count = min(max(target_train_count, 1), n_recordings - 1)

    candidates = list(range(1, n_recordings))

    def has_positive(rows: pd.DataFrame) -> bool:
        return bool((rows["seizure_count"] > 0).any())

    positive_candidates = [
        count
        for count in candidates
        if has_positive(subject_recordings.iloc[:count])
        and has_positive(subject_recordings.iloc[count:])
    ]
    pool = positive_candidates or candidates

    return min(
        pool,
        key=lambda count: (
            abs(count - target_train_count),
            abs(((n_recordings - count) / n_recordings) - test_fraction),
        ),
    )


def _base_window_metadata(features_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
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
        "sampling_rate",
        "duration_seconds",
        "channel_count",
    ]
    return features_df[columns].copy()


def build_evaluation_assignments(
    window_metadata: pd.DataFrame,
    data_dict: Dict,
    config: Dict,
) -> pd.DataFrame:
    """Create mode-specific train/test assignments without splitting EDF files."""
    evaluation = config.get("evaluation", {})
    test_fraction = float(evaluation.get("test_fraction_per_subject", 0.4))
    modes = set(
        evaluation.get(
            "modes",
            ["within_subject_chronological", "held_out_subject"],
        )
    )

    recordings_df = pd.DataFrame(data_dict["recordings"])[
        ["subject_id", "edf_file", "recording_order", "duration_seconds", "seizure_count"]
    ].copy()
    recordings_df["recording_key"] = [
        _recording_key(row.subject_id, row.edf_file) for row in recordings_df.itertuples()
    ]

    metadata = window_metadata.copy()
    metadata["recording_key"] = [
        _recording_key(row.subject_id, row.edf_file) for row in metadata.itertuples()
    ]

    assignments: List[pd.DataFrame] = []

    if "within_subject_chronological" in modes:
        for subject_id, subject_recordings in recordings_df.groupby("subject_id", sort=False):
            subject_recordings = subject_recordings.sort_values("recording_order").reset_index(drop=True)
            train_count = _choose_chronological_split(subject_recordings, test_fraction)
            split_by_recording = {
                row.recording_key: "train" if idx < train_count else "test"
                for idx, row in enumerate(subject_recordings.itertuples())
            }

            subject_rows = metadata[metadata["subject_id"] == subject_id].copy()
            subject_rows["split"] = subject_rows["recording_key"].map(split_by_recording)
            subject_rows["evaluation_mode"] = "within_subject_chronological"
            subject_rows["fold_id"] = f"within_{subject_id}"
            subject_rows["fold_subject_id"] = subject_id
            assignments.append(subject_rows)

    if "held_out_subject" in modes:
        for held_out_subject in data_dict["subjects"]:
            fold_rows = metadata.copy()
            fold_rows["split"] = np.where(
                fold_rows["subject_id"] == held_out_subject,
                "test",
                "train",
            )
            fold_rows["evaluation_mode"] = "held_out_subject"
            fold_rows["fold_id"] = f"loso_{held_out_subject}"
            fold_rows["fold_subject_id"] = held_out_subject
            assignments.append(fold_rows)

    if not assignments:
        raise ValueError("No evaluation assignments were created. Check evaluation.modes in config.")

    labels_df = pd.concat(assignments, ignore_index=True)
    labels_df = labels_df[
        [
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
            "fold_id",
            "fold_subject_id",
            "duration_seconds",
            "sampling_rate",
            "channel_count",
        ]
    ]
    return labels_df


def _sum_unique_recording_hours(rows: pd.DataFrame) -> float:
    unique_recordings = rows.drop_duplicates(["subject_id", "edf_file"])
    return float(unique_recordings["duration_seconds"].sum() / 3600.0)


def _aggregate_mean_std(df: pd.DataFrame, group_cols: Iterable[str], value_cols: Iterable[str]) -> pd.DataFrame:
    named_aggs = {"n_folds": ("fold_id", "nunique")}
    for column in value_cols:
        named_aggs[f"{column}_mean"] = (column, "mean")
        named_aggs[f"{column}_std"] = (column, "std")

    aggregate = df.groupby(list(group_cols), dropna=False).agg(**named_aggs).reset_index()
    std_cols = [column for column in aggregate.columns if column.endswith("_std")]
    aggregate[std_cols] = aggregate[std_cols].fillna(0.0)
    return aggregate


def _save_model_performance_tables(metrics_df: pd.DataFrame) -> pd.DataFrame:
    aggregate = _aggregate_mean_std(
        metrics_df,
        ["evaluation_mode", "model"],
        METRIC_COLUMNS,
    )

    within = aggregate[aggregate["evaluation_mode"] == "within_subject_chronological"].copy()
    loso = aggregate[aggregate["evaluation_mode"] == "held_out_subject"].copy()

    within.to_csv(
        TABLES_DIR / "table2a_within_subject_model_performance_real_eeg.csv",
        index=False,
    )
    loso.to_csv(
        TABLES_DIR / "table2b_loso_model_performance_real_eeg.csv",
        index=False,
    )
    aggregate.to_csv(TABLES_DIR / "table2_model_performance_real_eeg.csv", index=False)
    return aggregate


def _save_policy_tables(policy_df: pd.DataFrame) -> pd.DataFrame:
    by_subject = policy_df[
        [
            "fold_subject_id",
            "evaluation_mode",
            "fold_id",
            "model",
            "policy",
            "threshold",
            "total_stimulations",
            "stimulations_per_hour",
            "false_stimulations_per_hour",
            "window_level_sensitivity",
            "test_hours",
            "test_windows",
            "test_positive_windows",
        ]
    ].rename(columns={"fold_subject_id": "subject_id"})

    by_subject.to_csv(
        TABLES_DIR / "table3_policy_comparison_by_subject_real_eeg.csv",
        index=False,
    )

    aggregate = _aggregate_mean_std(
        policy_df,
        ["evaluation_mode", "model", "policy"],
        POLICY_COLUMNS,
    )
    aggregate.to_csv(TABLES_DIR / "table3_policy_comparison_real_eeg.csv", index=False)
    return aggregate


def _save_feature_tables(
    importance_df: pd.DataFrame,
    top_by_subject_df: pd.DataFrame,
) -> pd.DataFrame:
    aggregate = _aggregate_mean_std(
        importance_df,
        ["model", "feature"],
        ["importance"],
    ).rename(
        columns={
            "importance_mean": "importance_mean",
            "importance_std": "importance_std",
        }
    )

    aggregate["rank"] = (
        aggregate.groupby("model")["importance_mean"]
        .rank(method="first", ascending=False)
        .astype(int)
    )
    aggregate = aggregate.sort_values(["model", "rank"])
    top_global = aggregate[aggregate["rank"] <= 15].copy()
    top_global.to_csv(TABLES_DIR / "table_top_features_real_eeg.csv", index=False)

    top_by_subject_df.to_csv(
        TABLES_DIR / "table_top_features_by_subject_real_eeg.csv",
        index=False,
    )
    return top_global


def _write_xai_summary(top_features_df: pd.DataFrame, data_dict: Dict) -> None:
    report_path = REPORTS_DIR / "xai_summary_real_eeg.md"
    lines = [
        "# Explainable AI Summary for Real CHB-MIT EEG",
        "",
        "This analysis used only locally available CHB-MIT EDF and summary files.",
        f"Subjects analyzed: {', '.join(data_dict['subjects'])}.",
        f"Final harmonized channel count: {len(data_dict['common_channels'])}.",
        "",
        "The listed features are associated with model-predicted seizure-risk windows and should not be interpreted as causal mechanisms.",
        "",
    ]

    for model_name, model_features in top_features_df.groupby("model"):
        lines.append(f"## {model_name}")
        for row in model_features.sort_values("rank").head(10).itertuples():
            lines.append(
                f"- {row.rank}. `{row.feature}`: mean importance {row.importance_mean:.6f} "
                f"(SD {row.importance_std:.6f})"
            )
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")


def build_feature_table(data_dict: Dict, config: Dict, logger) -> pd.DataFrame:
    window_cfg = get_windowing_config(config)
    feature_flags = config.get("features", {})
    feature_frames: List[pd.DataFrame] = []

    for idx, recording in enumerate(data_dict["recordings"], start=1):
        logger.info(
            "Processing EDF %s/%s: %s %s",
            idx,
            len(data_dict["recordings"]),
            recording["subject_id"],
            recording["edf_file"],
        )
        windows, _, metadata, sfreq = create_sliding_windows_for_recording(
            recording,
            window_len_sec=window_cfg["window_length_sec"],
            overlap_pct=window_cfg["overlap"],
            preictal_sec=window_cfg["preictal_sec"],
            common_channels=data_dict["common_channels"],
            split="unassigned",
            evaluation_mode="feature_extraction",
        )
        if len(windows) == 0:
            logger.warning("No windows extracted from %s %s", recording["subject_id"], recording["edf_file"])
            continue

        features = extract_features(windows, sfreq, feature_flags)
        feature_frames.append(pd.concat([metadata.reset_index(drop=True), features], axis=1))

        del windows

    if not feature_frames:
        raise ValueError("No feature rows were extracted from the local EDF files.")

    features_df = pd.concat(feature_frames, ignore_index=True)
    feature_columns = [
        column
        for column in features_df.columns
        if column
        not in {
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
    ]
    features_df[feature_columns] = features_df[feature_columns].replace([np.inf, -np.inf], np.nan).fillna(0.0)
    return features_df


def run_pipeline(config_path: str | None = None, subject: str | None = None) -> None:
    ensure_directories()

    logger = setup_logger("pipeline")
    logger.info("Starting Explainable VNS Pipeline on REAL multi-subject CHB-MIT EEG")
    log_environment_info(logger)

    config = _read_config(config_path)
    set_global_seed(config.get("random_seed", 42))

    chbmit_data_dir = resolve_data_dir(config)
    logger.info("Using CHB-MIT data directory: %s", chbmit_data_dir)

    data_dict = load_chbmit_data(chbmit_data_dir, subject=subject, config=config)

    features_df = build_feature_table(data_dict, config, logger)
    features_path = PROCESSED_DATA_DIR / "chbmit_features.parquet"
    features_df.to_parquet(features_path, index=False)
    logger.info("Saved features to %s", features_path)

    label_assignments = build_evaluation_assignments(
        _base_window_metadata(features_df),
        data_dict,
        config,
    )
    labels_path = PROCESSED_DATA_DIR / "chbmit_window_labels.csv"
    label_assignments.to_csv(labels_path, index=False)
    logger.info("Saved window labels to %s", labels_path)

    metadata_columns = set(_base_window_metadata(features_df).columns)
    metadata_columns.update({"split", "evaluation_mode"})
    feature_columns = [column for column in features_df.columns if column not in metadata_columns]

    feature_matrix = features_df.set_index("window_id")[feature_columns]
    binary_labels = features_df.set_index("window_id")["binary_label"].astype(int)

    window_cfg = get_windowing_config(config)
    window_step_sec = window_cfg["window_length_sec"] * (1.0 - window_cfg["overlap"])
    models_to_train = config.get(
        "models_to_train",
        ["logistic_regression", "random_forest", "lightgbm"],
    )

    all_metrics: List[pd.DataFrame] = []
    all_policies: List[pd.DataFrame] = []
    all_importances: List[Dict] = []
    top_features_by_subject: List[Dict] = []

    fold_groups = label_assignments.groupby(
        ["evaluation_mode", "fold_id", "fold_subject_id"],
        sort=False,
    )

    for (evaluation_mode, fold_id, fold_subject_id), fold_rows in fold_groups:
        train_rows = fold_rows[fold_rows["split"] == "train"]
        test_rows = fold_rows[fold_rows["split"] == "test"]

        X_train = feature_matrix.loc[train_rows["window_id"]].copy()
        y_train = binary_labels.loc[train_rows["window_id"]].to_numpy()
        X_test = feature_matrix.loc[test_rows["window_id"]].copy()
        y_test = binary_labels.loc[test_rows["window_id"]].to_numpy()

        logger.info(
            "Fold %s: train windows=%s positives=%s; test windows=%s positives=%s",
            fold_id,
            len(y_train),
            int(y_train.sum()),
            len(y_test),
            int(y_test.sum()),
        )

        if len(X_train) == 0 or len(X_test) == 0:
            raise RuntimeError(f"Empty train or test set for fold {fold_id}.")
        if len(np.unique(y_train)) < 2:
            raise RuntimeError(f"Training set for fold {fold_id} does not contain both classes.")
        if len(np.unique(y_test)) < 2:
            raise RuntimeError(f"Test set for fold {fold_id} does not contain both classes.")

        train_hours = _sum_unique_recording_hours(train_rows)
        test_hours = _sum_unique_recording_hours(test_rows)
        context = {
            "evaluation_mode": evaluation_mode,
            "fold_id": fold_id,
            "fold_subject_id": fold_subject_id,
            "train_hours": train_hours,
            "test_hours": test_hours,
            "window_step_sec": window_step_sec,
        }

        train_results = train_models(
            X_train,
            y_train,
            models_to_train,
            config.get("random_seed", 42),
        )
        train_results.update(
            {
                "X_train": X_train,
                "y_train": y_train,
                "X_test": X_test,
                "y_test": y_test,
                "train_hours": train_hours,
                "test_hours": test_hours,
                "window_step_sec": window_step_sec,
            }
        )

        all_metrics.append(evaluate_models(train_results, context=context))
        all_policies.append(simulate_vns(train_results, config, context=context))

        for model_name, model in train_results["models"].items():
            importances = get_feature_importances(model, model_name, X_train)
            importances = importances.replace([np.inf, -np.inf], np.nan).fillna(0.0).sort_values(ascending=False)

            for feature, value in importances.items():
                all_importances.append(
                    {
                        "evaluation_mode": evaluation_mode,
                        "fold_id": fold_id,
                        "fold_subject_id": fold_subject_id,
                        "model": model_name,
                        "feature": feature,
                        "importance": float(value),
                    }
                )

            for rank, (feature, value) in enumerate(importances.head(15).items(), start=1):
                top_features_by_subject.append(
                    {
                        "subject_id": fold_subject_id,
                        "evaluation_mode": evaluation_mode,
                        "fold_id": fold_id,
                        "model": model_name,
                        "rank": rank,
                        "feature": feature,
                        "importance_score": float(value),
                    }
                )

    metrics_df = pd.concat(all_metrics, ignore_index=True)
    policy_df = pd.concat(all_policies, ignore_index=True)
    importance_df = pd.DataFrame(all_importances)
    top_by_subject_df = pd.DataFrame(top_features_by_subject)

    metrics_aggregate = _save_model_performance_tables(metrics_df)
    policy_aggregate = _save_policy_tables(policy_df)
    top_features_global = _save_feature_tables(importance_df, top_by_subject_df)
    _write_xai_summary(top_features_global, data_dict)

    generate_all_plots(
        metrics_aggregate,
        policy_aggregate,
        top_features_global,
        config,
    )

    logger.info("Pipeline completed successfully on real multi-subject CHB-MIT EEG data.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the Explainable VNS Pipeline on local real CHB-MIT EEG."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/chbmit.yaml",
        help="Path to YAML config file.",
    )
    parser.add_argument(
        "--subject",
        type=str,
        default=None,
        help="Optional fallback CHB-MIT subject if the config omits subjects.",
    )
    args = parser.parse_args()
    run_pipeline(args.config, args.subject)
