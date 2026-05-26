from __future__ import annotations

from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.utils.logging_utils import setup_logger
from src.utils.paths import FIGURES_DIR

logger = setup_logger("plot_results")


def _save_figure(fig, stem: str, formats: List[str]) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for fmt in formats:
        fig.savefig(FIGURES_DIR / f"{stem}.{fmt}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def _add_simulated_triggering_note(fig) -> None:
    fig.text(
        0.5,
        0.01,
        "Simulated triggering from real EEG model predictions; not therapeutic efficacy.",
        ha="center",
        va="bottom",
        fontsize=8,
        color="gray",
        style="italic",
    )
    fig.tight_layout(rect=[0, 0.04, 1, 1])


def plot_model_performance(metrics_df: pd.DataFrame, formats: List[str] | None = None) -> None:
    """Figure 2: model performance across evaluation modes."""
    formats = formats or ["png"]
    logger.info("Generating Figure 2: Model Performance")

    metrics = ["auroc", "auprc", "sensitivity", "specificity", "f1", "balanced_accuracy"]
    rows = []
    for _, row in metrics_df.iterrows():
        for metric in metrics:
            rows.append(
                {
                    "evaluation_mode": row["evaluation_mode"],
                    "model": row["model"],
                    "metric": metric.replace("_", " ").title(),
                    "score": row[f"{metric}_mean"],
                }
            )

    plot_df = pd.DataFrame(rows)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    modes = list(metrics_df["evaluation_mode"].drop_duplicates())

    for ax, mode in zip(axes, modes):
        sns.barplot(
            data=plot_df[plot_df["evaluation_mode"] == mode],
            x="metric",
            y="score",
            hue="model",
            ax=ax,
            palette="Set2",
        )
        ax.set_title(mode.replace("_", " ").title())
        ax.set_xlabel("")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="Model", fontsize=8)

    if len(modes) == 1:
        axes[1].axis("off")

    fig.suptitle("Model Performance Across Evaluation Modes", y=1.02)
    fig.tight_layout()
    _save_figure(fig, "figure2_model_performance_real_eeg", formats)


def plot_global_explainability(top_features_df: pd.DataFrame | Dict[str, pd.Series], formats: List[str] | None = None) -> None:
    """Figure 3: global explainability from aggregated feature importances."""
    formats = formats or ["png"]
    logger.info("Generating Figure 3: Global Explainability")

    if isinstance(top_features_df, dict):
        model_name = "lightgbm" if "lightgbm" in top_features_df else next(iter(top_features_df))
        plot_df = (
            top_features_df[model_name]
            .head(15)
            .rename("importance_mean")
            .reset_index()
            .rename(columns={"index": "feature"})
        )
        title_model = model_name
    else:
        source = top_features_df.copy()
        if "model" in source.columns and (source["model"] == "lightgbm").any():
            source = source[source["model"] == "lightgbm"]
            title_model = "lightgbm"
        else:
            title_model = str(source["model"].iloc[0]) if "model" in source.columns and len(source) else "model"
        if "rank" in source.columns:
            source = source[source["rank"] <= 15]
        plot_df = source.sort_values("importance_mean", ascending=False).head(15)

    plot_df = plot_df.sort_values("importance_mean", ascending=True)
    max_value = plot_df["importance_mean"].max() or 1.0
    plot_df["relative_importance"] = plot_df["importance_mean"] / max_value

    fig, ax = plt.subplots(figsize=(10, 7))
    ax.barh(plot_df["feature"], plot_df["relative_importance"], color="#4C78A8")
    ax.set_title(f"Global EEG Feature Importance ({title_model})")
    ax.set_xlabel("Relative Importance")
    ax.set_ylabel("")
    fig.tight_layout()
    _save_figure(fig, "figure3_global_explainability_real_eeg", formats)


def plot_policy_comparison(policy_df: pd.DataFrame, formats: List[str] | None = None) -> None:
    """Figure 4: false stimulations per hour by simulated triggering policy."""
    formats = formats or ["png"]
    logger.info("Generating Figure 4: Policy Comparison")

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.barplot(
        data=policy_df,
        x="policy",
        y="false_stimulations_per_hour_mean",
        hue="evaluation_mode",
        ax=ax,
        palette="Set3",
    )
    ax.set_title("False Stimulations per Hour by Triggering Policy")
    ax.set_ylabel("False Stimulations / Hour")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=15)
    ax.legend(title="Evaluation Mode", fontsize=8)
    _add_simulated_triggering_note(fig)
    _save_figure(fig, "figure4_policy_comparison_real_eeg", formats)


def plot_stimulation_burden_tradeoff(policy_df: pd.DataFrame, formats: List[str] | None = None) -> None:
    """Figure 5: stimulation burden versus sensitivity."""
    formats = formats or ["png"]
    logger.info("Generating Figure 5: Stimulation Burden Tradeoff")

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=policy_df,
        x="false_stimulations_per_hour_mean",
        y="window_level_sensitivity_mean",
        hue="policy",
        style="evaluation_mode",
        size="stimulations_per_hour_mean",
        sizes=(80, 260),
        ax=ax,
    )
    ax.set_title("Stimulation Burden Versus Window-Level Sensitivity")
    ax.set_xlabel("False Stimulations / Hour")
    ax.set_ylabel("Window-Level Sensitivity")
    ax.set_ylim(0, 1.05)
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0)
    _add_simulated_triggering_note(fig)
    _save_figure(fig, "figure5_stimulation_burden_tradeoff_real_eeg", formats)


def generate_all_plots(
    metrics_df: pd.DataFrame,
    policy_df: pd.DataFrame,
    top_features_df: pd.DataFrame,
    config: Dict,
) -> None:
    formats = config.get("figure_formats", ["png"])
    plot_model_performance(metrics_df, formats)
    plot_global_explainability(top_features_df, formats)
    plot_policy_comparison(policy_df, formats)
    plot_stimulation_burden_tradeoff(policy_df, formats)
