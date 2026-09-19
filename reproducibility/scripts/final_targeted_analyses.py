"""Final targeted sensitivity analyses requested before final revision.

1) Reclassify false authorization clusters using window-end decision availability
   instead of center-based operational labels, holding scores, thresholds, masks,
   and cluster construction fixed.
2) Diagnose the chb05 burden concentration at EDF level and compare calibrated
   negative-score distributions between the calibration block and held-out
   chb05 exposure.

This script does not fit or retune any model or operating parameter.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd

from policy_core import Stream, policy_mask

ROOT = Path(__file__).resolve().parents[1]
PRED = ROOT / "predictions"
RES = ROOT / "results"
EVENTS = pd.read_csv(ROOT / "inputs/derived_inputs/event_manifest.csv")
SEL = pd.read_csv(ROOT / "inputs/reference/trainonly_policy_selection_by_fold.csv")

CASES = ["chb01", "chb02", "chb03", "chb05", "chb08"]
FAMILIES = {
    "Calibrated threshold only": "threshold",
    "+ persistence": "persistence",
    "+ attribution overlap": "overlap",
    "+ persistence + overlap": "full",
}


def operational_labels(stream, clock):
    """Operational positive label evaluated at window center or window end."""
    t = stream.center if clock == "center" else stream.end
    out = np.zeros(stream.n, dtype=bool)
    for ev in stream.events.itertuples():
        key = f"{ev.subject_id}|{ev.edf_file}"
        lo = max(0.0, float(ev.seizure_onset) - 300.0)
        hi = float(ev.seizure_offset)
        out |= (stream.keys == key) & (t >= lo) & (t <= hi)
    return out


def cluster_endpoint_records(stream, mask, end_y, cooldown=0.0):
    """Same support clustering as policy_core, with center- and end-time truth."""
    idx = np.flatnonzero(mask)
    cols = [
        "subject_id", "edf_file", "support_start", "support_end",
        "decision_first", "decision_last", "n_windows",
        "n_positive_center", "n_positive_end", "false_center", "false_end",
    ]
    if len(idx) == 0:
        return pd.DataFrame(columns=cols)

    new = np.r_[
        True,
        (stream.gid[idx[1:]] != stream.gid[idx[:-1]])
        | (stream.start[idx[1:]] > stream.end[idx[:-1]] + cooldown),
    ]
    a = np.flatnonzero(new)
    b = np.r_[a[1:], len(idx)]
    rows = []
    for aa, bb in zip(a, b):
        ci = idx[aa:bb]
        first, last = ci[0], ci[-1]
        nc = int(stream.y[ci].sum())
        ne = int(end_y[ci].sum())
        rows.append(
            dict(
                subject_id=stream.rows.subject_id.iloc[first],
                edf_file=stream.rows.edf_file.iloc[first],
                support_start=float(stream.start[first]),
                support_end=float(stream.end[last]),
                decision_first=float(stream.end[first]),
                decision_last=float(stream.end[last]),
                n_windows=int(len(ci)),
                n_positive_center=nc,
                n_positive_end=ne,
                false_center=bool(nc == 0),
                false_end=bool(ne == 0),
            )
        )
    return pd.DataFrame(rows, columns=cols)


def support_seconds_for_group(stream, mask):
    idx = np.flatnonzero(mask)
    if len(idx) == 0:
        return 0.0
    # Same selected-support union calculation used by Stream.counts().
    increments = np.r_[
        4.0,
        np.where(
            stream.gid[idx[1:]] == stream.gid[idx[:-1]],
            stream.end[idx[1:]] - stream.end[idx[:-1]],
            4.0,
        ),
    ]
    return float(np.minimum(4.0, increments).sum())


def score_summary(name, values, theta):
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return dict(group=name, n=0)
    return dict(
        group=name,
        n=int(len(x)),
        mean=float(np.mean(x)),
        median=float(np.median(x)),
        q90=float(np.quantile(x, 0.90)),
        q95=float(np.quantile(x, 0.95)),
        q99=float(np.quantile(x, 0.99)),
        maximum=float(np.max(x)),
        fraction_gt_theta=float(np.mean(x > theta)),
        theta=float(theta),
    )


def main():
    sens_rows = []
    changed_rows = []
    chb05_edf = []
    chb05_score_shift = []
    chb05_summary = {}

    for case in CASES:
        fold = f"loso_{case}"
        df = pd.read_csv(PRED / f"{fold}.csv.gz", float_precision="round_trip")
        sel = SEL.loc[SEL.fold == fold].iloc[0]
        theta = float(sel.theta)

        held = df.loc[df.partition != "calibration"].copy()
        stream = Stream(held, EVENTS)

        center_y = operational_labels(stream, "center")
        end_y = operational_labels(stream, "end")

        # Guard against silently changing the historical operational label.
        if not np.array_equal(center_y.astype(int), stream.y.astype(int)):
            bad = int(np.sum(center_y.astype(int) != stream.y.astype(int)))
            raise AssertionError(f"{fold}: derived center labels disagree for {bad} windows")

        p = stream.rows.p_isotonic.to_numpy(float)

        for policy_name, family in FAMILIES.items():
            mask = policy_mask(stream, p, theta, family)
            cr = cluster_endpoint_records(stream, mask, end_y, cooldown=0.0)

            center_false = int(cr.false_center.sum()) if len(cr) else 0
            end_false = int(cr.false_end.sum()) if len(cr) else 0
            c2e_false = int(((~cr.false_center) & cr.false_end).sum()) if len(cr) else 0
            c2e_true = int((cr.false_center & (~cr.false_end)).sum()) if len(cr) else 0

            # Exact reproduction check against Stream.counts() center-label endpoint.
            _, reference_false, _ = stream.counts(mask, cooldown=0)
            if center_false != reference_false:
                raise AssertionError(
                    f"{fold} {policy_name}: center false {center_false} != {reference_false}"
                )

            sens_rows.append(
                dict(
                    fold=fold,
                    subject=case,
                    policy=policy_name,
                    theta=theta,
                    hours=stream.hours,
                    clusters=int(len(cr)),
                    center_false_clusters=center_false,
                    end_false_clusters=end_false,
                    center_false_clusters_per_hour=center_false / stream.hours,
                    end_false_clusters_per_hour=end_false / stream.hours,
                    absolute_rate_change=(end_false - center_false) / stream.hours,
                    relative_count_change=(
                        (end_false - center_false) / center_false
                        if center_false
                        else np.nan
                    ),
                    center_true_to_end_false=c2e_false,
                    center_false_to_end_true=c2e_true,
                    changed_clusters=c2e_false + c2e_true,
                )
            )

            if len(cr):
                changed = cr.loc[cr.false_center != cr.false_end].copy()
                if len(changed):
                    changed.insert(0, "fold", fold)
                    changed.insert(1, "policy", policy_name)
                    changed.insert(2, "theta", theta)
                    changed_rows.append(changed)

            if case == "chb05" and family == "full":
                # EDF-level diagnostics for the principal train-selected policy.
                cr_by_edf = {k: g.copy() for k, g in cr.groupby("edf_file")}
                for edf, g in stream.rows.groupby("edf_file", sort=True):
                    pos = g.index.to_numpy()
                    local_mask = mask[pos]
                    local_p = p[pos]
                    local_end_y = end_y[pos]
                    local_center_y = stream.y[pos]
                    cdf = cr_by_edf.get(edf)
                    hours = float(g.duration_seconds.iloc[0]) / 3600.0
                    if cdf is None:
                        fc = fe = nclusters = 0
                    else:
                        fc = int(cdf.false_center.sum())
                        fe = int(cdf.false_end.sum())
                        nclusters = int(len(cdf))
                    # Support seconds for one EDF using the exact local rows.
                    local_stream = Stream(g.copy(), EVENTS)
                    local_support = support_seconds_for_group(local_stream, local_mask)
                    chb05_edf.append(
                        dict(
                            edf_file=edf,
                            exposure=str(g.partition.iloc[0]),
                            hours=hours,
                            windows=int(len(g)),
                            positive_windows_center=int(local_center_y.sum()),
                            positive_windows_end=int(local_end_y.sum()),
                            calibrated_mean=float(np.mean(local_p)),
                            calibrated_median=float(np.median(local_p)),
                            calibrated_q90=float(np.quantile(local_p, 0.90)),
                            calibrated_q95=float(np.quantile(local_p, 0.95)),
                            calibrated_q99=float(np.quantile(local_p, 0.99)),
                            calibrated_max=float(np.max(local_p)),
                            fraction_gt_theta=float(np.mean(local_p > theta)),
                            authorized_windows=int(local_mask.sum()),
                            authorization_fraction=float(np.mean(local_mask)),
                            authorized_support_seconds=local_support,
                            authorized_support_fraction=local_support / (hours * 3600.0),
                            clusters=nclusters,
                            false_clusters_center=fc,
                            false_clusters_end=fe,
                            false_clusters_center_per_hour=fc / hours,
                            false_clusters_end_per_hour=fe / hours,
                        )
                    )

        if case == "chb05":
            # Compare the negative-score distribution that determined burden:
            # OOF calibration negatives vs final-calibrated held-out negatives.
            cal = df.loc[df.partition == "calibration"].copy()
            held = df.loc[df.partition != "calibration"].copy()
            theta = float(sel.theta)

            groups = [
                (
                    "calibration_negative_oof",
                    cal.loc[cal.binary_label == 0, "p_isotonic_oof"].to_numpy(float),
                ),
                (
                    "heldout_chb05_negative_combined",
                    held.loc[held.binary_label == 0, "p_isotonic"].to_numpy(float),
                ),
                (
                    "heldout_chb05_negative_original",
                    held.loc[
                        (held.binary_label == 0) & (held.partition == "original"),
                        "p_isotonic",
                    ].to_numpy(float),
                ),
                (
                    "heldout_chb05_negative_additional",
                    held.loc[
                        (held.binary_label == 0) & (held.partition == "additional"),
                        "p_isotonic",
                    ].to_numpy(float),
                ),
            ]
            chb05_score_shift.extend(score_summary(name, vals, theta) for name, vals in groups)

    sens = pd.DataFrame(sens_rows)
    sens.to_csv(RES / "decision_time_false_cluster_sensitivity.csv", index=False)

    if changed_rows:
        pd.concat(changed_rows, ignore_index=True).to_csv(
            RES / "decision_time_changed_clusters.csv", index=False
        )
    else:
        pd.DataFrame(
            columns=[
                "fold", "policy", "theta", "subject_id", "edf_file",
                "support_start", "support_end", "decision_first", "decision_last",
                "n_windows", "n_positive_center", "n_positive_end",
                "false_center", "false_end",
            ]
        ).to_csv(RES / "decision_time_changed_clusters.csv", index=False)

    edf = pd.DataFrame(chb05_edf).sort_values(
        ["false_clusters_center", "authorized_windows"], ascending=[False, False]
    )
    total_center = int(edf.false_clusters_center.sum())
    total_end = int(edf.false_clusters_end.sum())
    edf["center_false_cluster_share"] = (
        edf.false_clusters_center / total_center if total_center else 0.0
    )
    edf["end_false_cluster_share"] = (
        edf.false_clusters_end / total_end if total_end else 0.0
    )
    edf.to_csv(RES / "chb05_per_edf_diagnostic.csv", index=False)

    shift = pd.DataFrame(chb05_score_shift)
    shift.to_csv(RES / "chb05_score_shift_summary.csv", index=False)

    ranked_c = edf.false_clusters_center.sort_values(ascending=False).to_numpy()
    ranked_e = edf.false_clusters_end.sort_values(ascending=False).to_numpy()
    by_exposure = (
        edf.groupby("exposure", as_index=False)
        .agg(
            edfs=("edf_file", "count"),
            hours=("hours", "sum"),
            false_clusters_center=("false_clusters_center", "sum"),
            false_clusters_end=("false_clusters_end", "sum"),
            authorized_windows=("authorized_windows", "sum"),
        )
    )
    by_exposure["center_false_per_hour"] = (
        by_exposure.false_clusters_center / by_exposure.hours
    )
    by_exposure["end_false_per_hour"] = (
        by_exposure.false_clusters_end / by_exposure.hours
    )

    principal = sens.loc[sens.policy == "+ persistence + overlap"].copy()
    chb05_principal = principal.loc[principal.subject == "chb05"].iloc[0]

    chb05_summary = {
        "theta": float(chb05_principal.theta),
        "edfs_total": int(len(edf)),
        "edfs_with_center_false_cluster": int((edf.false_clusters_center > 0).sum()),
        "edfs_with_end_false_cluster": int((edf.false_clusters_end > 0).sum()),
        "edfs_with_authorization": int((edf.authorized_windows > 0).sum()),
        "center_false_clusters_total": total_center,
        "end_false_clusters_total": total_end,
        "center_false_rate_per_hour": float(chb05_principal.center_false_clusters_per_hour),
        "end_false_rate_per_hour": float(chb05_principal.end_false_clusters_per_hour),
        "top1_center_share": float(ranked_c[:1].sum() / total_center) if total_center else 0.0,
        "top3_center_share": float(ranked_c[:3].sum() / total_center) if total_center else 0.0,
        "top5_center_share": float(ranked_c[:5].sum() / total_center) if total_center else 0.0,
        "top1_end_share": float(ranked_e[:1].sum() / total_end) if total_end else 0.0,
        "top3_end_share": float(ranked_e[:3].sum() / total_end) if total_end else 0.0,
        "top5_end_share": float(ranked_e[:5].sum() / total_end) if total_end else 0.0,
        "max_center_false_edf": str(edf.iloc[0].edf_file),
        "max_center_false_count": int(edf.iloc[0].false_clusters_center),
        "max_center_false_rate_per_hour": float(edf.iloc[0].false_clusters_center_per_hour),
        "by_exposure": by_exposure.to_dict(orient="records"),
        "score_shift": shift.to_dict(orient="records"),
    }
    (RES / "chb05_diagnostic_summary.json").write_text(
        json.dumps(chb05_summary, indent=2), encoding="utf-8"
    )

    # Small pooled summary for the principal policy, weighted only for the pooled rate.
    pp = principal.copy()
    pooled_center = int(pp.center_false_clusters.sum())
    pooled_end = int(pp.end_false_clusters.sum())
    pooled_hours = float(pp.hours.sum())
    final_summary = {
        "principal_policy": "+ persistence + overlap",
        "case_false_clusters_center": dict(
            zip(pp.subject, pp.center_false_clusters.astype(int))
        ),
        "case_false_clusters_end": dict(
            zip(pp.subject, pp.end_false_clusters.astype(int))
        ),
        "pooled_center_false_clusters": pooled_center,
        "pooled_end_false_clusters": pooled_end,
        "pooled_hours": pooled_hours,
        "pooled_center_rate": pooled_center / pooled_hours,
        "pooled_end_rate": pooled_end / pooled_hours,
        "case_mean_center_rate": float(pp.center_false_clusters_per_hour.mean()),
        "case_mean_end_rate": float(pp.end_false_clusters_per_hour.mean()),
        "case_median_center_rate": float(pp.center_false_clusters_per_hour.median()),
        "case_median_end_rate": float(pp.end_false_clusters_per_hour.median()),
        "total_changed_clusters": int(pp.changed_clusters.sum()),
        "center_true_to_end_false": int(pp.center_true_to_end_false.sum()),
        "center_false_to_end_true": int(pp.center_false_to_end_true.sum()),
    }
    (RES / "decision_time_sensitivity_summary.json").write_text(
        json.dumps(final_summary, indent=2), encoding="utf-8"
    )

    print(json.dumps(final_summary, indent=2))
    print(json.dumps(chb05_summary, indent=2))


if __name__ == "__main__":
    main()

# PR-trigger marker for the final targeted run.
