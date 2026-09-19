from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]          # reproducibility/
REPO = ROOT.parent
BASE = ROOT / "results"
OUT = REPO / "results" / "figures"
MOUT = REPO / "manuscript" / "figures"
OUT.mkdir(parents=True, exist_ok=True)
MOUT.mkdir(parents=True, exist_ok=True)


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=300, bbox_inches="tight")
    fig.savefig(MOUT / f"{stem}.pdf", bbox_inches="tight")
    plt.close(fig)

# Figure 1: methodological coverage map
studies = ["Andrade et al.\n(2024)", "Batista et al.\n(2024)", "Ingolfsson et al.\n(2024)", "Vieira et al.\n(2023)", "Present study"]
cols = ["Score /\nmodel", "Probability\ncalibration", "Temporal\nrule", "Burden /\ntiming", "Attribution /\nXAI", "Simulated VNS\nauthorization"]
M = np.array([[1,0,1,1,0,0],[1,0,1,1,0,0],[1,0,1,1,0,0],[1,0,0,0,1,0],[1,1,1,1,1,1]], dtype=float)
fig, ax = plt.subplots(figsize=(9.2,4.0))
# Neutral structural map: use symbols rather than filled cells so the present-study
# row is not visually encoded as a performance ranking.
ax.set_xlim(-0.5, len(cols)-0.5)
ax.set_ylim(len(studies)-0.5, -0.5)
ax.set_xticks(range(len(cols)), cols, fontsize=10)
ax.set_yticks(range(len(studies)), studies, fontsize=10)
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, "●" if M[i,j] else "–", ha="center", va="center", fontsize=15)
ax.set_xticks(np.arange(-.5, len(cols), 1), minor=True)
ax.set_yticks(np.arange(-.5, len(studies), 1), minor=True)
ax.grid(which="minor", linewidth=0.8, alpha=0.45)
ax.tick_params(which="minor", bottom=False, left=False)
ax.tick_params(axis="x", pad=8)
for spine in ax.spines.values(): spine.set_visible(False)
fig.tight_layout(); save(fig, "figure1_methodological_coverage")

pol = pd.read_csv(BASE / "policies_by_fold.csv")
full = pol[(pol.exposure == "combined") & (pol.cooldown_s == 0) & (pol.policy == "+ persistence + overlap")].sort_values("subject")
subjects = full.subject.tolist(); vals = full.false_clusters_per_hour.to_numpy(); median = float(np.median(vals))

# Figure 3: final case burden only
fig, ax = plt.subplots(figsize=(8.4,4.8)); x = np.arange(len(subjects))
ax.scatter(x, vals, s=85, zorder=3); ax.set_yscale("log")
ax.axhline(2.0, linestyle="--", linewidth=1.3); ax.axhline(median, linestyle=":", linewidth=1.3)
for xi, y in zip(x, vals): ax.annotate(f"{y:.3f}", (xi,y), xytext=(0,8), textcoords="offset points", ha="center", fontsize=9)
ax.text(len(subjects)-0.05, 2.0, "training-side target: 2/h", ha="right", va="bottom", fontsize=9)
ax.text(len(subjects)-0.05, median, f"case median: {median:.3f}/h", ha="right", va="bottom", fontsize=9)
ax.set_xticks(x, subjects); ax.set_ylabel("False authorization clusters / recording-hour"); ax.set_xlabel("Held-out case")
ax.grid(axis="y", alpha=0.25, which="both"); fig.tight_layout(); save(fig, "figure3_case_burden")

# Figure 4: common-score factorial
order = ["Calibrated threshold only", "+ persistence", "+ attribution overlap", "+ persistence + overlap"]
labels = ["Calibrated\nthreshold", "+ persistence", "+ attribution\noverlap", "+ both"]
fac = pol[(pol.exposure == "combined") & (pol.cooldown_s == 0) & pol.policy.isin(order)].copy()
fig, ax = plt.subplots(figsize=(9.2,5.0))
for subject, grp in fac.groupby("subject", sort=True):
    grp = grp.set_index("policy").loc[order].reset_index()
    ax.plot(np.arange(4), grp.false_clusters_per_hour, marker="o", linewidth=1.4, label=subject)
meds = [fac[fac.policy == p].false_clusters_per_hour.median() for p in order]
ax.plot(np.arange(4), meds, marker="D", linewidth=2.4, linestyle="--", label="case median")
ax.axhline(2.0, linestyle=":", linewidth=1.2); ax.set_yscale("log"); ax.set_xticks(np.arange(4), labels)
ax.set_ylabel("False authorization clusters / recording-hour"); ax.grid(axis="y", alpha=0.25, which="both")
ax.legend(ncol=3, frameon=False, bbox_to_anchor=(0.5,1.02), loc="lower center")
fig.tight_layout(); save(fig, "figure4_factorial_complete")

# Figure 5: event timing + comparator benchmark
comp = pd.read_csv(BASE / "temporal_comparators_by_fold.csv")
comp300 = comp[(comp.exposure == "combined") & (comp.refractory_s == 300)]
summary = comp300.groupby("family", as_index=False).agg(false_h=("false_clusters_per_hour","mean"), coverage=("end_coverage","mean"))
family_names = {"threshold":"Threshold", "persistence":"Persistence", "full":"Persistence + overlap", "majority3":"Majority of 3", "firing_power":"Firing power"}
principal = full[["end_pre","end_post","end_missed"]].sum(); counts = [int(principal.end_pre), int(principal.end_post), int(principal.end_missed)]
fig, axes = plt.subplots(1,2, figsize=(11.0,4.6), gridspec_kw={"width_ratios":[0.9,1.25]})
ax = axes[0]; bars = ax.bar(["Pre-onset","At/after\nonset only","Missed"], counts)
for bar,c in zip(bars,counts): ax.text(bar.get_x()+bar.get_width()/2, c+0.35, f"{c}/27", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Seizures"); ax.set_ylim(0,max(counts)+3); ax.text(-0.08,1.03,"A",transform=ax.transAxes,fontweight="bold",fontsize=13); ax.grid(axis="y",alpha=.2)
ax = axes[1]
for _,r in summary.iterrows():
    name = family_names.get(r.family, r.family); ax.scatter(r.false_h,r.coverage,s=75)
    dy = -16 if name == "Firing power" else 5
    ax.annotate(name,(r.false_h,r.coverage),xytext=(5,dy),textcoords="offset points",fontsize=9)
ax.set_xlabel("False authorization clusters / recording-hour\n(case mean; 300 s refractory suppression)")
ax.set_ylabel("Event coverage (case mean)"); ax.set_ylim(.45,.76); ax.grid(alpha=.2); ax.text(-.08,1.03,"B",transform=ax.transAxes,fontweight="bold",fontsize=13)
fig.tight_layout(); save(fig, "figure5_operational_outcomes")

# Figure 6: complete-exposure attribution-lag result only
lag_fold = pd.read_csv(BASE / "attribution_lag_complete_by_fold.csv"); lag_sum = pd.read_csv(BASE / "attribution_lag_complete_summary.csv")
lag_fold = lag_fold[lag_fold.exposure == "combined"]; lag_sum = lag_sum[lag_sum.exposure == "combined"]
lags = np.array(sorted(lag_sum.lag_s.unique())); subjects = sorted(lag_fold.subject.unique()); offsets = np.linspace(-.14,.14,len(subjects))
fig, ax = plt.subplots(figsize=(8.4,4.8))
for off,sub in zip(offsets,subjects):
    d = lag_fold[lag_fold.subject == sub].set_index("lag_s").reindex(lags); ax.scatter(lags+off,d.difference,s=28,alpha=.55)
s = lag_sum.set_index("lag_s").reindex(lags); mean=s.difference_mean.to_numpy(); lo=s.difference_ci_low.to_numpy(); hi=s.difference_ci_high.to_numpy()
ax.errorbar(lags,mean,yerr=np.vstack([mean-lo,hi-mean]),marker="o",linewidth=2,capsize=4,label="case mean and 95% interval")
ax.axhline(0,linewidth=1); ax.axvline(4,linestyle="--",linewidth=1.2)
ax.text(4.15, ax.get_ylim()[1]*.95, "first non-overlapping separation", fontsize=9, va="top")
ax.set_xticks(lags); ax.set_xlabel("Window-center separation (s)"); ax.set_ylabel("Positive minus negative attribution-overlap difference")
ax.grid(alpha=.2); ax.legend(frameon=False,loc="upper right"); fig.tight_layout(); save(fig, "figure6_attribution_lag")

# Figure S1: fixed-gamma nominal threshold sweep, complete exposure, isotonic full policy
sw = pd.read_csv(BASE / "fixed_gamma_sweep_by_fold.csv")
sw = sw[(sw.exposure == "combined") & (sw.policy == "Isotonic full") & np.isclose(sw.gamma,0.4)]
agg = sw.groupby("theta",as_index=False).agg(false_h=("false_clusters_per_hour","mean"),coverage=("end_coverage","mean"))
fig, ax = plt.subplots(figsize=(7.8,4.8))
ax.plot(agg.false_h,agg.coverage,marker="o")
nom = agg[np.isclose(agg.theta,0.5)]
if len(nom): ax.scatter(nom.false_h.iloc[0],nom.coverage.iloc[0],marker="*",s=150,zorder=5)
for _,r in agg.iterrows(): ax.annotate(f"θ={r.theta:.1f}",(r.false_h,r.coverage),xytext=(4,4),textcoords="offset points",fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("False authorization clusters / recording-hour (case mean)"); ax.set_ylabel("Event coverage (case mean)")
ax.grid(alpha=.22,which="both"); fig.tight_layout(); save(fig,"figureS1_operating_sweep")

print(f"Saved final figures to {OUT} and {MOUT}")
