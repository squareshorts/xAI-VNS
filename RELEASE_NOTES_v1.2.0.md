# v1.2.0 - Final targeted-robustness resubmission release

This release accompanies the final revision of **“Burden-Constrained Policy-Layer Authorization for Simulated EEG-Guided Adaptive Vagus Nerve Stimulation.”**

## Final targeted analyses

Two targeted analyses were added without model refitting or operating-point retuning.

1. **Decision-time false-cluster sensitivity.** Authorization masks and support-defined clusters were held fixed while cluster truth was recomputed from operational labels at window-end decision availability rather than window centers. For the principal persistence-plus-attribution-overlap policy, no cluster changed classification in any held-out case: counts remain 2, 12, 4, 681, and 1, totaling **700 false clusters (4.050/h pooled)**. Across the wider factorial set, only one single-window chb05 cluster changes under thresholding alone and overlap alone; persistence-based configurations are unchanged.

2. **chb05 EDF-level diagnostic.** False clusters occur in **32/39 chb05 EDFs**. The largest EDF contributes 86/681 (**12.6%**) of the chb05 total; the three largest contribute 33.5% and the five largest 48.3%. The original 14-EDF analysis inventory contributes 66 false clusters over 14.00 h (**4.71/h**), whereas the 25 additional evaluation-only EDFs contribute 615 over 25.00 h (**24.6/h**), with false clusters in 22/25 added recordings. This establishes that the high chb05 burden is distributed across recordings rather than driven by one or two anomalous EDFs.

A descriptive score diagnostic shows a heavier upper tail among operationally negative chb05 windows in the added exposure: **7.06%** exceed the frozen threshold $\theta=0.48$, compared with **1.63%** in the original held-out inventory and **0.14%** in cross-fitted calibration negatives.

## Manuscript and response changes

- Revised title to **Burden-Constrained Policy-Layer Authorization for Simulated EEG-Guided Adaptive Vagus Nerve Stimulation**.
- Established seizure-responsive authorization framing before the Results because the operational target combines preictal and ictal labels.
- Tightened the abstract and Discussion around the policy-layer generalization result rather than framing the revision as a postmortem of earlier claims.
- Added the decision-time and chb05 diagnostics to Methods, Results, Discussion, supplement, and reviewer response.
- Added paired case-level comparator uncertainty to directly address the request for statistical evaluation.
- Renamed the 70-EDF set as the **original analysis inventory** where appropriate to avoid implying that held-out-case recordings entered model fitting.
- Removed the unmodeled human-review checkpoint from the pipeline schematic.
- Re-rendered the related-work coverage map without filled-cell visual ranking.
- Corrected reviewer-response historical burden values to **1.17 +/- 2.05/h** for the nominal held-out full-governed configuration.
- Clarified that the reviewer-referenced nested AUROC/AUPRC 0.612/0.133 used the original exposure, whereas the revised complete-inventory nested-core values are 0.601/0.082.

## Reproducibility

The release includes the new executable analysis script and outputs:

- `reproducibility/scripts/final_targeted_analyses.py`
- `reproducibility/results/decision_time_false_cluster_sensitivity.csv`
- `reproducibility/results/decision_time_changed_clusters.csv`
- `reproducibility/results/decision_time_sensitivity_summary.json`
- `reproducibility/results/chb05_per_edf_diagnostic.csv`
- `reproducibility/results/chb05_score_shift_summary.csv`
- `reproducibility/results/chb05_diagnostic_summary.json`

The revised manuscript and reviewer response compile successfully in the repository CI workflow.

## Data

Raw CHB-MIT EEG is not redistributed. CHB-MIT is available through PhysioNet (DOI `10.13026/C2K01R`). The release includes the public file registry/checksum provenance and derived reproducibility inputs.

## Archival DOI

The connected GitHub-Zenodo integration will assign a new version-specific DOI for v1.2.0 after the GitHub release is published. The previous v1.1.0 DOI is `10.5281/zenodo.22849640`.
