# Changelog

## v1.2.0 - 2026-09-19

Final targeted-robustness resubmission release.

- Retains the complete 175-EDF inventory and frozen train-only operating-point selection.
- Adds window-end false-cluster truth sensitivity; the principal persistence-plus-overlap endpoint is unchanged in every held-out case (700 false clusters; pooled 4.050/h).
- Adds EDF-level chb05 diagnostics showing false clusters in 32/39 EDFs; the largest EDF contributes 12.6% of the chb05 total, excluding a single-file explanation.
- Shows that the additional 25 chb05 evaluation-only EDFs contribute 615 false clusters over 25 h (24.6/h), compared with 66 over 14.00 h (4.71/h) in the original analysis inventory.
- Adds descriptive calibrated-score upper-tail summaries for chb05.
- Adds paired case-level uncertainty for temporal comparator burden under the common 300 s refractory rule.
- Reframes the task explicitly as seizure-responsive authorization because the operational target combines preictal and ictal windows; pre-onset authorization remains a separate event-timing endpoint.
- Renames the manuscript to “Burden-Constrained Policy-Layer Authorization for Simulated EEG-Guided Adaptive Vagus Nerve Stimulation.”
- Removes the unmodeled human-review checkpoint from the pipeline schematic and makes the related-work coverage map visually neutral.
- Updates and compile-checks the manuscript and response to reviewers.

## v1.1.0 - 2026-09-19

Major reproducibility and manuscript-resubmission release.

- Evaluates the complete 175-EDF inventory for the five analyzed CHB-MIT cases.
- Keeps model fitting, calibration, and operating-point selection restricted to the 70 model-development EDFs.
- Adds common-score factorial decomposition of temporal persistence and attribution-overlap constraints.
- Adds causal window-end event timing and explicit pre-onset / at-or-after-onset / missed categories.
- Adds common-score majority-voting and firing-power temporal comparators.
- Adds non-overlapping attribution-lag sensitivity.
- Adds case-level burden reporting, including median and individual held-out-case rates.
- Replaces historical/revision-history figures with six final scientific figures.
- Adds exact analysis inputs, fitted model artifacts, per-window predictions, event/cluster ledgers, parameter searches, tests, and provenance mappings.
- Removes legacy `archive/` material and obsolete result/manuscript snapshots from the release tree.

The complete-recording evaluation increases the estimated false authorization-cluster burden and reveals pronounced case heterogeneity; the release reports this result without post-hoc case exclusion or threshold retuning.

## v1.0.1 - 2026-05-26

Historical archival/reproducibility release. Zenodo DOI: 10.5281/zenodo.20403195.
