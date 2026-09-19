# Changelog

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
