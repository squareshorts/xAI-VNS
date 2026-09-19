# v1.1.0 - Complete-recording resubmission release

This release accompanies the major revision of **“Burden-Constrained Policy-Layer Explainability for Simulated EEG-Guided Adaptive Vagus Nerve Stimulation.”**

## Main changes

- Expanded held-out evaluation to all **175 EDF recordings (172.83 h)** available for the five analyzed CHB-MIT cases.
- Preserved the 70-EDF model-development partitions for model fitting, calibration, and operating-point selection; the other 105 EDFs enter evaluation only.
- Added a true common-score factorial analysis for persistence and attribution-overlap constraints.
- Added window-end decision timing and explicit pre-onset, at/after-onset-only, and missed-event categories.
- Added adapted majority-of-three and firing-power temporal-rule comparators on common calibrated score streams.
- Added attribution-overlap sensitivity at non-overlapping temporal separations.
- Replaced legacy/revision-history figures with a compact final scientific figure set.
- Added executable inputs, fitted models/calibrators, predictions, cluster/event ledgers, parameter searches, tests, and final figure generation.

## Complete-recording result

The principal persistence-plus-attribution-overlap policy yields a held-out case median of **0.105 false clusters/h** and a case mean of **3.601 +/- 7.748/h**. The pooled rate is **4.050/h**. Burden is highly heterogeneous: `chb05` contributes 681 of 700 false clusters, while the other four cases range from 0.049 to 0.340/h.

At complete-window decision availability, **2/27** seizures are authorized before onset, **18/27** only at/after onset, and **7/27** are missed. Median latency among covered events is **16 s** (IQR 8.5-28 s).

These results do not establish reliable advance seizure prediction, therapeutic VNS efficacy, or a subject-independent low-burden guarantee.

## Data

Raw CHB-MIT EEG is not redistributed. The release includes public file manifests/checksums and derived features required to reproduce the reported analyses. CHB-MIT must be obtained from PhysioNet for raw-signal re-extraction.

## Archival DOI

When the GitHub-Zenodo integration archives this release, Zenodo will assign a **new version-specific DOI** for v1.1.0. The previous immutable v1.0.1 DOI is `10.5281/zenodo.20403195`.
