# Acceptance-Readiness Report

## What changed

- Reframed the manuscript around governed authorization policies rather than classifier performance.
- Replaced the prior per-hour intervention wording with authorization windows, authorization clusters, false authorization burden, and event-level authorization coverage.
- Added authorization-cluster analysis, including consecutive-window clusters and 30 s, 60 s, and 120 s cooldown sensitivity.
- Regenerated Figure 2 as AUROC/AUPRC-only validation panels with prevalence/chance references.
- Regenerated Figure 3 with calibration curves, bin-count markers, probability-bin counts, Brier score, ECE, calibration slope/intercept, and maximum probability in the accompanying table.
- Replaced the generic explainability emphasis with an attribution-stability governance figure.
- Regenerated the burden-constrained operating-curve figure with log-scaled low-burden axes and selected governed operating-point markers.
- Rewrote Results, Discussion, Limitations, captions, and the abstract; added float barriers and fixed PDF layout.

## Central quantitative result

In held-out-subject validation, fixed-threshold LightGBM authorization produced 270.05 +/- 118.01 authorization windows/h, 88.38 +/- 30.05 authorization clusters/h, and 84.44 +/- 29.57 false authorization clusters/h. The full governed policy reduced this to 5.80 +/- 7.76 authorization windows/h, 1.72 +/- 2.10 authorization clusters/h, and 1.17 +/- 2.05 false authorization clusters/h, while fold-mean event-level authorization coverage decreased from 1.00 to 0.76 +/- 0.29.

With 60 s cooldown merging, held-out-subject false authorization clusters decreased from 9.79 +/- 1.39/h under fixed thresholding to 0.64 +/- 0.95/h under the full governed policy.

## Remaining weaknesses

- The cohort is still a five-subject pediatric CHB-MIT subset.
- The study remains retrospective and simulated; no delivered VNS therapy or longitudinal response is modeled.
- The central operating-point table is dense because it is carrying the reviewer-facing quantitative burden.
- Event coverage is sensitive to the 5 min operational labeling convention.

## Final score

Acceptance-readiness score: 9/10.
