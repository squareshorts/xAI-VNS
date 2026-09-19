# Reproducibility package - v1.1.0

This directory contains the executable complete-recording evaluation for the xAI-VNS manuscript.

## Contents

- `inputs/derived_inputs/`: derived feature matrices, labels, collection/event manifests, and feature-reproduction checks.
- `inputs/reference/`: fixed development partitions and archived numerical reference outputs used for reproduction checks.
- `inputs/public_metadata/`: CHB-MIT public manifests, checksums, and subject summary files.
- `models/`: fitted LightGBM boosters, parameters, and calibrator mappings.
- `predictions/`: per-window held-out scores and top-five contribution ranks.
- `results/`: policy, factorial, comparator, calibration, timing, cluster, sweep, and lag-sensitivity outputs.
- `scripts/`: executable analysis, tests, and final figure generation.

## Run

```bash
pip install -r requirements-evaluation.txt
python run_analysis.py
```

Existing prediction/result files are reused where the analysis script verifies they are present. Removing the cached predictions causes the LightGBM fits to be reconstructed from the supplied derived inputs.

## Analysis conventions

- 4 s windows, 2 s step.
- Operational positive label: 300 s preictal interval plus ictal interval, defined within each EDF.
- Held-out-subject model fitting/calibration/threshold selection uses the 70 model-development EDFs only.
- The full evaluation inventory includes all 175 EDFs from the five analyzed cases.
- False-cluster denominator: total unique analyzed recording duration.
- Principal event clock: window end, the earliest time at which the complete feature window is available.
- Exploratory intervals: exact enumeration of all 5^5 ordered case-bootstrap draws.

These materials support reproducibility of the reported retrospective simulated-authorization analyses. They do not establish clinical explanation validity, therapeutic VNS efficacy, or independent external validation.
