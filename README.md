# xAI-VNS

Reproducible analysis accompanying the manuscript **“Burden-Constrained Policy-Layer Authorization for Simulated EEG-Guided Adaptive Vagus Nerve Stimulation.”**

This repository evaluates how calibrated EEG model scores are converted into **simulated VNS authorization decisions**. The analysis separates model discrimination from operational policy behavior, including temporal persistence, attribution overlap, false authorization-cluster burden, event timing, and temporal-rule comparisons.

## Scope

The study uses five CHB-MIT cases (`chb01`, `chb02`, `chb03`, `chb05`, `chb08`). The complete evaluation inventory contains **175 EDF recordings, 172.83 h of EEG, 310,916 four-second windows, and 27 annotated seizures**.

The original 70-EDF analysis inventory supplies the fold-specific model-fitting, calibration, and development-test partitions. The fold-specific fitted pipelines are then evaluated over the full 175-EDF inventory; the other 105 EDFs enter evaluation only and are never used for fitting or parameter selection.

No VNS was delivered. The repository does not establish clinical efficacy, reliable advance seizure prediction, or a subject-independent low-burden guarantee.

## Principal findings in v1.2.0

- Full-inventory false authorization-cluster burden for the train-selected persistence-plus-attribution-overlap policy: case median **0.105/h**, case mean **3.601 +/- 7.748/h**, pooled **4.050/h**.
- Burden is strongly heterogeneous: `chb05` contributes 681 of 700 false clusters; the other four cases range from 0.049 to 0.340/h. The chb05 result is distributed across 32 of 39 EDFs rather than driven by a single file; the largest EDF contributes 12.6% of chb05 false clusters.
- In the common-score factorial analysis, temporal persistence accounts for most of the burden reduction; attribution overlap adds a smaller incremental restriction.
- At complete-window decision time: **2/27** seizures are authorized pre-onset, **18/27** only at/after onset, and **7/27** are missed; median latency among covered events is **16 s** (IQR 8.5-28 s).
- The attribution-overlap difference remains positive on average at the first non-overlapping comparison (4 s separation), but this is not clinical validation of explanation reliability.
- Reclassifying false-cluster truth at window-end decision availability leaves the principal 700-cluster burden result unchanged.
- Adapted majority-voting and firing-power rules occupy similar burden-coverage ranges under a common 300 s refractory rule; paired case-level uncertainty is reported descriptively and no comparator family is presented as a winner.

## Repository layout

- `manuscript/` - final LaTeX manuscript, response to reviewers, compiled PDFs, and the six retained manuscript/supplementary figures.
- `reproducibility/` - derived inputs, fitted models/calibrators, per-window predictions, cluster/event outputs, parameter searches, tests, and executable analysis scripts.
- `results/figures/` - final raster exports of manuscript figures.
- `src/`, `configs/`, `tests/` - core analysis code retained from the development pipeline.

Legacy `archive/` material and obsolete manuscript/result snapshots remain excluded from v1.2.0; they are available through Git history and prior immutable releases.

## Reproduce the complete-recording analysis

Create an isolated Python environment and install:

```bash
pip install -r reproducibility/requirements-evaluation.txt
```

Then run:

```bash
python reproducibility/run_analysis.py
```

The supplied derived inputs allow the evaluation to run without redistributing raw CHB-MIT EDFs. Raw EEG must be obtained directly from PhysioNet if feature extraction itself is to be repeated.

## Data

CHB-MIT Scalp EEG Database: PhysioNet, DOI `10.13026/C2K01R`.

Raw EEG is **not** redistributed in this repository. Public file manifests, subject summary files, checksums, and derived features used for reproducibility are included under `reproducibility/inputs/`.

## Versioning and archival record

- `v1.0.1`: historical manuscript archive; Zenodo DOI `10.5281/zenodo.20403195`.
- `v1.1.0`: complete-recording major-revision release; Zenodo DOI `10.5281/zenodo.22849640`.
- `v1.2.0`: final targeted-robustness resubmission release. A new version-specific Zenodo DOI is generated when this GitHub release is archived by the connected Zenodo integration.

## Authors

- Suzana Cescon - Signal Processing Laboratory, Institute of Technology, Federal University of Para
- Antonio Pereira - Signal Processing Laboratory, Institute of Technology, Federal University of Para; ORCID 0000-0002-0808-1058

## Citation

Use the version-specific Zenodo DOI associated with the GitHub release used in your analysis. See `CITATION.cff` and the GitHub release metadata.
