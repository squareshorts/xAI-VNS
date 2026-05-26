# README: Explainable Patient-Specific Seizure-Risk Modeling for Simulated Responsive VNS Triggering

## Project Aim
This repository provides a reproducible research pipeline for explainable seizure-risk modeling to support simulated responsive Vagus Nerve Stimulation (rVNS) decision policies using real scalp EEG data. The goal is to demonstrate how transparent, patient-specific Machine Learning can be integrated into clinical decision support systems for drug-resistant epilepsy.

This project is implemented as a single-subject feasibility study using CHB-MIT scalp EEG (subject `chb01`).

## Important Disclaimer: No Clinical Claims
All VNS stimulation outcomes in this repository are **simulated triggering policies** based on real scalp EEG data from the CHB-MIT database. 
This study does **not** claim to clinically optimize implanted VNS parameters, nor does it provide evidence of therapeutic efficacy. 
The focus is strictly on the computational framework, explainability (XAI), and algorithmic decision support.

## Dataset Access (CHB-MIT)
This project is designed to use the CHB-MIT Scalp EEG Database (specifically subject `chb01`).
To run the pipeline, the CHB-MIT EDF and summary files for `chb01` must be placed in:
`data/external/chbmit/chb01/`

You can download the data automatically using the provided download script:
```powershell
$env:PYTHONPATH = "C:\work\explainable-AI"
python scripts/download_chbmit_real.py
```

## How to Run

### Installation
```bash
pip install -r requirements.txt
# OR
conda env create -f environment.yml
```

### Full Pipeline (Real EEG)
To run the preprocessing, feature extraction, modeling, explainability analysis, and VNS triggering simulations:
```powershell
$env:PYTHONPATH = "C:\work\explainable-AI"
python scripts/run_pipeline.py --config configs/chbmit.yaml
```

## Repository Outputs
- `results/figures/`: Manuscript-ready plots (PNG) comparing model performance (`figure2`), global explainability (`figure3`), policy comparison (`figure4`), and the stimulation burden tradeoff (`figure5`).
- `results/tables/`: Machine-readable CSV tables with dataset summaries (`table1`), model metrics (`table2`), and policy comparisons (`table3`).
- `results/reports/`: Text summaries including the XAI feature summary (`xai_summary_real_eeg.md`).
- `manuscript/`: LaTeX manuscript (`main.tex`) and compiled PDF.
