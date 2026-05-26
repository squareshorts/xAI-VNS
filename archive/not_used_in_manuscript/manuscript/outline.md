# Explainable Patient-Specific Seizure-Risk Modeling for Ethical Personalization of Responsive Vagus Nerve Stimulation in Drug-Resistant Epilepsy

## 1. Introduction
- Background: Drug-resistant epilepsy and VNS.
- The need for patient-specific, responsive triggering rather than open-loop fixed duty cycles.
- The ethical imperative of Explainable AI (XAI) in healthcare: clinician trust, transparency, and avoiding automation bias.
- Study Aim: Propose a reproducible, explainable EEG-based seizure-risk modeling framework to support simulated VNS decision policies.

## 2. Methods
- **Data Source**: CHB-MIT Scalp EEG database (or synthetic demonstration).
- **Preprocessing & Windowing**: Sliding window segmentation, defining interictal, preictal, and ictal states.
- **Interpretable Feature Extraction**: Bandpower, Hjorth parameters, line length, etc.
- **Explainable Modeling**: Logistic regression, Random Forest, LightGBM, EBM.
- **XAI Framework**: SHAP values, feature importance, and model transparency.
- **Simulated VNS Policies**: Fixed threshold vs. patient-specific vs. explanation-constrained triggering.

## 3. Results
- **Model Performance**: AUROC, AUPRC, F1-scores across models (Table 2, Fig 2).
- **Global Explainability**: Key EEG features driving seizure-risk predictions (Fig 3).
- **Policy Comparison**: Trade-offs between sensitivity and false stimulation burden (Fig 4, Fig 5, Table 3).

## 4. Discussion: Ethics and XAI in Clinical Deployment
- Transparency and clinician-in-the-loop design.
- Risks of false alarms vs. missed seizures (The Risk Matrix).
- Limitations: This study simulates triggering policies on scalp EEG and does NOT claim clinical efficacy of implanted VNS.

## 5. Conclusion
- Summary of contributions towards transparent decision support for future neurostimulation devices.
