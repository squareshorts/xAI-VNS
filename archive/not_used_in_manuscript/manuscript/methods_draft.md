# Methods Draft

## Data and Preprocessing
Scalp EEG recordings were processed using a sliding window approach (4s windows, 50% overlap). Interictal, preictal (defined as 5 minutes prior to seizure onset), and ictal windows were extracted. 

## Feature Extraction
Interpretable time-frequency features were extracted per window, including absolute and relative bandpower across five standard physiological bands (delta, theta, alpha, beta, gamma), Hjorth parameters (activity, mobility, complexity), line length, variance, RMS amplitude, and zero-crossing rate.

## Explainable Modeling
To balance predictive performance and transparency, we evaluated Logistic Regression with L2 regularization, Random Forests, LightGBM, and Explainable Boosting Machines (EBM). Models were trained to distinguish seizure-risk windows from interictal baseline. Explainability was achieved via SHAP (SHapley Additive exPlanations) values and permutation feature importance, allowing global and local interpretation of feature contributions.

## Simulated Responsive Triggering
We simulated three VNS triggering policies on hold-out data:
1. **Fixed Threshold**: Triggers when predicted risk exceeds 0.5.
2. **Patient-Specific Threshold**: The threshold is optimized per patient on a validation set to constrain the false-stimulation rate to a target maximum (e.g., 2 stimulations per hour).
3. **Explanation-Constrained Policy**: Triggers only when risk exceeds the threshold across multiple consecutive windows, ensuring temporal stability of the underlying explanation.

*Note: Triggering policies were evaluated entirely in-silico. This study does not present clinical validation of implanted VNS.*
