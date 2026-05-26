# Explainable AI Summary

> **Disclaimer**: Simulated responsive stimulation-triggering policy — not evidence of therapeutic efficacy or biological causation.

The features listed below are *associated* with model-predicted seizure risk. They do not necessarily *cause* seizure onset.

## logistic_regression
Top 5 contributing features:
- **bp_theta**: 0.3070
- **variance**: 0.2351
- **hjorth_activity**: 0.2351
- **line_length**: 0.1543
- **kurtosis**: 0.1181

## random_forest
Top 5 contributing features:
- **rel_bp_beta**: 0.0643
- **bp_theta**: 0.0611
- **rel_bp_theta**: 0.0586
- **rms_amplitude**: 0.0563
- **hjorth_complexity**: 0.0556

## lightgbm
Top 5 contributing features:
- **skewness**: 228.0000
- **kurtosis**: 207.0000
- **rel_bp_beta**: 197.0000
- **zero_crossing_rate**: 192.0000
- **hjorth_complexity**: 188.0000

## ebm
Top 5 contributing features:
- **rel_bp_theta**: 0.0743
- **line_length**: 0.0693
- **kurtosis**: 0.0668
- **bp_gamma**: 0.0641
- **rel_bp_beta**: 0.0635

