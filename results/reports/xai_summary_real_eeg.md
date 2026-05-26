# Explainable AI Summary for Real CHB-MIT EEG

This analysis used only locally available CHB-MIT EDF and summary files.
Subjects analyzed: chb01, chb02, chb03, chb05, chb08.
Final harmonized channel count: 22.

The listed features are associated with model-predicted seizure-risk windows and should not be interpreted as causal mechanisms.

## lightgbm
- 1. `rel_bp_alpha`: mean importance 245.500000 (SD 40.075623)
- 2. `line_length`: mean importance 245.100000 (SD 83.437868)
- 3. `rel_bp_theta`: mean importance 228.000000 (SD 41.004065)
- 4. `hjorth_complexity`: mean importance 209.200000 (SD 21.688707)
- 5. `bp_gamma`: mean importance 208.100000 (SD 44.645393)
- 6. `zero_crossing_rate`: mean importance 196.600000 (SD 41.183060)
- 7. `kurtosis`: mean importance 172.000000 (SD 37.193189)
- 8. `hjorth_mobility`: mean importance 168.300000 (SD 38.297229)
- 9. `rel_bp_beta`: mean importance 165.700000 (SD 35.879583)
- 10. `rel_bp_delta`: mean importance 142.100000 (SD 33.127531)

## logistic_regression
- 1. `rms_amplitude`: mean importance 2.651995 (SD 1.063716)
- 2. `hjorth_activity`: mean importance 2.080579 (SD 0.832358)
- 3. `variance`: mean importance 2.080579 (SD 0.832358)
- 4. `line_length`: mean importance 1.739614 (SD 1.859720)
- 5. `rel_bp_delta`: mean importance 1.524162 (SD 0.826267)
- 6. `bp_delta`: mean importance 1.510170 (SD 0.951639)
- 7. `bp_gamma`: mean importance 1.443055 (SD 0.957275)
- 8. `spectral_entropy`: mean importance 1.306775 (SD 1.335507)
- 9. `rel_bp_gamma`: mean importance 1.297675 (SD 1.274883)
- 10. `bp_theta`: mean importance 1.262484 (SD 0.510362)

## random_forest
- 1. `rel_bp_alpha`: mean importance 0.076720 (SD 0.030127)
- 2. `line_length`: mean importance 0.065962 (SD 0.015158)
- 3. `bp_alpha`: mean importance 0.061278 (SD 0.014391)
- 4. `bp_gamma`: mean importance 0.061131 (SD 0.014254)
- 5. `zero_crossing_rate`: mean importance 0.060187 (SD 0.017922)
- 6. `rel_bp_theta`: mean importance 0.059521 (SD 0.021908)
- 7. `hjorth_mobility`: mean importance 0.052983 (SD 0.011395)
- 8. `rel_bp_gamma`: mean importance 0.052764 (SD 0.015871)
- 9. `bp_theta`: mean importance 0.051928 (SD 0.012421)
- 10. `hjorth_complexity`: mean importance 0.051773 (SD 0.008551)
