# Ethical and XAI Analysis for Responsive VNS

## 1. Transparency and Clinician Trust
Black-box models in healthcare face significant barriers to adoption due to their opacity. In epilepsy, where the cost of a missed seizure or the burden of continuous false stimulations is high, clinicians must trust the algorithm driving responsive stimulation. By restricting our feature space to interpretable EEG parameters (e.g., bandpower, Hjorth mobility) and employing SHAP values, we ensure that every simulated trigger can be traced back to physiological signal changes.

## 2. Automation Bias
There is a risk that clinicians may over-rely on algorithmic risk scores, a phenomenon known as automation bias. To mitigate this, our framework emphasizes *decision support* rather than *autonomous control*. The explanation-constrained policy demonstrates a method to gate automated triggers, ensuring that stimulation only occurs when the model's reasoning is stable over time, reducing sporadic, uninterpretable triggers.

## 3. The Risk Matrix (Simulated)
| Risk | Clinical Implication | Mitigation Strategy in Model |
|------|----------------------|------------------------------|
| False Alarms | Unnecessary stimulation burden, battery depletion, patient discomfort | Patient-specific thresholding to cap max false triggers per hour |
| Missed Seizures | Lack of therapeutic intervention | Tuning threshold for high sensitivity, constrained by acceptable FA rate |
| Model Opacity | Inability to audit failures | Global and local feature importance via SHAP / interpretable features |
| Poor Generalization | Model works for one patient but fails on another | Subject-wise cross-validation and patient-specific policy tuning |

## 4. Fundamental Limitations
It is critical to distinguish between algorithmic trigger accuracy and clinical efficacy. This study demonstrates a *computational framework* on scalp EEG data. We do not claim that triggering VNS based on these parameters will necessarily abort seizures in a clinical setting. Prospective clinical trials with implanted devices are required to validate the therapeutic benefit of such personalized, explainable policies.
