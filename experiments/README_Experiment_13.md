# Experiment 13 — False-Alarm-Controlled Embedding-Level Drift Detection

## Objective
Evaluate the trade-off between false-alarm frequency and detection sensitivity
for Euclidean and regularized Mahalanobis detectors under controlled synthetic shifts.

## Method
- 20 random seeds, starting at 202613
- 64-dimensional synthetic Gaussian embeddings
- 50 reference windows per seed
- 200 calibration windows per seed
- 500 independent control windows per seed
- 500 evaluation windows per shift condition
- Window size: 20 observations
- Mean shifts: 0.00, 0.25, 0.50, 0.75, 1.00
- Nominal false-alarm targets: 1%, 5%, 10%
- Covariance ridge parameter: 0.01

## Files
- `experiment13_trial_results.csv`: seed-level control and evaluation results
- `experiment13_control_fpr_summary.csv`: control false-alarm summary and 95% intervals
- `experiment13_detection_summary.csv`: detection sensitivity summary and 95% intervals
- `experiment13_single_seed_calibration.csv`: initial calibration thresholds
- `experiment13_single_seed_control.csv`: initial control evaluation
- `experiment13_detection_ci.png`: detection sensitivity plot with 95% intervals
- `Experiment_13_Reproducible_Report.md`: methodology, results, and limitations

## Main finding
Detection sensitivity increased with injected mean-shift strength. The Euclidean
detector generally achieved higher detection rates than the Mahalanobis detector
under this synthetic configuration. This does not establish universal superiority.

## Limitations
All data are synthetic Gaussian vectors. The experiment does not validate real
biometric embeddings, naturally occurring drift, identity-disjoint generalization,
demographic fairness, biometric recognition accuracy, or deployment readiness.
Windows are independent; sequential detection delay is not measured.

## Reproducibility
The notebook records the generation process and random seeds. Re-run the notebook
from a fresh runtime to verify that the saved results and figures can be regenerated.
