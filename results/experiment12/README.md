# Experiment 12 — Cross-Condition Generalisation of Embedding-Level Drift Detectors

This directory contains the outputs of the Experiment 12 synthetic embedding-drift evaluation.

## Artifacts

- `exp12_trial_results.csv` — Trial-level detector results
- `exp12_detection_summary.csv` — Detection performance by drift strength
- `exp12_control_results.csv` — Independent control-set evaluation
- `exp12_calibration_results.csv` — Calibrated detector thresholds
- `exp12_detection_rate.png` — Detection-rate plot
- `exp12_post_alarm_rate.png` — Post-drift alarm-rate plot
- `README_experiment12.txt` — Methodology and limitations

## Important limitations

The experiment uses artificial additive shifts, 30 calibration windows, and 10 synthetic directions per drift condition. The data partitions share the same identities, so this is image-level separation rather than identity-disjoint generalisation. Results are preliminary and do not establish real-world biometric drift performance.
