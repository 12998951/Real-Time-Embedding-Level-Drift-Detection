Experiment 12 — Cross-Condition Generalisation of Embedding-Level Drift Detectors

Contents
- exp12_trial_results.csv: trial-level results
- exp12_detection_summary.csv: condition-level summary
- exp12_control_results.csv: independent control-set results
- exp12_calibration_results.csv: calibrated thresholds
- exp12_detection_rate.png: detection rate plot
- exp12_post_alarm_rate.png: post-drift alarm-rate plot

Method
- 3,023 embedding rows, each with 768 dimensions
- Image-level partitions: reference, calibration, control, evaluation
- Window size: 20 embeddings
- 10 synthetic drift directions per positive drift strength
- Detectors: Euclidean and regularised Mahalanobis
- Thresholds calibrated using the calibration windows

Limitations
- All partitions contain the same 62 identities. This is image-level separation,
  not identity-disjoint generalisation.
- Only 30 calibration windows are available, so threshold calibration is coarse.
- The drift is an artificial additive shift, not naturally occurring biometric drift.
- Results are preliminary and should not be presented as real-world performance.
- The experiment does not establish fairness, bias mitigation, or robustness to
  naturally occurring face or voice changes.
