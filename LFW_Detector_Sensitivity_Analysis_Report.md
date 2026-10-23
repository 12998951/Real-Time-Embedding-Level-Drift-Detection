# Experiment 09: Detector Sensitivity Analysis

## 1. Objective

This experiment evaluates how window size, covariance regularisation, and
controlled embedding-space shifts affect Euclidean and Mahalanobis anomaly
detectors.

## 2. Dataset

- Dataset: Labeled Faces in the Wild (LFW) image embeddings
- Images: 3,023
- Embedding dimensions: 768
- Identity classes: 62
- Embedding model: Generic Vision Transformer (ViT)
- Missing feature values: 0

The embeddings were standardised using statistics estimated exclusively
from the reference pool.

## 3. Experimental design

The dataset was divided into disjoint image-level pools:

- Reference: 1209 images
- Calibration: 907 images
- Evaluation: 907 images

Window sizes: [10, 20, 40]

Mahalanobis covariance ridge values: [0.001, 0.01, 0.1]

Injected drift strengths: [0.0, 0.05, 0.1, 0.2, 0.35]

Drift directions: 10

Calibration thresholds used the empirical 95th percentile. Thresholds were
not recalibrated using shifted evaluation data.

## 4. Metrics

- Calibration alarm rate
- Held-out no-shift control alarm rate
- Pre-shift and post-shift alarm rates
- Change in post-shift alarm rate relative to the no-shift control
- First post-shift alarm delay, measured in windows

An alarm after the nominal shift point is not necessarily evidence of
successful drift detection, because the no-shift control also produces alarms.

## 5. Findings

The observed results show sensitivity to both window size and covariance
regularisation. The strongest increases in post-shift alarm rate were
observed at window size 40 with higher injected drift strengths for some
Mahalanobis configurations. Euclidean detection did not show a consistent
increase across all configurations.

Many configurations showed little or no improvement over the corresponding
no-shift control. This indicates that the detectors, thresholds, and current
embedding representation are not yet sufficiently reliable for strong
claims about real-world biometric drift detection.

## 6. Limitations

1. The embedding extractor is a generic ViT model, not a validated
   face-recognition embedding model.
2. LFW is not a chronological biometric stream.
3. Drift is injected synthetically into standardised embedding space.
4. Identity classes overlap across the image-level reference, calibration,
   and evaluation pools.
5. The same evaluation windows are reused across drift strengths and
   directions; these are not independent real-world trials.
6. Calibration pools contain relatively few windows, especially at larger
   window sizes, making threshold estimates unstable.
7. Alarm-rate differences are descriptive and have not established
   statistical significance.
8. The reported delay is measured in windows, not seconds.

## 7. Conclusion

Experiment 09 provides a preliminary sensitivity analysis of Euclidean and
Mahalanobis detectors under controlled embedding-space perturbations.
The results indicate that detection behaviour depends on window size,
covariance regularisation, and shift magnitude. However, the inconsistent
response and elevated control alarm rates prevent a claim of robust
real-world drift detection.

Future work should use a validated face-recognition embedding model,
larger independent calibration and control streams, multiple independently
sampled evaluation sets, uncertainty intervals, and realistic temporal
or domain-shift scenarios.

## 8. Reproducibility

Random seeds and experimental parameters are defined in the notebook.
Detailed trial data, the corrected summary table, and sensitivity plots are
saved separately in `results/LFW_Real_Data_Results/`.

Report generated: 2026-10-23
