# Experiment 13: False-Alarm-Controlled Embedding-Level Drift Detection

## 1. Objective

Evaluate the trade-off between false-alarm frequency and detection sensitivity
for Euclidean and regularised Mahalanobis embedding-level drift detectors
under controlled synthetic distribution shifts.

## 2. Experimental Configuration

- Random seeds: 20 consecutive seeds beginning at 202613
- Embedding dimension: 64
- Reference windows: 50 per seed, each containing 20 observations
- Calibration windows: 200 per seed
- Independent control windows: 500 per seed
- Evaluation windows: 500 per seed and shift condition
- Mean-shift strengths: 0.00, 0.25, 0.50, 0.75, 1.00
- Nominal calibration false-alarm targets: 1%, 5%, 10%
- Detectors: Euclidean and regularised Mahalanobis
- Covariance ridge parameter: 0.01

## 3. Methodology

Each seed generated separate reference, calibration, control, and evaluation
datasets. Detector parameters were fitted using reference data. Thresholds
were estimated using calibration scores only. Control and evaluation datasets
were scored without being used to fit thresholds.

Within each seed and shift condition, both detectors used the same evaluation
samples. The positive-shift conditions applied an additive shift to the first
embedding coordinate.

The evaluation unit was an embedding window. An alarm was counted when its
distance score exceeded the calibrated threshold.

## 4. Control False-Alarm Results

| detector    |   target_fpr |   mean |     sd |   n |     se |   ci95_lower |   ci95_upper |
|:------------|-------------:|-------:|-------:|----:|-------:|-------------:|-------------:|
| euclidean   |       0.0100 | 0.0136 | 0.0117 |  20 | 0.0026 |       0.0081 |       0.0191 |
| euclidean   |       0.0500 | 0.0565 | 0.0191 |  20 | 0.0043 |       0.0476 |       0.0654 |
| euclidean   |       0.1000 | 0.1077 | 0.0254 |  20 | 0.0057 |       0.0958 |       0.1196 |
| mahalanobis |       0.0100 | 0.0152 | 0.0093 |  20 | 0.0021 |       0.0109 |       0.0195 |
| mahalanobis |       0.0500 | 0.0513 | 0.0173 |  20 | 0.0039 |       0.0432 |       0.0594 |
| mahalanobis |       0.1000 | 0.0997 | 0.0222 |  20 | 0.0050 |       0.0893 |       0.1101 |

## 5. Detection Sensitivity Results

| detector    |   target_fpr |   mean_shift |   mean |     sd |   n |     se |   ci95_lower |   ci95_upper |
|:------------|-------------:|-------------:|-------:|-------:|----:|-------:|-------------:|-------------:|
| euclidean   |       0.0100 |       0.0000 | 0.0127 | 0.0106 |  20 | 0.0024 |       0.0077 |       0.0177 |
| euclidean   |       0.0100 |       0.2500 | 0.0202 | 0.0147 |  20 | 0.0033 |       0.0133 |       0.0271 |
| euclidean   |       0.0100 |       0.5000 | 0.0409 | 0.0253 |  20 | 0.0057 |       0.0291 |       0.0527 |
| euclidean   |       0.0100 |       0.7500 | 0.0981 | 0.0517 |  20 | 0.0116 |       0.0739 |       0.1223 |
| euclidean   |       0.0100 |       1.0000 | 0.2573 | 0.0879 |  20 | 0.0197 |       0.2162 |       0.2984 |
| euclidean   |       0.0500 |       0.0000 | 0.0504 | 0.0209 |  20 | 0.0047 |       0.0406 |       0.0602 |
| euclidean   |       0.0500 |       0.2500 | 0.0694 | 0.0258 |  20 | 0.0058 |       0.0573 |       0.0815 |
| euclidean   |       0.0500 |       0.5000 | 0.1220 | 0.0289 |  20 | 0.0065 |       0.1085 |       0.1355 |
| euclidean   |       0.0500 |       0.7500 | 0.2411 | 0.0584 |  20 | 0.0131 |       0.2138 |       0.2684 |
| euclidean   |       0.0500 |       1.0000 | 0.4748 | 0.0678 |  20 | 0.0152 |       0.4431 |       0.5065 |
| euclidean   |       0.1000 |       0.0000 | 0.1015 | 0.0243 |  20 | 0.0054 |       0.0901 |       0.1129 |
| euclidean   |       0.1000 |       0.2500 | 0.1265 | 0.0293 |  20 | 0.0066 |       0.1128 |       0.1402 |
| euclidean   |       0.1000 |       0.5000 | 0.2088 | 0.0375 |  20 | 0.0084 |       0.1913 |       0.2263 |
| euclidean   |       0.1000 |       0.7500 | 0.3669 | 0.0480 |  20 | 0.0107 |       0.3444 |       0.3894 |
| euclidean   |       0.1000 |       1.0000 | 0.6136 | 0.0522 |  20 | 0.0117 |       0.5892 |       0.6380 |
| mahalanobis |       0.0100 |       0.0000 | 0.0141 | 0.0099 |  20 | 0.0022 |       0.0095 |       0.0187 |
| mahalanobis |       0.0100 |       0.2500 | 0.0190 | 0.0107 |  20 | 0.0024 |       0.0140 |       0.0240 |
| mahalanobis |       0.0100 |       0.5000 | 0.0366 | 0.0144 |  20 | 0.0032 |       0.0299 |       0.0433 |
| mahalanobis |       0.0100 |       0.7500 | 0.0756 | 0.0326 |  20 | 0.0073 |       0.0604 |       0.0908 |
| mahalanobis |       0.0100 |       1.0000 | 0.1680 | 0.0583 |  20 | 0.0130 |       0.1407 |       0.1953 |
| mahalanobis |       0.0500 |       0.0000 | 0.0504 | 0.0234 |  20 | 0.0052 |       0.0395 |       0.0613 |
| mahalanobis |       0.0500 |       0.2500 | 0.0628 | 0.0204 |  20 | 0.0046 |       0.0532 |       0.0724 |
| mahalanobis |       0.0500 |       0.5000 | 0.0997 | 0.0313 |  20 | 0.0070 |       0.0851 |       0.1143 |
| mahalanobis |       0.0500 |       0.7500 | 0.1814 | 0.0552 |  20 | 0.0123 |       0.1556 |       0.2072 |
| mahalanobis |       0.0500 |       1.0000 | 0.3236 | 0.0860 |  20 | 0.0192 |       0.2834 |       0.3638 |
| mahalanobis |       0.1000 |       0.0000 | 0.0983 | 0.0291 |  20 | 0.0065 |       0.0847 |       0.1119 |
| mahalanobis |       0.1000 |       0.2500 | 0.1191 | 0.0291 |  20 | 0.0065 |       0.1055 |       0.1327 |
| mahalanobis |       0.1000 |       0.5000 | 0.1757 | 0.0392 |  20 | 0.0088 |       0.1574 |       0.1940 |
| mahalanobis |       0.1000 |       0.7500 | 0.2799 | 0.0654 |  20 | 0.0146 |       0.2493 |       0.3105 |
| mahalanobis |       0.1000 |       1.0000 | 0.4610 | 0.0947 |  20 | 0.0212 |       0.4167 |       0.5053 |

## 6. Interpretation

The observed mean control false-alarm rates were generally close to the
nominal targets, but individual-seed variability was substantial at the
1% target.

Detection rates increased as the injected mean shift increased. Across the
tested conditions, Euclidean generally achieved higher detection rates than
Mahalanobis, particularly at stronger shifts. This is a finding for this
specific synthetic setup and does not establish universal detector superiority.

At the 5% target and shift strength 1.00, the mean detection rates were
47.48% for Euclidean and 32.36% for Mahalanobis. At the 10% target, they
were 61.36% and 46.10%, respectively.

The confidence intervals describe variability across 20 seed-level results.
They do not establish performance on real biometric identities or deployed
systems.

## 7. Limitations

1. All observations were synthetic Gaussian vectors, not real biometric embeddings.
2. The injected drift was an additive shift in one coordinate.
3. The experiment used independent windows rather than a temporally dependent stream.
4. Detection delay and sequential alarm behaviour were not evaluated.
5. No demographic fairness, biometric recognition accuracy, or bias mitigation
   was evaluated.
6. Results depend on the selected dimension, covariance regularisation,
   sample sizes, and threshold calibration method.
7. The evaluation does not establish identity-disjoint or real-world
   generalisation.

## 8. Reproducibility

- Main trial results: `experiment13_trial_results.csv`
- Control summary and confidence intervals: `experiment13_control_fpr_summary.csv`
- Detection summary and confidence intervals: `experiment13_detection_summary.csv`
- Initial calibration results: `experiment13_single_seed_calibration.csv`
- Initial control results: `experiment13_single_seed_control.csv`
- Detection plot: `experiment13_detection_ci.png`

The experiment uses NumPy random generators with documented seeds. The
reported results should be regenerated from the saved notebook before
publication.

## 9. Conclusion

This synthetic study demonstrates a measurable trade-off between nominal
false-alarm targets and detection sensitivity. The results motivate further
evaluation using real embeddings, multiple drift mechanisms, and temporally
structured streams. They should be treated as preliminary methodological
evidence rather than real-world biometric validation.
