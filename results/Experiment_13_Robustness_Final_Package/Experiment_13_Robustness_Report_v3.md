# Experiment 13 — Scientific Robustness Evaluation

## 1. Purpose

Evaluate the sensitivity of synthetic embedding-level drift detectors
to embedding dimension and covariance regularisation.

This experiment uses synthetic data only.

## 2. Experimental Design

- Random seeds: 20
- Embedding dimensions: 16, 32, 64
- Covariance ridge values: 0.001, 0.01, 0.1
- Detectors: Euclidean and regularised Mahalanobis
- Calibration target false-positive rate: 5%
- Evaluation mean shift: 1.0 in the first coordinate
- Reference windows: 50
- Window size: 20
- Calibration windows: 200
- Control windows: 500
- Evaluation windows per condition: 500
- Total trial records: 360
- Total configurations: 18

## 3. Statistical Analysis

For each configuration, results were aggregated across 20 random seeds.

Means and sample standard deviations were calculated for control
false-positive rates and detection rates.

95% confidence intervals were calculated using the Student's
t-distribution across the 20 seed-level observations.

These intervals describe variation across the simulated seeds
under this experimental design.

## 4. Results

Overall mean control false-positive rate: 0.0520
(5.20%).

Overall mean detection rate: 0.5783
(57.83%).

### Detection rate by embedding dimension

| Dimension | Mean detection rate |
|---:|---:|
| 16 | 0.7774 |
| 32 | 0.5673 |
| 64 | 0.3903 |

### Detector and regularisation observations

- Euclidean results were invariant across the tested ridge values,
  as expected because Euclidean scoring does not use covariance.
- Mahalanobis detection rates varied with covariance regularisation.
- In this simulation, increasing the ridge value increased the
  observed Mahalanobis detection rates across the tested configurations.
- Detection rates decreased with increasing embedding dimension
  under the tested one-coordinate mean-shift condition.
- The overall mean control false-positive rate was close to the
  nominal 5% calibration target.

## 5. Numerical and Implementation Validation

The following checks were completed:

- 360 trial records and 18 configurations verified.
- 20 seeds per configuration verified.
- No duplicate trial records or missing numerical values.
- Saved summary statistics independently reconstructed from trial data.
- All 95% confidence interval statistics independently recalculated.
- Confidence interval bounds verified.
- Euclidean ridge invariance verified.
- Regularised covariance matrices verified as positive definite
  for dimensions 16, 32 and 64 in the corrected stability diagnostic
  using ridge = 0.01.
- Detector scores verified as finite.

## 6. Limitations

This experiment is a synthetic diagnostic using independent
randomly generated windows and an additive mean shift in one coordinate.

It does not establish:

- Real-world biometric recognition accuracy.
- Real-world distribution-shift detection performance.
- Temporal detection delay or sequential detection performance.
- Identity-disjoint generalisation.
- Fairness or demographic robustness.
- Deployment readiness.

The 20 random seeds represent repeated simulations, not 20 independent
real-world biometric populations.

## 7. Conclusion

The experiment demonstrates sensitivity of synthetic embedding-level
drift scores to embedding dimension and covariance regularisation.

The Euclidean detector is invariant to covariance ridge settings,
whereas the Mahalanobis detector responds to ridge regularisation.

The observed decline in detection sensitivity with increasing
embedding dimension is specific to the tested synthetic setup.

These findings provide a controlled numerical robustness assessment,
not evidence of real-world biometric system performance.
