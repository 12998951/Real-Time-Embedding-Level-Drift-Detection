# Experiment 13 — Stage 2.4
# Statistical Robustness Analysis

**Report date:** 4 November 2026  
**Status:** Statistical analysis completed

## 1. Objective

This analysis evaluates the sensitivity of Euclidean and regularised
Mahalanobis embedding-level drift detectors under different synthetic
embedding dimensions and Mahalanobis covariance ridge parameters.

The analysis examines:

1. The effect of covariance ridge regularisation on detection sensitivity.
2. Whether ridge changes are accompanied by changes in control false-alarm rates.
3. The effect of embedding dimension on detection sensitivity.

## 2. Experimental design

The robustness trial dataset contains 360 rows, generated using:

- 20 synthetic random seeds.
- Embedding dimensions: 16, 32 and 64.
- Covariance ridge values: 0.001, 0.01 and 0.1.
- Detectors: Euclidean and Mahalanobis.
- Nominal control false-alarm target: 5%.
- Synthetic evaluation shift magnitude: 1.0.
- Evaluation shift direction: first embedding coordinate.

Each seed and dimension combination uses generated reference,
calibration, control and evaluation data. Within a fixed dimension
and seed, these same datasets are evaluated across ridge settings.

The same seed number across different dimensions does not establish
matched observations across dimensions.

## 3. Statistical methods

### 3.1 Ridge sensitivity comparisons

Mahalanobis detection rates were compared between ridge 0.001 and
ridge 0.01, and between ridge 0.001 and ridge 0.1.

Comparisons were paired by seed within each embedding dimension.

Paired t-tests and two-sided Wilcoxon signed-rank tests were used.
Holm correction was applied separately to the six comparisons
for each test family.

### 3.2 Control false-alarm comparisons

Control false-alarm rates were compared using paired t-tests and
two-sided Wilcoxon signed-rank tests, paired by seed within dimension.

Holm correction was applied separately to the six comparisons
for each test family.

### 3.3 Embedding-dimension comparisons

At ridge 0.01, detection rates were compared across dimensions
16, 32 and 64 for each detector.

Welch independent-sample t-tests were used because the generated
data arrays differ across dimensions.

Holm correction was applied across the six dimension comparisons.

## 4. Ridge effect on detection sensitivity

All six comparisons showed positive detection-rate differences.

| Dimension | Comparison | Mean difference | Holm-adjusted Wilcoxon p |
|---|---|---:|---:|
| 16 | 0.01 vs 0.001 | +0.0562 | 0.00035 |
| 16 | 0.1 vs 0.001 | +0.1264 | 0.00006 |
| 32 | 0.01 vs 0.001 | +0.1240 | 0.00035 |
| 32 | 0.1 vs 0.001 | +0.2566 | 0.00035 |
| 64 | 0.01 vs 0.001 | +0.0828 | 0.00006 |
| 64 | 0.1 vs 0.001 | +0.1896 | 0.00035 |

All six comparisons remained statistically significant after
Holm correction using both paired t-tests and Wilcoxon tests.

The largest observed sensitivity improvement was 0.2566
(25.66 percentage points), for dimension 32 at ridge 0.1
relative to ridge 0.001.

These results indicate that higher tested ridge values improved
detection sensitivity under this particular synthetic configuration.

## 5. Ridge effect on control false-alarm rates

| Dimension | Comparison | Mean FPR difference | Holm-adjusted Wilcoxon p |
|---|---|---:|---:|
| 16 | 0.01 vs 0.001 | +0.0040 | 0.47326 |
| 16 | 0.1 vs 0.001 | +0.0043 | 1.00000 |
| 32 | 0.01 vs 0.001 | +0.0001 | 1.00000 |
| 32 | 0.1 vs 0.001 | -0.0018 | 1.00000 |
| 64 | 0.01 vs 0.001 | -0.0024 | 1.00000 |
| 64 | 0.1 vs 0.001 | -0.0015 | 1.00000 |

No comparison showed a statistically significant control-FPR
difference after Holm correction using either test family.

This is not proof of equivalence or non-inferiority. The analysis
did not conduct a formal equivalence test.

## 6. Embedding-dimension effect

Comparisons were conducted at ridge 0.01.

### 6.1 Euclidean detector

| Comparison | Mean detection rate, first dimension | Mean detection rate, second dimension | Difference | Holm-adjusted p |
|---|---:|---:|---:|---:|
| 16 vs 32 | 0.8144 | 0.6447 | +0.1697 | <0.000005 |
| 16 vs 64 | 0.8144 | 0.4550 | +0.3594 | <0.000005 |
| 32 vs 64 | 0.6447 | 0.4550 | +0.1897 | <0.000005 |

### 6.2 Mahalanobis detector

| Comparison | Mean detection rate, first dimension | Mean detection rate, second dimension | Difference | Holm-adjusted p |
|---|---:|---:|---:|---:|
| 16 vs 32 | 0.7357 | 0.4870 | +0.2487 | <0.000005 |
| 16 vs 64 | 0.7357 | 0.3176 | +0.4181 | <0.000005 |
| 32 vs 64 | 0.4870 | 0.3176 | +0.1694 | 0.00003 |

All six dimension comparisons remained statistically significant
after Holm correction.

Detection sensitivity decreased as embedding dimension increased
for both detectors under the tested synthetic configuration.

## 7. Main findings

1. Increasing the Mahalanobis ridge parameter from 0.001 to 0.01
   or 0.1 significantly increased synthetic detection sensitivity
   in all tested dimensions.

2. No statistically significant control false-alarm-rate increase
   was detected in the six ridge comparisons.

3. Euclidean results were invariant to ridge changes, consistent
   with the Euclidean scoring method not using the covariance ridge.

4. At ridge 0.01, both detectors showed significantly lower
   detection sensitivity at higher tested embedding dimensions.

5. The results demonstrate parameter and dimensionality sensitivity
   in the tested synthetic experimental setting. They do not establish
   universal detector superiority or an optimal ridge value.

## 8. Limitations

- The robustness experiment uses synthetic embedding vectors.
- Only 20 random seeds were evaluated per condition.
- Only one shift magnitude (1.0) and one shift direction
  (the first coordinate) were used in the robustness analysis.
- Dimension comparisons were conducted only at ridge 0.01.
- The analysis does not establish performance under naturally
  occurring biometric drift.
- The analysis does not measure biometric recognition accuracy,
  demographic fairness, or deployment readiness.
- Non-significant false-alarm comparisons do not establish
  statistical equivalence.
- No formal equivalence or non-inferiority test was conducted.
- These findings should not be generalised beyond the tested
  configurations without further experiments.

## 9. Conclusion

The statistical robustness analysis identifies substantial
sensitivity to Mahalanobis covariance ridge regularisation and
embedding dimensionality in the tested synthetic setting.

Higher ridge values improved detection sensitivity without a
statistically significant control-FPR increase being observed.
Higher embedding dimensions were associated with lower detection
sensitivity for both tested detectors.

These results provide controlled synthetic evidence about detector
parameter sensitivity. They are not evidence of real-world biometric
drift-detection effectiveness or deployment readiness.

## 10. Reproducibility reference

Primary input:
`results/experiment13_robustness_trials.csv`

This report documents the statistical analyses performed on that
existing dataset. It does not regenerate the original experiment
or modify its underlying trial results.
