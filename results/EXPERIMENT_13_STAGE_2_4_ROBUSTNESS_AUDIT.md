# Experiment 13 — Stage 2.4 Robustness Statistical Audit

**Audit date:** 4 November 2026  
**Status:** Completed  
**Scope:** Statistical analysis of the Experiment 13 synthetic robustness evaluation.

---

## 1. Audit objectives

This audit examines whether the observed detection sensitivity of Euclidean and
regularised Mahalanobis detectors changes with:

1. Covariance ridge regularisation.
2. Embedding dimensionality.
3. Control false-positive rates under these parameter changes.

The audit uses existing experiment results. It does not regenerate embeddings,
rerun the experiment, or modify the original result CSV files.

## 2. Experimental configuration

| Parameter | Configuration |
|---|---|
| Random seeds | 20 |
| Embedding dimensions | 16, 32, 64 |
| Covariance ridge values | 0.001, 0.01, 0.1 |
| Detectors | Euclidean, Mahalanobis |
| Calibration target FPR | 5% |
| Evaluation shift | Mean shift of 1.0 |
| Reference windows per seed | 50 |
| Calibration windows per seed | 200 |
| Control windows per seed | 500 |
| Evaluation windows per seed | 500 |
| Window size | 20 |

The robustness evaluation uses synthetic embedding vectors. The injected mean
shift is applied in a fixed direction.

## 3. Stage 2.4.1 — Structural integrity

The robustness results file contained:

- 360 experimental rows.
- 20 unique seeds.
- Three embedding dimensions.
- Three ridge values.
- Two detectors.
- 18 detector/dimension/ridge combinations.
- Exactly 20 observations per combination.
- Zero duplicate experimental rows.

All reported control FPRs and detection rates were within [0, 1].
No missing values were found, and all thresholds were finite.

Euclidean results were invariant to ridge values across all 60
seed/dimension combinations. Control FPR, detection rate, and threshold
did not change across ridge settings.

**Verdict: Structural checks passed.**

## 4. Stage 2.4.2 — Covariance ridge effect on detection sensitivity

Paired comparisons were performed within each embedding dimension, comparing
ridge 0.01 and 0.1 against ridge 0.001.

The paired t-test and Wilcoxon signed-rank test were used. Holm correction
was applied separately across the six comparisons for each test family.

| Dimension | Comparison | Mean detection difference | 95% CI |
|---:|---|---:|---:|
| 16 | 0.01 vs 0.001 | +0.0562 | [0.04124, 0.07116] |
| 16 | 0.1 vs 0.001 | +0.1264 | [0.08865, 0.16415] |
| 32 | 0.01 vs 0.001 | +0.1240 | [0.10869, 0.13931] |
| 32 | 0.1 vs 0.001 | +0.2566 | [0.20418, 0.30902] |
| 64 | 0.01 vs 0.001 | +0.0828 | [0.05921, 0.10639] |
| 64 | 0.1 vs 0.001 | +0.1896 | [0.14956, 0.22964] |

All six comparisons were significant after Holm correction under both
the paired t-test and Wilcoxon test.

The largest observed improvement was at dimension 32 with ridge 0.1,
where the mean detection-rate increase was 0.2566.

**Verdict:** Higher ridge values were associated with higher detection
sensitivity across all tested dimensions in this synthetic configuration.

## 5. Stage 2.4.3 — Covariance ridge effect on control FPR

Paired comparisons examined control FPR changes between ridge 0.01 or 0.1
and ridge 0.001 within each dimension.

| Dimension | Comparison | Mean FPR difference | 95% CI | Paired t p |
|---:|---|---:|---:|---:|
| 16 | 0.01 vs 0.001 | +0.0040 | [-0.000705, 0.008705] | 0.091157 |
| 16 | 0.1 vs 0.001 | +0.0043 | [-0.004771, 0.013371] | 0.333605 |
| 32 | 0.01 vs 0.001 | +0.0001 | [-0.004580, 0.004780] | 0.964794 |
| 32 | 0.1 vs 0.001 | -0.0018 | [-0.008902, 0.005302] | 0.601928 |
| 64 | 0.01 vs 0.001 | -0.0024 | [-0.008839, 0.004039] | 0.444948 |
| 64 | 0.1 vs 0.001 | -0.0015 | [-0.009766, 0.006766] | 0.708299 |

No comparison was statistically significant after Holm correction under
either the paired t-test or Wilcoxon test.

This is a failure to detect a statistically significant difference, not
proof that the FPRs are equivalent.

**Verdict:** No statistically significant ridge-related control FPR change
was detected.

## 6. Stage 2.4.4 — Embedding dimension effect on detection sensitivity

Welch independent-samples t-tests compared dimensions 16, 32, and 64.
The same numeric seed labels across dimensions were not treated as evidence
of paired datasets.

Holm correction was applied across all 12 planned comparisons.

| Detector | Ridge | Comparison | Mean detection difference | 95% CI |
|---|---:|---|---:|---:|
| Euclidean | 0.001 | 32D vs 16D | -0.1697 | [-0.211606, -0.127794] |
| Euclidean | 0.001 | 64D vs 16D | -0.3594 | [-0.391534, -0.327266] |
| Euclidean | 0.001 | 64D vs 32D | -0.1897 | [-0.235797, -0.143603] |
| Mahalanobis | 0.001 | 32D vs 16D | -0.3165 | [-0.388283, -0.244717] |
| Mahalanobis | 0.001 | 64D vs 16D | -0.4447 | [-0.508150, -0.381250] |
| Mahalanobis | 0.001 | 64D vs 32D | -0.1282 | [-0.197662, -0.058738] |
| Mahalanobis | 0.01 | 32D vs 16D | -0.2487 | [-0.318852, -0.178548] |
| Mahalanobis | 0.01 | 64D vs 16D | -0.4181 | [-0.478637, -0.357563] |
| Mahalanobis | 0.01 | 64D vs 32D | -0.1694 | [-0.241872, -0.096928] |
| Mahalanobis | 0.1 | 32D vs 16D | -0.1863 | [-0.225907, -0.146693] |
| Mahalanobis | 0.1 | 64D vs 16D | -0.3815 | [-0.417176, -0.345824] |
| Mahalanobis | 0.1 | 64D vs 32D | -0.1952 | [-0.238324, -0.152076] |

All 12 comparisons remained significant after Holm correction.

**Verdict:** Detection sensitivity declined consistently as embedding dimension
increased from 16 to 64 in the tested synthetic configuration.

## 7. Stage 2.4.5 — Embedding dimension effect on control FPR

Welch independent-samples t-tests compared control FPR across dimensions.
Holm correction was applied across all 12 comparisons.

No comparison was statistically significant after correction.

The observed mean FPR differences ranged from -0.0098 to +0.0062.
All 12 confidence intervals included zero.

**Verdict:** No statistically significant dimension-related control FPR
difference was detected.

This does not establish statistical equivalence between dimensions.

## 8. Overall findings

The robustness audit supports the following observations within the tested
synthetic configuration:

1. Increasing Mahalanobis covariance ridge was associated with increased
   detection sensitivity at dimensions 16, 32, and 64.
2. The ridge-related sensitivity improvements were statistically significant
   under both paired t-tests and Wilcoxon signed-rank tests after Holm
   correction.
3. No statistically significant ridge-related control FPR differences
   were detected.
4. Detection sensitivity decreased consistently with increasing embedding
   dimension for both Euclidean and Mahalanobis detectors.
5. No statistically significant dimension-related control FPR differences
   were detected.
6. Euclidean outputs were invariant to covariance ridge settings, as
   expected because covariance ridge regularisation does not affect the
   Euclidean scoring calculation.

## 9. Statistical and methodological limitations

- The experiment uses synthetic Gaussian embedding vectors, not naturally
  occurring biometric embedding streams.
- The positive shift is a fixed-direction mean shift. Other shift types,
  directions, and distributions were not established by this audit.
- Only 20 seeds were evaluated per condition.
- The dimension comparisons use independent-samples Welch tests. The same
  numeric seed labels across dimensions do not establish matched datasets.
- The reported confidence intervals are unadjusted 95% intervals; Holm
  correction was applied to the hypothesis-test p-values.
- Failure to reject an FPR difference does not demonstrate equivalence.
- The analysis does not establish real-world biometric recognition
  accuracy, demographic fairness, temporal drift performance, or
  deployment readiness.
- No conclusion is made that ridge 0.1 is universally optimal. Only three
  ridge values and one synthetic shift configuration were tested.

## 10. Final audit determination

**Stage 2.4 statistical robustness audit: completed.**

The structural checks and planned statistical comparisons were completed
using the existing Experiment 13 robustness results.

The evidence indicates that covariance ridge and embedding dimension
substantially affect synthetic detection sensitivity, while no statistically
significant control FPR changes were detected in the tested comparisons.

These conclusions are limited to the experimental configuration documented
above. They should not be generalized to real-world biometric systems
without further validation.

**No original experiment CSV files or notebook outputs were modified.**
