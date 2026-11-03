# Experiment 12 V2: Synthetic Embedding-Shift Sensitivity Analysis under Identity-Disjoint Evaluation

## 1. Experiment Overview

This experiment evaluates how Euclidean and regularised Mahalanobis
embedding-level anomaly detectors respond to controlled synthetic shifts
in embedding space.

The evaluation uses identity-disjoint reference, calibration, and
evaluation partitions. Synthetic shifts are applied to evaluation-window
means along multiple randomly generated directions.

The experiment is designed to examine detector sensitivity under
controlled perturbations. It does not establish real-world temporal drift
detection, biometric recognition accuracy, fairness, or deployment readiness.

## 2. Corrected Scope

The experiment is described as:

**Synthetic Embedding-Shift Sensitivity Analysis under Identity-Disjoint Evaluation.**

The dataset does not provide verified acquisition timestamps or a
validated chronological sequence. Consequently, the original row ordering
is not treated as temporal evidence.

The synthetic shifts are controlled perturbations, not observed changes
in real biometric acquisition conditions.

## 3. Dataset

- Dataset: LFW-derived embedding dataset
- Total embedding rows: 3,023
- Identity count: 62
- Embedding dimensions: 768
- Additional column: identity_label
- Missing values: none
- Duplicate rows: none

### Identity-disjoint partition

| Partition | Samples | Identities |
|---|---:|---:|
| Reference | 1,492 | 37 |
| Calibration | 1,071 | 12 |
| Evaluation | 460 | 13 |

The partitions use seeds 2024 and 2025. Identity overlap between
the three partitions was checked and was zero.

### Window construction

Window size: 20 samples.

| Partition | Complete windows | Discarded samples |
|---|---:|---:|
| Reference | 74 | 12 |
| Calibration | 53 | 11 |
| Evaluation | 23 | 0 |

## 4. Preprocessing

Embedding standardisation parameters were estimated using reference data
only. The same reference-derived transformation was applied to calibration
and evaluation data.

All transformed embeddings were finite.

The reference centroid was calculated from the reference-window means.

## 5. Detectors

Two detectors were evaluated.

### 5.1 Euclidean detector

The Euclidean detector measures the distance between an evaluation-window
mean and the reference centroid.

### 5.2 Regularised Mahalanobis detector

The Mahalanobis detector uses a covariance estimate fitted on reference
data only.

The empirical covariance matrix was ill-conditioned, so Ledoit-Wolf
shrinkage regularisation was used.

- Shrinkage coefficient: approximately 0.0221563
- Regularised covariance condition number: approximately 4802.76

The regularised covariance matrix was finite and symmetric, and its
inverse consistency check passed.

## 6. Calibration and Thresholds

Calibration used 53 complete windows.

Thresholds were estimated from the calibration score distributions
at the 90th, 95th, and 97.5th percentiles.

The thresholds were frozen before evaluation.

### Euclidean thresholds

| Percentile | Threshold |
|---|---:|
| 90th | 10.994498 |
| 95th | 11.484664 |
| 97.5th | 12.456855 |

### Mahalanobis thresholds

| Percentile | Threshold |
|---|---:|
| 90th | 9.696375 |
| 95th | 10.251740 |
| 97.5th | 10.846457 |

## 7. Synthetic Shift Design

Synthetic shifts were generated using 20 random unit directions in
the 768-dimensional embedding space.

Configuration:

- Random seed: 20261203
- Number of directions: 20
- Embedding dimension: 768
- Shift magnitudes: 0.00, 0.25, 0.50, 0.75, 1.00
- Evaluation windows: 23

For each direction and magnitude, the perturbation was applied to
each evaluation-window mean:

    z_shifted = z_original + alpha * direction

where alpha is the shift magnitude.

This produced 100 direction-magnitude conditions.

The zero-shift condition reconstructed the original evaluation scores
exactly for both detectors.

## 8. Evaluation Measures

The analysis reports:

1. Raw alarm rate across evaluation windows.
2. Additional alarm crossings among windows that were not alarms at baseline.
3. Lost alarms: baseline alarms that become non-alarms after perturbation.
4. Net alarm-count change: new alarms minus lost alarms.
5. Direction-level variability and the percentage of directions with
   positive, zero, or negative net alarm-count change.

These measures describe threshold-crossing behaviour under the
specified synthetic perturbations. They are not conventional recall,
sensitivity against labelled drift events, or real-world detection accuracy.

## 9. Results

### 9.1 Baseline alarm rates

Before synthetic shifts, the evaluation set already showed elevated
alarm rates.

| Detector | Threshold | Baseline alarms | Baseline alarm rate |
|---|---:|---:|---:|
| Euclidean | 90th | 9/23 | 39.13% |
| Euclidean | 95th | 8/23 | 34.78% |
| Euclidean | 97.5th | 5/23 | 21.74% |
| Mahalanobis | 90th | 19/23 | 82.61% |
| Mahalanobis | 95th | 11/23 | 47.83% |
| Mahalanobis | 97.5th | 9/23 | 39.13% |

These results indicate that calibration thresholds did not transfer
cleanly to the identity-disjoint evaluation partition.

The evaluation partition is not independently established as drift-free.
Therefore, these baseline alarms cannot automatically be classified
as false positives.

### 9.2 Raw alarm-rate response

At shift magnitude 1.00:

| Detector | Threshold | Mean raw alarm rate |
|---|---:|---:|
| Euclidean | 90th | 40.43% |
| Euclidean | 95th | 34.78% |
| Euclidean | 97.5th | 21.74% |
| Mahalanobis | 90th | 94.35% |
| Mahalanobis | 95th | 86.52% |
| Mahalanobis | 97.5th | 46.52% |

Mahalanobis shows a stronger increase in raw alarm rates than Euclidean.
However, raw rates include alarms already present at zero shift.

### 9.3 Additional alarm crossings

At shift magnitude 1.00, the mean additional alarm crossings were:

| Detector | Threshold | Mean additional alarms | Additional crossing rate among baseline non-alarms |
|---|---:|---:|---:|
| Euclidean | 90th | 0.30 | 2.14% |
| Euclidean | 95th | 0.00 | 0.00% |
| Euclidean | 97.5th | 0.00 | 0.00% |
| Mahalanobis | 90th | 2.70 | 67.50% |
| Mahalanobis | 95th | 8.90 | 74.17% |
| Mahalanobis | 97.5th | 1.70 | 12.14% |

The Mahalanobis 90th-percentile result uses only four baseline
non-alarming windows. Its percentage therefore has a very small
denominator.

### 9.4 Paired alarm transitions

Lost alarms were zero in nearly all tested conditions. The exception
was the Mahalanobis 95th-percentile threshold at shift magnitude 0.25,
where mean new alarms were 0.05 and mean lost alarms were 0.10 per
direction.

At magnitude 1.00, Mahalanobis produced positive net alarm-count
changes in all 20 directions at all three thresholds.

### 9.5 Direction-level consistency

At magnitude 1.00:

| Detector | Threshold | Directions with positive net alarm change |
|---|---:|---:|
| Euclidean | 90th | 30% |
| Euclidean | 95th | 0% |
| Euclidean | 97.5th | 0% |
| Mahalanobis | 90th | 100% |
| Mahalanobis | 95th | 100% |
| Mahalanobis | 97.5th | 100% |

The strongest mean response occurred for Mahalanobis at the
95th-percentile threshold, with 8.90 additional alarms per direction
on average.

The direction-level results indicate consistency under the tested
synthetic perturbations. They do not establish generalisation to
unseen real-world drift mechanisms.

## 10. Interpretation

Under this experimental configuration:

- Mahalanobis was more responsive to the applied synthetic shifts.
- Euclidean showed limited additional alarm crossings.
- Mahalanobis had substantially higher baseline alarm rates.
- The calibration-to-evaluation transfer problem remains unresolved.
- The strongest synthetic response should not be interpreted as proof
  that Mahalanobis is a better operational biometric drift detector.

The results demonstrate detector-specific responses to controlled
embedding-space perturbations, not validated real-world detection
performance.

## 11. Limitations

1. Synthetic shifts do not represent all possible real biometric
   distribution changes.
2. The dataset does not provide verified temporal ordering.
3. Evaluation uses only 23 complete windows.
4. There are only 20 synthetic shift directions.
5. The same 23 evaluation windows are reused across directions.
   Direction-condition combinations are not independent real-world samples.
6. The identity-disjoint evaluation partition has elevated baseline
   alarm rates.
7. No ground-truth drift-event labels are available.
8. No real-world biometric recognition accuracy or fairness evaluation
   is performed.
9. The experiment does not demonstrate deployment readiness.

## 12. Reproducibility Configuration

| Parameter | Value |
|---|---|
| Reference split seed | 2024 |
| Calibration split seed | 2025 |
| Synthetic shift seed | 20261203 |
| Window size | 20 |
| Embedding dimensions | 768 |
| Shift directions | 20 |
| Shift magnitudes | 0.00, 0.25, 0.50, 0.75, 1.00 |
| Detector types | Euclidean, regularised Mahalanobis |
| Threshold percentiles | 90, 95, 97.5 |

## 13. Output Files

The following files were generated with the `exp12_v2_` prefix.
The original Experiment 12 outputs were retained.

| File | Description |
|---|---|
| `exp12_v2_raw_sensitivity.csv` | Raw alarm-rate sensitivity results |
| `exp12_v2_baseline_adjusted_sensitivity.csv` | Additional alarm-crossing results |
| `exp12_v2_alarm_transitions.csv` | New, lost, and net alarm transitions |
| `exp12_v2_direction_robustness.csv` | Direction-level robustness summary |
| `exp12_v2_raw_sensitivity.png` | Raw alarm-rate plot |
| `exp12_v2_baseline_adjusted_sensitivity.png` | Baseline-adjusted sensitivity plot |

## 14. Conclusion

The experiment shows that regularised Mahalanobis scoring responds
more strongly than Euclidean scoring to the tested synthetic
embedding shifts.

However, the elevated baseline alarm rates, limited evaluation
windows, synthetic perturbation design, and absence of ground-truth
drift labels prevent claims of validated real-world drift-detection
performance.

The results should be treated as controlled preliminary evidence
about embedding-level detector sensitivity.
