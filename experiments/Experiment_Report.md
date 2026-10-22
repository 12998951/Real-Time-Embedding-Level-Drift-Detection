# Experimental Report: Comparing Embedding-Level Drift Detectors

## 1. Objective

This experiment compares three statistical detectors for identifying distribution changes in synthetic embedding vectors: Euclidean distance, regularized Mahalanobis distance, and a variance-sensitive score. The objective is to characterize their responses to controlled mean shifts and variance changes under a common calibration procedure.

**Scope:** These are synthetic Gaussian experiments. The vectors are not embeddings extracted from real face, voice, or other biometric systems.

## 2. Experimental Setup

| Parameter | Configuration |
|---|---:|
| Embedding dimensionality | 16 |
| Reference samples per seed | 500 |
| Calibration windows per seed | 2,000 |
| Test windows per scenario and seed | 1,000 |
| Independent validation windows per seed | 5,000 |
| Number of random seeds | 10 |
| Samples per window | 20 |
| Target false-alarm rate | 1% |
| Mean-shift levels | 0, 0.02, 0.05, 0.10, 0.15 |
| Variance scale factors | 1.0, 1.1, 1.25, 1.5, 2.0 |

The standardized experiment uses random seeds 2026–2035. Each seed generates reference data, calibration windows, test windows, and independent validation windows using separate random-number generators.

## 3. Detection Methods

### 3.1 Euclidean detector

The detector calculates the Euclidean distance between the reference mean embedding and the mean of the current window. An alert is generated when the score exceeds the calibrated threshold.

### 3.2 Mahalanobis detector

This detector measures the difference between the reference and current-window means relative to the reference covariance matrix. A ridge term of \(10^{-6}I\) is added to the covariance matrix for numerical stability.

### 3.3 Variance-sensitive detector

This detector calculates the ratio of the current window's average squared spread around its own mean to the corresponding reference-data spread. A score greater than the calibrated threshold indicates a potential variance change.

## 4. Calibration and Validation

Each detector is calibrated independently using 2,000 no-drift windows, with the target threshold set at the 99th percentile of calibration scores. In the corrected procedure, calibration windows are centered on the estimated reference mean to match the no-drift test distribution.

Independent validation uses 5,000 additional windows per seed.

| Detector | Mean false-alarm rate |
|---|---:|
| Euclidean | 1.10% |
| Mahalanobis | 1.01% |
| Variance-sensitive | 1.10% |

The observed averages are close to the 1% target. Individual seed-level rates vary, so calibration should be described as approximately achieving the target rather than guaranteeing an exact rate.

## 5. Detection Results

### 5.1 Mean drift

| Mean shift per dimension | Euclidean | Mahalanobis | Variance-sensitive |
|---:|---:|---:|---:|
| 0.00 | 1.20% | 1.05% | 0.99% |
| 0.02 | 1.23% | 1.22% | 0.86% |
| 0.05 | 1.75% | 1.44% | 0.83% |
| 0.10 | 4.29% | 3.89% | 0.91% |
| 0.15 | 13.36% | 11.38% | 1.01% |

Euclidean and Mahalanobis detection rates generally increase with the magnitude of the injected mean shift. The variance-sensitive detector remains close to its no-drift baseline, as expected for a score focused on spread rather than location. Detection of smaller mean shifts remains limited under the current configuration.

### 5.2 Variance drift

| Variance scale factor | Euclidean | Mahalanobis | Variance-sensitive |
|---:|---:|---:|---:|
| 1.00 | 1.12% | 1.11% | 0.97% |
| 1.10 | 4.73% | 4.44% | 53.83% |
| 1.25 | 21.13% | 19.43% | 99.92% |
| 1.50 | 57.98% | 55.85% | 100.00% |
| 2.00 | 95.09% | 94.27% | 100.00% |

The variance-sensitive detector shows a strong response to increasing spread, achieving 99.92% detection at a scale factor of 1.25 and 100% at scale factors of 1.5 and 2.0 in the reported experiment.

Euclidean and Mahalanobis also respond to variance increases because the distribution of the current-window mean changes when the underlying variance increases.

## 6. Discussion

The results indicate that detector performance depends on the type and magnitude of the distribution change. Euclidean and Mahalanobis scores respond to changes in the mean and can also respond to changes in variance. The variance-sensitive score is more targeted toward detecting spread changes and shows limited response to the tested mean-only shifts.

These results do not establish that one detector is universally superior. They demonstrate different behaviours under the specified synthetic conditions.

## 7. Limitations

1. **Synthetic data:** The vectors are sampled from Gaussian distributions; they are not extracted from face, voice, or other biometric datasets.
2. **Independent windows:** The experiment evaluates independent windows rather than a continuous temporal stream. It therefore does not yet measure online detection delay or temporal change-point performance.
3. **Controlled drift:** Mean shifts are applied equally across all 16 dimensions. Real embedding changes may be sparse, correlated, nonlinear, or modality-dependent.
4. **Limited reference sample:** Each reference distribution is estimated from 500 samples, so the estimated mean and covariance introduce variability.
5. **Finite validation:** False-alarm estimates are empirical and subject to sampling variation.
6. **Limited metrics:** The current experiment reports detection and false-alarm rates but does not yet evaluate detection delay, computational overhead, or biometric identification accuracy.

## 8. Conclusion

Under the current synthetic Gaussian setup, the variance-sensitive detector responds most strongly to the tested increases in distribution spread, while Euclidean and Mahalanobis detectors respond to both mean and variance changes. Independent validation indicates mean false-alarm rates close to the specified 1% target.

The next research stage should evaluate these methods on appropriately sourced biometric embeddings and continuous or sequential drift scenarios. Until that work is completed, the findings should be presented as preliminary synthetic-data evidence rather than validated biometric-system performance.

## 9. Reproducibility Record

- Notebook: `04_Euclidean_vs_Mahalanobis_Comparison(2).ipynb`
- Corrected detection results: `corrected_multiseed_detection_results.csv`
- Corrected summary: `corrected_detection_summary.csv`
- Independent validation: `independent_false_alarm_validation.csv`
- Random seeds: 2026–2035
- Calibration target: 1%

## Pre-publication checklist

- [ ] Re-run the notebook from a fresh runtime and confirm the corrected CSV outputs are regenerated.
- [ ] Confirm the report's tables match the saved CSV files.
- [ ] Keep corrected results separate from exploratory results.
- [ ] Label this experiment as synthetic-data evaluation in the README and report.
- [ ] Do not claim validation on real biometric embeddings until that evaluation has been performed.
