# Experiment 13: False-Alarm-Controlled Embedding-Level Drift Detection

## 1. Objective

This experiment evaluates the trade-off between false-alarm frequency and detection sensitivity for Euclidean and regularised Mahalanobis embedding-level drift detectors under controlled synthetic distribution shifts.

The primary objectives are:

1. Evaluate whether empirical control false-alarm rates remain close to their nominal calibration targets.
2. Measure detection sensitivity under progressively stronger synthetic mean shifts.
3. Compare Euclidean and regularised Mahalanobis distance-based detectors under identical evaluation conditions.
4. Examine variability across multiple independent random seeds.
5. Establish a reproducible experimental baseline for subsequent embedding-level drift detection studies.

The experiment is designed as a controlled synthetic evaluation. It does not claim real-world biometric recognition accuracy, fairness, or deployment readiness.

## 2. Experimental Configuration

The experiment used the following configuration:

| Parameter | Configuration |
|---|---|
| Random seeds | 20 consecutive seeds beginning at 202613 |
| Embedding dimension | 64 |
| Window size | 20 observations |
| Reference windows | 50 per seed |
| Calibration windows | 200 per seed |
| Independent control windows | 500 per seed |
| Evaluation windows | 500 per seed and shift condition |
| Mean-shift strengths | 0.00, 0.25, 0.50, 0.75, 1.00 |
| Nominal calibration false-alarm targets | 1%, 5%, 10% |
| Detectors | Euclidean and regularised Mahalanobis |
| Covariance ridge parameter | 0.01 |
| Number of random seeds | 20 |

Each window contained 20 observations, with each observation represented by a 64-dimensional vector.

The evaluation considered five mean-shift conditions and three nominal false-alarm targets for both detectors.

## 3. Methodology

### 3.1 Synthetic Data Generation

Synthetic embedding observations were generated from a 64-dimensional standard multivariate normal distribution:

\[
X \sim \mathcal{N}(0,I_{64})
\]

where:

- \(X\) represents a synthetic embedding observation.
- \(I_{64}\) represents the 64-dimensional identity covariance matrix.
- Each observation contains 64 dimensions.

A window consisted of 20 observations.

The reference dataset contained 50 windows per seed, corresponding to 1,000 observations.

For positive drift conditions, an additive mean shift was applied to the first coordinate:

\[
\mu_{\delta}=(\delta,0,\ldots,0)
\]

where:

\[
\delta \in \{0.00,0.25,0.50,0.75,1.00\}
\]

The remaining coordinates retained zero mean and unit variance.

This design introduced controlled distributional changes while keeping the experimental dimensionality and window size constant.

### 3.2 Window-Level Representation

The evaluation unit was an embedding window rather than an individual observation.

Each window was represented by its mean embedding:

\[
\bar{x}_i=\frac{1}{W}\sum_{j=1}^{W}x_{ij}
\]

where:

- \(W=20\) is the number of observations in a window.
- \(x_{ij}\) is the \(j\)-th observation in window \(i\).
- \(\bar{x}_i\) is the resulting window-level embedding representation.

Detector scores were calculated using these window-level representations.

### 3.3 Reference Parameter Estimation

For each random seed, the reference-window means were used to estimate the reference centre:

\[
\hat{\mu}=\frac{1}{N}\sum_{i=1}^{N}\bar{x}_i
\]

where \(N=50\) represents the number of reference windows.

The reference centre was used by both detectors.

For the Mahalanobis detector, the covariance matrix was estimated from the reference-window representations.

The regularised covariance matrix was calculated as:

\[
\hat{\Sigma}_{reg}=\hat{\Sigma}+0.01I
\]

where:

- \(\hat{\Sigma}\) is the estimated reference covariance matrix.
- \(I\) is the 64-dimensional identity matrix.
- 0.01 is the covariance ridge regularisation parameter.

The inverse regularised covariance matrix was calculated using the Moore–Penrose pseudoinverse.

### 3.4 Euclidean Drift Detector

The Euclidean detector measured the L2 distance between a window mean and the reference centre:

\[
d_E(x)=\|\bar{x}-\hat{\mu}\|_2
\]

A larger distance indicated greater deviation from the reference distribution.

The Euclidean detector did not explicitly account for covariance structure between embedding dimensions.

### 3.5 Regularised Mahalanobis Drift Detector

The regularised Mahalanobis detector incorporated the estimated reference covariance structure:

\[
d_M(x)=
\sqrt{
(\bar{x}-\hat{\mu})^T
\hat{\Sigma}_{reg}^{-1}
(\bar{x}-\hat{\mu})
}
\]

This distance accounts for the scale and covariance relationships of the reference dimensions.

The ridge regularisation parameter was fixed at 0.01 throughout the experiment.

### 3.6 Threshold Calibration

Detector thresholds were estimated independently for each random seed, detector, and nominal false-alarm target.

Calibration data were used exclusively for threshold estimation.

The threshold was calculated using the empirical upper-tail calibration-score quantile:

\[
\tau_{\alpha}=Q_{1-\alpha}(S_{cal})
\]

where:

- \(\tau_{\alpha}\) is the calibrated threshold.
- \(S_{cal}\) represents the calibration scores.
- \(\alpha\) is the nominal false-alarm target.
- \(Q_{1-\alpha}\) represents the corresponding empirical quantile.

The three nominal false-alarm targets were:

\[
\alpha \in \{0.01,0.05,0.10\}
\]

The empirical quantile calculation used NumPy's default quantile method.

The calibration datasets were not used for independent control evaluation or detection sensitivity estimation.

### 3.7 Independent Control Evaluation

After threshold calibration, each detector was evaluated using an independent control dataset containing 500 windows per seed.

The control observations were not used to estimate detector parameters or calibration thresholds.

An alarm was recorded whenever:

\[
S(x)>\tau_{\alpha}
\]

The observed control false-alarm rate was calculated as:

\[
FPR=\frac{\text{Number of control windows exceeding the threshold}}
{\text{Total number of control windows}}
\]

This evaluation measured how closely the observed control false-alarm rates matched the nominal calibration targets.

### 3.8 Detection Sensitivity Evaluation

Detection sensitivity was evaluated using 500 evaluation windows per seed and mean-shift condition.

Both detectors used the same evaluation samples within each seed and shift condition.

The five evaluated mean-shift strengths were:

\[
\delta \in \{0.00,0.25,0.50,0.75,1.00\}
\]

For each detector and calibrated threshold, the detection rate was calculated as:

\[
DR=\frac{\text{Number of evaluation windows exceeding the threshold}}
{\text{Total number of evaluation windows}}
\]

The zero-shift condition provided a baseline evaluation under the absence of an injected mean shift.

The positive-shift conditions measured the sensitivity of each detector to progressively stronger synthetic distribution changes.

### 3.9 Multi-Seed Evaluation

The experiment used 20 consecutive random seeds beginning at 202613.

For each seed, separate reference, calibration, independent control, and evaluation datasets were generated.

Detector parameters and thresholds were estimated independently within each seed.

Control false-alarm rates and detection rates were calculated separately for each seed before aggregation.

This design allowed the experiment to examine variability across independent random-seed runs rather than relying on a single synthetic dataset.

### 3.10 Statistical Analysis

The reported means and standard deviations describe variation across the 20 seed-level rates.

The sample mean was calculated as:

\[
\bar{x}=\frac{1}{n}\sum_{i=1}^{n}x_i
\]

The sample standard deviation was calculated using the sample standard deviation estimator.

The standard error was calculated as:

\[
SE=\frac{s}{\sqrt{n}}
\]

where:

- \(s\) is the sample standard deviation.
- \(n=20\) is the number of random seeds.

Two-sided 95% confidence intervals were calculated using the Student's t-distribution:

\[
CI_{95\%}=
\bar{x}\pm t_{0.975,n-1}SE
\]

The confidence interval bounds were clipped to the valid rate range:

\[
[0,1]
\]

These confidence intervals describe variability across seed-level rates. They should not be interpreted as evidence of generalisation to real biometric identities or deployed systems.

## 4. Control False-Alarm Results

The following table presents the observed false-alarm rates across 20 random seeds.

| Detector | Target FPR | Mean | SD | n | SE | 95% CI Lower | 95% CI Upper |
|:---|---:|---:|---:|---:|---:|---:|---:|
| Euclidean | 0.0100 | 0.0136 | 0.0117 | 20 | 0.0026 | 0.0081 | 0.0191 |
| Euclidean | 0.0500 | 0.0565 | 0.0191 | 20 | 0.0043 | 0.0476 | 0.0654 |
| Euclidean | 0.1000 | 0.1077 | 0.0254 | 20 | 0.0057 | 0.0958 | 0.1196 |
| Mahalanobis | 0.0100 | 0.0152 | 0.0093 | 20 | 0.0021 | 0.0109 | 0.0195 |
| Mahalanobis | 0.0500 | 0.0513 | 0.0173 | 20 | 0.0039 | 0.0432 | 0.0594 |
| Mahalanobis | 0.1000 | 0.0997 | 0.0222 | 20 | 0.0050 | 0.0893 | 0.1101 |

### 4.1 Control Evaluation Findings

The observed mean control false-alarm rates were generally close to the nominal calibration targets.

At the 1% target:

- Euclidean: 1.36%.
- Mahalanobis: 1.52%.

At the 5% target:

- Euclidean: 5.65%.
- Mahalanobis: 5.13%.

At the 10% target:

- Euclidean: 10.77%.
- Mahalanobis: 9.97%.

The observed rates indicate that calibration produced control false-alarm frequencies reasonably close to the requested nominal targets in this synthetic evaluation.

However, variability across random seeds was present, particularly at the 1% target.

These results should not be interpreted as a guarantee that the detectors will maintain the same false-alarm behaviour under real-world data distributions.

## 5. Detection Sensitivity Results

The following table presents the detection sensitivity of both detectors across the three nominal false-alarm targets and five mean-shift strengths.

All results represent mean detection rates across 20 random seeds.

| Detector | Target FPR | Mean Shift | Mean | SD | n | SE | 95% CI Lower | 95% CI Upper |
|:---|---:|---:|---:|---:|---:|---:|---:|---:|
| Euclidean | 0.0100 | 0.0000 | 0.0127 | 0.0106 | 20 | 0.0024 | 0.0077 | 0.0177 |
| Euclidean | 0.0100 | 0.2500 | 0.0202 | 0.0147 | 20 | 0.0033 | 0.0133 | 0.0271 |
| Euclidean | 0.0100 | 0.5000 | 0.0409 | 0.0253 | 20 | 0.0057 | 0.0291 | 0.0527 |
| Euclidean | 0.0100 | 0.7500 | 0.0981 | 0.0517 | 20 | 0.0116 | 0.0739 | 0.1223 |
| Euclidean | 0.0100 | 1.0000 | 0.2573 | 0.0879 | 20 | 0.0197 | 0.2162 | 0.2984 |
| Euclidean | 0.0500 | 0.0000 | 0.0504 | 0.0209 | 20 | 0.0047 | 0.0406 | 0.0602 |
| Euclidean | 0.0500 | 0.2500 | 0.0694 | 0.0258 | 20 | 0.0058 | 0.0573 | 0.0815 |
| Euclidean | 0.0500 | 0.5000 | 0.1220 | 0.0289 | 20 | 0.0065 | 0.1085 | 0.1355 |
| Euclidean | 0.0500 | 0.7500 | 0.2411 | 0.0584 | 20 | 0.0131 | 0.2138 | 0.2684 |
| Euclidean | 0.0500 | 1.0000 | 0.4748 | 0.0678 | 20 | 0.0152 | 0.4431 | 0.5065 |
| Euclidean | 0.1000 | 0.0000 | 0.1015 | 0.0243 | 20 | 0.0054 | 0.0901 | 0.1129 |
| Euclidean | 0.1000 | 0.2500 | 0.1265 | 0.0293 | 20 | 0.0066 | 0.1128 | 0.1402 |
| Euclidean | 0.1000 | 0.5000 | 0.2088 | 0.0375 | 20 | 0.0084 | 0.1913 | 0.2263 |
| Euclidean | 0.1000 | 0.7500 | 0.3669 | 0.0480 | 20 | 0.0107 | 0.3444 | 0.3894 |
| Euclidean | 0.1000 | 1.0000 | 0.6136 | 0.0522 | 20 | 0.0117 | 0.5892 | 0.6380 |
| Mahalanobis | 0.0100 | 0.0000 | 0.0141 | 0.0099 | 20 | 0.0022 | 0.0095 | 0.0187 |
| Mahalanobis | 0.0100 | 0.2500 | 0.0190 | 0.0107 | 20 | 0.0024 | 0.0140 | 0.0240 |
| Mahalanobis | 0.0100 | 0.5000 | 0.0366 | 0.0144 | 20 | 0.0032 | 0.0299 | 0.0433 |
| Mahalanobis | 0.0100 | 0.7500 | 0.0756 | 0.0326 | 20 | 0.0073 | 0.0604 | 0.0908 |
| Mahalanobis | 0.0100 | 1.0000 | 0.1680 | 0.0583 | 20 | 0.0130 | 0.1407 | 0.1953 |
| Mahalanobis | 0.0500 | 0.0000 | 0.0504 | 0.0234 | 20 | 0.0052 | 0.0395 | 0.0613 |
| Mahalanobis | 0.0500 | 0.2500 | 0.0628 | 0.0204 | 20 | 0.0046 | 0.0532 | 0.0724 |
| Mahalanobis | 0.0500 | 0.5000 | 0.0997 | 0.0313 | 20 | 0.0070 | 0.0851 | 0.1143 |
| Mahalanobis | 0.0500 | 0.7500 | 0.1814 | 0.0552 | 20 | 0.0123 | 0.1556 | 0.2072 |
| Mahalanobis | 0.0500 | 1.0000 | 0.3236 | 0.0860 | 20 | 0.0192 | 0.2834 | 0.3638 |
| Mahalanobis | 0.1000 | 0.0000 | 0.0983 | 0.0291 | 20 | 0.0065 | 0.0847 | 0.1119 |
| Mahalanobis | 0.1000 | 0.2500 | 0.1191 | 0.0291 | 20 | 0.0065 | 0.1055 | 0.1327 |
| Mahalanobis | 0.1000 | 0.5000 | 0.1757 | 0.0392 | 20 | 0.0088 | 0.1574 | 0.1940 |
| Mahalanobis | 0.1000 | 0.7500 | 0.2799 | 0.0654 | 20 | 0.0146 | 0.2493 | 0.3105 |
| Mahalanobis | 0.1000 | 1.0000 | 0.4610 | 0.0947 | 20 | 0.0212 | 0.4167 | 0.5053 |

## 6. Interpretation and Comparative Analysis

### 6.1 Relationship Between Drift Strength and Detection Sensitivity

Detection rates increased as the injected mean-shift strength increased.

This trend was observed for both Euclidean and regularised Mahalanobis detectors across all three nominal false-alarm targets.

At stronger mean shifts, both detectors identified a larger proportion of evaluation windows as deviating from the reference distribution.

The results therefore demonstrate measurable sensitivity to the specific synthetic mean-shift mechanism used in this experiment.

### 6.2 Comparison Between Euclidean and Mahalanobis Detectors

Across the tested conditions, Euclidean generally achieved higher detection rates than Mahalanobis, particularly at stronger shifts.

At the 5% nominal false-alarm target and mean-shift strength 1.00:

- Euclidean detection rate: 47.48%.
- Mahalanobis detection rate: 32.36%.

At the 10% nominal false-alarm target and mean-shift strength 1.00:

- Euclidean detection rate: 61.36%.
- Mahalanobis detection rate: 46.10%.

At the 1% target and mean-shift strength 1.00:

- Euclidean detection rate: 25.73%.
- Mahalanobis detection rate: 16.80%.

These results indicate that Euclidean produced higher observed sensitivity under the selected synthetic conditions.

This does not establish universal superiority of Euclidean distance. The relative performance may change under different covariance structures, drift directions, dimensionalities, or reference distributions.

### 6.3 False-Alarm and Sensitivity Trade-Off

Increasing the nominal false-alarm target generally increased detection sensitivity.

A higher target false-alarm rate permits lower detection thresholds, making the detectors more responsive to distributional changes but also increasing the frequency of alarms under control conditions.

The results illustrate the practical trade-off between conservative detection and sensitivity.

The appropriate operating threshold would depend on the intended application and the relative cost of false alarms and missed drift events.

### 6.4 Interpretation of Confidence Intervals

The confidence intervals represent uncertainty in the mean seed-level false-alarm and detection rates across the 20 random-seed runs.

They do not represent uncertainty across real biometric identities, demographic groups, operational environments, or deployed systems.

No claim of real-world generalisation should be derived from these intervals.

## 7. Limitations

The following limitations define the scope of the experiment.

1. **Synthetic observations:** All observations were generated from synthetic Gaussian distributions rather than real biometric embeddings.

2. **Restricted drift mechanism:** The injected drift consisted of an additive mean shift in a single embedding coordinate.

3. **Independent windows:** The experiment used independently generated windows rather than a temporally dependent data stream.

4. **No sequential detection evaluation:** Detection delay, sustained alarms, sequential alarm behaviour, and recovery after drift were not evaluated.

5. **No biometric recognition evaluation:** The experiment did not measure face-recognition accuracy, identity verification performance, or biometric matching errors.

6. **No fairness evaluation:** Demographic fairness, group-specific error rates, and bias mitigation were not evaluated.

7. **Parameter dependence:** Results depend on the selected embedding dimension, covariance regularisation, reference sample size, calibration sample size, and threshold estimation procedure.

8. **Limited generalisation:** The evaluation does not establish identity-disjoint, cross-dataset, or real-world generalisation.

9. **Synthetic dimensional structure:** The selected Gaussian distribution and single-coordinate shift do not represent the full complexity of real embedding distributions.

10. **Operational readiness:** The results do not establish suitability for production deployment or real-time biometric monitoring.

These limitations should be considered when interpreting the experimental findings and planning subsequent validation studies.

## 8. Reproducibility and Experimental Artifacts

### 8.1 Main Experimental Artifacts

The following files are associated with the experiment:

| Artifact | Description |
|---|---|
| `Experiment_13_Final.ipynb` | Main experimental notebook |
| `Experiment_13_Final_Reproducibility_Audit.ipynb` | Reproducibility and audit notebook |
| `Experiment_13_Reproducible_Report.md` | Consolidated experimental report |
| `README_Experiment_13.md` | Experiment-specific documentation |
| `experiment13_trial_results.csv` | Main trial-level results |
| `experiment13_control_fpr_summary.csv` | Control false-alarm summary and confidence intervals |
| `experiment13_detection_summary.csv` | Detection sensitivity summary and confidence intervals |
| `experiment13_single_seed_calibration.csv` | Initial single-seed calibration results |
| `experiment13_single_seed_control.csv` | Initial single-seed control results |
| `experiment13_detection_ci.png` | Detection sensitivity confidence-interval plot |

### 8.2 Random-Seed Reproducibility

The experiment used NumPy random generators with documented seeds beginning at 202613.

Separate reference, calibration, control, and evaluation datasets were generated for each seed.

The reported multi-seed results should be regenerated from the saved notebook to verify that the saved result files correspond to the documented configuration and implementation.

### 8.3 Reproducibility Scope

The reproducibility claim is limited to the documented synthetic experiment and its saved implementation.

Reproduction of the experiment does not establish that the synthetic data represent real biometric distributions.

The reported results should be interpreted within the scope of the specified random seeds, distributional assumptions, detector definitions, and calibration procedure.

## 9. Conclusion

This experiment evaluated false-alarm-controlled embedding-level drift detection using Euclidean and regularised Mahalanobis distance-based detectors under controlled synthetic distribution shifts.

Across 20 random seeds, the observed control false-alarm rates were generally close to the nominal calibration targets of 1%, 5%, and 10%.

Detection rates increased as the injected mean-shift strength increased. Euclidean generally achieved higher detection sensitivity than regularised Mahalanobis under the evaluated synthetic conditions, particularly at stronger mean shifts.

The findings demonstrate a measurable trade-off between false-alarm frequency and detection sensitivity in the selected experimental setting.

However, the results are preliminary methodological evidence. They do not establish real-world biometric recognition accuracy, demographic fairness, identity-disjoint generalisation, sequential detection performance, or deployment readiness.

Further evaluation should investigate real biometric embeddings, multiple drift mechanisms, temporally structured streams, cross-condition generalisation, and operational false-alarm behaviour.

Experiment 13 therefore provides a controlled synthetic baseline for subsequent research into embedding-level drift detection rather than a validated real-world biometric drift-detection solution.

---

**Experiment status:** Synthetic multi-seed evaluation completed; real-world biometric validation remains outside the scope of this experiment.
