# Experiment 19 — Consolidated Research Evidence

## Research Title

Disentangling Identity-Composition Drift from Within-Identity Embedding Drift in Biometric Representation Systems

## Research Questions

1. How do changes in identity composition affect global embedding distributions?
2. Can within-identity embedding distribution changes be detected independently of identity-composition changes?
3. How effectively can global and identity-conditioned monitoring distinguish mechanisms?
4. How do detection performance and false alarms change when both mechanisms occur simultaneously?

## Experimental Scenarios

### C0_no_drift
Stable identity composition and conditional distributions.

### C1_composition_drift
Changed identity proportions with original embeddings.

### C2_within_identity_drift
Stable identity proportions with low-illumination embedding variants.

### C3_combined_drift
Changed identity proportions and low-illumination embedding variants.

## Main Research Findings

### Global composition drift

**Evidence:** Global Euclidean and Mahalanobis detectors detected all 11 post-change windows in C1 and C3 at the 5% threshold.

**Interpretation:** The global embedding distribution was sensitive to the constructed identity-composition changes.

### Within-identity drift

**Evidence:** C2 produced greater identity-level displacement than C0, with a mean paired difference of 0.091008 and Holm-adjusted p-value of 0.001953.

**Interpretation:** The low-illumination condition produced measurable within-identity embedding displacement.

### Combined drift

**Evidence:** C3 versus C1 had a positive mean paired difference of 0.036232, but its Holm-adjusted p-value was 0.064453.

**Interpretation:** The additional within-identity effect under the combined condition was not statistically significant after Holm correction.

### Mathematical attribution

**Evidence:** The vector decomposition reconstructed the total mean shift with maximum numerical error 8.85e-17.

**Interpretation:** The decomposition is numerically consistent, but its estimated components include sampling noise.

## Identity-Level Statistical Results

| comparison   |   mean_paired_difference |   p_value_raw |   p_value_holm |   cohens_dz | significant_holm_0_05   |
|:-------------|-------------------------:|--------------:|---------------:|------------:|:------------------------|
| C2_vs_C0     |                0.0910079 |   0.000488281 |     0.00195312 |   1.32064   | True                    |
| C3_vs_C1     |                0.0362315 |   0.0214844   |     0.0644531  |   0.713132  | False                   |
| C1_vs_C0     |                0.0111501 |   0.735352    |     0.735352   |   0.0886288 | False                   |
| C3_vs_C2     |               -0.0436263 |   0.243896    |     0.487793   |  -0.475693  | False                   |

## Drift Attribution Results

| scenario                 |   baseline_observations |   post_change_observations |   composition_TV_distance |   composition_vector_norm |   within_identity_vector_norm |   total_observed_mean_shift_norm |   weighted_within_identity_displacement |   composition_within_vector_cosine |   vector_reconstruction_error |
|:-------------------------|------------------------:|---------------------------:|--------------------------:|--------------------------:|------------------------------:|---------------------------------:|----------------------------------------:|-----------------------------------:|------------------------------:|
| C0_no_drift              |                     220 |                        220 |                  0        |                  0        |                     0.0477609 |                        0.0477609 |                                0.114926 |                         nan        |                   3.95388e-17 |
| C1_composition_drift     |                     220 |                        220 |                  0.586364 |                  0.481641 |                     0.0855986 |                        0.443126  |                                0.250974 |                          -0.520822 |                   8.84779e-17 |
| C2_within_identity_drift |                     220 |                        220 |                  0        |                  0        |                     0.117821  |                        0.117821  |                                0.195178 |                         nan        |                   4.10229e-17 |
| C3_combined_drift        |                     220 |                        220 |                  0.586364 |                  0.481641 |                     0.118808  |                        0.471671  |                                0.287245 |                          -0.206389 |                   8.29262e-17 |

## Detector Evaluation

| scenario                 | detector             |   target_false_alarm_rate |   threshold |   baseline_alarms |   post_change_detections |   post_change_detection_rate |   overall_alarm_rate |   first_detection_post_window |   detection_latency_windows |   baseline_mean_score |   post_change_mean_score |
|:-------------------------|:---------------------|--------------------------:|------------:|------------------:|-------------------------:|-----------------------------:|---------------------:|------------------------------:|----------------------------:|----------------------:|-------------------------:|
| C1_composition_drift     | global_euclidean     |                      0.1  |     1.05784 |                 1 |                       11 |                    1         |            0.545455  |                             1 |                           0 |              0.962123 |                 1.26569  |
| C1_composition_drift     | global_euclidean     |                      0.05 |     1.08149 |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              0.962123 |                 1.26569  |
| C1_composition_drift     | global_euclidean     |                      0.01 |     1.12589 |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              0.962123 |                 1.26569  |
| C1_composition_drift     | global_mahalanobis   |                      0.1  |     8.23952 |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              7.56567  |                 9.44827  |
| C1_composition_drift     | global_mahalanobis   |                      0.05 |     8.3763  |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              7.56567  |                 9.44827  |
| C1_composition_drift     | global_mahalanobis   |                      0.01 |     8.65062 |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              7.56567  |                 9.44827  |
| C1_composition_drift     | identity_conditioned |                      0.1  |     1.07071 |                 0 |                       10 |                    0.909091  |            0.454545  |                             1 |                           0 |              0.998882 |                 1.10403  |
| C1_composition_drift     | identity_conditioned |                      0.05 |     1.08541 |                 0 |                        7 |                    0.636364  |            0.318182  |                             1 |                           0 |              0.998882 |                 1.10403  |
| C1_composition_drift     | identity_conditioned |                      0.01 |     1.11655 |                 0 |                        5 |                    0.454545  |            0.227273  |                             2 |                           1 |              0.998882 |                 1.10403  |
| C2_within_identity_drift | global_euclidean     |                      0.1  |     1.05784 |                 1 |                        1 |                    0.0909091 |            0.0909091 |                            10 |                           9 |              0.962123 |                 0.969657 |
| C2_within_identity_drift | global_euclidean     |                      0.05 |     1.08149 |                 0 |                        1 |                    0.0909091 |            0.0454545 |                            10 |                           9 |              0.962123 |                 0.969657 |
| C2_within_identity_drift | global_euclidean     |                      0.01 |     1.12589 |                 0 |                        0 |                    0         |            0         |                           nan |                         nan |              0.962123 |                 0.969657 |
| C2_within_identity_drift | global_mahalanobis   |                      0.1  |     8.23952 |                 0 |                        0 |                    0         |            0         |                           nan |                         nan |              7.56567  |                 7.80312  |
| C2_within_identity_drift | global_mahalanobis   |                      0.05 |     8.3763  |                 0 |                        0 |                    0         |            0         |                           nan |                         nan |              7.56567  |                 7.80312  |
| C2_within_identity_drift | global_mahalanobis   |                      0.01 |     8.65062 |                 0 |                        0 |                    0         |            0         |                           nan |                         nan |              7.56567  |                 7.80312  |
| C2_within_identity_drift | identity_conditioned |                      0.1  |     1.07071 |                 0 |                        2 |                    0.181818  |            0.0909091 |                             6 |                           5 |              0.998882 |                 1.03602  |
| C2_within_identity_drift | identity_conditioned |                      0.05 |     1.08541 |                 0 |                        1 |                    0.0909091 |            0.0454545 |                             6 |                           5 |              0.998882 |                 1.03602  |
| C2_within_identity_drift | identity_conditioned |                      0.01 |     1.11655 |                 0 |                        0 |                    0         |            0         |                           nan |                         nan |              0.998882 |                 1.03602  |
| C3_combined_drift        | global_euclidean     |                      0.1  |     1.05784 |                 1 |                       11 |                    1         |            0.545455  |                             1 |                           0 |              0.962123 |                 1.28118  |
| C3_combined_drift        | global_euclidean     |                      0.05 |     1.08149 |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              0.962123 |                 1.28118  |
| C3_combined_drift        | global_euclidean     |                      0.01 |     1.12589 |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              0.962123 |                 1.28118  |
| C3_combined_drift        | global_mahalanobis   |                      0.1  |     8.23952 |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              7.56567  |                 9.38811  |
| C3_combined_drift        | global_mahalanobis   |                      0.05 |     8.3763  |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              7.56567  |                 9.38811  |
| C3_combined_drift        | global_mahalanobis   |                      0.01 |     8.65062 |                 0 |                       11 |                    1         |            0.5       |                             1 |                           0 |              7.56567  |                 9.38811  |
| C3_combined_drift        | identity_conditioned |                      0.1  |     1.07071 |                 0 |                        9 |                    0.818182  |            0.409091  |                             1 |                           0 |              0.998882 |                 1.10065  |
| C3_combined_drift        | identity_conditioned |                      0.05 |     1.08541 |                 0 |                        7 |                    0.636364  |            0.318182  |                             1 |                           0 |              0.998882 |                 1.10065  |
| C3_combined_drift        | identity_conditioned |                      0.01 |     1.11655 |                 0 |                        4 |                    0.363636  |            0.181818  |                             1 |                           0 |              0.998882 |                 1.10065  |

## No-Drift Control

| detector             |   target_false_alarm_rate |   threshold |   baseline_alarms |   post_change_alarms |   total_alarms |   baseline_empirical_alarm_rate |   post_empirical_alarm_rate |   overall_empirical_alarm_rate |
|:---------------------|--------------------------:|------------:|------------------:|---------------------:|---------------:|--------------------------------:|----------------------------:|-------------------------------:|
| global_euclidean     |                      0.1  |     1.05784 |                 1 |                    1 |              2 |                       0.0909091 |                   0.0909091 |                      0.0909091 |
| global_euclidean     |                      0.05 |     1.08149 |                 0 |                    0 |              0 |                       0         |                   0         |                      0         |
| global_euclidean     |                      0.01 |     1.12589 |                 0 |                    0 |              0 |                       0         |                   0         |                      0         |
| global_mahalanobis   |                      0.1  |     8.23952 |                 0 |                    0 |              0 |                       0         |                   0         |                      0         |
| global_mahalanobis   |                      0.05 |     8.3763  |                 0 |                    0 |              0 |                       0         |                   0         |                      0         |
| global_mahalanobis   |                      0.01 |     8.65062 |                 0 |                    0 |              0 |                       0         |                   0         |                      0         |
| identity_conditioned |                      0.1  |     1.07071 |                 0 |                    0 |              0 |                       0         |                   0         |                      0         |
| identity_conditioned |                      0.05 |     1.08541 |                 0 |                    0 |              0 |                       0         |                   0         |                      0         |
| identity_conditioned |                      0.01 |     1.11655 |                 0 |                    0 |              0 |                       0         |                   0         |                      0         |

## Methodological Limitations

- The experiment uses a filtered LFW-derived subset containing 13 identities.
- Identity 14 accounts for approximately 56.69% of the original paired cohort.
- Bootstrap sampling with replacement means repeated draws are not independent images.
- The evaluation contains only 22 windows per scenario.
- Calibration thresholds are based on 181 calibration observations and bootstrap windows.
- Identity-conditioned monitoring requires identity labels.
- The within-identity drift scenario uses low-illumination embedding variants.
- The post-change samples are independently resampled rather than strictly paired image observations.
- The statistical analysis uses only 13 identity-level paired observations.
- The results do not establish real-world biometric recognition accuracy.
- The results do not establish operational fairness or deployment readiness.
- The experiment does not establish causal isolation of drift mechanisms.

## Reproducibility Audit

- Status: PASS
- Checks passed: 55/55
- Failed checks: 0

## Generated Figures

- `plots/experiment19_figure1_drift_attribution.png`
- `plots/experiment19_figure2_identity_displacement.png`
- `plots/experiment19_figure3_statistical_significance.png`

## Interpretation Boundary

These findings are limited to the controlled LFW-derived embedding experiment. Identity labels are used for scenario construction and identity-conditioned analysis. The results do not demonstrate label-free operational monitoring, real-world biometric recognition accuracy, fairness, causal mechanism isolation, or deployment readiness.

This report consolidates previously saved results. No experiment or statistical test was rerun.