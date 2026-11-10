# Experiment 19: Disentangling Identity-Composition Drift from Within-Identity Embedding Drift in Biometric Representation Systems

## Overview

Experiment 19 investigates the distinction between identity-composition drift and within-identity embedding drift in biometric representation systems.

The experiment examines how changes in the proportions of represented identities and changes in embedding distributions within individual identities affect global embedding distributions and drift-monitoring methods.

The study uses controlled scenarios constructed from an LFW-derived FaceNet embedding subset.

## Research Questions

1. How do changes in identity composition affect global embedding distributions?
2. Can within-identity embedding distribution changes be detected independently of identity-composition changes?
3. How effectively can global and identity-conditioned monitoring distinguish these mechanisms?
4. How do detection performance and false alarms change when both mechanisms occur simultaneously?

## Experimental Design

Four controlled scenarios were constructed.

| Scenario | Description |
|---|---|
| C0 — No drift | Identity composition and conditional distributions remain stable. |
| C1 — Composition drift | Identity proportions change while original embeddings are used. |
| C2 — Within-identity drift | Identity proportions remain stable while low-illumination embedding variants are introduced. |
| C3 — Combined drift | Identity proportions change and low-illumination embedding variants are introduced. |

Each scenario contains 440 observations:

- 220 baseline observations.
- 220 post-change observations.
- 22 windows per scenario.
- Window size of 20 observations.

The scenarios share the same baseline observations.

Bootstrap sampling with replacement was used to construct the controlled scenarios because of substantial identity imbalance in the source cohort.

## Dataset and Embeddings

The experiment uses a filtered LFW-derived biometric embedding subset.

- Unique identities: 13
- Embedding dimension: 512
- Original paired observations: 889
- Embedding model: InceptionResnetV1
- Pretrained weights: VGGFace2
- Within-identity perturbation: Low-illumination embedding variants

Identity 14 accounts for approximately 56.69% of the original paired cohort.

The bootstrap-generated observations are not independent images. This limitation is considered when interpreting the results.

## Drift Monitoring Methods

Three monitoring approaches were evaluated:

1. **Global Euclidean detector:** Measures normalized Euclidean distance from the global reference centroid.
2. **Global Mahalanobis detector:** Measures covariance-adjusted distance from the global reference distribution, using covariance shrinkage of 0.10.
3. **Identity-conditioned detector:** Measures standardized Euclidean displacement relative to identity-specific reference centroids and dispersion.

The identity-conditioned approach uses identity labels. It should not be interpreted as a label-free operational monitoring method.

## Calibration and Evaluation

The reference observations were divided into model-fitting and calibration partitions.

- Model-fitting observations: 261
- Calibration observations: 181
- Evaluation observations: 447

The model-fitting and calibration partitions were disjoint.

Calibration thresholds were estimated using bootstrap windows of calibration scores. The thresholds are sample-dependent and should not be interpreted as operationally validated false-positive rates.

## Main Results

### 1. Composition drift

At the 5% calibration threshold:

- Global Euclidean detected 11/11 post-change windows in C1.
- Global Mahalanobis detected 11/11 post-change windows in C1.
- Both global detectors detected 11/11 post-change windows in C3.

This indicates strong sensitivity to the constructed identity-composition changes.

### 2. Within-identity drift

For C2:

- Global Euclidean detected 1/11 post-change windows.
- Global Mahalanobis detected 0/11.
- Identity-conditioned monitoring detected 1/11.

The low-illumination condition produced measurable identity-level embedding displacement, although window-level detection was weak.

### 3. Combined drift

For C3 at the 5% threshold:

- Global Euclidean detected 11/11 post-change windows.
- Global Mahalanobis detected 11/11.
- Identity-conditioned monitoring detected 7/11.

The combined scenario was strongly detected by global monitoring, primarily in the presence of the constructed composition change.

### 4. Identity-level statistical analysis

Paired Wilcoxon signed-rank tests were performed across 13 identities, with Holm correction across four comparisons.

| Comparison | Mean paired difference | Holm-adjusted p-value | Significant at 0.05 |
|---|---:|---:|---|
| C2 vs C0 | 0.091008 | 0.001953 | Yes |
| C3 vs C1 | 0.036232 | 0.064453 | No |
| C1 vs C0 | 0.011150 | 0.735352 | No |
| C3 vs C2 | -0.043626 | 0.487793 | No |

The C2 versus C0 comparison provides evidence of increased identity-level displacement under the within-identity drift condition.

The C3 versus C1 comparison did not remain statistically significant after Holm correction.

### 5. Drift attribution

The global mean-shift decomposition was:

\[
\Delta\mu =
\sum_i(p_{1,i}-p_{0,i})\mu_{0,i}
+
\sum_i p_{1,i}(\mu_{1,i}-\mu_{0,i})
\]

The two terms represent the composition component and within-identity component, respectively.

The maximum numerical reconstruction error was approximately \(8.85\times10^{-17}\).

This confirms numerical consistency of the decomposition, not causal isolation of the underlying mechanisms.

## Visual Results

The following figures are included in this experiment:

- `plots/experiment19_figure1_drift_attribution.png`
- `plots/experiment19_figure2_identity_displacement.png`
- `plots/experiment19_figure3_statistical_significance.png`

## Reproducibility

The experiment includes:

- Environment and configuration records.
- Sampling and scenario-generation protocols.
- Detector configuration and calibration thresholds.
- Scenario validation outputs.
- Detection and attribution results.
- Identity-level statistical results.
- SHA-256 reproducibility manifest.
- Consolidated research evidence.

The independent artifact audit passed all 55 checks.

## Limitations

1. The experiment uses only 13 identities from a filtered LFW-derived subset.
2. The identity distribution is substantially imbalanced.
3. Bootstrap sampling with replacement creates repeated observations.
4. Each scenario contains only 22 evaluation windows.
5. Calibration is based on 181 observations.
6. Identity-conditioned monitoring requires identity labels.
7. Low illumination is the specific within-identity perturbation studied.
8. Post-change samples are independently resampled rather than strictly paired image observations.
9. Statistical inference is based on only 13 identity-level paired observations.
10. The experiment does not establish real-world biometric recognition accuracy, operational fairness, or deployment readiness.

## Interpretation Boundary

The findings apply to the controlled LFW-derived embedding experiment.

They do not establish:

- Label-free operational monitoring.
- Generalisation across biometric datasets or recognition models.
- Causal isolation of drift mechanisms.
- Real-world biometric recognition performance.
- Deployment readiness.

## Data Availability

The experiment uses biometric-derived embeddings and identity labels. Raw embeddings and identity-level data should not be publicly redistributed without verifying the source dataset's licensing, applicable ethical conditions, and data-sharing permissions.

Aggregated results, figures, methodological descriptions, and reproducibility documentation can be provided separately.

## Research Status

**Experiment 19: Completed.**

The experiment provides controlled evidence that identity-composition changes and within-identity embedding displacement can affect global embedding distributions differently, while also demonstrating limitations in distinguishing these mechanisms using aggregate monitoring alone.
