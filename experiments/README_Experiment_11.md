# Experiment 11 — Illumination Drift Detection

## 1. Overview

Experiment 11 investigates whether changes in controlled image brightness can be detected through shifts in image embeddings.

The experiment uses a subset of the Labeled Faces in the Wild (LFW) dataset and ImageNet-pretrained ResNet-18 embeddings. It evaluates Euclidean distance and ridge-regularized Mahalanobis distance using identity-disjoint reference, calibration, and evaluation partitions.

This is a controlled image-transformation experiment. It is not a validation of naturally occurring illumination drift or a complete biometric recognition system.

## 2. Dataset and Embedding Configuration

| Parameter | Configuration |
|---|---|
| Dataset | Labeled Faces in the Wild (LFW) subset |
| Images | 3,023 |
| Identities | 62 |
| Minimum faces per identity | 20 |
| Embedding model | ImageNet-pretrained ResNet-18 |
| Embedding dimension | 512 |
| Device | CPU |
| Image transformations | Dim, Original, Bright |
| Brightness factors | 0.60, 1.00, 1.40 |

The ResNet-18 classification head was removed to obtain 512-dimensional feature embeddings.

### Identity-disjoint partition

| Partition | Images | Identities |
|---|---:|---:|
| Reference | 2,101 | 37 |
| Calibration | 523 | 12 |
| Evaluation | 399 | 13 |

The three partitions contain no overlapping identities.

## 3. Detection Methodology

The experiment compares four detectors:

- Euclidean distance.
- Mahalanobis distance with ridge regularization 0.001.
- Mahalanobis distance with ridge regularization 0.01.
- Mahalanobis distance with ridge regularization 0.1.

Evaluation uses windows of 20 images. The imposed transition begins at evaluation window index 9.

### Corrected calibration protocol

The authoritative calibration procedure calculates scores using window-mean embeddings, matching the evaluation scoring procedure.

Thresholds were calibrated using 26 calibration windows at the 95th percentile, with the thresholds fixed before evaluation.

| Detector | Calibration windows | Quantile | Threshold |
|---|---:|---:|---:|
| Euclidean | 26 | 0.95 | 6.905388 |
| Mahalanobis ridge 0.001 | 26 | 0.95 | 7.592309 |
| Mahalanobis ridge 0.01 | 26 | 0.95 | 7.166080 |
| Mahalanobis ridge 0.1 | 26 | 0.95 | 5.593615 |

The earlier per-image calibration procedure was inconsistent with the window-mean evaluation scores. Its results are retained for transparency but are not treated as authoritative.

## 4. Corrected Controlled Evaluation

The corrected evaluation produced the following results:

| Stream | Detector | Pre-drift alarms | Post-drift alarms | Detection |
|---|---|---:|---:|---|
| Control | Euclidean | 2 | 0 | No drift detected |
| Control | Mahalanobis 0.001 | 2 | 0 | No drift detected |
| Control | Mahalanobis 0.01 | 3 | 0 | No drift detected |
| Control | Mahalanobis 0.1 | 3 | 0 | No drift detected |
| Dim | Euclidean | 2 | 10 | Detected |
| Dim | Mahalanobis 0.001 | 2 | 8 | Detected |
| Dim | Mahalanobis 0.01 | 3 | 9 | Detected |
| Dim | Mahalanobis 0.1 | 3 | 10 | Detected |
| Bright | Euclidean | 2 | 10 | Detected |
| Bright | Mahalanobis 0.001 | 2 | 10 | Detected |
| Bright | Mahalanobis 0.01 | 3 | 10 | Detected |
| Bright | Mahalanobis 0.1 | 3 | 10 | Detected |

All four detectors first alarmed at window 9 in both controlled brightness-drift streams. The recorded detection delay was zero windows.

These results show detection of the imposed brightness transformations in the tested evaluation streams. They do not establish generalisation to other illumination conditions or datasets.

## 5. Randomized-Order Evaluation

A randomized image-order evaluation was performed using seed 2026. The same permutation was applied across the Original, Dim, and Bright conditions.

| Stream | Detector | Pre-transition false alarms (%) | Post-transition alarms |
|---|---|---:|---:|
| Control | Euclidean | 11.11 | 1 |
| Control | Mahalanobis 0.001 | 44.44 | 2 |
| Control | Mahalanobis 0.01 | 44.44 | 2 |
| Control | Mahalanobis 0.1 | 55.56 | 1 |
| Dim | Euclidean | 11.11 | 10 |
| Dim | Mahalanobis 0.001 | 44.44 | 10 |
| Dim | Mahalanobis 0.01 | 44.44 | 10 |
| Dim | Mahalanobis 0.1 | 55.56 | 10 |
| Bright | Euclidean | 11.11 | 10 |
| Bright | Mahalanobis 0.001 | 44.44 | 10 |
| Bright | Mahalanobis 0.01 | 44.44 | 10 |
| Bright | Mahalanobis 0.1 | 55.56 | 10 |

The randomized control stream contains no imposed drift. Its post-transition alarms must therefore be interpreted as control false alarms, not successful drift detections. First post-transition alarm positions in this control stream represent control alarm events only. Post-drift detection percentage and detection delay are not applicable because no drift was imposed.

## 6. Multi-Permutation Control Analysis

A further control-order sensitivity analysis evaluated 20 permutations using seed 20261024.

The permutations reuse the same evaluation images. They are not 20 independent datasets.

| Detector | Mean control alarms | Standard deviation | Minimum | Maximum | Mean alarm rate |
|---|---:|---:|---:|---:|---:|
| Euclidean | 1.40 | 0.88 | 0 | 4 | 7.37% |
| Mahalanobis 0.001 | 1.95 | 1.05 | 0 | 4 | 10.26% |
| Mahalanobis 0.01 | 2.35 | 1.04 | 0 | 4 | 12.37% |
| Mahalanobis 0.1 | 3.40 | 0.99 | 1 | 5 | 17.89% |

Euclidean produced the lowest mean control alarm rate in this particular analysis. This does not establish universal detector superiority.

The results also show that control alarm counts varied with image ordering, especially for the Mahalanobis configurations.

## 7. Interpretation

The corrected controlled evaluation detected both dimming and brightening transitions at the first post-transition window.

However, the control and randomized-order analyses reveal false alarms and sensitivity to image ordering. The low calibration sample size also limits confidence in threshold stability.

Consequently, detection sensitivity must be interpreted jointly with false-alarm behaviour. A detector that responds to controlled brightness changes is not necessarily reliable under unchanged operating conditions.

## 8. Limitations

1. Brightness changes are synthetic pixel transformations, not naturally captured illumination variation.
2. The embedding model is ImageNet-pretrained ResNet-18, not a dedicated face-recognition encoder.
3. The evaluation uses a limited identity-disjoint LFW subset.
4. Calibration uses only 26 windows, making threshold stability uncertain.
5. The 20 randomized permutations reuse the same evaluation images and are not independent datasets.
6. The evaluation does not establish biometric recognition accuracy, fairness, bias mitigation, spoofing detection, or production readiness.
7. A detected embedding distribution shift does not necessarily indicate a recognition failure or security incident.
8. Full from-scratch numerical reproducibility remains incomplete because the original embedding archive was not recovered during the audit.

## 9. Reproducibility and Artifact Status

The recovered experiment artifacts include:

- Notebook: `notebooks/11_Illumination_Drift_Detection.ipynb`
- Corrected detection results: `results/experiment11_corrected_detection_results.csv`
- Randomized-order results: `results/experiment11_randomized_order_results.csv`
- Multi-permutation control results: `results/experiment11_multi_permutation_control.csv`
- Multi-permutation summary: `results/experiment11_multi_permutation_summary.csv`
- Corrected calibration thresholds: `results/experiment11_consistent_window_thresholds.csv`
- Combined evaluation: `results/experiment11_combined_evaluation_results.csv`
- Metadata: `results/experiment11_metadata.json`
- Archive manifest: `results/experiment11_archive_manifest.json`

Figures are stored in `figures/`.

**Missing artifact:** `data/experiment11_resnet18_embeddings.npz` was not recovered from the checked Colab runtime, mounted Drive locations, or fetched Git repository history. The original notebook's archive-generation step therefore cannot be assumed to be reproducible from the recovered files alone.

The original notebook also contains an archive-packaging step that expects this embedding file. That step must be adjusted to handle the missing file explicitly rather than failing or silently claiming a complete archive.

## 10. Overall Conclusion

Experiment 11 provides preliminary evidence that controlled brightness transformations can produce detectable changes in ResNet-18 embedding-window statistics.

The results also expose false-alarm behaviour, sensitivity to image ordering, and uncertainty associated with limited calibration data.

The findings justify further methodological investigation but do not establish dependable real-world biometric drift detection.

**Status:** Controlled experiment completed; selected numerical outputs recovered and checked; full from-scratch reproducibility remains incomplete.


### Archive audit update

The original archive manifest is preserved as a historical record. It listed `data/experiment11_resnet18_embeddings.npz`, but this embedding archive was not recovered during the November 2026 audit. The corrected manifest records this discrepancy. The original embedding archive remains unavailable, so full from-scratch numerical reproducibility has not been established.
