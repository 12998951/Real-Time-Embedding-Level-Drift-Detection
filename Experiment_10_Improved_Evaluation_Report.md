# Experiment 10: Improved Drift Evaluation Methodology

## 1. Objective

Evaluate whether Euclidean and Mahalanobis detectors distinguish
controlled embedding-space shifts from ordinary variation.

## 2. Dataset

- Dataset: Labeled Faces in the Wild (LFW)
- Images: 3023
- Embedding dimensions: 768
- Identity classes: 62
- Embedding extractor: Generic Vision Transformer (ViT)

## 3. Data partitioning

| Pool | Images |
|---|---:|
| Reference | 1209 |
| Calibration | 604 |
| Control | 605 |
| Evaluation | 605 |

The four pools have no overlapping image indices. All contain the same
identity classes, so this is not an identity-disjoint generalisation test.

Standardisation statistics were fitted using the reference pool only.
Calibration thresholds were fixed before control and shifted evaluation.

## 4. Experimental configuration

- Window size: 20
- Calibration quantile: 0.95
- Mahalanobis ridge values: [0.001, 0.01, 0.1]
- Drift strengths: [0.0, 0.05, 0.1, 0.2, 0.35]
- Drift directions: 10
- Evaluation windows: 30
- Shift begins at window: 15

## 5. Independent no-shift control

| Detector | Ridge | Threshold | Control alarms | Control rate |
|---|---:|---:|---:|---:|
| Euclidean | - | 7.5478 | 3/30 | 10.00% |\n| Mahalanobis | 0.001 | 14.0490 | 1/30 | 3.33% |\n| Mahalanobis | 0.010 | 11.1010 | 1/30 | 3.33% |\n| Mahalanobis | 0.100 | 6.9832 | 0/30 | 0.00% |\n
## 6. Results

The Euclidean detector reports alarms in the post-shift portion even
when the injected drift strength is zero. Its reported detection rate
therefore cannot be interpreted as successful drift detection.

The tested Mahalanobis configurations report no detections under the
current detection definition, including at the strongest tested shift.
This suggests that the present setup does not reliably distinguish
the injected perturbations from the detector's threshold behaviour.

The results are descriptive. The same evaluation windows are reused
across directions and strengths, so the trials are not independent
real-world observations.

## 7. Limitations

1. LFW images do not constitute a chronological biometric stream.
2. The feature extractor is a generic ViT, not a validated face-recognition model.
3. Shifts are synthetic perturbations of standardised embedding-space window means.
4. Only 30 windows are available per pool; estimates are uncertain.
5. All pools contain the same identities.
6. Ten drift directions reuse the same underlying evaluation windows.
7. A post-shift alarm is not proof that drift was detected.
8. No statistical-significance claim is made.

## 8. Conclusion

This experiment does not establish reliable drift detection. It identifies
a mismatch between the current injected shifts, score distributions,
thresholds, and detection metric. Further work should inspect raw scores,
validate the perturbation scale, use larger independent evaluation sets,
and evaluate identity-disjoint and realistic temporal/domain-shift scenarios.

## 9. Reproducibility

Trial-level results, summary tables, control results, and plots are saved
under `results/LFW_Real_Data_Results/`.

Report generated: 2026-10-23\n