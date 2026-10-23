# LFW Independent Threshold Validation

## Objective
Evaluate Euclidean and regularized Mahalanobis drift scores using separate
reference, threshold-calibration, and no-drift validation image pools.

## Dataset and split
- Dataset: Labeled Faces in the Wild (LFW)
- Extractor: google/vit-base-patch16-224
- Features: 768-dimensional generic ViT image features
- Reference images: 1,497
- Calibration images: 737
- Validation images: 789
- Window size: 20 images
- Reference windows: 74
- Calibration windows: 36
- Validation windows: 39
- Image-index overlap between pools: 0

The extractor is a general-purpose vision model, not a validated face-recognition
model. The experiment therefore evaluates feature-distribution drift, not
validated biometric identity drift.

## Threshold calibration
The reference pool was used to estimate feature statistics. Detector thresholds
were selected using the separate calibration pool at the empirical 99th
percentile. The validation pool was not used to choose thresholds.

## No-drift validation results

| Detector | Threshold | Validation windows | False alarms | Observed false-alarm rate |
|---|---:|---:|---:|---:|
| Euclidean | 8.4724 | 39 | 0 | 0.0000 |
| Mahalanobis | 14.3602 | 39 | 0 | 0.0000 |

## Statistical uncertainty
Both detectors produced zero observed false alarms in 39 validation windows.
The approximate one-sided 95% binomial upper bound for the false-alarm
probability is 7.39% when zero alarms are observed in 39 independent trials.

This bound depends on an independence assumption. Windows formed from images
in the same dataset may not be fully independent, particularly because identity
composition and image acquisition conditions can be related.

The results do not establish a 1% false-alarm guarantee. Only 39 non-overlapping
validation windows were available.

## Limitations
1. LFW does not provide a natural chronological biometric stream.
2. The feature extractor is a generic ViT rather than a validated face-recognition model.
3. The validation sample is too small to certify a low false-alarm target.
4. Identity composition and acquisition conditions may affect feature distributions.
5. Zero observed false alarms does not imply a zero underlying false-alarm probability.

## Conclusion
This is a preliminary independent-threshold-calibration exercise on real-image
features. The observed false-alarm rate was zero for both detectors in the
available validation sample, but uncertainty remains substantial. A larger,
carefully designed follow-up study is required before making stronger claims.

## Reproducibility
Notebook: 07_LFW_Independent_Threshold_Validation.ipynb

Results file: lfw_independent_threshold_validation.csv
