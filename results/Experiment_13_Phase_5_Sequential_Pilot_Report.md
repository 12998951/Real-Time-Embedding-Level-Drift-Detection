# Experiment 13 — Phase 5 Sequential Detection Pilot

## Objective

Evaluate whether a fixed sequential detector detects controlled brightness-induced
changes in FaceNet embedding-window scores while avoiding alarms on an unchanged
control stream.

## Stream Design

- Dataset: Labeled Faces in the Wild (LFW).
- Embeddings: 512-dimensional FaceNet embeddings.
- Window size: 50 images.
- Evaluation streams: one control stream, one brightness-0.7 stream, and one
  brightness-1.3 stream.
- Change onset: window 6 (zero-indexed).
- Windows 0–5: original embeddings in all streams.
- Windows 6–11: original embeddings for control; transformed embeddings for
  the two perturbation streams.
- Evaluation: 12 windows per stream.
- Identity separation: reference and evaluation identities were disjoint.

## Calibration

An identity-balanced calibration set was created from 2,012 calibration identities.
One image per selected identity was used in each 50-image calibration window.
The saved calibration set contains 100 windows.

The sequential detector used the calibration median and robust scale of Mahalanobis
window scores, with allowance k = 0.5 and decision threshold h = 5.0.
These settings are provisional and do not guarantee a specified false-alarm rate.

## Results

The fixed one-sided CUSUM did not alarm on the control stream or either perturbed
stream. The two-sided CUSUM diagnostic also produced no alarms.

The maximum positive CUSUM observed after onset was:

- Control: 0.425
- Brightness 0.7: 1.018
- Brightness 1.3: 2.498

The brightness-1.3 stream showed a larger cumulative score increase, but its
maximum remained below the fixed threshold of 5.0. No detection delay can be
reported for either perturbed stream because neither was detected.

## Interpretation

The results demonstrate a score shift under the tested controlled brightness
perturbations, particularly for brightness factor 1.3, but the current sequential
detector did not trigger an alarm.

This is a negative detection result for the tested configuration. It does not
establish that sequential embedding-level drift detection is generally ineffective.

## Limitations

- Only one stream construction was evaluated per condition.
- Twelve windows per stream are insufficient for robust false-alarm-rate or
  detection-rate estimates.
- Calibration windows are assembled from static LFW images, not independent
  temporal recordings.
- Brightness transformations are controlled perturbations, not natural temporal
  biometric drift.
- No claim is supported about recognition accuracy, fairness, or deployment
  readiness.

## Recommended Next Steps

1. Construct repeated independent control and perturbed stream instances.
2. Keep fitting, calibration, and evaluation identities separated.
3. Prespecify any alternative sequential statistic and its calibration procedure.
4. Estimate control false-alarm rate, detection probability, and detection delay
   over repeated streams.
5. Report non-detections rather than tuning the threshold against evaluation
   outcomes.
