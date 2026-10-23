# Scientific Validation Report - Experiment 10

## 1. Objective

This report documents diagnostic checks performed on Experiment 10, which evaluates Euclidean and regularized Mahalanobis detectors under controlled synthetic embedding-space shifts.

## 2. Dataset and experimental limitations

The evaluation uses 3,023 LFW images represented by 768-dimensional embeddings generated using `google/vit-base-patch16-224`. This is a generic vision transformer, not a validated face-recognition embedding model.

The images are not chronological observations. The injected shifts are synthetic and should not be interpreted as realistic temporal biometric drift.

The evaluation uses 30 windows, including 15 designated pre-shift windows and 15 designated post-shift windows. Ten perturbation directions reuse the same evaluation windows and therefore do not represent ten independent datasets.

## 3. Threshold validation

The stored thresholds matched NumPy's 95th-percentile calculation using `method="higher"`. Each detector had one calibration score above its stored threshold.

With only 30 calibration windows, percentile estimates are uncertain. The observed calibration alarm count should not be interpreted as a precise estimate of the detector's long-run false-alarm probability.

## 4. Zero-shift control

In the unmodified evaluation windows, the Euclidean detector produced one alarm at window index 24. The three Mahalanobis configurations produced no alarms.

The Euclidean alarm also occurred when no synthetic shift was injected. Consequently, its alarm cannot be treated as evidence of successful drift detection in this experiment.

## 5. Sensitivity to synthetic shifts

The tested shift strengths were 0.00, 0.10, 0.20, 0.35, 0.50, 0.75, 1.00, 1.50, and 2.00.

The Mahalanobis detectors showed increasing alarm activity at larger perturbation strengths:

- Ridge 0.001: 9 of 10 directions had at least one post-shift alarm at strength 1.00; 10 of 10 did at strength 1.50.
- Ridge 0.010: 9 of 10 directions had at least one post-shift alarm at strength 1.00; 10 of 10 did at strength 1.50.
- Ridge 0.100: 10 of 10 directions had at least one post-shift alarm at strength 1.50.

At strength 2.00, the mean post-shift alarm rates were 0.9933, 0.9800, and 0.6600 for ridge parameters 0.001, 0.010, and 0.100, respectively.

The Euclidean detector's direction detection fraction was 1.00 at every strength, including zero. Because the same baseline alarm persists at every strength, this metric does not demonstrate sensitivity to the injected shift.

## 6. Detection delay

Detection delay uses a one-based definition: the first post-shift window is delay 1, and the second is delay 2.

For Mahalanobis ridge 0.001, the mean delay among detected directions was 2.0 windows at strengths 0.75, 1.00, and 1.50, and 1.1 windows at strength 2.00.

For ridge 0.010, the corresponding mean delays were 2.0 windows at strengths 0.75, 1.00, and 1.50, and 1.2 windows at strength 2.00.

For ridge 0.100, the mean delay was 2.0 windows at strengths 1.00, 1.50, and 2.00 among the directions that triggered alarms.

These values are conditional on detection. Undetected directions were left without a delay value.

## 7. Interpretation

The results provide preliminary evidence that the tested Mahalanobis detectors respond to sufficiently large synthetic embedding-space perturbations. The ridge parameter affects the observed sensitivity.

The experiment does not establish reliable real-world biometric drift detection, fairness improvement, or generalization to chronological biometric data.

## 8. Recommended next steps

1. Re-evaluate using a validated face-recognition embedding model.
2. Use temporally ordered or carefully designed shift scenarios.
3. Increase the number of calibration windows and independent evaluation trials.
4. Report control alarm rates and detection rates with uncertainty estimates.
5. Define detection delay consistently and evaluate it over independent trials.
6. Preserve the original Experiment 10 outputs and record future methodological revisions as a separate experiment.

## Conclusion

Experiment 10 is a preliminary synthetic sensitivity study. The Mahalanobis detectors show increasing alarm activity under larger injected shifts, but further independent and realistic validation is necessary before making claims about trustworthy real-world biometric drift detection.
