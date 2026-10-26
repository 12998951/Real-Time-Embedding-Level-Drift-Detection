# Experiment 13 — Scientific Robustness Evaluation

## Purpose
Test sensitivity to embedding dimension and covariance regularisation
using synthetic embeddings and independent calibration/control samples.

## Design
- Random seeds: 20
- Embedding dimensions: 16, 32, 64
- Mahalanobis covariance ridge: 0.001, 0.01, 0.1
- Detectors: Euclidean and regularised Mahalanobis
- Calibration target false-alarm rate: 5%
- Evaluation shift: additive mean shift of 1.0 in the first coordinate
- Reference windows: 50; window size: 20
- Calibration windows: 200; control windows: 500; evaluation windows: 500

## Interpretation
The mean control false-alarm rates were broadly near the nominal 5% target.
Detection rates decreased with increasing embedding dimension in this setup.
Mahalanobis performance varied with ridge regularisation, while Euclidean
results were unchanged by ridge because that detector does not use covariance.

## Limitations
This is a synthetic diagnostic using independent windows and a one-coordinate
mean shift. The confidence intervals describe variation across 20 seeds under
this simulation design; they do not establish real-world biometric performance.
No temporal detection delay, identity-disjoint generalisation, fairness, or
deployment readiness is established by this experiment.
