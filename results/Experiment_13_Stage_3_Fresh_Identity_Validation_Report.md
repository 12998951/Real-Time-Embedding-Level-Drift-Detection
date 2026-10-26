# Experiment 13 — Stage 3: Fresh Identity-Stream Validation

## Objective

Evaluate the previously frozen one-sided CUSUM detector on a newly constructed
LFW stream containing identities not used in the earlier 600-image matched stream.

## Dataset and stream design

- Dataset: Labeled Faces in the Wild (LFW), loaded using TensorFlow Datasets.
- Embeddings: 512-dimensional FaceNet embeddings.
- Fresh stream: 1,000 images from 1,000 distinct identities.
- Window size: 50 images; 20 windows per stream.
- Conditions: unchanged control, brightness factor 0.70, and brightness factor 1.30.
- Change onset: window 10 (zero-based), the 11th window.
- Windows 0–9 use original embeddings in all conditions.
- Windows 10–19 use transformed-image embeddings in the two brightness conditions.
- The fresh identities do not overlap with identities in the earlier 600-image stream.
- Reference, calibration, and evaluation partitions remain identity-disjoint as
  defined in the existing experiment.

The fresh stream uses identities from the same LFW dataset and evaluation partition.
This is not validation on an independent external dataset.

## Frozen detector configuration

- Baseline median: 1.041908290
- Robust scale: 0.099255882
- CUSUM allowance, k: 0.5
- Alarm threshold, h: 5.0
- Window size: 50

No detector parameters were retuned on the fresh evaluation stream.

## Results

| Condition | First detection window | Pre-onset alarm | Detected after onset | Detection delay (windows) | Maximum CUSUM |
|---|---:|---:|---:|---:|---:|
| control | none | 0 | N/A | N/A | 2.842 |
| brightness_070 | none | 0 | 0.000 | N/A | 2.842 |
| brightness_130 | 14 | 0 | 1.000 | 4.000 | 9.762 |

## Interpretation

The detector did not alarm on the unchanged control stream or the brightness-0.70
stream. It detected the brightness-1.30 stream at window 14, four windows after
the imposed onset at window 10.

The control and brightness-0.70 streams have identical embeddings in their first
10 windows and therefore have the same maximum CUSUM in this evaluation.

This is a positive detection for one fresh stream under the stronger controlled
brightness transformation. It is not sufficient to estimate a reliable detection
rate, false-alarm rate, or generalisation performance.

## Limitations

- This is one stream construction per condition; the result does not establish
  repeatable performance across independent stream constructions.
- The evaluation uses static LFW images with controlled pixel-brightness changes,
  not naturally occurring temporal biometric drift.
- The fresh identities come from the same dataset, not an external dataset.
- Matching identity labels verifies the selected identity mapping but does not
  independently prove exact image-level alignment for identities with multiple images.
- No biometric recognition accuracy, demographic fairness, bias mitigation, or
  deployment-readiness claim is supported.
- The detector's threshold and CUSUM settings are provisional and do not guarantee
  a specified operational false-alarm rate.

## Saved outputs

- `lfw_stage3_fresh_stream_manifest.csv`
- `lfw_stage3_fresh_cusum_results.csv`
- `lfw_stage3_fresh_cusum_trace.csv`

The original and brightness-shifted embedding arrays are retained locally for
reproducibility and should not be uploaded publicly without a privacy, licensing,
and consent review.
