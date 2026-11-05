# Experiment 13 — Stage 3 Bootstrap Comparison (Corrected)

## Objective

Compare first CUSUM alarm timing between unchanged control streams and
paired brightness-transformed streams across 20 resampling replicates.

## Method

- Dataset: Labeled Faces in the Wild (LFW).
- Embeddings: 512-dimensional FaceNet embeddings.
- Conditions: unchanged control, brightness factor 0.70, brightness factor 1.30.
- Replicates: 20 paired resampling replicates.
- Windows per stream: 20.
- Window size: 50 images.
- Imposed change onset: window 10 (zero-based).
- Detector parameters remained frozen.

The same underlying resampling cohorts were used across conditions.
These replicates are paired resampling evaluations, not 20 independent datasets.

## Mutually exclusive first-alarm classifications

| Condition | First alarm after onset | No alarm | Pre-onset alarm only | Pre-onset followed by post-onset alarm | Total |
|---|---:|---:|---:|---:|---:|
| control | 8 | 4 | 1 | 7 | 20 |\n| brightness_070 | 10 | 2 | 0 | 8 | 20 |\n| brightness_130 | 12 | 0 | 0 | 8 | 20 |\n
## Paired first-alarm timing comparison

The following comparison includes only replicates in which both the
unchanged control and transformed stream alarmed.

| Condition | Both alarmed | Transformed earlier | Same first-alarm window | Transformed later |
|---|---:|---:|---:|---:|
| brightness_070 | 16 | 6 | 10 | 0 |\n| brightness_130 | 16 | 8 | 8 | 0 |\n
## Interpretation

The first alarm occurred at or after the imposed change onset in 8/20
control streams, 10/20 brightness-0.70 streams, and 12/20
brightness-1.30 streams.

### Detection-delay interpretation

Detection delay is calculated as the first detection window minus the
imposed change-onset window (10). The reported mean delays are conditional
on the first alarm occurring at or after onset. Therefore, the delay
summaries include 8/20 control streams, 10/20 brightness-0.70 streams,
and 12/20 brightness-1.30 streams. Streams whose first alarm occurred
before onset are excluded from this calculation, even when a subsequent
post-onset alarm occurred. The conditional mean delays were 4.375,
3.000, and 2.083 windows, respectively. These values should not be
interpreted independently of the pre-onset alarm rates.


All three conditions also had 8/20 streams with pre-onset alarms.
Among transformed streams, eight replicates in each condition had a
pre-onset alarm followed by at least one later alarm.

In paired replicates where both streams alarmed, the transformed stream
had an earlier first alarm in 6/16 brightness-0.70 comparisons and
8/16 brightness-1.30 comparisons. No transformed stream had a later
first alarm in these paired comparisons.

These timing differences are exploratory. The unchanged control also
produced first alarms after the nominal change point. Therefore, an
alarm occurring after onset cannot automatically be attributed to the
brightness transformation.

The results suggest a stronger response under brightness factor 1.30
than under brightness factor 0.70, but they do not establish reliable
drift detection or a controlled operational false-alarm rate.

## Limitations

- The 20 replicates are paired resampling evaluations, not independent
  external datasets.
- Eight streams in every condition had pre-onset alarms.
- The unchanged control also produced substantial post-onset alarm activity.
- First-alarm timing alone does not establish that an alarm was caused
  by the imposed brightness change.
- The experiment uses static LFW images and synthetic pixel-brightness
  transformations.
- No claims are supported about biometric recognition accuracy,
  demographic fairness, bias mitigation, or deployment readiness.
- External-dataset generalisation has not been established.

## Provenance and outputs

This corrected report is based on the existing bootstrap result and trace
CSV files. It does not regenerate embeddings, rerun the experiment, or
modify the original reports or numerical results.

The original report and all raw results remain unchanged.
