# Experiment 13 — Stage 3 Bootstrap Paired Comparison

## Objective

Compare the first CUSUM alarm window in each brightness-transformed stream
with its paired unchanged control stream across 20 resampling replicates.

## Method

The comparison uses the existing 1,000-image Stage 3 embedding arrays and
frozen CUSUM parameters. Twenty paired resamples were evaluated for control,
brightness factor 0.70, and brightness factor 1.30. Resampling uses the same
underlying cohorts, so the replicates are not independent datasets.

The imposed change onset is window 10 (zero-based). An alarm before window 10
is a pre-onset alarm and must not be counted as a successful post-change
detection. A control alarm at or after window 10 is a late alarm, not evidence
of a true change, because no change was imposed on the control.

## Paired first-alarm comparison

| Condition | Both streams alarm | Transformed earlier | Same first-alarm window | Transformed later | Transformed no alarm | Control no alarm | Transformed pre-onset alarms |
|---|---:|---:|---:|---:|---:|---:|---:|
| brightness_070 | 16 | 6 | 10 | 0 | 2 | 4 | 8 |
| brightness_130 | 16 | 8 | 8 | 0 | 0 | 4 | 8 |

## Interpretation

In paired replicates where both streams alarmed, the transformed streams
never had a later first alarm than their paired controls in this resampling
run. Brightness factor 0.70 alarmed earlier in 6 paired replicates and at
the same window in 10; brightness factor 1.30 alarmed earlier in 8 and at
the same window in 8.

However, the detector also produced pre-onset alarms, and the unchanged
control produced late alarms. These findings indicate that the detector can
respond to stream variability as well as to the imposed brightness change.
This comparison is exploratory and does not establish reliable drift
detection or independent generalisation.

The detector parameters were not retuned. No claims are made about biometric
recognition accuracy, demographic fairness, bias mitigation, or deployment
readiness.

## Output

- `lfw_stage3_bootstrap_paired_comparison.csv`
