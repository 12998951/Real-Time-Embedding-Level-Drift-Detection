# Experiment 13 — Phase 5B: Repeated-Stream CUSUM Evaluation

## Objective

Evaluate the frozen sequential CUSUM detector across 20 resampled versions
of the matched LFW embedding stream. Controlled brightness transformations
begin at window 6; windows 1–5 are pre-onset.

## Frozen detector

- Baseline median: 1.0419083
- Robust scale: 0.0992559
- Reference allowance, k: 0.5
- Alarm threshold, h: 5.0
- Shift onset: window 6
- Conditions: original control, brightness 0.70, brightness 1.30

No detector parameters were retuned after examining these repeated-stream results.

## Results

| Condition | Replicates | Pre-onset alarms | Pre-onset alarm rate | Post-onset detection rate | Mean delay (windows) | Mean maximum CUSUM |
|---|---:|---:|---:|---:|---:|---:|
| control | 20 | 2/20 | 0.100 | N/A | N/A | 3.657 |
| brightness_070 | 20 | 2/20 | 0.100 | 0.400 | 2.500 | 5.063 |
| brightness_130 | 20 | 2/20 | 0.100 | 0.550 | 2.364 | 7.344 |

## Interpretation

The detector raised pre-onset alarms in 2 of 20 replicates (10%) in each
condition. The same two resampling replicates alarmed in the control and
both transformed conditions, at windows 3 and 4. These results indicate that
some alarms were driven by stream variability before the imposed shift,
rather than uniquely by the brightness transformation.

Among transformed streams without a prior alarm, the reported post-onset
detection rates were 40% for brightness 0.70 and 55% for brightness 1.30.
Mean detection delays among detected streams were approximately 2.5 and
2.36 windows, respectively. These conditional figures must be interpreted
alongside the pre-onset alarms; they do not establish reliable deployment
performance.

## Limitations

- These are controlled transformations of static LFW images, not naturally
  occurring temporal biometric drift.
- The 20 streams are bootstrap/resampling replicates drawn from the same
  600-image pool. They overlap and are not independent datasets.
- Sampling with replacement can duplicate identities within a window, unlike
  the identity-balanced calibration windows.
- Only 20 replicates were evaluated; the estimated rates are uncertain.
- Detection means the first threshold crossing. A pre-onset alarm is counted
  as a false alarm for that stream; subsequent crossings are not treated as
  a successful first detection.
- These experiments do not establish biometric recognition accuracy,
  fairness, or deployment readiness.

## Conclusion

The frozen CUSUM configuration shows some sensitivity to the stronger
brightness transformation, but pre-onset alarms and modest detection rates
limit the evidence for dependable sequential detection. Further validation
should use a protocol that preserves identity constraints and tests false
alarms and detection delay on independently constructed streams.
