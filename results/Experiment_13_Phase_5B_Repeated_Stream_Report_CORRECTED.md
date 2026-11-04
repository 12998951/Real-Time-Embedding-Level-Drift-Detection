# Experiment 13 — Phase 5B: Repeated-Stream CUSUM Evaluation
## Corrected Report

### 1. Objective

Evaluate the frozen sequential CUSUM detector across 20 bootstrap-resampled
versions of the matched LFW embedding stream.

The three conditions are control, brightness 0.70, and brightness 1.30.
The imposed brightness transformation begins at window 6 (one-based
window numbering); windows 1–5 are pre-onset.

### 2. Frozen detector configuration

- Baseline median: 1.0419083
- Robust scale: 0.0992559
- Reference allowance, k: 0.5
- Alarm threshold, h: 5.0
- Shift onset: window 6
- Replicates: 20
- Window size: 50 images
- Total windows per stream: 12
- Bootstrap random seed: 202621

No detector parameters were retuned after examining these results.

### 3. Results

| Condition | Replicates | Pre-onset alarms | Pre-onset alarm rate | Detection rate / all replicates | Detection rate / eligible replicates | Mean delay (windows) | Mean maximum CUSUM |
|---|---:|---:|---:|---:|---:|---:|---:|
| control | 20 | 2/20 | 0.100 | N/A | N/A | N/A | 3.657 |
| brightness_070 | 20 | 2/20 | 0.100 | 0.400 | 0.444 | 2.500 | 5.063 |
| brightness_130 | 20 | 2/20 | 0.100 | 0.550 | 0.611 | 2.364 | 7.344 |

**Denominator definitions:**

- Detection rate / all replicates = post-onset detections divided by 20.
- Detection rate / eligible replicates = post-onset detections divided by
  the replicates without a pre-onset alarm.
- Mean detection delay is calculated among replicates whose first alarm
  occurred after the imposed shift.

For brightness 0.70, 8 of 20 replicates detected the shift.
The overall rate is 40.00%; the eligible-stream rate is 44.44%
(8 of 18).

For brightness 1.30, 11 of 20 replicates detected the shift.
The overall rate is 55.00%; the eligible-stream rate is 61.11%
(11 of 18).

### 4. Pre-onset alarm analysis

The same two replicate IDs, 2 and 16, produced pre-onset alarms in
all three conditions.

This is consistent with shared resampling variability affecting the
matched streams. It does not independently establish the cause of
those alarms.

A pre-onset first alarm is treated as a false alarm for that stream.
A subsequent threshold crossing is not counted as a successful
first detection.

### 5. Trace availability and evidence scope

The original repeated-stream implementation records trial-level outcomes
and maximum CUSUM values. It does not save individual window-level
scores or CUSUM trajectories.

Therefore, the 60-row repeated-stream result file supports the reported
trial-level outcomes, but does not provide a 720-row window-level trace.

The separate 36-row sequential detector trace represents three pilot
streams, not all 20 repeated-stream replicates.

No claim of complete repeated-stream window-level trace availability
is made in this report.

### 6. Limitations

- These are controlled transformations of static LFW images, not
  naturally occurring temporal biometric drift.
- The 20 streams are bootstrap/resampling replicates drawn from the
  same 600-image pool. They overlap and are not independent datasets.
- Sampling with replacement can duplicate identities within a window.
- Only 20 replicates were evaluated; estimated rates are uncertain.
- A first threshold crossing is used as the detection event.
- These experiments do not establish biometric recognition accuracy,
  fairness, or deployment readiness.
- Exact reproduction of the original Mahalanobis scores remains
  dependent on the availability of the original reference centre
  and precision matrix.

### 7. Conclusion

The frozen CUSUM configuration showed greater post-onset sensitivity
under the stronger brightness transformation in these 20 resampling
replicates.

However, 10% of replicates alarmed before onset, and overall
post-onset detection rates were 40% and 55% for brightness 0.70
and 1.30, respectively.

The results provide preliminary evidence of sensitivity to controlled
brightness changes, not evidence of dependable operational detection.

Further validation should use independently constructed streams,
identity-preserving evaluation protocols, and explicit assessment
of false alarms and detection delay.

### 8. Provenance

This is a corrected interpretation of the existing Phase 5B results.
The original raw result and summary CSV files were not modified.

The original Phase 5B report is preserved separately.
