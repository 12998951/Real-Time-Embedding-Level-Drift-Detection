# Experiment 13 — Stage 1 Methodology Audit

**Audit date:** 4 November 2026  
**Scope:** Phase 5B repeated-stream analysis and Stage 3 bootstrap CUSUM evaluation  
**Status:** Stage 1 audit artifacts created; further Experiment 13 stages remain to be audited.

## 1. Purpose

This audit checks the interpretation of the existing CUSUM results. It does not rerun the experiment, retune detector parameters, or modify original experimental artifacts.

## 2. Stage 3 raw-trace verification

The saved first-alarm windows were reconstructed from the raw CUSUM trace.

- Saved first-alarm records checked: 60
- Records matching reconstructed first alarms: 60
- All saved first alarms match the raw traces: **Yes**
- Imposed change onset: window 10 (zero-based)
- Windows before onset: 0 to 9
- Valid first alarms at or after onset: window 10 onward

## 3. Corrected first-alarm classification

A stream counts as having a valid first post-onset alarm only when its first alarm occurs at or after the imposed change onset. A stream whose first alarm occurs before onset is classified as a pre-onset alarm, even if later trace rows also contain alarms.

| condition      |   replicates |   no_alarm |   first_alarm_before_onset |   valid_first_alarm_at_or_after_onset |   valid_post_onset_rate |   mean_valid_detection_delay |   median_valid_detection_delay |
|:---------------|-------------:|-----------:|---------------------------:|--------------------------------------:|------------------------:|-----------------------------:|-------------------------------:|
| brightness_070 |           20 |          2 |                          8 |                                    10 |                     0.5 |                      3       |                            2.5 |
| brightness_130 |           20 |          0 |                          8 |                                    12 |                     0.6 |                      2.08333 |                            2   |
| control        |           20 |          4 |                          8 |                                     8 |                     0.4 |                      4.375   |                            4   |

The control stream has no imposed brightness change. Its post-onset alarms are therefore control alarms, not successful detections of brightness drift.

## 4. Corrected paired comparison

The paired comparison below considers only streams whose first alarm occurs at or after onset in both the control and transformed conditions.

| condition      |   both_valid_alarm_count |   transformed_earlier_count |   same_window_count |   transformed_later_count |   median_difference_windows |
|:---------------|-------------------------:|----------------------------:|--------------------:|--------------------------:|----------------------------:|
| brightness_070 |                        8 |                           6 |                   2 |                         0 |                        -1   |
| brightness_130 |                        8 |                           8 |                   0 |                         0 |                        -2.5 |

A negative difference means the transformed stream alarmed earlier than control. A zero difference means both alarmed in the same window. A positive difference means the transformed stream alarmed later.

These comparisons are conditional on both streams having valid first post-onset alarms. They do not represent overall detection performance.

## 5. Interpretation

The results show a preliminary association between brightness perturbation and earlier CUSUM alarms in paired replicates where both control and transformed streams alarmed after onset.

However:

- 8 of 20 streams in each condition had their first alarm before onset.
- 8 of 20 control streams had their first alarm at or after onset, despite no imposed brightness change.
- The valid first post-onset alarm rates are not sufficiently separated from the control rate to establish reliable drift detection.
- Earlier alarms in paired cases are exploratory evidence, not proof of robust detection.

## 6. Phase 5B repeated-stream audit findings

The Phase 5B repeated-stream results were previously checked separately:

- 20 resamples were evaluated per condition.
- Pre-onset alarms occurred in 2/20 streams per condition.
- Post-onset alarms were recorded in 8/20 brightness_070 streams and 11/20 brightness_130 streams.
- The corresponding unconditional rates are 40% and 55%.
- Among the 18 streams per condition without a pre-onset alarm, the conditional rates are 8/18 = 44.4% and 11/18 = 61.1%.
- The original wording describing 40% and 55% as rates among streams without prior alarms used the wrong denominator.
- The resamples were drawn with replacement from the same 600-image pool and are not 20 independent datasets.
- No Phase 5B confidence intervals were reported. The separate calibration-transfer confidence-interval file does not provide confidence intervals for these Phase 5B outcomes.

## 7. Methodological limitations

1. The repeated bootstrap resamples overlap and must not be treated as independent datasets.
2. Row-level resampling with replacement can duplicate identities and omit others within a resampled stream.
3. The control condition has a substantial alarm rate, including alarms before and after the nominal onset.
4. The Stage 3 paired timing analysis is conditional on both streams having valid first post-onset alarms.
5. No confidence intervals are calculated here because the resamples are not independent experimental datasets.
6. This audit does not establish real-world biometric drift detection accuracy, demographic fairness, recognition accuracy, or deployment readiness.

## 8. Files created

- `stage1_corrected_first_alarm_classification.csv`
- `stage1_corrected_paired_comparison.csv`
- `STAGE_1_METHODOLOGY_AUDIT.md`

These are new derived audit artifacts. Original experiment CSVs, traces, reports, and notebooks were not overwritten.

## 9. Audit conclusion

The Stage 3 saved first-alarm results are consistent with the raw trace. The corrected interpretation separates pre-onset first alarms from valid first post-onset alarms and treats control alarms as false alarms for the imposed-change evaluation.

The results support only a preliminary, exploratory brightness-sensitivity observation. They do not yet establish reliable drift detection.

**Stage 1 audit status: completed for the checks documented above. Other Experiment 13 stages still require review.**
