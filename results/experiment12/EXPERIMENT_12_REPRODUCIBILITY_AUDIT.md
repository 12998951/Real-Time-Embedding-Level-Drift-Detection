# Experiment 12 — Reproducibility and Provenance Audit

**Audit date:** 2026-11-04
**Experiment:** Cross-Condition Generalisation of Embedding-Level Drift Detectors
**Historical result version:** V2
**Audit outcome:** Historical V2 results preserved; independent reproduction not established.

## 1. Purpose

This audit records the provenance and reproducibility limitations identified
during a review of Experiment 12 V2. It does not replace, modify, or regenerate
the historical V2 results.

## 2. Repository and artifact status

At the time of this audit:

- The Git working tree was clean.
- The checked-out commit was `ed6c4199bdd9f4e0835a8daa8a42d979ddfbd1ef`.
- All seven original V2 artifacts listed below were present.
- SHA-256 hashes were calculated for all seven files.

## 3. Source-code investigation

The following investigation was performed:

1. Inspected the Experiment 12 notebook, which contains 33 cells.
2. Searched the notebook for V2-specific terms and implementation markers.
3. Inspected the Git history and available branches.
4. Inspected commit `0b0b32a`, titled
   `Add Experiment 12 v2 synthetic embedding shift analysis`.
5. Searched reachable Git history for distinctive V2 configuration values,
   output filenames, and implementation terms.
6. Inspected saved notebook outputs and metadata.

The V2 commit added a README, four CSV files, and two plot images.
It did not add the code used to generate those results.

The inspected notebook contains earlier Experiment 12 work, but the
historical V2 Mahalanobis and 20-direction implementation was not recovered.

No claim is made that every possible external copy or uncommitted execution
environment has been searched.

## 4. Historical V2 configuration documented in its README

The historical V2 README records:

- Dataset: LFW-derived embeddings.
- Embedding rows: 3,023.
- Identities: 62.
- Embedding dimensions: 768.
- Reference partition: 1,492 samples, 37 identities.
- Calibration partition: 1,071 samples, 12 identities.
- Evaluation partition: 460 samples, 13 identities.
- Window size: 20.
- Synthetic shift directions: 20.
- Shift magnitudes: 0.00, 0.25, 0.50, 0.75, 1.00.
- Detectors: Euclidean and regularised Mahalanobis.
- Threshold percentiles: 90, 95, and 97.5.
- Documented synthetic shift seed: 20261203.

These values are recorded as historical documentation, not as an
independently reproduced implementation.

## 5. Calibration discrepancy

The historical V2 README reports the following Mahalanobis thresholds:

| Percentile | Historical V2 threshold |
|---|---:|
| 90th | 9.696375 |
| 95th | 10.251740 |
| 97.5th | 10.846457 |

A later restored diagnostic calculation produced:

| Percentile | Restored diagnostic threshold |
|---|---:|
| 90th | 9.712607 |
| 95th | 10.411112 |
| 97.5th | 11.033033 |

The restored diagnostic used the identity-disjoint partition and a
Ledoit-Wolf covariance model. Its shrinkage coefficient was approximately
0.0221563, matching the value documented in the historical README.

However, the thresholds differ. The original V2 95th-percentile baseline
reports 11 alarms out of 23 evaluation windows, whereas the restored
diagnostic reports 10 out of 23.

Changing the comparison operator did not resolve the discrepancy.

The exact cause has not been established. Possible differences in the
historical preprocessing, scoring, calibration, or evaluation implementation
remain unverified.

The restored diagnostic must not be presented as an exact reproduction
of historical V2.

## 6. Original V2 artifact integrity

The following SHA-256 hashes were recorded during the audit:

| Artifact | SHA-256 |
|---|---|
| README_experiment12_v2.md | `9a062180b0667cd81d4e3dc42c00a33ed4ded49ef2db179edb9e7e583a2f46fa` |
| exp12_v2_alarm_transitions.csv | `1cb095ba30a97fb1fd918ddd4aef3c1bf6ecf33f68d4053c587b54f5f6fe500b` |
| exp12_v2_baseline_adjusted_sensitivity.csv | `bc1ab6d434c1778d05cac590adb2b36afcd7e743be9801e95e60588bec9ca37e` |
| exp12_v2_baseline_adjusted_sensitivity.png | `9cb390f8b914fccb8ce925315f2709ab97f835cb49cee8702bc5c895d8a9a330` |
| exp12_v2_direction_robustness.csv | `5e57cbac3c5a9542056ff4cdf99dbbe797dfe81da9aa4d38210419b0e2d6eb40` |
| exp12_v2_raw_sensitivity.csv | `b0d400ba25c31a22e4f746bba666ddf18f7e1fce52c58312ad0c59977736fc32` |
| exp12_v2_raw_sensitivity.png | `e75b5726245cfe68a274833a152102ef893cdb98d705415b4b2bcfd630b7399a` |

These hashes provide a baseline for detecting subsequent changes.
They do not prove that the historical calculations are reproducible.

## 7. Scientific interpretation and limitations

The historical V2 outputs should be treated as preserved preliminary
results of a documented synthetic embedding-shift analysis.

They do not independently establish:

- Real-world temporal biometric drift detection.
- Detection accuracy against ground-truth drift events.
- Real-world biometric recognition accuracy.
- Fairness or bias mitigation.
- Deployment readiness.

The evaluation contains 23 windows and reuses them across synthetic
directions. The synthetic direction experiments are not independent
real-world observations.

The calibration-to-evaluation transfer problem remains unresolved.
Baseline alarms cannot automatically be classified as false positives
because the evaluation partition is not independently established
as drift-free.

## 8. Final audit determination

| Audit item | Determination |
|---|---|
| Historical V2 artifacts present | Verified |
| Artifact SHA-256 hashes recorded | Verified |
| Historical V2 README preserved | Verified |
| Original V2 generation code recovered | No |
| Exact historical thresholds reproduced | No |
| 95th-percentile discrepancy resolved | No |
| Historical V2 independently reproducible | Not established |
| Original V2 results modified by this audit | No |

**Final determination:** The historical V2 results are preserved and their
provenance limitations are documented. Their exact generating implementation
was not recovered, so independent reproducibility cannot be claimed.

A future independently implemented evaluation must be versioned separately.
It must not silently replace or retroactively relabel the historical V2
results.

---

**Audit classification:** Documentation and provenance audit.
**This document does not certify numerical reproduction of historical V2.**
