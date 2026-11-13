# Real-Time Embedding-Level Drift Detection for Trustworthy Biometric Systems

A research project investigating label-free distributional drift detection in embedding spaces, with the long-term goal of improving the reliability and trustworthiness of biometric identification systems under changing operating conditions.

**Current status:** Research prototype with completed experimental evaluations across Experiments 11–17, covering synthetic embedding distributions and exploratory controlled LFW/FaceNet embedding experiments. Experiments 14–17 investigate temporal drift patterns, covariance-sensitive detector behaviour, cross-condition calibration transfer, and target-condition threshold recalibration. Experiment 15 includes paired statistical comparisons with Holm correction and Bonferroni-adjusted confidence intervals. Experiment 13 includes exploratory sequential CUSUM evaluations and recovered bootstrap artifacts. The real-image experiments do not establish biometric recognition performance, dependable operational false-alarm control, fairness, or deployment readiness. Exact reproduction of the original Experiment 13 Mahalanobis scores remains unverified because the original reference-model artifacts have not been recovered.

## Overview

Machine learning systems can encounter changes in their input data after deployment. In biometric applications, these changes may arise from variations in illumination, ageing, camera or microphone characteristics, environmental noise, and other operating conditions.

Such changes can alter the feature representations produced by face or voice encoders. Monitoring these representations may provide an early indication that the data distribution has changed, even when ground-truth labels or recognition accuracy measurements are unavailable.

This repository contains an original Euclidean-distance synthetic prototype and subsequent experiments comparing multiple statistical drift detectors. Later experiments include regularized Mahalanobis and variance-sensitive scores, plus an exploratory sequential CUSUM pilot using pretrained FaceNet embeddings extracted from controlled LFW image streams.

The repository therefore contains experiments at different maturity levels: synthetic evaluations, controlled transformations of static face images, and a preliminary sequential detector study. These experiments do not constitute a complete face or voice recognition pipeline and do not establish dependable real-world biometric drift detection.

## Research Objectives

- Investigate whether embedding-level statistics can provide a practical signal for detecting distributional changes without requiring labels.
- Evaluate the relationship between drift magnitude, detection sensitivity, and false-alarm behaviour.
- Examine how detector performance varies across independently generated reference samples.
- Establish a reproducible experimental pipeline that can later be extended to real face and voice embeddings.
- Investigate the limitations of simple statistical drift signals under noise and changing operating conditions.

## Current Implementation

The initial Euclidean-only prototype and the later multi-detector/FaceNet experiments are separate stages of the project; results from one stage should not be treated as validation of another.

The repository includes:

- A reference window containing synthetic embedding vectors.
- Current windows of newly generated synthetic vectors.
- A Euclidean distance score between reference and current window means.
- Threshold-based drift decisions.
- Synthetic experiments involving stable data and controlled mean shifts.
- Threshold calibration using simulated stable windows.
- Repeated evaluations across independently generated reference windows.
- CSV export of experimental results and generation of a detection-rate plot.

The implementation is a window-based statistical prototype. It does not yet establish production-grade streaming performance or validate a complete biometric recognition system.

## Methodology

### Reference representation

A reference window is generated from a baseline synthetic distribution. Its mean vector represents the initial reference state.

Let \(R\) denote the reference window and \(C\) the current window.

### Drift scoring

The prototype calculates the Euclidean distance between the two window means:

$$
s_t = \|\mu_R - \mu_C\|_2
$$

where:

- \(\mu_R\) is the mean vector of the reference window.
- \(\mu_C\) is the mean vector of the current window.
- \(s_t\) is the resulting drift score.

### Threshold-based detection

The detector raises an alarm when the drift score exceeds a calibrated threshold:

$$
A_t =
\begin{cases}
1, & s_t > \tau \\
0, & s_t \leq \tau
\end{cases}
$$

Here, \(\tau\) is estimated from simulated stable calibration windows. It is not a universal constant and may need recalibration for different embedding models, modalities, reference distributions, and operating conditions.

### Synthetic evaluation

Controlled experiments vary the mean of the generated embedding distribution to investigate how detection sensitivity changes with drift magnitude. Repeated reference samples are used to examine variability in the observed results.

## Experimental Configuration

The consolidated synthetic evaluation uses the following configuration:

| Parameter | Value |
|---|---:|
| Independent reference windows | 30 |
| Embedding dimensions | 16 |
| Vectors per window | 20 |
| Synthetic noise standard deviation | 0.10 |
| Calibration windows per reference | 500 |
| Test windows per reference and shift | 300 |
| Calibration percentile | 99th percentile |
| Shift values | 0.00, 0.02, 0.04, 0.05, 0.07, 0.10, 0.15 |

The script uses fixed random seeds for the reference trials to support repeatable synthetic experiments.

## Preliminary Synthetic Results

In the recorded consolidated evaluation, the mean detection rates across 30 independent reference trials were:

| Synthetic mean shift per dimension | Mean detection rate |
|---:|---:|
| 0.00 | 1.11% |
| 0.02 | 17.32% |
| 0.04 | 86.64% |
| 0.05 | 97.98% |
| 0.07 | 100.00% |
| 0.10 | 100.00% |
| 0.15 | 100.00% |

These results were reproduced in Google Colab using the committed evaluation script. The generated summary and per-reference CSV files matched the committed versions exactly in that run. The results remain preliminary and apply only to the synthetic data and experimental configuration described here.

## Interpretation

Under the tested synthetic configuration, the mean detection rate generally increased as the imposed distribution shift became larger. Small shifts were more difficult to detect, while larger shifts were detected consistently in the recorded trials.

## Limitations of These Results

- The experiments use synthetic Gaussian vectors, not real face or voice embeddings.
- The zero-shift detection rate is an empirical result, not a guaranteed false-positive rate.
- The 100% detection rates apply only to the tested synthetic windows and configuration.
- Results depend on the reference samples, noise assumptions, embedding dimensionality, window size, and threshold calibration procedure.
- The current experiment evaluates individual windows; it does not demonstrate production-grade streaming behaviour or detection delay in a live system.
- A detected distribution shift does not necessarily indicate a recognition failure, bias, spoofing attempt, or cyberattack.

## Evaluation Outputs

The synthetic evaluation script generates the following outputs:

| File | Purpose |
|---|---|
| [drift_results_summary.csv](results/drift_results_summary.csv) | Aggregated detection results by synthetic shift magnitude |
| [drift_results_by_reference.csv](results/drift_results_by_reference.csv) | Detection results for each independent reference window |
| [drift_detection_curve.png](assets/drift_detection_curve.png) | Plot of mean detection rate against synthetic shift magnitude |

The outputs should be interpreted alongside the experimental configuration and limitations described in this README.

## Repository Structure

```text
Real-Time-Embedding-Level-Drift-Detection/
├── README.md
├── drift_detector/
│   └── drift_signal.py
├── experiments/
│   ├── synthetic_drift_evaluation.py
│   ├── 04_euclidean_mahalanobis_variance_comparison.ipynb
│   └── Experiment_Report.md
├── embeddings/
├── utils/
│   └── visualization/
├── results/
│   ├── drift_results_summary.csv
│   ├── drift_results_by_reference.csv
│   └── corrected/
│       ├── README.md
│       ├── corrected_detection_summary.csv
│       ├── corrected_multiseed_detection_results.csv
│       └── independent_false_alarm_validation.csv
├── assets/
│   └── drift_detection_curve.png
├── requirements.txt
├── RUN_EXPERIMENTS.txt
├── CITATION.cff
├── LICENSE
└── .gitignore
```

This tree provides a simplified overview of the repository; some subfolder contents are omitted for readability.

## Reproducing the Synthetic Evaluation

### Requirements

- Python 3
- NumPy
- pandas
- Matplotlib

The dependencies are listed in [requirements.txt](requirements.txt).

### Run locally

Clone the repository and enter its directory:

```bash
git clone https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection.git
cd Real-Time-Embedding-Level-Drift-Detection
```

Install the dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the evaluation script:

```bash
python experiments/synthetic_drift_evaluation.py
```

The script is intended to save the summary CSV, per-reference CSV, and detection curve under the repository's `results/` and `assets/` directories.

**Reproducibility note:** The evaluation script was run successfully in Google Colab. The generated summary and per-reference CSV files matched the committed versions exactly in that environment. Reproducibility across other environments has not yet been established.

### Evaluation Metrics

The current evaluation reports detection rates for controlled synthetic shifts. Further work should examine:

- **Detection rate:** Proportion of shifted test windows that trigger an alarm.
- **False-alarm rate:** Proportion of stable test windows incorrectly flagged as drift.
- **Reference sensitivity:** Variation in results across independently generated reference samples.
- **Detection delay:** Time or observations between drift introduction and the first alarm in a sequential experiment.
- **Computational cost:** Runtime and memory requirements under defined conditions.
- **Calibration quality:** Whether the observed stable-window alarm rate agrees with the intended calibration level.

These metrics should be evaluated using clearly defined protocols and independent test data.

## Technology

### Currently used

- Python
- NumPy
- pandas
- Matplotlib
- Google Colab

### Potential extensions

- PyTorch or other embedding frameworks.
- Documented face and voice encoders.
- Suitable face and voice datasets, subject to access and licensing conditions.
- Online change-point detection methods.
- Streaming and monitoring tools.

Potential extensions are research plans, not claims of completed integration.

## Planned Work

### Phase 1 — Reproducibility and synthetic evaluation

**Completed or partially completed work:**

- Run and record multiple synthetic evaluations, including multi-detector comparisons and robustness checks.
- Compare detection behaviour across selected shifts, dimensions, regularization settings, and random seeds.
- Preserve selected result tables, plots, and experiment reports in the repository.

**Outstanding work:**

- Verify all README artifact links against the live repository.
- Recover or regenerate missing artifacts only through a documented, reproducible procedure.
- Recover the original reference centre and regularized precision matrix, or clearly document that exact score reproduction remains unavailable.
- Compare against appropriate statistical baselines using a pre-specified protocol.
- Quantify uncertainty across independent evaluation streams and assess calibration and false-alarm behaviour more thoroughly.

The current audit verified selected files and environment/model prerequisites, but did not establish full end-to-end reproducibility of Experiment 13.

### Phase 2 — Real biometric embeddings

- Select suitable face and voice datasets, subject to their access and licensing conditions.
- Generate embeddings using documented, reproducible encoders.
- Define controlled evaluation scenarios for illumination changes, acoustic noise, and other relevant shifts.
- Compare detected distributional changes against known experimental conditions.
- Evaluate each modality separately before investigating multimodal monitoring.

### Phase 3 — Trustworthiness and operational evaluation

- Investigate whether detected drift corresponds to meaningful changes in biometric system behaviour.
- Evaluate false alarms, detection delay, sensitivity, and computational overhead.
- Examine differences across relevant data groups where the dataset and research design permit.
- Investigate online change-point detection or complementary drift statistics where justified.
- Explore integration with a streaming monitoring pipeline if the experimental findings support it.

These phases are planned research activities, not completed capabilities.

## Research Scope and Limitations

This project investigates statistical drift in embedding distributions. Distributional drift is not equivalent to biometric recognition failure, bias, spoofing, or a cyberattack.

A detected shift may occur during benign environmental changes, while a security-relevant event may occur without a large change in the monitored mean. Establishing the operational significance of a drift signal requires additional evaluation against relevant system outcomes and threat scenarios.

The current prototype has not yet demonstrated:

- Validation using real face and voice embeddings.
- A complete multimodal biometric identification pipeline.
- Real-time production deployment.
- Improved biometric recognition accuracy.
- Demonstrated mitigation of bias or security threats.

## Research Direction

The broader research direction is to investigate how lightweight, interpretable monitoring of representation changes could contribute to trustworthy biometric systems and, potentially, human-guided cyber defence.

This work is an evolving research prototype. Conclusions will be updated as additional experiments, baselines, and real-data evaluations become available.

## Additional Experiment: Multi-Detector Drift Comparison

A supplementary synthetic-data experiment compares three embedding-level drift detectors:

- **Euclidean distance** — detects changes in the mean of the reference and current embedding windows.
- **Regularized Mahalanobis distance** — measures mean differences relative to the reference covariance.
- **Variance-sensitive score** — monitors changes in the spread of the embedding distribution.

### Experimental Configuration

The standardized evaluation uses 16-dimensional synthetic Gaussian vectors, 500 reference samples per seed, 2,000 calibration windows, 1,000 test windows per scenario, and 10 independent random seeds. An independent validation stage uses 5,000 no-drift windows per seed.

### Main Findings

In the tested synthetic scenarios, the variance-sensitive detector responded strongly to increasing variance, while Euclidean and Mahalanobis scores responded to both mean shifts and variance changes. The mean false-alarm rates across the 10 seeds were approximately 1.10%, 1.01%, and 1.10%, respectively, against a 1% target.

These findings are preliminary and apply only to the specified synthetic Gaussian setup. They do not establish performance on real biometric embeddings or a deployed biometric identification system.

### Files

- Notebook: [`04_euclidean_mahalanobis_variance_comparison.ipynb`](experiments/04_euclidean_mahalanobis_variance_comparison.ipynb)
- Experimental report: [`Experiment_Report.md`](experiments/Experiment_Report.md)
- Corrected summary: [`corrected_detection_summary.csv`](results/corrected/corrected_detection_summary.csv)
- Per-seed results: [`corrected_multiseed_detection_results.csv`](results/corrected/corrected_multiseed_detection_results.csv)
- Independent validation: [`independent_false_alarm_validation.csv`](results/corrected/independent_false_alarm_validation.csv)

The original Euclidean-only evaluation and its results are retained separately. This supplementary experiment is not a replacement for the original evaluation.

## Experiment 11: Illumination Drift Detection

Evaluates ResNet-18 embedding sensitivity to controlled illumination changes using LFW images, calibrated distance-based detectors, and randomized-order control tests.

- **Notebook:** [`11_Illumination_Drift_Detection.ipynb`](experiments/11_Illumination_Drift_Detection.ipynb)
- **Results:** Experiment 11 CSV files and metadata in `results/`
- **Figures:** [Detector score timeline](assets/experiment11_detector_score_timeline.png) and [Illumination preview](assets/experiment11_illumination_preview.png)
- **Methods:** Euclidean distance, ridge-regularized Mahalanobis distance, identity-disjoint evaluation, and 20-permutation control analysis.
- **Status:** Preliminary controlled evaluation. Full from-scratch reproducibility remains unverified.

### Limitations

The experiment tests controlled brightness transformations rather than naturally occurring illumination changes. It uses pretrained ImageNet ResNet-18 features rather than a dedicated face-recognition encoder. The results do not establish biometric recognition accuracy, fairness, production readiness, or general detector superiority.

## Results-file clarification

The corrected controlled-drift evaluation is recorded in [`experiment11_corrected_detection_results.csv`](results/experiment11_corrected_detection_results.csv). The consolidated table, [`experiment11_combined_evaluation_results.csv`](results/experiment11_combined_evaluation_results.csv), includes the corrected original-order post-drift detection percentages.

The older [`experiment11_controlled_detection_results.csv`](results/experiment11_controlled_detection_results.csv) is retained for transparency. It uses the earlier per-image calibration thresholds, which are inconsistent with the window-mean scores used in the controlled evaluation. It should not be interpreted as the corrected evaluation.

Post-drift detection percentage represents the proportion of post-drift windows that triggered an alarm. A detected distribution shift does not, by itself, establish a biometric recognition failure, fairness degradation, or security incident.

## Experiment 12 V1 (Historical): Original Synthetic Embedding-Shift Evaluation

This experiment compares Euclidean and ridge-regularized Mahalanobis detectors under controlled additive shifts to embedding windows.

- **Results directory:** [`results/experiment12/`](results/experiment12/)
- **Detection summary:** [`exp12_detection_summary.csv`](results/experiment12/exp12_detection_summary.csv)
- **Trial-level results:** [`exp12_trial_results.csv`](results/experiment12/exp12_trial_results.csv)
- **Control-set results:** [`exp12_control_results.csv`](results/experiment12/exp12_control_results.csv)
- **Calibration thresholds:** [`exp12_calibration_results.csv`](results/experiment12/exp12_calibration_results.csv)
- **Detection-rate plot:** [`exp12_detection_rate.png`](results/experiment12/exp12_detection_rate.png)
- **Post-drift alarm-rate plot:** [`exp12_post_alarm_rate.png`](results/experiment12/exp12_post_alarm_rate.png)

### Preliminary findings

In this synthetic evaluation, Mahalanobis with ridge 0.001 detected 90% of trials at drift strength 1.0 and 100% at strengths 1.5–4.0. The Euclidean detector showed baseline alarms even when no drift was injected, which limits the interpretation of its apparent detection rate.

### Limitations

The evaluation uses 10 synthetic directions per positive drift condition, with 30 calibration windows, 30 control windows, and 30 evaluation windows.

These findings are preliminary and should not be interpreted as evidence of real-world biometric recognition accuracy, fairness, or deployment readiness.

## Numerical Stability Check

A numerical diagnostic was performed on the regularized covariance matrices used by the Mahalanobis detectors, using 1,209 reference samples with 768 embedding dimensions.

| Ridge regularization | Condition number | Inverse residual |
|---:|---:|---:|
| 0.001 | 1.005 × 10⁵ | 9.984 × 10⁻¹² |
| 0.01 | 1.005 × 10⁴ | 3.307 × 10⁻¹² |
| 0.1 | 1.006 × 10³ | 6.640 × 10⁻¹³ |

All computed inverses contained finite values, and the inverse residuals were small. Increasing the ridge regularization reduced the condition number under the tested configuration.

This diagnostic supports the numerical computability of the regularized inverses but does not establish statistical calibration quality, optimal ridge selection, or generalisation to real biometric data. The calibration set contains only 30 windows, so threshold uncertainty remains a limitation. Full end-to-end reproducibility from a clean runtime has not yet been independently verified.

## Experiment 12 V2: Synthetic Embedding-Shift Sensitivity Analysis under Identity-Disjoint Evaluation

This revised evaluation examines Euclidean and regularised Mahalanobis
embedding-level anomaly detectors under controlled synthetic shifts.
Reference, calibration, and evaluation partitions are identity-disjoint.
The dataset does not provide verified acquisition timestamps, so this
experiment does not claim temporal drift detection.

- **Detailed methodology and report:** [README_experiment12_v2.md](results/experiment12/README_experiment12_v2.md)
- **Results directory:** [results/experiment12/](results/experiment12/)

### Dataset and evaluation design

- 3,023 LFW-derived embedding rows, 62 identities, and 768 dimensions.
- Identity-disjoint partitions: reference 1,492 samples / 37 identities;
  calibration 1,071 / 12 identities; evaluation 460 / 13 identities.
- Window size: 20 samples; 74 reference, 53 calibration, and 23 evaluation windows.
- Standardisation parameters were estimated from reference data only.
- Twenty random unit directions and five shift magnitudes
  (0.00, 0.25, 0.50, 0.75, 1.00) were used.
- Calibration thresholds were frozen at the 90th, 95th, and 97.5th percentiles.

### Preliminary findings

At zero synthetic shift, the identity-disjoint evaluation partition already
showed elevated alarm rates:

| Detector | 90th percentile | 95th percentile | 97.5th percentile |
|---|---:|---:|---:|
| Euclidean | 39.13% | 34.78% | 21.74% |
| Mahalanobis | 82.61% | 47.83% | 39.13% |

At shift magnitude 1.00, Mahalanobis showed stronger additional alarm
crossings than Euclidean. For example, at the 95th-percentile threshold,
the mean additional crossings were 8.90 per direction for Mahalanobis
and zero for Euclidean. Mahalanobis produced positive net alarm-count
changes in all 20 tested directions at all three thresholds.

However, the elevated baseline alarm rates show that calibration thresholds
did not transfer cleanly to the identity-disjoint evaluation partition.
The evaluation data is not independently established as drift-free, so
baseline alarms cannot automatically be classified as false positives.

### V2 output files

- **Detailed report:** [README_experiment12_v2.md](results/experiment12/README_experiment12_v2.md)
- **Raw sensitivity results:** [exp12_v2_raw_sensitivity.csv](results/experiment12/exp12_v2_raw_sensitivity.csv)
- **Baseline-adjusted sensitivity:** [exp12_v2_baseline_adjusted_sensitivity.csv](results/experiment12/exp12_v2_baseline_adjusted_sensitivity.csv)
- **Alarm transitions:** [exp12_v2_alarm_transitions.csv](results/experiment12/exp12_v2_alarm_transitions.csv)
- **Direction robustness:** [exp12_v2_direction_robustness.csv](results/experiment12/exp12_v2_direction_robustness.csv)
- **Raw sensitivity plot:** [exp12_v2_raw_sensitivity.png](results/experiment12/exp12_v2_raw_sensitivity.png)
- **Baseline-adjusted sensitivity plot:** [exp12_v2_baseline_adjusted_sensitivity.png](results/experiment12/exp12_v2_baseline_adjusted_sensitivity.png)

### Limitations and interpretation

These are controlled synthetic perturbation results, not conventional
recall or validated real-world detection accuracy. The evaluation contains
only 23 windows, the same windows are reused across shift directions, and
there are no ground-truth drift-event labels. The results do not establish
real-world biometric recognition accuracy, fairness, temporal drift
detection, or deployment readiness.

Mahalanobis was more responsive to the tested synthetic shifts, but its
baseline alarm behaviour remains a significant unresolved limitation.
These findings should be treated as preliminary evidence about
embedding-level detector sensitivity.


## Experiment 13: False-Alarm-Controlled Embedding-Level Drift Detection

Experiment 13 investigates threshold-calibrated and sequential drift detection using synthetic embedding distributions and an exploratory real-image embedding pilot. The work progresses from synthetic robustness analysis to controlled LFW brightness transformations, sequential CUSUM evaluation, fresh-identity stream validation, and repeated bootstrap resampling.

### Synthetic evaluation

The synthetic evaluation examines detection sensitivity, false-alarm behaviour, and robustness across embedding dimensions, covariance regularization settings, and random seeds. A supplementary robustness evaluation covers 360 configurations.

Results are specific to the synthetic data-generating process and tested parameter settings. They do not establish performance on real biometric recognition systems.

Available synthetic evaluation artifacts include:

- [`experiment13_trial_results.csv`](results/experiment13_trial_results.csv)
- [`experiment13_detection_summary.csv`](results/experiment13_detection_summary.csv)
- [`experiment13_control_fpr_summary.csv`](results/experiment13_control_fpr_summary.csv)
- [`experiment13_single_seed_calibration.csv`](results/experiment13_single_seed_calibration.csv)
- [`experiment13_single_seed_control.csv`](results/experiment13_single_seed_control.csv)
- [`experiment13_robustness_summary.csv`](results/experiment13_robustness_summary.csv)
- [`experiment13_robustness_summary_ci.csv`](results/experiment13_robustness_summary_ci.csv)
- [`experiment13_robustness_trials.csv`](results/experiment13_robustness_trials.csv)
- [`experiment13_detection_ci.png`](results/experiment13_detection_ci.png)
- [`experiment13_robustness_detection.png`](results/experiment13_robustness_detection.png)

### LFW controlled-brightness pilot

A preliminary pilot uses pretrained FaceNet embeddings extracted from the Labeled Faces in the Wild (LFW) dataset. Controlled brightness transformations are applied to images to examine changes in embedding-level detector scores.

The evaluation uses matched image streams and identity-disjoint reference and evaluation partitions. These experiments examine representation-score changes, not biometric recognition accuracy or naturally occurring temporal drift.

The sequential detector uses Mahalanobis distances calculated from window-mean embeddings. Scores are standardized using calibration-derived statistics and passed to a one-sided cumulative sum (CUSUM) detector.

#### Frozen detector configuration

The saved parameter file records the following values:

| Parameter | Value |
|---|---:|
| Baseline median | 1.0419082897513998 |
| Robust scale | 0.09925588181635098 |
| CUSUM allowance, \(k\) | 0.5 |
| Alarm threshold, \(h\) | 5.0 |
| Calibration windows | 100 |
| Window size | 50 images |
| Calibration method | Identity-balanced windows |

These parameters were recorded as frozen settings for the subsequent sequential evaluations. The allowance and threshold are provisional settings and do not guarantee a specified operational false-alarm rate.

Available pilot artifacts:

- [`lfw_sequential_detector_parameters_v2.csv`](results/lfw_sequential_detector_parameters_v2.csv)
- [`lfw_sequential_detector_results_v2.csv`](results/lfw_sequential_detector_results_v2.csv)
- [`lfw_sequential_detector_trace_v2.csv`](results/lfw_sequential_detector_trace_v2.csv)
- [`lfw_sequential_cusum_diagnostics_v2.csv`](results/lfw_sequential_cusum_diagnostics_v2.csv)
- [`lfw_identity_balanced_calibration_diagnostics.csv`](results/lfw_identity_balanced_calibration_diagnostics.csv)

### Sequential CUSUM evaluation

The sequential pilot evaluates the frozen CUSUM configuration on controlled streams, with brightness transformations introduced at window 6. A repeated-stream sensitivity analysis uses 20 bootstrap/resampling replicates drawn from the same 600-image pool.

| Condition | Replicates | Pre-onset alarm rate | Post-onset alarm rate (all replicates) |
|---|---:|---:|---:|
| Control | 20 | 10% | Not applicable |
| Brightness 0.70 | 20 | 10% | 40% |
| Brightness 1.30 | 20 | 10% | 55% |

The two pre-onset alarms occurred in the same resampling replicates across the control and transformed conditions. This indicates that stream variability can trigger alarms before the imposed brightness change.

The resampled streams overlap and are not independent datasets. These detection rates are exploratory and do not establish reliable deployment performance.

Reports and results:

- Sequential pilot report: [`Experiment_13_Phase_5_Sequential_Pilot_Report.md`](results/Experiment_13_Phase_5_Sequential_Pilot_Report.md)
- Corrected repeated-stream report: [`Experiment_13_Phase_5B_Repeated_Stream_Report_CORRECTED.md`](results/Experiment_13_Phase_5B_Repeated_Stream_Report_CORRECTED.md)
- Original report retained for historical reference: [`Experiment_13_Phase_5B_Repeated_Stream_Report.md`](results/Experiment_13_Phase_5B_Repeated_Stream_Report.md)
- Repeated-stream results: [`lfw_repeated_stream_cusum_results.csv`](results/lfw_repeated_stream_cusum_results.csv)
- Repeated-stream summary: [`lfw_repeated_stream_cusum_summary.csv`](results/lfw_repeated_stream_cusum_summary.csv)

### Stage 3: Fresh-identity stream validation

Stage 3 evaluates the previously frozen one-sided CUSUM detector on a new stream of 1,000 LFW images representing 1,000 distinct identities that were not used in the earlier 600-image matched-stream experiment.

#### Configuration

- Embeddings: 512-dimensional pretrained FaceNet representations.
- Window size: 50 images; 20 windows per condition.
- Conditions: unchanged control, brightness factor 0.70, and brightness factor 1.30.
- Change onset: window 10, using zero-based indexing.
- Frozen baseline median: 1.041908290.
- Frozen robust scale: 0.099255882.
- CUSUM allowance: \(k=0.5\).
- Alarm threshold: \(h=5.0\).
- No detector retuning on the fresh evaluation streams.

#### Results

| Condition | First detection window | Pre-onset alarm | Detected after onset | Delay |
|---|---:|---|---|---|
| Control | None | No | Not applicable | Not applicable |
| Brightness 0.70 | None | No | No | Not applicable |
| Brightness 1.30 | 14 | No | Yes | 4 windows |

The detector detected the brightness-1.30 stream four windows after the imposed onset. It did not alarm on the control or brightness-0.70 stream. This is a positive result for one stream under one controlled transformation, not an estimate of reliable detection performance.

The fresh identities came from the same LFW dataset and evaluation partition, rather than an independent external dataset.

#### Stage 3 artifacts

- Fresh-identity validation report: [`Experiment_13_Stage_3_Fresh_Identity_Validation_Report.md`](results/Experiment_13_Stage_3_Fresh_Identity_Validation_Report.md)
- CUSUM results: [`lfw_stage3_fresh_cusum_results.csv`](results/lfw_stage3_fresh_cusum_results.csv)
- CUSUM trace: [`lfw_stage3_fresh_cusum_trace.csv`](results/lfw_stage3_fresh_cusum_trace.csv)

### Stage 3: Repeated bootstrap resampling and paired comparison

A further exploratory evaluation performs 20 paired resampling replicates for each condition, using the existing 1,000-image Stage 3 embedding pool and the same frozen detector configuration. The same sampled positions are used across the control and brightness-transformed streams within each replicate.

#### Any alarm at or after window 10

| Condition | Replicates | Pre-onset alarm rate | Replicates with an alarm at/after window 10 |
|---|---:|---:|---:|
| Control | 20 | 40% | 75% |
| Brightness 0.70 | 20 | 40% | 90% |
| Brightness 1.30 | 20 | 40% | 100% |

These figures count any alarm at or after window 10, including alarms that occurred after an earlier pre-onset alarm. They must not be interpreted as valid first post-onset alarm rates.

#### Mutually exclusive first-alarm classification

| Condition | First alarm after onset | No alarm | Pre-onset only | Pre-onset then post-onset |
|---|---:|---:|---:|---:|
| Control | 8 | 4 | 1 | 7 |
| Brightness 0.70 | 10 | 2 | 0 | 8 |
| Brightness 1.30 | 12 | 0 | 0 | 8 |

Eight of 20 replicates in each condition produced a pre-onset alarm. The unchanged control also produced first alarms after onset in eight replicates. These results show that normal stream variability can trigger the detector and that a later alarm cannot automatically be attributed to the imposed brightness transformation.

#### Paired first-post-onset comparison

The following comparison is restricted to replicates in which both the transformed stream and its paired control had their first alarm at or after the imposed onset. This excludes pairs affected by pre-onset first alarms.

| Condition | Both streams with valid first post-onset alarms | Transformed stream alarms earlier | Same first-alarm window | Transformed stream alarms later |
|---|---:|---:|---:|---:|
| Brightness 0.70 vs control | 8 | 6 | 2 | 0 |
| Brightness 1.30 vs control | 8 | 8 | 0 | 0 |

This is an exploratory paired timing comparison, not evidence of dependable drift detection. The number of eligible pairs is small, and the control condition itself produces post-onset first alarms.

The bootstrap replicates reuse the same underlying image pool and are not independent datasets. These findings measure sensitivity to resampling rather than generalisation to new identities or external datasets.

#### Bootstrap artifacts

The following artifacts have been recovered and verified in the repository:

- Bootstrap results: [`lfw_stage3_bootstrap_cusum_results.csv`](results/lfw_stage3_bootstrap_cusum_results.csv)
- Bootstrap trace: [`lfw_stage3_bootstrap_cusum_trace.csv`](results/lfw_stage3_bootstrap_cusum_trace.csv)
- Paired comparison: [`lfw_stage3_bootstrap_paired_comparison.csv`](results/lfw_stage3_bootstrap_paired_comparison.csv)
- Corrected bootstrap report: [`Experiment_13_Stage_3_Bootstrap_Comparison_Report_CORRECTED.md`](results/Experiment_13_Stage_3_Bootstrap_Comparison_Report_CORRECTED.md)

The original bootstrap report is retained as a historical record. The corrected report and audited trace-derived classifications should be used when interpreting these results.

### Visualization

The Stage 3 CUSUM plot shows the frozen threshold, imposed onset, and CUSUM traces for the control and both brightness conditions.

[Experiment 13 Stage 3 CUSUM plot](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/blob/main/results/lfw_stage3_fresh_cusum_plot.png) ([view raw image](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/raw/main/results/lfw_stage3_fresh_cusum_plot.png))

### Reproducibility audit

A separate Colab runtime was used to check selected environment and model prerequisites without rerunning the complete experiment.

The audit verified that:

- LFW could be downloaded and prepared in the fresh runtime.
- FaceNet initialized successfully.
- A sample LFW image produced a finite 512-dimensional embedding.
- The frozen detector parameter CSV could be retrieved from GitHub.
- The Stage 3 fresh-identity result CSV and CUSUM trace could be retrieved and parsed.
- The saved notebook contains the code used to calculate the reference centre and regularized precision matrix.

However, the original reference centre and precision matrix were not located among the artifacts checked. Consequently, exact reproduction of the original Mahalanobis scores has not been independently verified.

The Stage 3 bootstrap results, trace, and paired-comparison CSVs have since been recovered and verified in the repository. Their presence and structure have been checked, and the trace-derived first-alarm classifications have been audited. The fresh-runtime smoke test verifies selected prerequisites, not the complete end-to-end experiment.

Environment and model configuration records:

- [`Experiment_13_requirements_observed.txt`](results/Experiment_13_requirements_observed.txt)
- [`Experiment_13_FaceNet_Model_Configuration.txt`](results/Experiment_13_FaceNet_Model_Configuration.txt)
- [`Experiment_13_LFW_Dataset_Configuration.txt`](results/Experiment_13_LFW_Dataset_Configuration.txt)

### Embedding Provenance and Reproducibility Audit

#### 1. Audit Objective

A provenance audit was conducted to investigate the correspondence between the regenerated embedding matrix, its associated metadata, and the original Labeled Faces in the Wild (LFW) images.

The original embedding-generation pipeline preserved identity labels and sequential dataset indices but did not record the corresponding image filenames or paths. Therefore, direct verification of the original image-to-embedding mapping was not possible from the saved metadata alone.

#### 2. Verified Dataset and Model Configuration

| Property | Verified result |
|---|---|
| Dataset | Labeled Faces in the Wild (LFW) |
| Total images | 13,233 |
| Total identities | 5,749 |
| Embedding dimension | 512 |
| Embedding data type | float32 |
| Embedding model | FaceNet / Inception-ResNet-v1 |
| `keras-facenet` version | 0.3.2 |

The LFW source archive was verified using SHA-256:

```text
055f7d9c632d7370e6fb4afc7468d40f970c34a80d4c6f50ffec63f5a8d536c0
```

The pretrained FaceNet weights were verified using SHA-256:

```text
8b71e7045497e841c00ee568f031d1a4d30908fceadf6884aef2dec4d545202b
```

#### 3. Reconstruction and Matching Method

Fresh FaceNet embeddings were independently generated from the verified LFW image archive.

The regenerated embedding matrix was compared against the independently recomputed embedding matrix using identity-constrained cosine similarity matching. A one-to-one Hungarian assignment was performed within each identity group to establish image-level correspondence.

#### 4. Image-Level Correspondence Results

| Validation metric | Result |
|---|---:|
| Total embedding rows matched | 13,233 |
| Unmatched rows | 0 |
| Unique image assignments | True |
| Identity consistency | 13,233 / 13,233 |
| Minimum cosine similarity | 0.997429 |
| Median cosine similarity | 0.999820 |
| Mean cosine similarity | 0.999795 |
| Maximum cosine similarity | 0.999970 |

All 13,233 embedding rows were assigned to unique images, with identity consistency maintained throughout the matching process.

#### 5. Numerical Similarity Analysis

After aligning the independently generated embeddings with their assigned image positions, numerical differences were evaluated.

| Metric | Result |
|---|---:|
| Median L2 difference | 0.018968 |
| Mean L2 difference | 0.019624 |
| Maximum L2 difference | 0.071707 |
| Mean absolute element-wise difference | 0.000696 |
| Median cosine similarity | 0.999820 |
| Minimum cosine similarity | 0.997429 |

The regenerated and independently recomputed embedding matrices exhibited nearly identical vector norm distributions, with norms approximately equal to 1.

Despite the high cosine similarity, the embedding vectors were not numerically identical. Exact numerical reproducibility has therefore not been established.

#### 6. Dataset and Metadata Verification

The regenerated metadata contained 13,233 unique, consecutive dataset indices ranging from 0 to 13,232.

Additional checks established that:

- All 5,749 identities were represented.
- Identity counts matched the reconstructed LFW dataset for all 5,749 identities.
- The total image count matched 13,233.
- Direct positional alignment with the reconstructed archive ordering was not established.

The original pipeline did not preserve image paths, and its exact TFDS enumeration order could not be independently reconstructed with sufficient certainty.

#### 7. Diagnostic Image Mapping Artifact

A separate diagnostic CSV was generated to preserve the reconstructed image-level correspondence.

**Diagnostic CSV:** [`lfw_embedding_image_mapping_diagnostic.csv`](experiments/lfw_embedding_image_mapping_diagnostic.csv)

**Repository location:** `experiments/lfw_embedding_image_mapping_diagnostic.csv`

The CSV contains 13,233 records and seven columns:

| Column | Description |
|---|---|
| `recovered_row` | Row position in the regenerated embedding matrix |
| `dataset_index` | Original metadata dataset index |
| `metadata_identity` | Identity recorded in the metadata |
| `matched_identity` | Identity associated with the matched image |
| `matched_image_path` | Reconstructed LFW image path |
| `cosine_similarity` | Similarity score for the assigned correspondence |
| `identity_match` | Boolean identity consistency indicator |

The saved CSV was reloaded and verified. All checks passed, including identity consistency, unique image paths, finite similarity values, and preservation of dataset indices, image paths, and similarity scores.

This CSV is a diagnostic provenance artifact. It does not replace or modify the existing embedding or metadata artifacts.

#### 8. Reproducibility Limitations

The audit identified the following limitations:

1. The original extraction pipeline did not preserve image filenames or image paths alongside embedding vectors.
2. Exact original TFDS enumeration order could not be independently established.
3. The regenerated embeddings were not numerically identical to the independently recomputed embeddings.
4. The precise cause of the remaining numerical differences has not been isolated.
5. High cosine similarity does not independently establish real-world biometric recognition accuracy, fairness, or deployment readiness.

#### 9. Audit Conclusion

The provenance audit established one-to-one image-level correspondence for all 13,233 regenerated embedding rows against the verified LFW source images.

The reconstructed embeddings demonstrated consistently high cosine similarity, with a mean of 0.999795 and a minimum of 0.997429.

These findings provide strong evidence of embedding-level correspondence. However, exact reproduction of the original dataset enumeration order and numerical embedding values has not been demonstrated.

The regenerated artifacts and diagnostic mapping CSV should therefore be treated as reconstructed research artifacts rather than exact recovered copies of the original pipeline outputs.

**Audit status:** Image-level correspondence verified; exact numerical and ordering reproducibility not established.

### Limitations

These experiments use synthetic distributions and controlled transformations of static face images, not naturally occurring temporal biometric drift. Bootstrap/resampling replicates overlap and are not independent observations. Stage 3 uses one stream per condition, and the repeated-resampling control results show substantial late alarms and pre-onset alarms.

The detector has not demonstrated dependable operational false-alarm control. The experiments do not establish real-world biometric recognition accuracy, demographic fairness, successful bias mitigation, cyberattack detection, or deployment readiness. The CUSUM configuration is provisional and does not guarantee a specified operational false-alarm rate.

The fresh-identity evaluation uses the same LFW dataset and evaluation partition rather than an independent external dataset. The original reference centre and precision matrix have not been recovered, so exact reproduction of the original Mahalanobis scores remains unverified. Although the Stage 3 bootstrap CSVs have been recovered and verified, this does not resolve the original reference-model reproducibility limitation.

Raw face images, embedding arrays, and identity-level metadata should not be published without appropriate licensing, privacy, and consent review.

### Conclusion

Experiment 13 investigates false-alarm-controlled embedding-level drift detection through synthetic evaluations and an exploratory LFW FaceNet pilot. The fresh-identity evaluation recorded a threshold crossing under the stronger controlled brightness transformation, while the repeated-stream analyses exposed important pre-onset and control-condition alarm concerns.

The evidence supports continued methodological investigation, not a claim of reliable real-world biometric drift detection. The frozen detector parameters, Stage 3 fresh-identity results, and Stage 3 bootstrap results and trace are available in the repository. The corrected bootstrap analysis identifies substantial pre-onset and control-condition alarm concerns. Exact reproduction of the original Mahalanobis scores remains unverified because the original reference-model artifacts were not recovered.

**Overall status:** exploratory experiment completed; selected outputs verified; full numerical reproducibility remains incomplete.


## Experiment 14: Drift Detection Under Gradual and Recurring Distribution Shifts

### Objective

Evaluate the behaviour of embedding-level drift detectors under abrupt, gradual, and recurring distribution shifts. The experiment examines detection rate, detection delay, stable-period false alarms, and agreement between Euclidean and Mahalanobis detectors.

### Experimental Design

- **Drift patterns:** Abrupt, gradual, and recurring.
- **Detectors:** Euclidean and Mahalanobis.
- **Evaluation:** Synthetic embedding distributions with controlled temporal drift patterns.
- **Independent seeds:** 20.
- **Calibration:** Frozen detector thresholds.
- **Target false-alarm rate:** 5%.
- **Additional analysis:** Independent control evaluation, recurring-shift gap analysis, seed-level bootstrap confidence intervals, and numerical-stability diagnostics.

### Frozen Threshold Configuration

| Parameter | Value |
|---|---:|
| Target false-alarm rate | 5% |
| Calibration windows | 200 |
| Euclidean threshold | 2.0702929416972333 |
| Mahalanobis threshold | 2.0746417163536846 |
| Calibration method | Empirical upper-tail quantile |
| Reference seed | 20261401 |
| Calibration seed | 20261402 |

The thresholds were frozen before independent evaluation. The confidence-interval analysis did not modify the original experiment data or thresholds.

### Key Results

| Metric | Result |
|---|---:|
| Detector alarm-decision agreement | 100% across 3,600 comparisons |
| Abrupt-shift detection rate | 100% |
| Gradual-shift detection rate | 95.5% |
| Recurring-shift detection rate | 100% |
| Mean abrupt-shift detection delay | 0 windows |
| Mean gradual-shift detection delay | 1.45 windows |
| Mean recurring-shift detection delay | 0 windows per episode |
| Independent control false-alarm rate | 4.97% |
| Independent control false-alarm standard deviation | 0.78 percentage points |

The independent control evaluation recorded 497 alarms per detector across 10,000 control windows, corresponding to a pooled false-alarm rate of 4.97%.

### 95% Confidence Interval Analysis

Seed-level percentile bootstrap confidence intervals were calculated using 20,000 bootstrap repetitions and 20 independent seeds. The independent evaluation unit was the seed.

| Metric | Condition | Mean | 95% CI |
|---|---|---:|---:|
| Detection rate | Abrupt | 100.00% | 100.00–100.00% |
| Detection rate | Gradual | 95.50% | 94.75–96.25% |
| Detection rate | Recurring | 100.00% | 100.00–100.00% |
| Stable false-alarm rate | Abrupt | 6.75% | 4.25–9.25% |
| Stable false-alarm rate | Gradual | 6.75% | 4.25–9.25% |
| Stable false-alarm rate | Recurring | 5.75% | 4.25–7.125% |
| Independent control false-alarm rate | Independent control | 4.97% | 4.65–5.32% |
| Detection delay | Abrupt | 0.00 windows | 0.00–0.00 |
| Detection delay | Gradual | 1.45 windows | 1.05–1.80 |
| Detection delay | Recurring | 0.00 windows | 0.00–0.00 |

The confidence intervals were calculated across independent seeds rather than treating individual windows as independent observations.

For recurring detection delay, the episode-level delays were first averaged within each seed before calculating the seed-level bootstrap interval.

The zero-width intervals for 100% detection rates and zero-window delays reflect identical observations across the evaluated seeds. They do not establish guaranteed performance outside the tested synthetic configuration.

### Recurring-Shift Analysis

Recurring-shift gap analysis showed:

| Stable period | Mean false-alarm rate |
|---|---:|
| Recurring-shift gaps | 6.0% |
| Final baseline period | 3.5% |

Nine of 20 seeds had no alarms during recurring-shift gaps, while 13 of 20 seeds had no alarms during the final baseline period.

The recurring aggregate stable false-alarm rate was 5.75%. This aggregate includes the initial baseline, stable gaps, and final stable period.

These findings indicate that false alarms can persist during nominally stable periods and that alarm behaviour varies across random seeds.

### Main Findings

1. Both detectors produced matching alarm decisions in all 3,600 reported comparisons.
2. Abrupt and recurring shifts were detected at a 100% rate in the evaluated synthetic scenarios.
3. Gradual shifts were detected at a 95.5% rate, with a mean detection delay of 1.45 windows.
4. The gradual detection-rate confidence interval was 94.75–96.25%.
5. Independent control false-alarm performance was close to the nominal 5% target, with a mean of 4.97%.
6. Stable-period false alarms remained present despite frozen calibration thresholds.
7. Seed-level confidence intervals quantify variation across the 20 simulated seeds but do not establish generalisation to real biometric data.

### Interpretation and Limitations

The results provide preliminary evidence about detector behaviour under controlled synthetic temporal shifts.

Matching alarm decisions do not establish that either detector is universally superior. Both detectors produced identical decisions under the tested configuration.

The independent control false-alarm rate was close to the nominal target. However, the stable-period false-alarm rates observed during temporal evaluation were higher in some conditions. This distinction matters when interpreting the operational behaviour of the detectors.

The zero-width confidence intervals for some metrics arise because all 20 seeds produced the same measured result. They should not be interpreted as proof of zero uncertainty under other data-generating processes.

The experiment uses synthetic embedding distributions and controlled drift patterns. It does not establish:

- Real-world biometric recognition accuracy.
- Fairness or successful bias mitigation.
- Dependable drift detection on real face or voice embeddings.
- Detection of spoofing attempts or cyberattacks.
- Production deployment readiness.

Further evaluation on real embedding datasets, independent acquisition conditions, and additional drift scenarios is required.

### Reproducibility and Integrity Verification

The final experiment audit verified all 18 expected result artifacts.

- **Expected artifacts:** 18.
- **Artifacts present:** 18.
- **Missing artifacts:** 0.
- **SHA-256 hashes:** Generated for all 18 expected artifacts.
- **Frozen threshold validation:** Passed.
- **Confidence interval output:** Verified, 10 rows and 8 columns.
- **Artifact inventory:** Saved.
- **Integrity manifest:** Saved successfully.

The experiment data and frozen thresholds were not modified during the confidence-interval calculation or final integrity audit.

#### Expected Result Artifacts

1. `environment.json`
2. `calibration_scores.csv`
3. `frozen_thresholds.json`
4. `independent_control_results.csv`
5. `independent_control_summary.csv`
6. `baseline_detector_score_comparison.csv`
7. `temporal_drift_evaluation.csv`
8. `temporal_drift_trace.csv`
9. `temporal_drift_summary.csv`
10. `detector_alarm_agreement.csv`
11. `detection_delay_by_episode.csv`
12. `detection_delay_summary.csv`
13. `recurring_alarm_persistence.csv`
14. `recurring_episode_separation_summary.csv`
15. `stable_false_alarm_by_seed.csv`
16. `stable_false_alarm_seed_summary.csv`
17. `corrected_segment_alarm_diagnostic.csv`
18. `confidence_intervals.csv`

The separate audit outputs are:

- `artifact_inventory.csv`
- `experiment_14_integrity_manifest.csv`

The integrity manifest records file sizes and SHA-256 hashes. Passing the integrity checks establishes file presence and recorded hash consistency, not scientific validity or independent reproduction of the experiment.

### Conclusion

Experiment 14 evaluates embedding-level drift detection under abrupt, gradual, and recurring synthetic distribution shifts.

Both detectors achieved identical alarm decisions in the tested scenarios. Detection rates were high, and the independent control false-alarm rate was close to the nominal 5% target. However, stable-period false alarms and the synthetic nature of the evaluation remain important limitations.

The seed-level confidence intervals provide additional information about variability across the 20 independent seeds.

**Overall status:** Synthetic experiment completed; confidence intervals calculated; all 18 expected artifacts verified; real-world biometric validation remains outstanding.

# Experiment 15: Detector Discrimination Under Anisotropic and Correlated Embedding Distributions

## Objective

Evaluate how covariance structure and drift direction affect the behaviour of Euclidean and regularized Mahalanobis embedding-level drift detectors under controlled synthetic conditions.

## Experimental Design

- **Covariance conditions:** Isotropic, anisotropic, and correlated.
- **Embedding dimension:** 64.
- **Reference samples:** 1,000 per seed and condition.
- **Independent calibration windows:** 200 per seed and condition.
- **Independent control windows:** 500 per seed and condition.
- **Drift evaluation windows:** 500 per seed, condition, and direction.
- **Independent seeds:** 20.
- **Target false-alarm rate:** 5%.
- **Mean-shift magnitude:** 0.5 along normalized directions.
- **Detectors:** Euclidean and Ledoit–Wolf-regularized Mahalanobis.
- **Calibration:** Detector-specific empirical 95th-percentile thresholds, frozen before independent evaluation.

Drift directions were defined using covariance eigenvectors: high-variance, low-variance, and mixed directions. In the isotropic condition, these labels do not represent genuinely different population variances.

## Multi-Seed Independent Control Results

| **Covariance condition** | **Euclidean FPR** | **Mahalanobis FPR** |
| ------------------------ | ----------------- | ------------------- |
| Isotropic                | 4.54%             | 4.54%               |
| Anisotropic              | 4.86%             | 4.95%               |
| Correlated                | 5.23%             | 4.69%               |

The pooled false-alarm rates were close to the 5% target. However, seed-level rates varied, with observed values ranging from 1.4% to 8.6%. Calibration performance should therefore not be assumed to remain exactly at the target on every independent sample.

## Multi-Seed Drift Detection Results

Mean detection rates across 20 seeds:

| **Condition** | **Direction** | **Euclidean** | **Mahalanobis** |
| ------------- | ------------- | ------------- | --------------- |
| Isotropic     | High variance | 10.86%        | 10.86%          |
| Isotropic     | Low variance  | 11.19%        | 11.22%          |
| Isotropic     | Mixed         | 10.81%        | 10.78%          |
| Anisotropic   | High variance | 7.97%         | 6.15%           |
| Anisotropic   | Low variance  | 6.91%         | 29.88%          |
| Anisotropic   | Mixed         | 7.30%         | 15.87%          |
| Correlated    | High variance | 9.69%         | 6.24%           |
| Correlated    | Low variance  | 7.79%         | 39.42%          |
| Correlated    | Mixed         | 8.15%         | 19.56%          |

## Across-Seed Detector Comparison

Paired detector comparisons were performed using the 20 seed-level detection rates for each of the nine condition-direction combinations.

The difference was defined as:

**Mahalanobis detection rate − Euclidean detection rate.**

| **Condition** | **Direction** | **Mean difference** | **95% confidence interval** |
| ------------- | ------------- | ------------------- | --------------------------- |
| Anisotropic   | High variance | −1.82 pp             | [−2.80, −0.84]              |
| Anisotropic   | Low variance  | +22.97 pp            | [+20.66, +25.28]             |
| Anisotropic   | Mixed         | +8.57 pp             | [+6.98, +10.16]              |
| Correlated    | High variance | −3.45 pp             | [−4.65, −2.25]               |
| Correlated    | Low variance  | +31.63 pp            | [+28.24, +35.02]             |
| Correlated    | Mixed         | +11.41 pp            | [+9.22, +13.60]              |
| Isotropic     | High variance | 0.00 pp              | [−0.06, +0.06]               |
| Isotropic     | Low variance  | +0.03 pp             | [−0.02, +0.08]               |
| Isotropic     | Mixed         | −0.03 pp             | [−0.08, +0.02]               |

These are the original, unadjusted 95% confidence intervals.

## Statistical Significance Analysis

Paired one-sample t-tests were conducted on the seed-level differences across the nine condition-direction combinations.

- **Independent seeds per comparison:** 20.
- **Statistical test:** One-sample t-test on paired Mahalanobis-minus-Euclidean detection-rate differences across seeds.
- **Multiple-comparison correction:** Holm correction across nine comparisons.
- **Simultaneous confidence intervals:** Bonferroni-adjusted 95% confidence intervals.
- **Effect size:** Cohen's \(d_z\).

A positive mean difference indicates a higher detection rate for Mahalanobis, while a negative mean difference indicates a higher detection rate for Euclidean.

### Multiple-Comparison Results

| Condition | Direction | Mean Difference | Holm-Adjusted p-value | Cohen's dz | Result |
|---|---|---:|---:|---:|---|
| Anisotropic | High variance | -0.0182 | 0.003842 | -0.872 | Significant |
| Anisotropic | Low variance | +0.2297 | 1.39 × 10⁻¹³ | +4.653 | Significant |
| Anisotropic | Mixed | +0.0857 | 4.93 × 10⁻⁹ | +2.528 | Significant |
| Correlated | High variance | -0.0345 | 0.0000435 | -1.345 | Significant |
| Correlated | Low variance | +0.3163 | 3.91 × 10⁻¹³ | +4.368 | Significant |
| Correlated | Mixed | +0.1141 | 7.76 × 10⁻⁹ | +2.437 | Significant |
| Isotropic | High variance | 0.0000 | 1.000000 | 0.000 | Not significant |
| Isotropic | Low variance | +0.0003 | 0.559234 | +0.307 | Not significant |
| Isotropic | Mixed | -0.0003 | 0.559234 | -0.307 | Not significant |

**Six of the nine comparisons remained statistically significant after Holm correction at the 0.05 significance level.**

### Bonferroni-Adjusted 95% Confidence Intervals

| Condition | Direction | Lower Bound | Upper Bound | Excludes Zero |
|---|---|---:|---:|---|
| Anisotropic | High variance | -0.032787 | -0.003613 | Yes |
| Anisotropic | Low variance | 0.195186 | 0.264214 | Yes |
| Anisotropic | Mixed | 0.062000 | 0.109400 | Yes |
| Correlated | High variance | -0.052434 | -0.016566 | Yes |
| Correlated | Low variance | 0.265673 | 0.366927 | Yes |
| Correlated | Mixed | 0.081366 | 0.146834 | Yes |
| Isotropic | High variance | -0.000907 | 0.000907 | No |
| Isotropic | Low variance | -0.000384 | 0.000984 | No |
| Isotropic | Mixed | -0.000984 | 0.000384 | No |

The six statistically significant comparisons also had Bonferroni-adjusted simultaneous 95% confidence intervals that excluded zero.

The three isotropic comparisons did not show statistically significant differences. This does not establish statistical equivalence between the detectors.

### Main Statistical Findings

1. Under anisotropic conditions, Mahalanobis detection was significantly higher for low-variance and mixed-direction drift, while Euclidean detection was significantly higher for high-variance drift.
2. Under correlated conditions, Mahalanobis detection was significantly higher for low-variance and mixed-direction drift, while Euclidean detection was significantly higher for high-variance drift.
3. Under isotropic conditions, no statistically significant detector differences were observed.
4. The largest standardized paired effects occurred for low-variance drift under anisotropic and correlated conditions.
5. Detector performance depended on the covariance structure and direction of distribution shift. Neither detector universally outperformed the other.

## Interpretation and Limitations

The results support the hypothesis that covariance structure and drift direction influence detector behaviour. Mahalanobis distance accounts for covariance geometry and can be more sensitive to shifts in directions with low reference variance, whereas Euclidean distance does not explicitly account for covariance geometry.

These results do not establish that Mahalanobis is universally superior. Detection rates were modest in most conditions, the evaluation used synthetic Gaussian distributions and mean shifts, and independent false-alarm rates varied across seeds.

The statistical analysis uses 20 independent seeds per comparison. The reported effect sizes and significance tests should be interpreted within this experimental design.

The experiment does not establish real-world biometric recognition accuracy, fairness, robustness to all forms of distribution shift, or deployment readiness. Further work should investigate other drift magnitudes, covariance structures, calibration sample sizes, and real embedding datasets.

## Reproducibility and Outputs

The experiment used 20 seeds and independent reference, calibration, control, and evaluation streams. Frozen thresholds were retained during independent control and drift evaluation.

The final saved-data integrity audit passed all 24 checks, including expected row counts, finite scores and thresholds, alarm consistency, confidence interval validation, and threshold consistency across calibration, control, and evaluation records.

This audit verifies the structural and numerical consistency of the saved experimental outputs. It does not constitute independent reproduction of the complete experiment from a clean runtime.

### Primary Outputs

- `config.json`
- `environment.json`
- `covariance_diagnostics.csv`
- `reference_dataset_diagnostics.csv`
- `drift_direction_diagnostics.csv`
- `multiseed_reference_diagnostics.csv`
- `multiseed_calibration_scores.csv`
- `multiseed_frozen_thresholds.csv`
- `multiseed_independent_control_results.csv`
- `multiseed_control_seed_summary.csv`
- `multiseed_control_summary.csv`
- `multiseed_drift_evaluation_results.csv`
- `multiseed_drift_seed_summary.csv`
- `multiseed_drift_evaluation_summary.csv`
- `multiseed_detection_rate_differences.csv`
- `multiseed_detection_difference_summary.csv`
- `multiseed_detection_difference_ci95.csv`
- `multiseed_detection_difference_inference.csv`
- `experiment_15_integrity_audit.csv`

All results are synthetic experimental evidence and should be interpreted within the stated limitations.

## Final Experiment Status

- Multi-seed evaluation: Completed.
- Independent control evaluation: Completed.
- Across-seed detector comparison: Completed.
- Unadjusted 95% confidence intervals: Completed.
- Paired statistical significance analysis: Completed.
- Holm multiple-comparison correction: Completed.
- Bonferroni simultaneous confidence intervals: Completed.
- Final saved-data integrity audit: 24/24 checks passed.

**Experiment 15 is complete with respect to the implemented synthetic experimental workflow and saved-data integrity validation.**

## Experiment 16: Cross-Condition Calibration Transfer for Embedding-Level Drift Detection

### Objective

Experiment 16 investigates whether thresholds calibrated on one embedding distribution remain reliable when applied to a different target distribution.

It evaluates Euclidean distance and Ledoit–Wolf-regularized Mahalanobis distance under three synthetic covariance conditions: isotropic, anisotropic, and correlated.

The experiment also compares frozen source-condition calibration against target-condition threshold recalibration and target-condition detector adaptation.

### Experimental Setup

| Parameter | Configuration |
|---|---:|
| Independent random seeds | 20 |
| Random seed range | 202616–202635 |
| Embedding dimension | 64 |
| Window size | 20 |
| Reference samples | 1,000 per seed and condition |
| Source calibration windows | 200 |
| Independent control windows | 500 |
| Drift evaluation windows | 500 |
| Nominal false-alarm rate | 5% |
| Mean-shift strength | 0.5 |
| Covariance conditions | Isotropic, anisotropic, correlated |
| Source-target scenarios | 9 |
| Detectors | Euclidean, Ledoit–Wolf-regularized Mahalanobis |

### Methodology

Three covariance conditions were evaluated:

- **Isotropic:** Identity covariance.
- **Anisotropic:** Diagonal covariance with unequal variances.
- **Correlated:** Correlated covariance structure.

All nine source-target combinations were evaluated, including three same-condition and six cross-condition scenarios.

#### Original frozen-threshold evaluation

Detector parameters and thresholds were calibrated using source-condition data. The thresholds were frozen before evaluation on independent target-condition control and drift windows.

This stage examined whether source-condition calibration transfers reliably to different target distributions.

#### Three-method comparative evaluation

A further benchmark compared three calibration strategies using shared independent evaluation datasets.

| Method | Detector fitting | Threshold calibration | Interpretation |
|---|---|---|---|
| A | Source condition | Frozen source threshold | Zero-shot calibration transfer |
| B | Source condition | Target-specific threshold | Target threshold recalibration |
| C | Target condition | Target-specific threshold | Target-adapted detector and threshold |

Method B keeps the source-fitted detector unchanged and recalibrates only its threshold using target-condition calibration data.

Method C fits the detector using target-condition reference data and calibrates its threshold using target-condition calibration data. It is a target-adapted approach, not zero-shot transfer.

The three methods were evaluated on the same control and drift windows for each seed and target condition, enabling paired comparisons.

### Original Frozen-Threshold Findings

Same-condition calibration generally produced false-alarm rates near the nominal 5% target. Cross-condition transfer sometimes caused substantial false-alarm inflation or suppression.

Selected results from the original frozen-threshold evaluation:

| Source → Target | Euclidean FPR | Mahalanobis FPR |
|---|---:|---:|
| Anisotropic → Isotropic | 0.02% | 34.08% |
| Correlated → Isotropic | 0.84% | 96.67% |
| Correlated → Anisotropic | 32.84% | 99.50% |
| Anisotropic → Correlated | 0.28% | 35.38% |

These results demonstrate that a frozen threshold's behaviour depends on the relationship between the calibration and target distributions.

A high detection rate alone is not evidence of useful drift discrimination when the false-alarm rate is also high.

### Three-Method Comparative Findings

The shared-window benchmark showed that target-specific recalibration substantially improved false-alarm control in several cross-condition scenarios.

Selected comparative results:

| Source → Target | Detector | Method A FPR | Method B FPR | Method C FPR |
|---|---|---:|---:|---:|
| Anisotropic → Correlated | Euclidean | 0.26% | 5.29% | 5.15% |
| Anisotropic → Correlated | Mahalanobis | 35.52% | 5.04% | 5.57% |
| Correlated → Anisotropic | Mahalanobis | 99.59% | 5.91% | 4.95% |
| Correlated → Isotropic | Mahalanobis | 96.31% | 5.55% | 5.70% |
| Isotropic → Anisotropic | Euclidean | 55.03% | 5.31% | 5.05% |

Method A sometimes produced extremely high false-alarm rates under cross-condition transfer.

Methods B and C brought false-alarm rates much closer to the nominal 5% target in these examples.

However, target recalibration did not produce uniform improvements in detection sensitivity.

For example, in the anisotropic-to-correlated Mahalanobis scenario:

- Method A detection rate: 46.13%.
- Method B detection rate: 7.69%.
- Method C detection rate: 22.01%.

The higher detection rate of Method A must be interpreted alongside its 35.52% false-alarm rate. It does not demonstrate better drift discrimination.

### Statistical Analysis

The comparative statistical analysis evaluated:

- False-alarm-rate differences.
- Detection-rate differences.
- Method A versus Method B.
- Method A versus Method C.
- Method B versus Method C.

Paired comparisons were conducted across 20 independent random seeds.

The analysis used:

- Wilcoxon signed-rank tests.
- 95% bootstrap confidence intervals for paired mean differences, using 10,000 resamples.
- Holm multiple-comparison correction, applied separately within each metric family.

A total of 108 pairwise statistical comparisons were performed:

- 54 false-alarm-rate comparisons.
- 54 detection-rate comparisons.
- 20 paired seed-level observations per comparison.

**51 of the 108 comparisons were statistically significant after Holm correction.**

This count includes both same-condition and cross-condition comparisons. Same-condition methods produced identical results in the tested design.

The correction was applied separately to the false-alarm and detection-rate families, not globally across all 108 comparisons.

Statistical significance does not automatically establish practical superiority. The results must be interpreted alongside false-alarm control, detection sensitivity, and the experimental limitations.

### Integrity and Reproducibility

The original experiment integrity audit passed all 15 programmed checks.

The comparative benchmark additionally verified:

- 1,080 control evaluation records.
- 1,080 drift evaluation records.
- 54 aggregated comparative result groups.
- 108 paired statistical comparisons.
- Finite evaluation rates and valid rate ranges.
- Complete comparative evaluation groups.

The shared evaluation design ensured that Methods A, B, and C were assessed using the same independent control and drift windows for each seed and target condition.

These checks establish structural and numerical consistency of the recorded outputs. They do not independently establish scientific validity or real-world generalisability.

### Primary Output Files

The Experiment 16 results are stored in the repository's `results/` directory.

| File | Description |
|---|---|
| `frozen_thresholds.csv` | Original source-condition calibration thresholds |
| `comparative_control_evaluation.csv` | Comparative independent control evaluation |
| `comparative_drift_evaluation.csv` | Comparative drift evaluation |
| `paired_statistical_comparisons.csv` | Paired statistical comparisons and corrected p-values |
| `experiment_16_integrity_audit.csv` | Original experiment integrity audit |

The complete experiment outputs are also retained in the downloaded `experiment_16_results.zip` archive.

### Limitations

- The experiment uses synthetic multivariate Gaussian embeddings rather than real biometric embeddings.
- Drift is introduced through controlled mean shifts.
- Only three covariance conditions and two detector types are evaluated.
- Results depend on the selected covariance structures, random seeds, calibration procedure, and shift magnitude.
- Method B and Method C require target-condition calibration data.
- Method C additionally requires target-condition reference data to fit the detector.
- Bootstrap confidence intervals are based on 20 seed-level paired differences.
- Holm correction was applied separately by metric family rather than across all 108 comparisons.
- The experiment does not establish real-world biometric recognition accuracy, demographic fairness, successful bias mitigation, or production deployment readiness.

### Conclusion

Experiment 16 demonstrates that calibration thresholds may not transfer reliably between embedding distributions with different covariance structures.

The comparative benchmark shows that target-condition threshold recalibration can substantially improve false-alarm control in several cross-condition scenarios. Target-adapted detector fitting also provides an alternative, although it requires additional target-condition data and does not consistently outperform threshold recalibration.

The findings highlight the importance of evaluating false-alarm control and detection sensitivity together rather than ranking detectors using detection rates alone.

**Overall status:** Synthetic cross-condition calibration-transfer experiment completed; three-method comparative evaluation completed; paired statistical analysis completed; integrity checks passed; real-world biometric validation remains outstanding.

# Experiment 17: Target-Condition Recalibration for Embedding-Level Drift Detection

## Overview

This experiment evaluates whether recalibrating detection thresholds using target-condition calibration data improves false-alarm control and drift detection when embedding distributions change.

The experiment compares source-condition threshold calibration against target-condition threshold recalibration while keeping the source-fitted detector fixed. It evaluates Euclidean and Mahalanobis distance detectors across isotropic, anisotropic, and correlated synthetic embedding distributions.

## Research Objective

To investigate how target-condition threshold recalibration affects:

- False-alarm rates under no-drift conditions.
- Drift detection rates under a controlled distribution shift.
- Performance across different source and target distribution conditions.
- Sensitivity to the amount of target-condition calibration data.

## Experimental Configuration

| Parameter | Value |
|---|---|
| Experiment ID | 17 |
| Random seeds | 20 |
| Embedding dimension | 64 |
| Window size | 20 |
| Reference samples | 1,000 |
| Source calibration windows | 200 |
| Target calibration sample sizes | 25, 50, 100, 200, 500 windows |
| Independent control windows | 500 |
| Evaluation windows per condition | 500 |
| Target false-alarm rate | 0.05 |
| Mean-shift strength | 0.5 |
| Detectors | Euclidean, Mahalanobis |
| Distribution conditions | Isotropic, anisotropic, correlated |

## Methodology

### 1. Synthetic embedding distributions

Three covariance conditions are used:

- **Isotropic:** identity covariance.
- **Anisotropic:** diagonal covariance with variances spanning 0.25 to 4.
- **Correlated:** Toeplitz covariance with correlation parameter 0.6.

### 2. Source-condition detector fitting

Reference data are generated for each random seed and condition. Euclidean and regularized Mahalanobis detector configurations are fitted using source-condition reference data.

### 3. Source threshold calibration

Source thresholds are calibrated using 200 source-condition no-drift windows at a nominal false-alarm rate of 5%.

### 4. Target-condition recalibration

For each source-target condition pair, the fitted source detector is kept fixed. Only its threshold is recalibrated using target-condition no-drift calibration data at five sample sizes: 25, 50, 100, 200, and 500 windows.

This design evaluates threshold recalibration rather than refitting the detector to the target distribution.

### 5. Independent evaluation

Independent target-condition no-drift windows are used to estimate false-alarm rates. Separate drifted evaluation windows are used to estimate detection rates under a mean shift of 0.5 in the first embedding dimension.

### 6. Paired statistical analysis

Recalibrated results are compared with source-threshold results across 20 random seeds. Mean paired changes and unadjusted 95% t-confidence intervals are reported for false-alarm and detection rates.

## Results Summary

The results show that target-condition recalibration has condition-dependent effects rather than providing uniform improvement.

- Recalibration can reduce false alarms when the source threshold is poorly matched to the target distribution.
- Recalibration can also increase false alarms, particularly with limited target calibration data.
- Changes in detection rate do not always follow changes in false-alarm rate.
- In some scenarios, improved false-alarm control is accompanied by reduced drift detection.
- The observed effects vary with detector type, source-target condition pair, and calibration sample size.

The paired confidence intervals quantify uncertainty across simulated seeds. Since 90 scenario comparisons are reported and the intervals are not adjusted for multiple comparisons, individual results should be interpreted as exploratory.

Consult `experiment_17_aggregated_summary.csv` and `paired_recalibration_confidence_intervals.csv` for the numerical results.

## Integrity and Reproducibility

The experiment's recorded integrity audit passed 15 of 15 checks. These checks cover expected data structures, finite values, valid rates, matched evaluation records, and aggregated results.

Passing these checks confirms numerical and structural consistency; it does not independently establish scientific validity.

The reproducibility manifest records the experiment configuration and package information. The output archive contains the experiment's recorded CSV and JSON result files.

## Output Files

| File | Description |
|---|---|
| `experiment_17_config.json` | Experiment configuration |
| `source_calibration_thresholds.csv` | Source-condition calibrated thresholds |
| `target_recalibrated_thresholds.csv` | Target-condition recalibrated thresholds |
| `independent_control_fpr_results.csv` | Independent no-drift false-alarm results |
| `independent_drift_detection_results.csv` | Independent drift-detection results |
| `joint_recalibration_comparison.csv` | Combined paired recalibration comparisons |
| `experiment_17_aggregated_summary.csv` | Aggregated scenario-level metrics |
| `paired_recalibration_confidence_intervals.csv` | Paired mean changes and 95% confidence intervals |
| `experiment_17_integrity_audit.csv` | Recorded integrity checks |
| `experiment_17_manifest.json` | Reproducibility metadata |

## Limitations

- The data are synthetic and do not establish performance on real biometric datasets.
- Only three covariance conditions and one specified mean-shift strength are evaluated.
- Target recalibration assumes access to representative target-condition no-drift calibration data.
- Small calibration samples can produce unstable empirical thresholds.
- Confidence intervals are unadjusted for multiple comparisons.
- Numerical integrity checks do not establish robustness, fairness, real-world biometric accuracy, or deployment readiness.

## Conclusion

Experiment 17 evaluates the trade-off between false-alarm control and drift detection when recalibrating thresholds for a fixed source-fitted detector. The findings indicate that recalibration should be evaluated jointly on false-alarm and detection rates, across relevant condition pairs and calibration sample sizes, rather than treated as a universally beneficial operation.

**Status:** Synthetic experimental evaluation completed; real-world validation has not been established.

## Experiment 18: Controlled Illumination Drift Analysis Using FaceNet Embeddings

### Objective

Experiment 18 investigates how controlled illumination changes affect face embeddings and embedding-level drift detector responses using the Labeled Faces in the Wild (LFW) dataset.

The experiment evaluates Euclidean and Mahalanobis distance-based detectors under low- and high-illumination conditions. It also examines historical paired illumination displacement across identities.

### Dataset and Evaluation Cohorts

The experiment uses three distinct data cohorts for different analytical purposes:

| Cohort | Images / Embeddings | Dimensions | Purpose |
|---|---:|---:|---|
| Full original evaluation archive | 956 | — | Original evaluation archive |
| Historical paired illumination analysis | 889 | 512 | Illumination displacement analysis |
| Corrected identity-level detector evaluation | 229 | 512 | Paired detector-response analysis |

The historical displacement analysis and corrected detector evaluation must be interpreted separately. Both involve 13 identities, but they use different image cohorts.

### Methodology

- Use FaceNet embeddings to represent LFW face images.
- Examine embedding displacement under controlled low- and high-illumination conditions.
- Evaluate Euclidean and Mahalanobis detector responses.
- Perform paired identity-level statistical comparisons.
- Independently recalculate displacement statistics and verify saved experimental artifacts.
- Conduct data-integrity and numerical consistency audits.

### Corrected Detector Evaluation: 229-Image Cohort

The corrected identity-level evaluation compares paired detector responses under low- and high-illumination conditions.

| Metric | Euclidean | Mahalanobis |
|---|---:|---:|
| Low-illumination mean | 0.126002 | 2.389304 |
| High-illumination mean | 0.156053 | 3.111986 |
| Mean paired difference (High − Low) | 0.030051 | 0.722682 |
| Wilcoxon signed-rank p-value | 0.021484 | 0.002441 |
| Positive identities | 10 | 11 |
| Negative identities | 3 | 2 |

Both detectors showed higher mean responses under high illumination in this corrected evaluation.

The paired statistical comparisons are exploratory and based on only 13 identities.

### Historical Illumination Displacement Analysis: 889-Image Cohort

The separate historical analysis examined paired embedding displacement across 889 images.

| Metric | Result |
|---|---:|
| Low-illumination mean displacement | 0.234665 |
| High-illumination mean displacement | 0.318514 |
| Mean difference (High − Low) | 0.083849 |
| Image-level Wilcoxon p-value | 4.708938 × 10⁻⁸⁰ |
| Identity-level mean displacement p-value | 0.001221 |
| Identities with positive displacement difference | 11 |
| Identities with negative displacement difference | 2 |

The 889 image observations are clustered within 13 identities. They must not be interpreted as 889 independent subjects. Identity 14 contributes 504 of the 889 images, making identity-level interpretation particularly important.

### Statistical Interpretation

The corrected detector evaluation found statistically significant paired differences for both Euclidean and Mahalanobis responses at the nominal 0.05 level.

The historical displacement analysis also found higher displacement under high illumination.

These findings indicate sensitivity of the evaluated embedding representations and detector scores to the controlled illumination conditions. They do not demonstrate that the detectors can reliably identify real-world biometric failures.

### Reproducibility and Integrity Verification

The final research package includes:

- FaceNet embedding data.
- Illumination evaluation results.
- Identity-level summary statistics.
- Illumination metadata.
- Consolidated Markdown research evidence.
- Structured JSON research evidence.
- A ZIP archive containing the research artifacts.

The recorded validation checks covered saved data consistency, independent displacement recalculation, statistical result reproduction, and artifact integrity.

These checks establish consistency of the recorded outputs; they do not independently establish complete end-to-end reproduction of the image-processing and embedding-extraction pipeline.

### Research Artifacts

**Notebook:**

[Experiment_18_Genuine_Face_Recognition_Embedding_Validation.ipynb](experiments/Experiment_18_Genuine_Face_Recognition_Embedding_Validation.ipynb)

**Results:**

- [`experiment18_facenet_lfw_embeddings.npz`](results/experiment18_facenet_lfw_embeddings.npz)
- [`experiment18_illumination_results.npz`](results/experiment18_illumination_results.npz)
- [`illumination_identity_summary.csv`](results/illumination_identity_summary.csv)
- [`illumination_metadata.json`](results/illumination_metadata.json)
- [`experiment18_final_research_evidence.md`](results/experiment18_final_research_evidence.md)
- [`experiment18_final_research_evidence.json`](results/experiment18_final_research_evidence.json)
- [`Experiment_18_Final_Research_Package.zip`](results/Experiment_18_Final_Research_Package.zip)

### Limitations

- The corrected detector evaluation covers only 13 identities.
- The historical displacement analysis contains repeated observations from the same identities.
- Identity 14 contributes a substantial proportion of the historical image cohort.
- The illumination transformations are controlled experimental conditions, not naturally occurring environmental changes.
- Statistical significance does not establish generalisation to unseen identities.
- The experiment does not establish real-world biometric recognition accuracy, fairness, successful bias mitigation, or deployment readiness.

### Conclusion

Experiment 18 provides an exploratory evaluation of illumination-related embedding displacement and Euclidean and Mahalanobis detector responses.

The corrected 229-image detector evaluation and historical 889-image displacement analysis provide separate evidence of sensitivity to the tested illumination conditions.

Further evaluation across more identities, independent datasets, and naturally occurring acquisition conditions is required.

**Overall status:** Controlled illumination experiment completed; statistical analyses and saved-artifact verification completed; real-world biometric validation remains outstanding.


## Experiment 19: Disentangling Identity-Composition Drift from Within-Identity Embedding Drift in Biometric Representation Systems

### Objective

Experiment 19 investigates the distinction between identity-composition drift and within-identity embedding drift in biometric representation systems.

The experiment examines how changes in the proportions of represented identities and changes in embedding distributions within individual identities affect global embedding distributions and drift-monitoring methods.

The study uses controlled scenarios constructed from an LFW-derived FaceNet embedding subset.

### Research Questions

The experiment addresses four research questions:

1. How do changes in identity composition affect global embedding distributions?
2. Can within-identity embedding distribution changes be detected independently of identity-composition changes?
3. How effectively can global and identity-conditioned monitoring distinguish these mechanisms?
4. How do detection performance and false alarms change when both mechanisms occur simultaneously?

### Experimental Design

Four controlled scenarios were constructed:

| Scenario | Description |
|---|---|
| C0 — No drift | Identity composition and conditional distributions remain stable. |
| C1 — Composition drift | Identity proportions change while original embeddings are used. |
| C2 — Within-identity drift | Identity proportions remain stable while low-illumination embedding variants are introduced. |
| C3 — Combined drift | Identity proportions change and low-illumination embedding variants are introduced. |

Each scenario contains 440 observations:

- 220 baseline observations.
- 220 post-change observations.
- 22 windows per scenario.
- Window size of 20 observations.

The scenarios share the same baseline observations.

Bootstrap sampling with replacement was used to construct the controlled scenarios because of substantial identity imbalance in the source cohort.

### Dataset and Embeddings

The experiment uses a filtered LFW-derived biometric embedding subset.

| Property | Value |
|---|---:|
| Unique identities | 13 |
| Embedding dimension | 512 |
| Original paired observations | 889 |
| Embedding model | InceptionResnetV1 |
| Pretrained weights | VGGFace2 |
| Within-identity perturbation | Low-illumination embedding variants |

Identity 14 accounts for approximately 56.69% of the original paired cohort.

The bootstrap-generated observations are not independent images. This limitation is considered when interpreting the results.

### Drift Monitoring Methods

Three monitoring approaches were evaluated:

1. **Global Euclidean detector:** Measures normalized Euclidean distance from the global reference centroid.
2. **Global Mahalanobis detector:** Measures covariance-adjusted distance from the global reference distribution, using covariance shrinkage of 0.10.
3. **Identity-conditioned detector:** Measures standardized Euclidean displacement relative to identity-specific reference centroids and dispersion.

The identity-conditioned approach uses identity labels. It should not be interpreted as a label-free operational monitoring method.

### Calibration and Evaluation

The reference observations were divided into model-fitting and calibration partitions.

| Partition | Observations |
|---|---:|
| Model fitting | 261 |
| Calibration | 181 |
| Evaluation | 447 |

The model-fitting and calibration partitions were disjoint.

Calibration thresholds were estimated using bootstrap windows of calibration scores. The thresholds are sample-dependent and should not be interpreted as operationally validated false-positive rates.

### Main Results

#### 1. Identity-composition drift

At the 5% calibration threshold:

| Detector | C1: Composition drift | C3: Combined drift |
|---|---:|---:|
| Global Euclidean | 11/11 | 11/11 |
| Global Mahalanobis | 11/11 | 11/11 |

Both global detectors detected all 11 post-change windows in the composition-drift and combined-drift scenarios.

This indicates strong sensitivity to the constructed identity-composition changes.

#### 2. Within-identity drift

For C2, the within-identity drift scenario:

| Detector | Detected post-change windows |
|---|---:|
| Global Euclidean | 1/11 |
| Global Mahalanobis | 0/11 |
| Identity-conditioned | 1/11 |

The low-illumination condition produced measurable identity-level embedding displacement, although window-level detection was weak.

#### 3. Combined drift

For C3 at the 5% calibration threshold:

| Detector | Detected post-change windows |
|---|---:|
| Global Euclidean | 11/11 |
| Global Mahalanobis | 11/11 |
| Identity-conditioned | 7/11 |

The combined scenario was strongly detected by global monitoring, primarily in the presence of the constructed composition change.

#### 4. Identity-level statistical analysis

Paired Wilcoxon signed-rank tests were performed across 13 identities, with Holm correction across four comparisons.

| Comparison | Mean paired difference | Holm-adjusted p-value | Significant at 0.05 |
|---|---:|---:|---|
| C2 vs C0 | 0.091008 | 0.001953 | Yes |
| C3 vs C1 | 0.036232 | 0.064453 | No |
| C1 vs C0 | 0.011150 | 0.735352 | No |
| C3 vs C2 | -0.043626 | 0.487793 | No |

The C2 versus C0 comparison provides evidence of increased identity-level displacement under the within-identity drift condition.

The C3 versus C1 comparison did not remain statistically significant after Holm correction.

#### 5. Drift attribution

The global mean-shift decomposition was:

\[
\Delta\mu =
\sum_i(p_{1,i}-p_{0,i})\mu_{0,i}
+
\sum_i p_{1,i}(\mu_{1,i}-\mu_{0,i})
\]

The two terms represent the composition component and within-identity component, respectively.

The maximum numerical reconstruction error was approximately \(8.85\times10^{-17}\).

This confirms numerical consistency of the decomposition, not causal isolation of the underlying mechanisms.

### Statistical Interpretation

The findings demonstrate that identity-composition changes and within-identity embedding displacement can affect global embedding distributions differently.

The composition-drift scenarios produced strong detection responses from both global detectors. In contrast, within-identity drift produced measurable identity-level displacement but weak window-level detection.

The identity-level statistical comparison identified a significant difference between C2 and C0 after Holm correction. However, the C3 versus C1 comparison was not statistically significant after correction.

These findings highlight the limitations of aggregate global monitoring when attempting to distinguish different sources of biometric embedding drift.

### Reproducibility and Integrity Verification

The experiment includes:

- Environment and configuration records.
- Sampling and scenario-generation protocols.
- Detector configuration and calibration thresholds.
- Scenario validation outputs.
- Detection and attribution results.
- Identity-level statistical results.
- SHA-256 reproducibility manifest.
- Consolidated research evidence.

The independent artifact audit passed all 55 checks.

These checks support the integrity and consistency of the recorded research artifacts. They do not establish independent reproduction of the complete experimental pipeline or generalisation to external datasets.

### Research Artifacts

**Visual results:**

- [Figure 1 — Drift Attribution](plots/experiment19_figure1_drift_attribution.png)
- [Figure 2 — Identity Displacement](plots/experiment19_figure2_identity_displacement.png)
- [Figure 3 — Statistical Significance](plots/experiment19_figure3_statistical_significance.png)

The experiment also includes sampling protocols, detector configurations, calibration thresholds, statistical results, scenario validation outputs, and reproducibility documentation.

### Limitations

- The experiment uses only 13 identities from a filtered LFW-derived subset.
- The identity distribution is substantially imbalanced.
- Bootstrap sampling with replacement creates repeated observations.
- Each scenario contains only 22 evaluation windows.
- Calibration is based on 181 observations.
- Identity-conditioned monitoring requires identity labels.
- Low illumination is the specific within-identity perturbation studied.
- Post-change samples are independently resampled rather than strictly paired image observations.
- Statistical inference is based on only 13 identity-level paired observations.
- The experiment does not establish real-world biometric recognition accuracy, operational fairness, or deployment readiness.

### Interpretation Boundary

The findings apply to the controlled LFW-derived embedding experiment.

They do not establish:

- Label-free operational monitoring.
- Generalisation across biometric datasets or recognition models.
- Causal isolation of drift mechanisms.
- Real-world biometric recognition performance.
- Deployment readiness.

### Data Availability

The experiment uses biometric-derived embeddings and identity labels. Raw embeddings and identity-level data should not be publicly redistributed without verifying the source dataset's licensing, applicable ethical conditions, and data-sharing permissions.

Aggregated results, figures, methodological descriptions, and reproducibility documentation can be provided separately.

### Conclusion

Experiment 19 provides controlled evidence that identity-composition changes and within-identity embedding displacement can affect global embedding distributions differently.

The global Euclidean and Mahalanobis detectors showed strong sensitivity to the constructed composition-drift scenarios, while detection of within-identity drift was substantially weaker.

The identity-level statistical analysis identified increased displacement in the within-identity drift scenario compared with the no-drift scenario. However, the combined-drift comparison did not remain statistically significant after Holm correction.

The results demonstrate the importance of distinguishing identity-composition changes from within-identity representation changes when designing biometric drift-monitoring systems.

Further evaluation across larger identity cohorts, independent biometric datasets, and additional acquisition conditions is required.

**Overall status:** Experiment completed; controlled drift scenarios evaluated; statistical analysis and 55 artifact integrity checks completed; external generalisation and operational validation remain outstanding.


## Experiment 20: Paired Identity-Level Evaluation of Within-Identity Embedding Drift Detection Under Multiple Controlled Biometric Acquisition Changes

### Objective

Experiment 20 investigates reference-relative face embedding displacement across illumination geometry conditions using the Cropped Extended Yale B dataset.

The experiment evaluates within-identity embedding displacement across multiple biometric acquisition conditions using a paired identity-level evaluation design.

### Dataset and Evaluation Cohort

The experiment uses the Cropped Extended Yale B dataset with the following recorded characteristics:

| Property | Value |
|---|---:|
| Dataset | Cropped Extended Yale B |
| Identities | 38 |
| Total images | 2,414 |
| Common acquisition conditions | 57 |
| Non-reference observations | 2,128 |
| Non-reference conditions | 56 |
| Embedding dimension | 512 |
| Embedding model | FaceNet InceptionResnetV1 VGGFace2 |

The evaluation examines non-reference observations relative to identity-specific reference captures across the available illumination conditions.

### Methodology

- Use the Cropped Extended Yale B dataset for identity-level biometric evaluation.
- Generate 512-dimensional face embeddings using FaceNet InceptionResnetV1 VGGFace2.
- Evaluate reference-relative embedding displacement across illumination geometry conditions.
- Compare within-identity observations across multiple acquisition conditions.
- Apply paired identity-level statistical analysis.
- Evaluate statistical differences using the Friedman test and Holm-adjusted pairwise comparisons.

### Statistical Results

The experiment evaluated differences across the illumination conditions using the Friedman test.

| Statistical measure | Result |
|---|---:|
| Friedman chi-square statistic | 76.0 |
| Friedman test p-value | 3.139 × 10⁻¹⁷ |
| Kendall's coefficient of concordance (W) | 1.0 |
| Holm-adjusted pairwise comparisons | Remained significant |

The Friedman test identified statistically significant differences across the evaluated conditions.

The Kendall's W value indicates complete concordance in the observed condition rankings within this evaluation. These findings describe the evaluated dataset and experimental design; they do not establish generalisation to other biometric datasets or systems.

### Statistical Interpretation

The results indicate systematic differences in reference-relative embedding displacement across the evaluated illumination geometry conditions.

The statistically significant Friedman result and Holm-adjusted pairwise comparisons provide evidence of differences among the tested conditions.

However, statistical significance does not independently establish biometric recognition failure, reliable operational drift detection, or successful detection of naturally occurring acquisition changes.

### Reproducibility and Integrity Verification

The experiment's research package includes:

- Frozen source dataset archive.
- Face embedding artifacts.
- Experimental configuration records.
- Dataset and embedding audit records.
- SHA-256 checksum manifest.
- Research figures.
- Integrated research manuscript.
- Supporting audit and administrative closure records.

The Phase 4.7 manuscript audit verified the conclusion insertion but reported one structural check requiring review. Therefore, the overall automated manuscript audit must not be described as fully passed.

The Phase 5.5 project closure record documents administrative closure. It does not establish independent scientific validation.

### Research Artifacts

**Notebook:**

[Experiment 20 Notebook](notebook/Experiment_20.ipynb)

**Final research manuscript:**

[Phase 4.7 Integrated Manuscript](manuscript/experiment20_integrated_manuscript_v2_phase47_final.md)

**Research figures:**

[Figures directory](figures/)

The figures directory contains four frozen research figures.

**Supporting records:**

[Supporting audits and closure records](supporting_records/)

**Dataset package:**

The frozen dataset package contains the source archive, embeddings, configuration, audits, and checksum manifest.

The dataset ZIP is maintained separately because of its size and is not included in this repository.

### Limitations

- The acquisition captures are not controlled transformations.
- The dataset archive has not been authenticated against the official UCSD checksum.
- Exact model-weight checksum preservation is incomplete.
- One reference capture per identity was used.
- Only one embedding architecture was evaluated.
- The study does not establish biometric recognition accuracy.
- The experiment does not demonstrate operational drift detection.
- The results do not establish deployment readiness or generalisation to independent biometric datasets.

### Conclusion

Experiment 20 provides an identity-level evaluation of reference-relative face embedding displacement across illumination geometry conditions using the Cropped Extended Yale B dataset.

The Friedman test identified statistically significant differences across the evaluated conditions, with a chi-square statistic of 76.0, a p-value of 3.139 × 10⁻¹⁷, and Kendall's W of 1.0. Holm-adjusted pairwise comparisons also remained significant.

These findings provide evidence of systematic embedding displacement differences across the tested acquisition conditions. However, the study does not establish real-world biometric recognition accuracy, operational drift detection, or deployment readiness.

Further evaluation using independently authenticated datasets, multiple embedding architectures, and additional reference captures is required.

**Overall status:** Experiment completed and administratively closed; statistical results and research artifacts documented; independent scientific validation and operational drift detection remain outstanding.


## Author

Moien Shaik

Research interests: trustworthy machine learning, distributional drift detection, biometric systems, out-of-distribution detection, and cybersecurity.

## License

This repository includes a `LICENSE` file. Review its terms before reusing or distributing the code. Third-party dependencies, datasets, and model components remain subject to their respective licenses.
