# Real-Time Embedding-Level Drift Detection for Trustworthy Biometric Systems

A research project investigating label-free distributional drift detection in embedding spaces, with the long-term goal of improving the reliability and trustworthiness of biometric identification systems under changing operating conditions.

**Current status:** Research prototype with preliminary synthetic evaluations and exploratory controlled LFW/FaceNet embedding experiments. The real-image pilot does not establish biometric recognition performance, dependable operational false-alarm control, fairness, or deployment readiness. Full numerical reproducibility remains incomplete because the original reference-model arrays and Stage 3 bootstrap CSVs were not recovered during the documented audit.

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

## Experiment 12: Cross-Condition Generalisation of Embedding-Level Drift Detectors

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

| Condition | Replicates | Pre-onset alarm rate | Post-onset detection rate |
|---|---:|---:|---:|
| Control | 20 | 10% | Not applicable |
| Brightness 0.70 | 20 | 10% | 40% |
| Brightness 1.30 | 20 | 10% | 55% |

The two pre-onset alarms occurred in the same resampling replicates across the control and transformed conditions. This indicates that stream variability can trigger alarms before the imposed brightness change.

The resampled streams overlap and are not independent datasets. These detection rates are exploratory and do not establish reliable deployment performance.

Reports and results:

- Sequential pilot report: [`Experiment_13_Phase_5_Sequential_Pilot_Report.md`](results/Experiment_13_Phase_5_Sequential_Pilot_Report.md)
- Repeated-stream report: [`Experiment_13_Phase_5B_Repeated_Stream_Report.md`](results/Experiment_13_Phase_5B_Repeated_Stream_Report.md)
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

These late alarms must not automatically be interpreted as successful drift detections. The unchanged control also produces late alarms, and eight of 20 replicates have pre-onset alarms in each condition.

#### Paired first-alarm comparison

| Condition | Both streams alarm | Transformed stream alarms earlier | Same first-alarm window | Transformed stream alarms later |
|---|---:|---:|---:|---:|
| Brightness 0.70 vs control | 16 | 6 | 10 | 0 |
| Brightness 1.30 vs control | 16 | 8 | 8 | 0 |

Among replicates where both streams alarmed, the transformed stream never had a later first alarm than its paired control in this run. However, this comparison does not eliminate the false-alarm problem: pre-onset alarms and late alarms on the unchanged control indicate that normal stream variability can trigger the detector.

The bootstrap replicates reuse the same underlying image pool and are not independent datasets. These findings measure sensitivity to resampling rather than generalisation to new identities or external datasets.

#### Bootstrap artifact availability

The following two files were not found at the checked GitHub paths or in the checked Google Drive locations during the reproducibility audit:

- `lfw_stage3_bootstrap_cusum_results.csv`
- `lfw_stage3_bootstrap_cusum_trace.csv`

The numerical summaries above reflect previously recorded outputs, but the raw bootstrap CSVs were not independently recovered. They should not be treated as independently reproducible from the currently available artifacts alone.

The paired comparison report is also not linked here because its current GitHub availability has not been verified.

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

The bootstrap result and trace CSVs were also not recovered from the checked GitHub paths and Google Drive locations. The fresh-runtime smoke test verifies selected prerequisites, not the complete end-to-end experiment.

Environment and model configuration records:

- [`Experiment_13_requirements_observed.txt`](results/Experiment_13_requirements_observed.txt)
- [`Experiment_13_FaceNet_Model_Configuration.txt`](results/Experiment_13_FaceNet_Model_Configuration.txt)
- [`Experiment_13_LFW_Dataset_Configuration.txt`](results/Experiment_13_LFW_Dataset_Configuration.txt)

### Limitations

These experiments use synthetic distributions and controlled transformations of static face images, not naturally occurring temporal biometric drift. Bootstrap/resampling replicates overlap and are not independent observations. Stage 3 uses one stream per condition, and the repeated-resampling control results show substantial late alarms and pre-onset alarms.

The detector has not demonstrated dependable operational false-alarm control. The experiments do not establish real-world biometric recognition accuracy, demographic fairness, successful bias mitigation, cyberattack detection, or deployment readiness. The CUSUM configuration is provisional and does not guarantee a specified operational false-alarm rate.

The fresh-identity evaluation uses the same LFW dataset and evaluation partition rather than an independent external dataset. The original reference centre and precision matrix have not been recovered, so exact score reproduction remains unverified. The Stage 3 bootstrap CSVs also remain unavailable at the locations checked during the audit.

Raw face images, embedding arrays, and identity-level metadata should not be published without appropriate licensing, privacy, and consent review.

### Conclusion

Experiment 13 investigates false-alarm-controlled embedding-level drift detection through synthetic evaluations and an exploratory LFW FaceNet pilot. The fresh-identity evaluation recorded a threshold crossing under the stronger controlled brightness transformation, while the repeated-stream analyses exposed important pre-onset and control-condition alarm concerns.

The evidence supports continued methodological investigation, not a claim of reliable real-world biometric drift detection. The frozen detector parameters and Stage 3 fresh-identity results are available in the repository, but exact reproduction of the original Mahalanobis scores remains unverified because the reference-model artifacts were not located. The Stage 3 bootstrap CSVs also remain unavailable at the checked locations.

**Overall status:** exploratory experiment completed; selected outputs verified; full numerical reproducibility remains incomplete.

## Experiment 14: Drift Detection Under Gradual and Recurring Distribution Shifts

### Objective

Evaluate the behaviour of embedding-level drift detectors under abrupt, gradual, and recurring distribution shifts. The experiment examines detection rate, detection delay, stable-period false alarms, and agreement between Euclidean and Mahalanobis detectors.

### Experimental Design

- **Drift patterns:** abrupt, gradual, and recurring.
- **Detectors:** Euclidean and Mahalanobis.
- **Evaluation:** synthetic embedding distributions with controlled temporal drift patterns.
- **Calibration:** frozen detector thresholds.
- **Additional analysis:** independent control evaluation, recurring-shift gap analysis, and numerical-stability diagnostics.

### Key Results

| Metric | Result |
|---|---:|
| Detector alarm-decision agreement | 100% across 3,600 comparisons |
| Abrupt-shift detection rate | 100% |
| Gradual-shift detection rate | 95.5% |
| Recurring-shift detection rate | 100% |
| Mean gradual-shift detection delay | 1.45 windows |
| Independent control false-alarm rate | 4.97% |
| Independent control false-alarm standard deviation | 0.78 percentage points |

The independent control evaluation recorded 497 alarms per detector across 10,000 control windows, corresponding to a pooled false-alarm rate of 4.97%.

### Recurring-Shift Analysis

Recurring-shift gap analysis showed a mean alarm rate of 6.0% during recurring-shift gaps, compared with 3.5% in the final baseline period. Nine of 20 seeds had no alarms during the recurring-shift gaps, while 13 of 20 seeds had no alarms during the final baseline period.

These findings indicate that false alarms can persist during nominally stable periods and that alarm behaviour varies across random seeds.

### Main Findings

1. Both detectors produced matching alarm decisions in the reported 3,600 comparisons.
2. Abrupt and recurring shifts were detected at a 100% rate in the evaluated synthetic scenarios.
3. Gradual shifts were detected at a 95.5% rate, with a mean detection delay of 1.45 windows.
4. Stable-period false alarms remained present despite frozen calibration thresholds.
5. Recurring-shift results require careful interpretation because alarm persistence can affect the episode-level detection-delay definition.

### Interpretation and Limitations

The results provide preliminary evidence about detector behaviour under controlled synthetic temporal shifts. Matching alarm decisions do not establish that either detector is universally superior or that their decisions generalize to real biometric data.

The results do not establish real-world biometric recognition accuracy, fairness, or deployment readiness. Further evaluation on real embedding datasets and additional drift conditions is required.

### Reproducibility

The experiment used frozen thresholds and independent control evaluation. A numerical-stability diagnostic was also performed on the regularized covariance matrices used by the Mahalanobis detectors.

The evaluation remains synthetic and should be interpreted within its experimental assumptions.

## Experiment 15: Detector Discrimination Under Anisotropic and Correlated Embedding Distributions

### Objective

Evaluate how covariance structure and drift direction affect the behaviour of Euclidean and regularized Mahalanobis embedding-level drift detectors under controlled synthetic conditions.

### Experimental Design

- **Covariance conditions:** isotropic, anisotropic, and correlated.
- **Embedding dimension:** 64.
- **Reference samples:** 1,000 per seed and condition.
- **Independent calibration windows:** 200 per seed and condition.
- **Independent control windows:** 500 per seed and condition.
- **Drift evaluation windows:** 500 per seed, condition, and direction.
- **Independent seeds:** 20.
- **Target false-alarm rate:** 5%.
- **Mean-shift magnitude:** 0.5 along normalized directions.
- **Detectors:** Euclidean and Ledoit–Wolf-regularized Mahalanobis.
- **Calibration:** detector-specific empirical 95th-percentile thresholds, frozen before independent evaluation.

Drift directions were defined using covariance eigenvectors: high-variance, low-variance, and mixed directions. In the isotropic condition, these labels do not represent genuinely different population variances.

### Multi-Seed Independent Control Results

| Covariance condition | Euclidean FPR | Mahalanobis FPR |
|---|---:|---:|
| Isotropic | 4.54% | 4.54% |
| Anisotropic | 4.86% | 4.95% |
| Correlated | 5.23% | 4.69% |

The pooled false-alarm rates were close to the 5% target. However, seed-level rates varied, with observed values ranging from 1.4% to 8.6%. Calibration performance should therefore not be assumed to remain exactly at the target on every independent sample.

### Multi-Seed Drift Detection Results

Mean detection rates across 20 seeds:

| Condition | Direction | Euclidean | Mahalanobis |
|---|---|---:|---:|
| Isotropic | High variance | 10.86% | 10.86% |
| Isotropic | Low variance | 11.19% | 11.22% |
| Isotropic | Mixed | 10.81% | 10.78% |
| Anisotropic | High variance | 7.97% | 6.15% |
| Anisotropic | Low variance | 6.91% | 29.88% |
| Anisotropic | Mixed | 7.30% | 15.87% |
| Correlated | High variance | 9.69% | 6.24% |
| Correlated | Low variance | 7.79% | 39.42% |
| Correlated | Mixed | 8.15% | 19.56% |

### Main Findings

1. Under isotropic conditions, the detectors exhibited nearly identical detection rates.
2. Under anisotropic conditions, Mahalanobis detection was substantially higher for low-variance and mixed-direction drift.
3. Under correlated conditions, Mahalanobis detection was substantially higher for low-variance and mixed-direction drift.
4. Euclidean detection was higher on average for high-variance drift under both anisotropic and correlated conditions.
5. The across-seed 95% confidence intervals for the low-variance and mixed-direction detection-rate differences excluded zero in the anisotropic and correlated conditions. These intervals were not adjusted for multiple comparisons.

### Interpretation and Limitations

The results support the hypothesis that covariance structure and drift direction influence detector behaviour. Mahalanobis distance can be more sensitive to shifts in directions with low reference variance, whereas Euclidean distance does not account for the covariance geometry.

These results do not establish that Mahalanobis is universally superior. Detection rates were modest in most conditions, the evaluation used synthetic Gaussian distributions and mean shifts, and independent false-alarm rates varied across seeds.

The experiment does not establish real-world biometric recognition accuracy, fairness, robustness to all forms of distribution shift, or deployment readiness. Further work should investigate other drift magnitudes, covariance structures, calibration sample sizes, and real embedding datasets.

### Reproducibility and Outputs

The experiment used 20 seeds and independent reference, calibration, control, and evaluation streams. Frozen thresholds were retained during independent control and drift evaluation.

The final integrity audit passed all 15 checks, including expected row counts, finite scores and thresholds, and threshold consistency.

Primary outputs include:

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
- `experiment_15_integrity_audit.csv`

All results are synthetic experimental evidence and should be interpreted within the stated limitations.

## Experiment 16: Cross-Condition Calibration Transfer for Embedding-Level Drift Detection

### Objective

Experiment 16 investigates whether thresholds calibrated on one embedding distribution remain reliable when applied to a different target distribution. It compares Euclidean distance and Ledoit–Wolf-regularized Mahalanobis distance under isotropic, anisotropic, and correlated synthetic embedding distributions.

### Experimental Setup

- **Random seeds:** 20
- **Embedding dimension:** 64
- **Window size:** 20 embeddings
- **Reference samples:** 1,000 per seed and condition
- **Calibration windows:** 200 per seed, source condition, and detector
- **Control windows:** 500 per seed, source-target scenario, and detector
- **Drift evaluation windows:** 500 per seed, source-target scenario, and detector
- **Nominal false-alarm rate:** 5%
- **Source-target scenarios:** 9 (3 same-condition and 6 cross-condition)
- **Detectors:** Euclidean and Ledoit–Wolf-regularized Mahalanobis

### Methodology

The detectors are calibrated on source-condition data, and their thresholds are frozen before evaluation on independent target-condition control and drift windows. Same-condition scenarios provide a baseline for comparison with cross-condition transfers.

Performance is assessed using independent control false-alarm rates, drift detection rates, paired detector comparisons across seeds, and unadjusted 95% confidence intervals.

### Key Findings

Same-condition calibration generally produced false-alarm rates near the nominal 5% target. Cross-condition transfer sometimes caused substantial false-alarm inflation or suppression.

Examples include:

| Source → Target | Euclidean false-alarm rate | Mahalanobis false-alarm rate |
|---|---:|---:|
| Anisotropic → Isotropic | 0.02% | 34.08% |
| Correlated → Isotropic | 0.84% | 96.67% |
| Correlated → Anisotropic | 32.84% | 99.50% |
| Anisotropic → Correlated | 0.28% | 35.38% |

These results demonstrate that a frozen threshold's behavior depends on the relationship between the calibration and target distributions. A high detection rate alone is not evidence of useful drift discrimination when the false-alarm rate is also high.

### Statistical Analysis

Detector detection-rate differences were calculated as Mahalanobis minus Euclidean, paired across the 20 random seeds for each source-target scenario. Confidence intervals were unadjusted for multiple comparisons.

The largest apparent sensitivity advantages for Mahalanobis occurred in several transfer scenarios with substantial false-alarm inflation. These results should therefore be interpreted together with the corresponding control metrics.

### Integrity and Reproducibility

All 15 programmed integrity checks passed. The audit verified expected output counts, finite scores and thresholds, and valid false-alarm and detection-rate ranges.

The notebook, configuration, environment information, calibration outputs, control results, drift evaluation results, statistical comparisons, and integrity audit are retained in the experiment outputs.

### Limitations

This experiment uses synthetic multivariate Gaussian embeddings and controlled mean shifts. It does not establish real-world biometric recognition accuracy, fairness, or deployment readiness. The reported confidence intervals are not adjusted for multiple comparisons.

The findings support further investigation of calibration transfer and threshold reliability; they do not establish that either detector is universally superior.


## Author

Moien Shaik

Research interests: trustworthy machine learning, distributional drift detection, biometric systems, out-of-distribution detection, and cybersecurity.

## License

This repository includes a `LICENSE` file. Review its terms before reusing or distributing the code. Third-party dependencies, datasets, and model components remain subject to their respective licenses.
