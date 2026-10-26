Real-Time Embedding-Level Drift Detection for Trustworthy Biometric Systems

A research project investigating label-free distributional drift detection in embedding spaces, with the long-term goal of improving the reliability and trustworthiness of biometric identification systems under changing operating conditions.

Current status: Synthetic-data prototype with preliminary evaluation. Validation using real biometric embeddings remains future work.

## Overview

Machine learning systems can encounter changes in their input data after deployment. In biometric applications, these changes may arise from variations in illumination, ageing, camera or microphone characteristics, environmental noise, and other operating conditions.

Such changes can alter the feature representations produced by face or voice encoders. Monitoring these representations may provide an early indication that the data distribution has changed, even when ground-truth labels or recognition accuracy measurements are unavailable.

This project explores a lightweight statistical approach that compares reference and current embedding windows to produce a drift score. The current evaluation uses the Euclidean distance between window-level mean representations, with thresholds calibrated using synthetic reference data.

The present implementation evaluates controlled synthetic distribution shifts. It does not yet validate a complete face or voice recognition pipeline.

## Research Objectives

- Investigate whether embedding-level statistics can provide a practical signal for detecting distributional changes without requiring labels.
- Evaluate the relationship between drift magnitude, detection sensitivity, and false-alarm behaviour.
- Examine how detector performance varies across independently generated reference samples.
- Establish a reproducible experimental pipeline that can later be extended to real face and voice embeddings.
- Investigate the limitations of simple statistical drift signals under noise and changing operating conditions.

## Current Implementation

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

Let $R$ denote the reference window and $C$ the current window.

### Drift scoring

The prototype calculates the Euclidean distance between the two window means:

$$
s_t = |\mu_R - \mu_C|_2
$$

where:

- $\mu_R$ is the mean vector of the reference window.
- $\mu_C$ is the mean vector of the current window.
- $s_t$ is the resulting drift score.

### Threshold-based detection

The detector raises an alarm when the drift score exceeds a calibrated threshold:

$$
A_t =
\begin{cases}
1, & s_t > \tau \
0, & s_t \leq \tau
\end{cases}
$$

Here, $\tau$ is estimated from simulated stable calibration windows. It is not a universal constant and may need recalibration for different embedding models, modalities, reference distributions, and operating conditions.

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
The script is intended to save the summary CSV, per-reference CSV, and detection curve under the repository's "results/" and "assets/" directories.

Reproducibility note: The evaluation script was run successfully in Google Colab. The generated summary and per-reference CSV files matched the committed versions exactly in that environment. Reproducibility across other environments has not yet been established.

### Evaluation Metrics

The current evaluation reports detection rates for controlled synthetic shifts. Further work should examine:

- Detection rate: Proportion of shifted test windows that trigger an alarm.
- False-alarm rate: Proportion of stable test windows incorrectly flagged as drift.
- Reference sensitivity: Variation in results across independently generated reference samples.
- Detection delay: Time or observations between drift introduction and the first alarm in a sequential experiment.
- Computational cost: Runtime and memory requirements under defined conditions.
- Calibration quality: Whether the observed stable-window alarm rate agrees with the intended calibration level.

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

### Phase 1 — Strengthen synthetic evaluation

- Reproduce the committed results from a clean environment.
- Verify that the generated CSV files match the reported summary.
- Compare the detector with appropriate statistical baselines.
- Examine uncertainty and variability across independent reference samples.
- Test different window sizes, noise levels, and drift magnitudes.
- Evaluate calibration and false-alarm behaviour more thoroughly.

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

Experiment 13 investigates threshold-calibrated and sequential drift detection using synthetic embedding distributions and an exploratory real-image embedding pilot. The experiment progresses from synthetic robustness analysis to controlled LFW brightness transformations and fresh-identity stream validation using a frozen sequential CUSUM detector.

### Synthetic evaluation

The synthetic evaluation examines detection sensitivity, false-alarm behaviour, and robustness across embedding dimensions, covariance regularization settings, and random seeds. A supplementary robustness evaluation covers 360 configurations.

Results are specific to the synthetic data-generating process and tested parameter settings. They do not establish performance on real biometric recognition systems.

### LFW controlled-brightness pilot

A preliminary pilot uses pretrained FaceNet embeddings extracted from the Labeled Faces in the Wild (LFW) dataset. Controlled brightness transformations are applied to images to examine changes in embedding-level detector scores.

The evaluation uses matched image streams and identity-disjoint reference and evaluation partitions. These experiments examine representation-score changes, not biometric recognition accuracy or naturally occurring temporal drift.

### Sequential CUSUM evaluation

The sequential pilot evaluates a frozen CUSUM configuration on controlled streams, with brightness transformations introduced at window 6. A repeated-stream sensitivity analysis uses 20 bootstrap/resampling replicates drawn from the same 600-image pool.

| **Condition** | **Replicates** | **Pre-onset alarm rate** | **Post-onset detection rate** |
|---|---:|---:|---:|
| Control | 20 | 10% | Not applicable |
| Brightness 0.70 | 20 | 10% | 40% |
| Brightness 1.30 | 20 | 10% | 55% |

The two pre-onset alarms occurred in the same resampling replicates across the control and transformed conditions. This indicates that stream variability can trigger alarms before the imposed brightness change.

The resampled streams overlap and are not independent datasets. The detection rates are exploratory and do not establish reliable deployment performance.

### Stage 3: Fresh-identity stream validation

Stage 3 evaluates the previously frozen one-sided CUSUM detector on a newly constructed stream of LFW identities that were not used in the earlier 600-image matched-stream experiment.

**Experimental setup**

- Dataset: Labeled Faces in the Wild (LFW), using 512-dimensional FaceNet embeddings.
- Fresh stream: 1,000 images representing 1,000 distinct identities.
- Stream configuration: 20 windows of 50 images per condition.
- Conditions: unchanged control, brightness factor 0.70, and brightness factor 1.30.
- Change onset: window 10 (zero-based; the 11th window).
- Windows 0–9 use original embeddings in all conditions. Windows 10–19 use embeddings extracted from brightness-transformed images in the two transformed conditions.
- The fresh stream excludes identities used in the earlier 600-image matched stream.
- The fresh identities come from the same LFW dataset and evaluation partition; this is not independent external-dataset validation.

**Frozen detector configuration**

| Parameter | Value |
|---|---:|
| Baseline median | 1.041908290 |
| Robust scale | 0.099255882 |
| CUSUM allowance, \(k\) | 0.5 |
| Alarm threshold, \(h\) | 5.0 |
| Window size | 50 images |

Detector parameters were not retuned on the fresh evaluation streams.

**Results**

| **Condition** | **First detection window** | **Pre-onset alarm** | **Detected after onset** | **Detection delay** | **Maximum CUSUM** |
|---|---:|---|---|---:|---:|
| Control | None | No | Not applicable | Not applicable | 2.842 |
| Brightness 0.70 | None | No | No | Not applicable | 2.842 |
| Brightness 1.30 | 14 | No | Yes | 4 windows | 9.762 |

The detector did not alarm on the unchanged control stream or the brightness-0.70 stream. It detected the brightness-1.30 stream at window 14, four windows after the imposed onset at window 10.

The control and brightness-0.70 streams share the same original embeddings before the onset, which explains their identical maximum CUSUM values in this evaluation.

**Interpretation**

Stage 3 provides a preliminary positive detection for one fresh stream under the stronger controlled brightness transformation. It does not establish reliable detection rates, false-alarm rates, or generalisation performance because only one stream construction was evaluated per condition.

### Reports and results

**Synthetic and sequential evaluation**

- Sequential pilot report: [`Experiment_13_Phase_5_Sequential_Pilot_Report.md`](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/blob/main/results/Experiment_13_Phase_5_Sequential_Pilot_Report.md)
- Repeated-stream report: [`Experiment_13_Phase_5B_Repeated_Stream_Report.md`](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/blob/main/results/Experiment_13_Phase_5B_Repeated_Stream_Report.md)
- Repeated-stream results: [`lfw_repeated_stream_cusum_results.csv`](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/blob/main/results/lfw_repeated_stream_cusum_results.csv)
- Repeated-stream summary: [`lfw_repeated_stream_cusum_summary.csv`](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/blob/main/results/lfw_repeated_stream_cusum_summary.csv)
- Identity-balanced calibration diagnostics: [`lfw_identity_balanced_calibration_diagnostics.csv`](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/blob/main/results/lfw_identity_balanced_calibration_diagnostics.csv)

**Stage 3 fresh-identity validation**

- Fresh-identity validation report: [`Experiment_13_Stage_3_Fresh_Identity_Validation_Report.md`](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/blob/main/results/Experiment_13_Stage_3_Fresh_Identity_Validation_Report.md)
- CUSUM results: [`lfw_stage3_fresh_cusum_results.csv`](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/blob/main/results/lfw_stage3_fresh_cusum_results.csv)
- CUSUM trace: [`lfw_stage3_fresh_cusum_trace.csv`](https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection/blob/main/results/lfw_stage3_fresh_cusum_trace.csv)

Additional sequential detector parameters, traces, and diagnostics are stored in `results/`.

### Limitations

These experiments use synthetic distributions and controlled transformations of static face images. They do not represent naturally occurring temporal biometric drift.

The repeated-stream bootstrap/resampling replicates overlap and may contain repeated identities within a window. Their detection rates are exploratory. Stage 3 uses one stream per condition, which is insufficient to estimate dependable detection or false-alarm rates.

The fresh-identity evaluation uses the same LFW dataset and evaluation partition rather than an independent external dataset. The experiments do not establish real-world biometric recognition accuracy, demographic fairness, successful bias mitigation, cyberattack detection, or deployment readiness. The CUSUM threshold and configuration are provisional and do not guarantee a specified operational false-alarm rate.

Raw face images, embedding arrays, and identity-level metadata should not be published without appropriate licensing, privacy, and consent review.

## Author

Moien Shaik

Research interests: trustworthy machine learning, distributional drift detection, biometric systems, out-of-distribution detection, and cybersecurity.

## License

This repository includes a `LICENSE` file. Review its terms before reusing or distributing the code. Third-party dependencies, datasets, and model components remain subject to their respective licenses.
