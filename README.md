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
s_t = \|\mu_R - \mu_C\|_2
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
1, & s_t > \tau \\
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

These values are preliminary observations from the recorded synthetic evaluation, not guarantees of future performance. They should be compared against the committed CSV outputs and independently reproduced before being treated as final experimental results.

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
│   └── synthetic_drift_evaluation.py
├── embeddings/
├── utils/
│   └── visualization/
├── results/
│   ├── drift_results_summary.csv
│   ├── drift_results_by_reference.csv
│   └── .gitkeep
├── assets/
│   └── drift_detection_curve.png
├── requirements.txt
├── RUN_EXPERIMENTS.txt
├── CITATION.cff
├── LICENSE
└── .gitignore
```
The tree reflects the repository structure after removing the redundant root-level result files. The contents of "embeddings/" and "utils/visualization/" have not yet been verified.

## Reproducing the Synthetic Evaluation

### Requirements

- Python 3
- NumPy
- pandas
- Matplotlib

The dependencies are listed in [requirements.txt](requirements.txt)

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

Reproducibility note: Verify that the committed script uses paths relative to the repository root and that a fresh run completes successfully. The original experiment was also run in Google Colab. Re-running the script is necessary to confirm that the committed results match the reported summary.

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

## Author

Moien Shaik

Research interests: trustworthy machine learning, distributional drift detection, biometric systems, out-of-distribution detection, and cybersecurity.

 ## License

This repository includes a `LICENSE` file. Review its terms before reusing or distributing the code. Third-party dependencies, datasets, and model components remain subject to their respective licenses.
