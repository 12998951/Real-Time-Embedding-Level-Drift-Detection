\# LFW Real-Image Feature Drift Experiment



\## 1. Objective



This experiment investigates whether distance-based detectors can identify controlled distribution shifts in feature vectors extracted from real face images.



The experiment compares Euclidean distance and regularized Mahalanobis distance under three artificial shift magnitudes.



\## 2. Dataset



\- Dataset: Labeled Faces in the Wild (LFW)

\- Images loaded: 3,023

\- Unique identities: 62

\- Feature extractor: `google/vit-base-patch16-224`

\- Feature dimensions: 768

\- Image resolution supplied to the model: 224 × 224

\- Missing feature values: 0

\- Non-finite feature values: 0



The feature extractor is a general-purpose vision model, not a dedicated face-recognition model. Consequently, these features should not be described as validated biometric identity embeddings.



\## 3. Feature Extraction



Images were processed using the Hugging Face Transformers library and a pretrained Vision Transformer (ViT). The first-token representation from the model's final hidden state was used as the image feature vector.



The resulting feature matrix contained 3,023 rows and 768 numerical features per image. Identity labels were retained separately for dataset analysis.



\## 4. Experimental Design



The images were split separately within each identity into reference and test pools. This produced 1,497 reference images and 1,526 test images, with all 62 identities represented in both pools.



Feature standardization used the reference pool's means and standard deviations.



The controlled stream contained 76 windows, each containing 20 feature vectors. The first 40 windows formed the baseline period. An artificial shift was introduced at sample index 800, corresponding to the start of window 40.



The experiment evaluated three additive shift magnitudes:



\- 0.10

\- 0.25

\- 0.50



Ten random seeds, from 2026 through 2035, were used for each shift magnitude.



\## 5. Detectors



\### Euclidean mean-shift detector



The Euclidean detector measured the distance between each window's mean feature vector and the reference mean.



\### Regularized Mahalanobis detector



The Mahalanobis detector measured the multivariate distance between each window mean and the reference mean, using a regularized reference covariance matrix.



The regularization parameter was 0.001.



Both detectors used a nominal 99th-percentile threshold estimated from the 40 baseline windows in each trial.



\## 6. Observed Results



| Shift magnitude | Detector | Detections | Detection rate | Mean delay among detected trials |

|---:|---|---:|---:|---:|

| 0.10 | Euclidean | 4/10 | 40% | 10.75 windows |

| 0.10 | Mahalanobis | 10/10 | 100% | 0 windows |

| 0.25 | Euclidean | 10/10 | 100% | 0.30 windows |

| 0.25 | Mahalanobis | 10/10 | 100% | 0 windows |

| 0.50 | Euclidean | 10/10 | 100% | 0 windows |

| 0.50 | Mahalanobis | 10/10 | 100% | 0 windows |



A window contained 20 samples. Detection delay was measured relative to the known artificial change point.



These are preliminary observations from controlled experiments, not evidence of performance on natural biometric drift.



\## 7. Limitations



1\. The feature extractor is a generic ViT model rather than a validated face-recognition model.

2\. The shifts were artificially introduced by adding a constant to every standardized feature.

3\. Only ten trials were conducted per shift magnitude.

4\. Thresholds were estimated using the same baseline windows on which baseline alarms were counted. This makes the observed zero baseline alarm counts unsuitable as independent evidence of a 1% false-alarm guarantee.

5\. The random image ordering does not represent a real chronological biometric stream.

6\. No independent no-drift validation set was used to establish false-alarm performance.

7\. The experiment did not evaluate identity-recognition accuracy, demographic fairness, or natural changes in face images.



\## 8. Next Experiments



Future work should include:



\- Independent no-drift threshold calibration and validation.

\- Repeated trials with independent stream constructions.

\- Controlled image perturbations, such as illumination changes, blur, and compression.

\- Comparison with a suitable pretrained face-recognition model.

\- Evaluation of detection rate, detection delay, and false-alarm rate on held-out data.

\- Investigation of online change-point detectors such as ADWIN and Page-Hinkley.



\## 9. Reproducibility



The notebook and experiment outputs are maintained in the GitHub repository:



https://github.com/12998951/Real-Time-Embedding-Level-Drift-Detection



The associated notebook is `experiments/06\_LFW\_Real\_Data\_Drift\_Validation.ipynb`.



The trial-level and summary CSV files are stored under `results/LFW\_Real\_Data\_Results/`.



\## 10. Conclusion



The preliminary experiments show that both distance-based detectors responded to the injected feature shifts, with Mahalanobis distance detecting all tested shifts in the current trials. The Euclidean detector was less sensitive to the smallest tested shift.



These findings motivate further investigation but do not establish generalization to real-world biometric drift. Independent calibration, more realistic drift scenarios, and evaluation using a suitable face-recognition model are necessary before drawing stronger conclusions.



