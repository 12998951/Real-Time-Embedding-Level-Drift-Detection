\# Experiment 08: Robust Drift Detection Evaluation



\## 1. Objective



This experiment evaluates Euclidean-distance and regularised Mahalanobis-distance detectors under controlled perturbations of LFW image embeddings. The aim is to examine detection behaviour across several perturbation strengths and identify limitations in threshold calibration.



\## 2. Dataset and features



\- Dataset: Labeled Faces in the Wild (LFW)

\- Images represented: 3,023

\- Identity classes: 62

\- Feature dimensions: 768

\- Feature extractor: `google/vit-base-patch16-224`

\- Feature type: Generic Vision Transformer image embeddings, not embeddings from a validated face-recognition model.



The source CSV contained 3,023 rows and 769 columns, including the identity label. No missing feature values or non-finite feature values were found.



\## 3. Data partitioning



The image rows were divided into three disjoint pools:



| Pool | Images | Identities represented |

|---|---:|---:|

| Reference | 1,511 | 62 |

| Calibration | 756 | 61 |

| Evaluation | 756 | 61 |



No image index was shared between the pools. Identities could occur in multiple pools, so this was not an unseen-identity generalisation evaluation.



Features were standardised using statistics computed from the reference pool only.



\## 4. Detector calibration



Two detectors were evaluated:



1\. Euclidean distance from the reference mean.

2\. Regularised Mahalanobis distance using a covariance matrix with ridge regularisation of 0.01.



Initial thresholds were calibrated from 37 calibration-window scores. A diagnostic showed elevated alarm rates on the held-out control stream. The calibration and control pools were then processed using the same shuffled-window construction, and revised thresholds were calculated using the shuffled calibration windows.



| Detector | Initial threshold | Revised threshold |

|---|---:|---:|

| Euclidean | 7.175519 | 9.120568 |

| Mahalanobis | 9.567704 | 10.952770 |



The revised thresholds are empirical calibration values. They do not guarantee a 1% false-alarm rate.



\## 5. Evaluation protocol



\- Window size: 20 images

\- Complete evaluation windows per stream: 37

\- Trials per drift condition: 30

\- Drift strengths: 0.00, 0.05, 0.10, 0.20, 0.35

\- Total detector-trial records: 300

\- Injected shift: a controlled perturbation in a fixed direction in standardised feature space, beginning halfway through the stream.

\- Thresholds remained fixed during the revised evaluation.



The perturbations were artificial and do not represent naturally observed temporal biometric drift.



\## 6. Revised detection results



| Drift strength | Detector | Trials | Detection rate (%) | Mean delay (windows) |

|---:|---|---:|---:|---:|

| 0.00 | Euclidean | 30 | 26.667 | — |

| 0.00 | Mahalanobis | 30 | 16.667 | — |

| 0.05 | Euclidean | 30 | 26.667 | 5.250 |

| 0.05 | Mahalanobis | 30 | 16.667 | 11.600 |

| 0.10 | Euclidean | 30 | 23.333 | 7.143 |

| 0.10 | Mahalanobis | 30 | 30.000 | 7.556 |

| 0.20 | Euclidean | 30 | 16.667 | 9.600 |

| 0.20 | Mahalanobis | 30 | 6.667 | 5.500 |

| 0.35 | Euclidean | 30 | 16.667 | 9.000 |

| 0.35 | Mahalanobis | 30 | 13.333 | 13.500 |



Detection delay is measured in post-shift windows, with zero representing detection in the first post-shift window. Delay averages are calculated only among detected trials.



\## 7. Interpretation



Detection rates were low and did not increase consistently with injected drift strength. In the no-drift control, 26.667% of Euclidean trials and 16.667% of Mahalanobis trials produced at least one alarm.



These results do not establish reliable drift detection. They indicate that the current detectors, calibration procedure, and controlled perturbation design need further investigation.



The control evaluation included only 37 windows per stream. Therefore, the results are insufficient to establish a precise long-run false-alarm rate.



\## 8. Limitations



1\. The feature extractor is a generic Vision Transformer rather than a validated face-recognition embedding model.

2\. The LFW dataset is not a chronological stream, and the experiment does not reproduce naturally occurring temporal drift.

3\. The artificial shift follows one fixed direction in feature space.

4\. Calibration and control windows are limited in number.

5\. Identity classes overlap across pools, so unseen-identity generalisation was not tested.

6\. The repeated trials reuse the same underlying evaluation images; they are not independent datasets.

7\. The revised detection results are preliminary and should not be presented as proof of a robust biometric drift-detection method.



\## 9. Conclusion



Experiment 08 provides a reproducible preliminary comparison of Euclidean and regularised Mahalanobis distance under controlled shifts in LFW image embeddings. The results reveal weak detection performance and a need for better calibration and experimental design.



Further work should evaluate larger independent control streams, more diverse perturbation directions, sensitivity to window size and regularisation, and a validated face-recognition embedding model. Claims about trustworthy real-world biometric drift detection should wait until such evaluations are completed.



\## 10. Reproducibility artifacts



\- `experiments/08\_LFW\_Robust\_Drift\_Detection\_Evaluation.ipynb`

\- `results/LFW\_Real\_Data\_Results/exp08\_trial\_results\_v2.csv`

\- `results/LFW\_Real\_Data\_Results/exp08\_summary\_results\_v2.csv`

\- `results/LFW\_Real\_Data\_Results/exp08\_detection\_rate\_comparison.png`



These artifacts preserve the experiment configuration, revised trial-level results, summary metrics, and plot.



