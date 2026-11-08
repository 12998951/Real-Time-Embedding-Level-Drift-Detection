# Experiment 18 — Final Research Evidence Package

## 1. Experiment Overview

- Dataset: LFW filtered subset
- Model: InceptionResnetV1
- Pretrained weights: VGGFace2
- Embedding dimension: 512
- Brightness factors: 0.5, 1.0, 1.5
- Historical paired illumination images: 889
- Identities in historical paired cohort: 13
- Corrected detector evaluation cohort: 229 images

## 2. Dataset Composition

The historical paired illumination cohort contains 889
images from 13 identities.

Identity 14 contributes 504
images (56.69%) of the historical paired cohort.

| Identity | Paired image count |
|---:|---:|
| 1 | 33 |
| 2 | 21 |
| 8 | 24 |
| 12 | 114 |
| 13 | 21 |
| 14 | 504 |
| 33 | 22 |
| 36 | 19 |
| 43 | 18 |
| 45 | 29 |
| 47 | 22 |
| 49 | 22 |
| 60 | 40 |

## 3. Embedding Displacement

| Statistic | Low illumination | High illumination |
|---|---:|---:|
| Mean | 0.234665304 | 0.318513930 |
| Median | 0.222675174 | 0.306733727 |
| Standard deviation | 0.064857408 | 0.119036496 |

Mean high-minus-low displacement:
0.083848625

Median high-minus-low displacement:
0.075521380

Displacement reproduction maximum absolute error:
- Low: 0.0
- High: 0.0

## 4. Historical Displacement Statistical Tests

| Analysis | Unit | n | Wilcoxon statistic | p-value |
|---|---|---:|---:|---:|
| Image-level | Paired images | 889 | 52705.0 | 4.708938281282196e-80 |
| Identity-level means | Identities | 13 | 3.0 | 0.001220703125 |
| Identity-level medians | Identities | 13 | 3.0 | 0.001220703125 |

The image-level observations are clustered within identities and
must not be interpreted as 889 independent subjects.

## 5. Aggregate Detector Results

        Condition  Euclidean  Mahalanobis
         Original   0.053114     0.797920
 Low illumination   0.094771     1.735851
High illumination   0.082996     1.871540

## 6. Identity-Level Detector Results

 Identity         Condition  Sample Count  Euclidean  Mahalanobis
        1  Low illumination             9   0.154546     2.801872
        1 High illumination             9   0.216245     3.790312
        2  Low illumination             6   0.126820     2.445043
        2 High illumination             6   0.157551     3.127515
        8  Low illumination             6   0.116416     2.180474
        8 High illumination             6   0.181826     4.464461
       12  Low illumination            29   0.103735     2.125610
       12 High illumination            29   0.164707     2.937519
       13  Low illumination             6   0.107625     2.040383
       13 High illumination             6   0.149594     2.857727
       14  Low illumination           126   0.115388     2.279956
       14 High illumination           126   0.138949     2.546591
       33  Low illumination             6   0.152135     2.854331
       33 High illumination             6   0.136978     3.066314
       36  Low illumination             6   0.108945     1.901431
       36 High illumination             6   0.093113     1.708182
       43  Low illumination             5   0.159294     2.686244
       43 High illumination             5   0.164591     3.098066
       45  Low illumination             8   0.120118     2.495526
       45 High illumination             8   0.239573     4.814947
       47  Low illumination             6   0.166553     2.991859
       47 High illumination             6   0.126455     2.650389
       49  Low illumination             6   0.105617     2.114737
       49 High illumination             6   0.126860     2.737765
       60  Low illumination            10   0.100831     2.143482
       60 High illumination            10   0.132249     2.656033

## 7. Corrected Detector Statistical Tests and Effect Sizes

   Detector  Wilcoxon Statistic  P-value  Identities  Mean Difference  Median Difference  SD of Differences  Cohen's dz  Rank-Biserial Correlation  Mean CI Lower  Mean CI Upper  Median CI Lower  Median CI Upper
  Euclidean                13.0 0.021484          13         0.030051           0.030731           0.041980    0.715850                   0.714286       0.009059       0.052363         0.005297         0.060972
Mahalanobis                 5.0 0.002441          13         0.722682           0.623028           0.799337    0.904102                   0.890110       0.334228       1.170385         0.266635         0.817344

The corrected Euclidean paired detector-response test has
Wilcoxon statistic 13 and p-value 0.021484.

The corrected Mahalanobis paired detector-response test has
Wilcoxon statistic 5 and p-value 0.002441.

Both tests are below the nominal 0.05 significance threshold.
These results are based on 13 identity-level observations and
have not been adjusted for multiple comparisons.

## 8. Identity-Level Detector Direction

- Euclidean high-versus-low response increased for
  10 of 13 identities
  and decreased for 3.
- Mahalanobis high-versus-low response increased for
  11 of 13 identities
  and decreased for 2.

## 9. Reproducibility Audit

Step 17.10.6.4: 54 checks passed; 0 failed.

The audit checked saved files, array shapes and integrity,
displacement reproduction, statistical summaries, CSV consistency,
metadata, and detector-table availability.

It did not independently rerun original image detection,
preprocessing, embedding extraction, or detector calibration.

## 10. Software Environment

      Component                              Version
         python                              3.13.18
       platform Linux-6.6.159+-x86_64-with-glibc2.39
          numpy                                2.4.6
         pandas                                2.2.3
          scipy                               1.16.3
        pytorch                           2.14.1+cpu
    torchvision                           0.29.1+cpu
   scikit_learn                                1.6.1
         pillow                               11.3.0
facenet_pytorch                                2.6.0

## 11. Evidence Classification

### Validated findings

- Saved embedding arrays and dimensions were verified.
- Saved low- and high-illumination displacement arrays were reproduced exactly.
- Saved displacement summary statistics were reproduced.
- Historical image-level and identity-level displacement tests were reproduced.
- Saved identity summary CSV was checked for consistency.
- The reproducibility audit passed 54 checks.

### Exploratory findings

- Aggregate Euclidean and Mahalanobis detector responses varied across illumination conditions.
- Identity-level detector responses varied across identities.
- Corrected Euclidean and Mahalanobis detector-response tests were below the nominal 0.05 threshold.
- Detector-response findings are based on only 13 identity-level observations.

### Claims not established

- Real-world biometric recognition accuracy.
- Generalisation to other biometric datasets or models.
- Demographic fairness.
- Deployment readiness.
- Causal explanation of detector-response differences.

## 12. Limitations

1. The corrected detector evaluation cohort contains 229 images from 13 identities.
2. The historical paired illumination cohort contains 889 images from 13 identities.
3. Identity 14 contributes 56.7% of the historical paired images.
4. Image observations are clustered within identities and must not be treated as independent subjects.
5. Identity-level detector analyses have only 13 identity-level units.
6. The illumination changes were controlled brightness transformations.
7. Corrected detector-response p-values are nominal and have not been adjusted for multiple comparisons.
8. Detector-response findings rely on previously validated tables and were not recalibrated in this documentation step.

## 13. Provenance and Scope

This evidence package consolidates previously generated and audited
Experiment 18 outputs.

It does not perform new image detection, preprocessing, embedding
extraction, model training, detector calibration, or statistical testing.

Original experiment archives and metadata are retained unchanged.
