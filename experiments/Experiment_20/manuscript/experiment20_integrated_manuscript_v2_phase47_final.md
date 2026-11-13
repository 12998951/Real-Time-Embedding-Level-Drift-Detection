# Reference-Relative Face Embedding Displacement Across Illumination Geometry Conditions: An Identity-Level Empirical Study

## Abstract

Illumination variation remains a significant challenge for
face-recognition systems, potentially altering the representations
produced by deep neural networks. This study empirically examines
reference-relative face embedding displacement across illumination
geometry conditions using a fixed pretrained FaceNet architecture.
Experiments were conducted on the Cropped Extended Yale B dataset,
comprising 38 identities and 2,414 images. A common reference capture
was established for each identity, and 2,128 evaluation observations
across 56 non-reference illumination conditions were analysed using
512-dimensional embeddings. Euclidean displacement and cosine
distance were calculated relative to the corresponding identity
reference. Illumination conditions were grouped into near-frontal,
oblique, and large-angular categories.

Both distance measures exhibited progressively greater displacement
across the three illumination groups. Mean Euclidean displacement
increased from 0.402 in near-frontal conditions to 0.709 in oblique
conditions and 1.118 in large-angular conditions. Corresponding mean
cosine distances increased from 0.087 to 0.262 and 0.649. Friedman
tests indicated significant differences across groups for both
metrics (chi-square = 76.0, p = 3.14e-17, Kendall's W = 1.0).
All six pairwise comparisons remained significant following Holm
correction. An exploratory hemisphere comparison also showed greater
mean displacement under beyond-frontal conditions.

These findings characterise reference-relative embedding displacement
within the evaluated dataset and model. However, the analysis uses
different captures rather than exact transformations of identical
source images, limiting causal attribution to illumination alone.
The results do not establish recognition accuracy, operational drift
detection, or generalisation across architectures.

## Keywords

Face recognition; Face embeddings; Illumination variation; Embedding displacement; Euclidean distance; Cosine distance

# 1. Introduction

## 1.1 Background: Face Recognition and Deep Face Embeddings

Face recognition systems increasingly use deep learning models to represent facial images as numerical feature vectors, commonly referred to as face embeddings. These representations encode facial characteristics in a high-dimensional feature space, where distances or similarities between embeddings can be used to examine relationships between facial images.

The geometry of this embedding space is relevant to face-recognition research because changes in an input image may be reflected in the position of its corresponding representation. Consequently, comparing embeddings can provide a way to examine how image variations are associated with changes in learned facial representations.

A particularly important source of image variation is illumination. Changes in lighting direction and intensity can alter the appearance of facial regions, producing differences between images of the same individual. The effects of illumination on face recognition have been investigated in earlier research, including work examining changes in illumination direction (Moses et al., 1994).

In this context, examining reference-relative embedding displacement provides an empirical way to describe how the representations of different captures of an individual vary relative to a selected reference capture. The present study applies this approach to a fixed pretrained InceptionResnetV1 model and examines Euclidean displacement and cosine distance across groups of illumination conditions.

This investigation focuses on the geometry of the resulting embeddings. It does not, by itself, establish whether a face-recognition system correctly identifies individuals under those conditions.

## 1.2 Illumination Variation and Its Effect on Facial Representations

Illumination variation is a longstanding challenge in face-recognition research. Changes in the direction of incident light can produce substantial differences in facial appearance, affecting the distribution of shadows, highlights, and visible facial details. These variations can occur even when the images represent the same individual. Earlier research by Moses et al. (1994) specifically examined the problem of compensating for changes in illumination direction in face recognition.

Subsequent research has investigated the relationship between illumination and facial appearance using different analytical approaches. Beveridge et al. (2009) examined subject illumination spaces and the role of principal angles in distinguishing them. Other studies have explored illumination preprocessing and methods intended to reduce the effects of lighting variation on face recognition (Han et al., 2013; Ochoa-Villegas et al., 2015). Research using Yale Face Database B has also classified illumination conditions according to the angle between the light-source direction and camera axis. Ochoa-Villegas et al. (2015) describe five angular subsets: 0°–12°, 13°–25°, 26°–50°, 51°–77°, and above 77°. This provides precedent for angular grouping in illumination studies. The present study adopts a different three-group classification, with near-frontal conditions defined as θ ≤ 30°, oblique conditions as 30° < θ ≤ 60°, and large-angular-deviation conditions as θ > 60°. These thresholds represent the analytical grouping adopted in Experiment 20 and are not the same as the five subsets reported in the survey.

Deep face-recognition models represent facial images through learned feature embeddings. Changes in facial appearance may therefore be accompanied by changes in embedding-space position. However, the magnitude and consistency of such changes depend on the model, the input images, and the conditions under which the representations are evaluated. An observed embedding displacement should consequently be interpreted as a change in representation rather than direct evidence of incorrect identity recognition.

In the present study, illumination conditions are grouped according to their angular geometry relative to a common reference condition. Three groups are considered: near-frontal illumination, oblique illumination, and large-angular-deviation illumination. These categories provide a structured basis for comparing reference-relative embedding displacement across different lighting directions. They represent angular geometry rather than illumination intensity.

This distinction is important because the experiment uses different captures of the same individuals under different illumination conditions. The images are identity-matched observations, not exact transformations of identical source images. Therefore, the resulting differences in embedding space are examined as reference-relative associations with illumination geometry, rather than as effects attributable exclusively to illumination.

## 1.3 Research Gap and Motivation

Existing research has examined the effects of illumination variation on face recognition through several approaches, including illumination preprocessing, feature representation, subject-specific illumination spaces, and recognition-performance evaluation. Studies using Yale B and Extended Yale B datasets have also investigated changes in facial representations under different lighting directions. More recent work has explored deep face-recognition architectures and the geometric properties of learned embedding spaces.

Despite this established body of research, there remains a need for focused empirical characterisation of how reference-relative embedding displacement varies across illumination geometry conditions at the individual-identity level within a fixed pretrained face-recognition model. In particular, comparing multiple distance measures relative to a common reference capture provides a structured way to examine whether displacement patterns differ across illumination groups and whether these differences are consistent across identities.

This study addresses that specific empirical question using a fixed pretrained InceptionResnetV1 model and the Cropped Extended Yale B dataset. Rather than developing an illumination-invariant representation or evaluating identity-classification performance, the investigation examines Euclidean displacement and cosine distance between identity-specific reference embeddings and embeddings from other captures of the same individuals.

The motivation is to provide a transparent, identity-level description of embedding-space variation across defined illumination geometry conditions. By examining both observation-level distributions and repeated-measures differences across identities, the study seeks to characterise the consistency and magnitude of reference-relative displacement within the evaluated dataset.

The resulting analysis is deliberately descriptive. Because the evaluation uses different image captures rather than exact transformations of identical source images, observed embedding differences cannot be attributed exclusively to illumination. The findings are also specific to the selected dataset, preprocessing procedure, and pretrained model. Accordingly, the study is intended to contribute a focused empirical characterisation rather than a general claim about all face-recognition systems.

## 1.4 Research Questions and Objectives

Based on the identified research gap, this study investigates how reference-relative face embedding displacement varies across illumination geometry conditions and how consistently these differences are observed across individual identities. The investigation is guided by the following research questions:

RQ1: How do Euclidean embedding displacement and cosine distance vary across near-frontal, oblique, and large-angular illumination conditions?

RQ2: Are the differences between illumination geometry groups consistent across evaluated identities?

RQ3: How do reference-relative embedding displacement patterns differ between frontal and beyond-frontal illumination conditions?

To address these questions, the study has five objectives:

1. To generate and examine fixed pretrained face embeddings for identity-matched captures under different illumination conditions.
2. To quantify reference-relative embedding displacement using Euclidean distance and cosine distance.
3. To classify common illumination conditions into near-frontal, oblique, and large-angular geometry groups.
4. To compare displacement measurements across illumination groups using identity-level repeated-measures statistical analysis.
5. To explore differences between frontal and beyond-frontal illumination conditions while accounting for the unequal number of evaluated conditions.

The first two research questions form the primary analytical focus of the study. The third question provides an additional exploratory examination of illumination geometry. The statistical hypotheses associated with these comparisons were formulated after the experimental analyses and are therefore treated as retrospective analytical expectations rather than preregistered hypotheses.

## 1.5 Study Contributions and Scope

This study makes four empirical contributions.

First, it provides an identity-level characterisation of reference-relative face embedding displacement across three illumination geometry groups: near-frontal, oblique, and large-angular conditions. The analysis examines how the magnitude of displacement varies across individual identities within the evaluated dataset.

Second, it compares two complementary embedding-space distance measures: Euclidean displacement and cosine distance. Using both measures provides a broader descriptive examination of changes in embedding representations relative to an identity-specific reference capture.

Third, it applies repeated-measures statistical analysis to examine differences between the three illumination geometry groups across the evaluated identities. This analysis considers both the magnitude of group differences and the consistency of their ordering across identities.

Fourth, it provides an exploratory comparison of frontal and beyond-frontal illumination conditions, examining whether reference-relative displacement differs between these broader geometric categories.

The scope of the study is deliberately limited. The evaluation uses the Cropped Extended Yale B dataset, comprising 38 identities and 2,414 standard-illumination images, and a fixed pretrained InceptionResnetV1 model. Of the 2,414 standard-illumination images, 2,166 observations belong to the 57 common illumination conditions. After excluding the reference condition, 2,128 non-reference observations across 56 evaluation conditions are included in the displacement analysis. Each identity is represented by one reference capture, against which embeddings from other captures are compared. The evaluation therefore examines different identity-matched captures rather than exact transformations of identical source images.

The study does not evaluate face-recognition accuracy, develop an illumination-invariant representation, establish operational drift-detection capability, or validate generalisation across independent recognition architectures. Its findings should be interpreted as an empirical characterisation of reference-relative embedding displacement under the specific dataset, preprocessing procedure, and model used in this investigation.

# 2. Methodology

## 2.1 Research Design and Experimental Overview

Experiment 20 employed a descriptive empirical design to examine reference-relative displacement in deep face-recognition embeddings across illumination geometry conditions. The analysis focused on the extent to which the embedding representation of an identity's facial image differed from that identity's embedding under a designated reference illumination condition. Rather than developing or evaluating an operational drift-detection algorithm, the experiment characterised these differences using two distance measures: Euclidean displacement and cosine distance.

The evaluation used the Cropped Extended Yale B dataset, comprising 2,414 standard-illumination images from 38 identities. A common-condition selection procedure identified 57 illumination conditions represented across all 38 identities, yielding 2,166 identity-condition observations. One observation per identity corresponded to the designated reference condition, A+000E+00. These 38 reference observations were used to establish the identity-specific reference embeddings. After excluding the reference condition, the primary evaluation comprised 2,128 non-reference observations across 56 illumination conditions.

The primary unit of observation was an identity-condition pair, while identity served as the repeated-measures unit for group-level statistical comparisons. This structure allowed displacement measurements from the same 38 identities to be examined across near-frontal, oblique, and large-angular-deviation illumination groups. The observations represent identity-condition-matched captures under different illumination conditions; they are not pairs produced by applying controlled transformations to identical source images.

The analysis was conducted using 512-dimensional embeddings generated by a fixed pretrained InceptionResnetV1 model with VGGFace2 weights. Each evaluation embedding was compared with the corresponding identity's reference embedding. Euclidean displacement and cosine distance were then summarised across illumination geometry groups and examined using identity-level repeated-measures statistical procedures.

The study was intentionally restricted to descriptive embedding-space analysis. It did not estimate calibrated detection thresholds, operational false-alarm rates, or recognition accuracy. It also did not evaluate an independent face-recognition architecture. Consequently, the results characterise reference-relative embedding displacement within the selected dataset, preprocessing procedure, and pretrained model; they do not establish that illumination alone caused the observed differences or that the findings generalise to other recognition systems.

## 2.2 Dataset and Acquisition Provenance

The external evaluation dataset used in Experiment 20 was the Cropped Extended Yale B frontal-pose illumination subset. It contains cropped grayscale facial images captured under different illumination conditions for 38 identities. This subset should be distinguished from the complete multi-pose Extended Yale B dataset.

The dataset archive was acquired through a community-hosted Academic Torrents listing. The acquisition record identifies the source listing and records an archive size of 58,493,820 bytes. The downloaded ZIP archive passed its integrity test and contained 2,586 members. Its SHA-256 checksum was recorded as 8bbabf3836d077d60626c9ee3c54c2d2851a8d2c15c8b7e13b9bc162e86ea573. The archive provenance remains subject to an important limitation: it was not independently authenticated against an official UCSD checksum. Accordingly, the acquisition record and checksum provide traceability for the downloaded archive but do not establish official dataset authenticity.

The archive contained 2,452 PGM image files, all of which were successfully decoded during the image-integrity audit. Filename and condition validation identified 2,414 standard-illumination images and 38 ambient reference images. The ambient images were excluded from the standard-illumination analysis. The retained standard-illumination images represented 64 distinct illumination conditions across the 38 identities. Seven identities did not contain every condition in the overall set.

To support comparisons with complete identity coverage, the analysis used the 57 illumination conditions shared by all 38 identities. This common-condition subset comprised 2,166 identity-condition observations. The remaining conditions that were not shared by every identity were not included in analyses requiring complete identity-condition coverage. One observation per identity corresponded to the reference condition, A+000E+00; after its exclusion, 2,128 non-reference observations across 56 conditions formed the primary evaluation cohort.

The dataset was frozen for the Stage 2 analysis, with acquisition records, image-integrity results, condition manifests, embedding artifacts, and associated checksums retained in the project archive. These records support reproducibility and traceability of the analysed cohort. They do not remove the limitation arising from the community-hosted source or establish authentication against an official dataset checksum.

## 2.3 Subject Selection and Illumination-Condition Coverage

The subject cohort comprised 38 identities represented in the Cropped Extended Yale B frontal-pose illumination subset. The standard-illumination image collection contained 64 distinct illumination conditions across the cohort; however, not every condition was available for every identity. Seven identities had incomplete condition coverage. To preserve a consistent identity set across the primary condition-based comparisons, the analysis retained only conditions represented for all 38 identities.

This complete-case condition selection produced 57 common illumination conditions and 2,166 identity-condition observations, corresponding to 38 observations for each condition. The selection was based on shared condition availability rather than on embedding displacement values or statistical outcomes. Conditions not represented across all 38 identities were excluded from analyses requiring complete identity-condition coverage.

The condition A+000E+00 was designated as the reference condition. One reference observation was available for each of the 38 identities, providing an identity-specific reference embedding. The reference observations were retained for calculating reference-relative distances but excluded from the non-reference evaluation set. Consequently, the primary evaluation comprised the remaining 56 common conditions and 2,128 identity-condition observations.

The 56 non-reference conditions were assigned to three illumination geometry groups using the frozen condition-grouping manifest: near-frontal, oblique, and large-angular-deviation. The corresponding evaluation counts were 17, 11, and 28 conditions, respectively, representing 646, 418, and 1,064 observations. The reference condition belonged to the near-frontal group but was excluded from these evaluation counts.

The geometry groups represent the angular deviation of the illumination direction relative to the camera axis, as defined by the study's condition-grouping procedure. They do not represent directly measured illumination intensity. The same 38 identities were represented in each of the three evaluation groups, enabling identity-level repeated-measures comparisons. The resulting observations are identity-condition-matched captures under different lighting conditions, not exact image-level pairs generated from a single source image.

## 2.4 Face-Image Preprocessing

The Cropped Extended Yale B images were supplied as grayscale PGM files. Before embedding extraction, each image was converted to an RGB representation by replicating its grayscale channel across the three colour channels. The resulting images were resized directly to 160 × 160 pixels and converted to tensors using torchvision.transforms.ToTensor().

Face detection and facial alignment were not applied during this preprocessing stage. An initial MTCNN compatibility test detected only one face in twelve tested Yale B images. Consequently, the experiment used direct image resizing rather than MTCNN-based detection and alignment. This decision was consistent with the cropped-image format of the selected dataset, but it means that the preprocessing procedure should not be described as a face-detection or landmark-alignment pipeline.

The original resize operation did not explicitly specify interpolation or antialiasing arguments. The recorded torchvision version was 0.29.1+cpu, for which the applicable Resize defaults were bilinear interpolation and antialiasing enabled. These settings are therefore documented as library defaults rather than as manually specified experimental parameters.

This preprocessing procedure differs from the MTCNN-based preprocessing used in Experiment 18. Accordingly, results obtained from the two experimental stages should not be treated as directly interchangeable without accounting for their different image-processing procedures.

## 2.5 FaceNet Architecture and Embedding Extraction

Face embeddings were generated using the InceptionResnetV1 architecture implemented in facenet-pytorch, with pretrained weights associated with the VGGFace2 training dataset. The model was used as a fixed pretrained feature extractor rather than being fine-tuned on the Cropped Extended Yale B images. The same model configuration was applied throughout the Stage 2 embedding extraction procedure.

The model produced a 512-dimensional embedding for each preprocessed image. Embedding extraction was performed with the model in evaluation mode, and gradient computation was disabled. These settings were used to obtain feature representations without updating model parameters during inference. The extracted embeddings were L2-normalised, providing the representations used in the subsequent reference-relative distance calculations.

The recorded software environment included Python 3.13.18, PyTorch 2.14.1+cpu, torchvision 0.29.1+cpu, and facenet-pytorch 2.6.0. The embedding-generation procedure used the pretrained InceptionResnetV1 model configured for VGGFace2 weights.

Although the model architecture, pretrained training source, software versions, output dimensionality, and inference settings were recorded, the exact pretrained weight-file identity and its SHA256 checksum were not preserved. Consequently, the model configuration is documented at the architecture and pretrained-source level, but exact weight-file reproducibility cannot be independently established from the retained records.

The resulting embeddings were retained with row-aligned image and identity metadata, including filenames, subject identifiers, illumination azimuths, elevations, and condition labels. These records enabled each non-reference embedding to be associated with the corresponding identity-specific reference representation during the displacement analysis.

## 2.6 Embedding Normalisation and Reference-Relative Distance Calculation

The extracted 512-dimensional embeddings were L2-normalised before the reference-relative displacement analysis. For each of the 38 identities, the embedding associated with the reference illumination condition A+000E+00 served as that identity's reference representation. This produced 38 identity-specific reference embeddings.

Each non-reference embedding was matched to the reference embedding belonging to the same identity. The evaluation therefore compared identity-condition observations with their corresponding within-identity reference representations rather than using a single reference vector shared across all subjects. After excluding the reference observations, the analysis included 2,128 non-reference observations across 56 common evaluation conditions.

Two distance measures were used to characterise reference-relative embedding displacement. Euclidean displacement measured the L2 distance between an evaluation embedding and its corresponding identity-specific reference embedding. Cosine distance was calculated as one minus the cosine similarity between the same pair of embedding vectors. Both measures were calculated for each evaluation observation and retained alongside its identity and illumination-condition metadata. Because the embeddings were L2-normalised, the two distances are mathematically related: squared Euclidean distance equals twice cosine distance. They therefore provide related, rather than independent, measures of embedding displacement.

The resulting observation-level distances were subsequently grouped by illumination geometry for descriptive and identity-level statistical analysis. The distance calculations quantify changes in embedding representation relative to the selected reference capture. They do not, by themselves, establish a change in recognition accuracy, identity verification performance, or an operational biometric drift event.

The analysis was descriptive: no operational detection thresholds, calibrated false-alarm rates, or classification decisions were derived from these distances. The reference-relative comparisons should also be interpreted in light of the use of different image captures across illumination conditions, rather than exact transformations of identical source images.

## 2.7 Illumination Geometry Classification

Illumination conditions were grouped according to the angular deviation of the light source from the camera-facing direction. The angular deviation, denoted by θ, was calculated from the recorded illumination azimuth and elevation angles using θ = cos⁻¹(cos(elevation) × cos(azimuth)). The angular values were expressed in degrees for interpretation and grouping.

Three study-specific illumination geometry groups were defined. Near-frontal conditions comprised observations with θ ≤ 30°. Oblique conditions comprised observations with 30° < θ ≤ 60°. Large-angular conditions comprised observations with θ > 60°. These mutually exclusive intervals assigned each evaluated illumination condition to one geometry group.

Across the 56 non-reference evaluation conditions, 17 conditions were classified as near-frontal, 11 as oblique, and 28 as large-angular. Across the 38 identities, these groups contained 646, 418, and 1,064 evaluation observations, respectively. All three groups therefore included observations from the same 38 identities, supporting the repeated-measures comparisons conducted at the identity level.

The groups represent the angular geometry of the illumination direction rather than its intensity. They should not be interpreted as brightness levels, exposure categories, or measurements of lighting power. The three angular boundaries were selected for this study and are not presented as universal or standard illumination categories. Previous face-recognition research has used angular subsets to examine illumination variation, but the thresholds and grouping scheme used here are study-specific.

## 2.8 Statistical Analysis

Statistical comparisons were conducted using identity-level summaries of the reference-relative distances. For each identity, the mean Euclidean displacement and mean cosine distance were calculated separately within the near-frontal, oblique, and large-angular illumination geometry groups. This aggregation provided three repeated measurements for each metric from each of the 38 identities.

Differences across the three illumination geometry groups were evaluated using the Friedman test, a non-parametric repeated-measures procedure. The test was applied separately to Euclidean displacement and cosine distance. Kendall's coefficient of concordance (W) was reported as the corresponding effect-size measure.

Pairwise differences between illumination geometry groups were examined using Wilcoxon signed-rank tests on the paired identity-level summaries. Three pairwise comparisons were conducted for each distance metric: near-frontal versus oblique, near-frontal versus large-angular, and oblique versus large-angular. Holm's step-down procedure was applied across the combined family of six pairwise tests, comprising the three Euclidean comparisons and the three cosine comparisons, to account for multiple testing.

As a sensitivity analysis, exact two-sided paired sign tests were additionally conducted on the identity-level mean differences for each of the six primary pairwise comparisons. The sign tests evaluated whether the direction of the paired differences was consistently positive or negative across identities, without relying on the symmetry assumption associated with the Wilcoxon signed-rank test. Tied differences, if present, were excluded from the sign-test calculation. Holm's step-down correction was applied across the same combined family of six primary sign tests. These analyses were used to assess the robustness of the directional findings and did not replace the primary Wilcoxon signed-rank tests.

Bootstrap confidence intervals were calculated for the identity-level mean differences in the pairwise comparisons using 20,000 bootstrap iterations with random seed 2026274. Resampling was performed at the identity level, preserving the paired structure of the repeated-measures observations. The reported intervals are 95% bootstrap confidence intervals.

An exploratory comparison between frontal and beyond-frontal illumination conditions was additionally conducted using identity-level summaries. This comparison was interpreted cautiously because the two hemispheric categories contained unequal numbers of illumination conditions and observations and were associated with differences in angular geometry.

The research hypotheses were formulated retrospectively after the experimental results had been examined. They were not preregistered and should therefore be interpreted as hypotheses used to organise the analysis rather than as prospectively specified tests. Statistical significance was assessed using the corrected pairwise p-values, while effect sizes and confidence intervals were considered alongside the test results.

## 2.9 Exploratory Hemisphere-Based Analysis

A secondary exploratory analysis compared reference-relative embedding displacement between frontal and beyond-frontal illumination conditions. The grouping was based on the angular hemisphere classification recorded in the Stage 2 analysis. The frontal category comprised 45 non-reference evaluation conditions, corresponding to 1,710 observations across the 38 identities. The beyond-frontal category comprised 11 evaluation conditions and 418 observations.

For each identity, mean Euclidean displacement and mean cosine distance were calculated separately for the frontal and beyond-frontal categories. The resulting identity-level summaries were compared using paired statistical tests, with the identity serving as the unit of comparison. This approach retained the within-identity structure of the analysis.

As an additional sensitivity analysis, exact two-sided paired sign tests were performed separately for Euclidean displacement and cosine distance using the identity-level differences between beyond-frontal and frontal conditions. Holm correction was applied across these two hemisphere sign tests as a separate exploratory correction family. This correction family was kept separate from the six primary illumination-geometry comparisons because the hemisphere analysis was secondary and exploratory.

The hemisphere comparison was treated as exploratory and secondary to the three-group illumination geometry analysis. The two categories contained unequal numbers of illumination conditions and observations. In addition, hemisphere membership was associated with differences in angular illumination geometry. Consequently, the comparison does not isolate a hemisphere effect independently of the underlying angular distribution and should not be interpreted as evidence of a causal effect of illumination direction.

The results are therefore reported as descriptive evidence of differences between the evaluated frontal and beyond-frontal condition sets. They are not used to establish a generalised performance claim or an operational biometric drift-detection capability.

## 2.10 Methodological Limitations and Reproducibility

Several methodological limitations should be considered when interpreting the findings and assessing reproducibility. First, the Cropped Extended Yale B subset was obtained through a community-hosted Academic Torrents listing. Although the acquired archive and frozen dataset were recorded using SHA256 checksums and integrity checks, the source was not independently authenticated against an official UCSD checksum. The dataset should therefore be described according to its documented acquisition provenance rather than as an independently authenticated official release.

Second, the exact pretrained InceptionResnetV1 weight-file identity and its SHA256 checksum were not preserved. The architecture, VGGFace2 pretrained source, software versions, and inference configuration were recorded, but these records do not establish that an independent reproduction would load byte-identical model weights.

Third, the illumination conditions comprise different image captures of the same identities rather than exact transformations of identical source images. The observed embedding differences therefore cannot be attributed exclusively to illumination; other capture-related variation may also contribute. In addition, the analysis used one reference embedding per identity, which does not provide sufficient observations to estimate identity-specific covariance reliably.

Fourth, the selected cropped images were processed using direct resizing without MTCNN face detection or facial alignment. This procedure differs from the MTCNN-based preprocessing used in Experiment 18, limiting direct comparability between the two experimental stages. The Stage 2 evaluation also used a single pretrained architecture and did not include independent architecture validation.

Finally, the study characterises reference-relative embedding displacement rather than biometric recognition performance or an operational drift-detection system. It does not establish verification accuracy, fairness, deployment readiness, or generalisation to other datasets and recognition architectures. These boundaries should be retained when interpreting the empirical findings.

# 3. Results

## 3.1 Evaluation cohort and analysis coverage

The source dataset contained 2,414 standard-illumination images from 38 identities. Of these, 2,166 observations belonged to 57 common illumination conditions. One reference capture was selected for each identity using condition A+000E+00. After excluding the 38 reference observations, the evaluation comprised 2,128 non-reference observations across 56 common evaluation conditions. All 38 identities were represented in the evaluation.

The evaluation observations were classified into near-frontal, oblique, and large-angular-deviation illumination groups. These groups contained 646, 418, and 1,064 observations, respectively.


## 3.2 Overall reference-relative embedding displacement

Across 2,128 evaluation observations, mean Euclidean displacement was 0.820496 (SD = 0.362940; median = 0.817311), with values ranging from 0.151681 to 1.522514. Mean cosine distance was 0.402439 (SD = 0.309758; median = 0.333999), ranging from 0.011504 to 1.159025.


## 3.3 Displacement across illumination geometry groups

Both distance measures increased progressively across the illumination geometry groups. Mean Euclidean displacement increased from 0.401897 under near-frontal conditions to 0.708976 under oblique conditions and 1.118457 under large-angular deviation conditions. Mean cosine distance followed the same ordering, increasing from 0.086877 to 0.262274 and 0.649095, respectively. The identity-level patterns are shown in Figures 1 and 2.


### Table 1. Descriptive statistics by illumination geometry group

| Illumination group | Observations | Identities | Conditions | Euclidean mean | Euclidean SD | Cosine mean | Cosine SD |
|---|---:|---:|---:|---:|---:|---:|---:|
| Near-frontal | 646 | 38 | 17 | 0.4019 | 0.1107 | 0.0869 | 0.0479 |
| Oblique | 418 | 38 | 11 | 0.7090 | 0.1482 | 0.2623 | 0.1062 |
| Large-angular deviation | 1064 | 38 | 28 | 1.1185 | 0.2175 | 0.6491 | 0.2388 |


![Identity-level Euclidean displacement across near-frontal, oblique, and large-angular-deviation illumination groups.](../step_3_3/step_3330/figures/experiment20_figure1_euclidean_illumination_groups.png)

*Figure 1. Identity-level Euclidean displacement across illumination geometry groups. Thin lines represent the mean displacement for each of the 38 identities within each group; the thick olive line represents the mean across identities.*

![Identity-level cosine distance across near-frontal, oblique, and large-angular-deviation illumination groups.](../step_3_3/step_3330/figures/experiment20_figure2_cosine_illumination_groups.png)

*Figure 2. Identity-level cosine distance across illumination geometry groups. Thin lines represent the mean distance for each of the 38 identities within each group; the thick olive line represents the mean across identities.*


## 3.4 Identity-level repeated-measures analysis

Friedman repeated-measures tests identified differences among the three illumination geometry groups for Euclidean displacement (χ²(2) = 76.0, p = 3.139e-17, Kendall's W = 1.0) and cosine distance (χ²(2) = 76.0, p = 3.139e-17, Kendall's W = 1.0). Each analysis included 38 identities measured across three groups.

Pairwise identity-level comparisons used Wilcoxon signed-rank tests. Confidence intervals for mean paired differences were estimated using 20,000 identity-level bootstrap iterations with random seed 2026274. Holm correction was applied across one combined family of six comparisons covering both distance measures. All six comparisons remained statistically significant.

In the sign-test sensitivity analysis, all 38 identities showed positive paired differences for each of the six primary comparisons, with no negative differences or ties. The exact two-sided sign-test p-value was 7.276 × 10⁻¹² for each comparison. Following Holm correction across the six-test family, all adjusted p-values were 4.366 × 10⁻¹¹. This consistent direction of differences provides sensitivity evidence supporting the ordering observed in the primary Wilcoxon analysis.


### Table 2. Identity-level pairwise comparisons

| Metric | Comparison | Mean paired difference | 95% bootstrap CI | Raw p | Six-test Holm p |
|---|---|---:|---:|---:|---:|
| Euclidean displacement | Near-frontal vs Oblique | 0.307079 | [0.285544, 0.328413] | 7.276e-12 | 4.366e-11 |
| Euclidean displacement | Near-frontal vs Large-angular deviation | 0.716560 | [0.691110, 0.741689] | 7.276e-12 | 4.366e-11 |
| Euclidean displacement | Oblique vs Large-angular deviation | 0.409481 | [0.387346, 0.431761] | 7.276e-12 | 4.366e-11 |
| Cosine distance | Near-frontal vs Oblique | 0.175397 | [0.159082, 0.191703] | 7.276e-12 | 4.366e-11 |
| Cosine distance | Near-frontal vs Large-angular deviation | 0.562219 | [0.536391, 0.588301] | 7.276e-12 | 4.366e-11 |
| Cosine distance | Oblique vs Large-angular deviation | 0.386821 | [0.364906, 0.408321] | 7.276e-12 | 4.366e-11 |


## 3.5 Exploratory frontal and beyond-frontal comparison

In the exploratory hemisphere analysis, mean Euclidean displacement was 0.706220 under frontal conditions and 1.287990 under beyond-frontal conditions. The mean paired difference (beyond-frontal minus frontal) was 0.581770 (95% bootstrap CI: [0.561986, 0.601954]; Holm-adjusted p = 1.455e-11). (Figure 3).

For cosine distance, the corresponding means were 0.295816 and 0.838624. The mean paired difference was 0.542807 (95% bootstrap CI: [0.516419, 0.568516]; Holm-adjusted p = 1.455e-11). (Figure 4).

In the exploratory hemisphere sign-test sensitivity analysis, all 38 identities showed positive differences between beyond-frontal and frontal conditions for both Euclidean displacement and cosine distance, with no negative differences or ties. The exact two-sided sign-test p-value was 7.276 × 10⁻¹² for each metric. Following Holm correction across the separate two-test hemisphere family, both adjusted p-values were 1.455 × 10⁻¹¹. These results provide additional evidence of directional consistency but do not establish an independent or causal hemisphere effect.

![Paired identity-level Euclidean displacement for frontal and beyond-frontal illumination hemispheres.](../step_3_3/step_3330/figures/experiment20_figure3_euclidean_hemisphere_comparison.png)

*Figure 3. Paired identity-level Euclidean displacement by illumination hemisphere. Each thin line connects the frontal and beyond-frontal mean displacement for one identity; the thick olive line represents the mean across the 38 identities. This comparison is exploratory.*

![Paired identity-level cosine distance for frontal and beyond-frontal illumination hemispheres.](../step_3_3/step_3330/figures/experiment20_figure4_cosine_hemisphere_comparison.png)

*Figure 4. Paired identity-level cosine distance by illumination hemisphere. Each thin line connects the frontal and beyond-frontal mean distance for one identity; the thick olive line represents the mean across the 38 identities. This comparison is exploratory.*


This comparison is exploratory. The frontal and beyond-frontal categories contained unequal numbers of illumination conditions, and hemisphere membership is related to angular illumination geometry. Therefore, the observed differences should not be interpreted as an isolated causal effect of hemisphere direction.


## 3.6 Summary of findings

The results show progressively greater reference-relative embedding displacement across near-frontal, oblique, and large-angular-deviation illumination groups. The identity-level repeated-measures analysis identified statistically significant group differences for both distance measures, and all six pairwise comparisons remained significant following combined six-test Holm correction. The exploratory hemisphere analysis also showed higher mean displacement under beyond-frontal conditions. These findings characterise embedding displacement within the evaluated dataset and fixed pretrained model; they do not establish recognition accuracy, operational drift-detection performance, or generalisation across architectures.

# 4. Discussion

## 4.1 Principal findings

This study examined reference-relative displacement in 512-dimensional FaceNet embeddings across illumination geometry conditions using a fixed pretrained InceptionResnetV1 model. Three main findings emerged. First, both Euclidean displacement and cosine distance increased progressively from near-frontal to oblique and large-angular-deviation illumination conditions. Second, the ordering of the three groups was consistent across the evaluated identities, with statistically significant repeated-measures differences for both distance measures. Third, the exploratory hemisphere comparison showed greater mean displacement under beyond-frontal conditions than under frontal conditions.

Across 2,128 non-reference observations, mean Euclidean displacement was 0.820496, while mean cosine distance was 0.402439. The group-level means showed a clear ordered pattern: Euclidean displacement increased from 0.401897 in the near-frontal group to 0.708976 in the oblique group and 1.118457 in the large-angular deviation group. Cosine distance followed the same ordering. These findings characterise differences in embedding position relative to an identity-specific reference; they do not directly measure successful or failed identity recognition.

## 4.2 Interpretation of illumination geometry and embedding displacement

The progressive increase in both distance measures suggests that the angular geometry of illumination conditions is associated with increasingly different FaceNet representations relative to the selected reference captures. The largest mean displacement occurred in the large-angular-deviation group, while the near-frontal group showed the smallest mean displacement. This pattern is consistent with the broader face-recognition literature, which has long identified illumination direction as an important source of variation in facial appearance and recognition conditions (Moses et al., 1994; Ochoa-Villegas et al., 2015).

The geometry groups used here describe the angular direction of the illumination source. They do not represent illumination intensity. Consequently, the observed ordering should be interpreted as an association with the defined illumination geometry categories, rather than evidence that increasing illumination intensity produces increasing embedding displacement.

The analysis also compares different captures of each identity. Changes between captures may include image variation beyond illumination alone. Therefore, the results do not isolate a causal illumination effect. A controlled experiment using relit versions of the same source images would be required to isolate that effect more directly.

## 4.3 Identity-level consistency and statistical evidence

The Friedman tests identified differences among the three illumination groups for Euclidean displacement (χ²(2) = 76.0, p = 3.139e-17) and cosine distance (χ²(2) = 76.0, p = 3.139e-17). Kendall's W was 1.0 for both metrics. In this analysis, W = 1.0 indicates complete agreement in the ranking of the three group conditions across the evaluated identities. It should not be interpreted as perfect recognition performance, perfect embedding quality, or detector accuracy.

All six pairwise identity-level comparisons remained statistically significant after Holm correction across the combined family of six comparisons. The consistency of the ordering across identities strengthens the descriptive evidence that the observed group pattern was not confined to a small subset of identities.

However, the statistical evidence should be interpreted within the study design. The descriptive statistics are based on unequal numbers of image observations across the three illumination groups (646 near-frontal, 418 oblique, and 1,064 large-angular-deviation observations). In contrast, the repeated-measures analysis uses identity-level group means from the same 38 identities across all three groups. The inferential sample therefore consists of 38 matched identities, rather than the total number of image observations. This distinction is important when interpreting the descriptive and inferential results.

## 4.4 Relationship to previous illumination research

Previous research has examined the effects of illumination direction on face recognition and has investigated subject-specific variation in illumination-related face-image representations. Moses et al. (1994) addressed the challenge of compensating for changes in illumination direction, while Beveridge et al. (2009) examined subject-specific illumination spaces using principal-angle analysis. These studies establish that illumination-related variation and identity-dependent representation structure are longstanding concerns in face recognition.

Other work has investigated illumination preprocessing as a means of addressing changes in facial appearance. Han et al. (2013) compared illumination preprocessing approaches, and Ochoa-Villegas et al. (2015) surveyed illumination challenges in two-dimensional face recognition. This prior work provides context for evaluating how illumination conditions relate to facial representations, but it does not establish that the specific reference-relative FaceNet displacement measurements reported here are equivalent to the recognition or preprocessing outcomes examined in those studies.

The present study differs in its empirical emphasis: it characterises Euclidean and cosine displacement relative to one common reference capture per identity and compares those measurements across three study-defined illumination geometry groups. This is a difference in experimental framing, not proof of research novelty. Existing work has already explored illumination variation, identity-dependent illumination structure, and distance-based face representation analysis. A broader systematic prior-art review would be needed before making a definitive novelty claim.

## 4.5 Implications for face-embedding evaluation

The findings indicate that reference-relative embedding displacement can provide a descriptive way to examine how representations vary across illumination geometry conditions within a fixed recognition model. Reporting both Euclidean displacement and cosine distance offers two familiar views of the relationship between reference and evaluation embeddings.

These measures are not independent forms of validation in this experiment. Because the embeddings were L2-normalised, squared Euclidean distance is mathematically related to cosine distance: squared Euclidean distance equals twice the cosine distance. The parallel trends in the two measures should therefore not be presented as two independent confirmations of the same finding.

The results may motivate further controlled evaluation of embedding variation under changing capture conditions. However, displacement alone does not establish whether a face-recognition system accepts or rejects an identity correctly. Recognition or verification performance would require separate evaluation using appropriate genuine and impostor comparisons and task-specific performance measures. Similarly, the present analysis does not implement or validate an operational drift detection system.

## 4.6 Limitations

Several limitations define the scope of these findings.

**Different image captures.** The evaluation compares different captures of the same identities under different illumination conditions. These are not exact transformations of identical source images. Other capture-related differences may contribute to the observed displacement, preventing exclusive attribution to illumination.

**Single reference capture.** Each identity was represented by one reference embedding. This design provides a common reference for the descriptive comparisons, but it does not estimate within-identity reference variability or the distribution of possible reference embeddings.

**Single recognition architecture.** Embeddings were generated using one fixed pretrained InceptionResnetV1 model trained on VGGFace2. The results cannot establish whether the same displacement pattern occurs in other architectures or embedding models.

**Dataset coverage and provenance.** The evaluation used a cropped frontal-pose subset of Extended Yale B, rather than the complete multi-pose dataset. The acquired archive was obtained through a community-hosted Academic Torrents listing and was not independently authenticated against an official dataset checksum. These factors should be considered when assessing reproducibility and external validity.

**Preprocessing and alignment.** The evaluation used the documented direct-resize preprocessing pipeline. A full-cohort face-alignment procedure was not established for this dataset. The results should therefore be understood in the context of the specific preprocessing pipeline used in this experiment.

**No recognition-performance evaluation.** The study did not measure identification accuracy, verification accuracy, false acceptance, false rejection, or recognition performance under operational conditions. Embedding displacement cannot be substituted for these outcomes.

**No operational drift detection.** The experiment did not calibrate detection thresholds, estimate operational false-alarm rates, or evaluate a streaming drift detection method. The findings are descriptive and should not be interpreted as evidence of deployment readiness.

**Exploratory hemisphere comparison.** The frontal and beyond-frontal categories contained unequal numbers of illumination conditions. Hemisphere membership is also related to angular geometry. The hemisphere comparison is consequently exploratory and does not isolate an independent hemisphere effect.

**Related distance measures.** Euclidean displacement and cosine distance are mathematically related for L2-normalised embeddings. Their agreement does not constitute independent validation using unrelated measurement constructs.


**Independence between identities.** Identity served as the unit of inference for the repeated-measures analyses. However, the available dataset and acquisition records do not independently establish statistical independence between identities. The inferential results should therefore be interpreted with this limitation in mind.

## 4.7 Future research directions

Future work should first investigate illumination variation using controlled transformations of identical source images. Such a design would help separate illumination effects from differences between independent captures. Multiple reference captures per identity could also be examined to assess the sensitivity of reference-relative displacement to reference selection.

A second direction is to evaluate additional face-recognition architectures using a consistent cohort and clearly documented preprocessing. This would determine whether the observed geometry-related displacement pattern is specific to the evaluated FaceNet model or is also present in other learned representations.

A further extension would connect embedding displacement to recognition outcomes. Genuine and impostor comparisons, verification metrics, and identity-level performance analysis would be necessary to determine whether larger displacement corresponds to a measurable reduction in recognition performance.

Finally, a separate drift-detection study could investigate calibrated thresholds, temporal monitoring, false-alarm behaviour, and detection delay under controlled changes in the input distribution. Such work would require its own experimental protocol and validation; it is not demonstrated by the present descriptive analysis.

## 4.8 Overall discussion

Experiment 20 provides an identity-level descriptive analysis of reference-relative FaceNet embedding displacement across illumination geometry conditions in the evaluated Cropped Extended Yale B cohort. The results show an ordered increase in both distance measures from near-frontal to large-angular-deviation conditions, with consistent group rankings across the 38 evaluated identities. The exploratory hemisphere comparison showed a similar increase under beyond-frontal conditions, although its interpretation is limited by unequal condition coverage and its relationship to angular geometry.

The findings contribute a focused empirical characterisation of embedding displacement under the specified dataset, reference-selection procedure, preprocessing pipeline, and pretrained model. They should be interpreted as descriptive evidence rather than proof of recognition failure, causal illumination effects, generalisation across architectures, or operational drift-detection capability.

# 5. Conclusion

This study examined reference-relative face embedding displacement across illumination geometry conditions using a fixed pretrained FaceNet/InceptionResnetV1 model and a cropped subset of the Extended Yale B dataset. The analysis focused on 38 identities and compared 512-dimensional face embeddings relative to one common reference capture per identity. The objective was to characterise how embedding displacement varied across near-frontal, oblique, and large-angular-deviation illumination conditions, rather than to evaluate face-recognition accuracy or develop an operational detection system.

The results demonstrated a consistent increase in both Euclidean displacement and cosine distance across the three illumination geometry groups. Identity-level repeated-measures analyses identified statistically significant differences for both measures, with all six pairwise comparisons remaining significant following the combined six-test Holm correction. The exploratory frontal and beyond-frontal comparison also showed greater mean displacement under beyond-frontal conditions. These findings provide descriptive evidence of systematic variation in reference-relative embedding representations across the evaluated illumination conditions.

The contribution of this study is its identity-level empirical characterisation of embedding displacement using a documented reference-selection procedure, a fixed pretrained recognition architecture, and complementary distance measurements. It also demonstrates the importance of distinguishing embedding-space variation from actual recognition performance. The agreement between Euclidean displacement and cosine distance should not be interpreted as independent validation because the two measures are mathematically related for L2-normalised embeddings.

Several limitations constrain the interpretation and generalisability of these findings. The analysis used different image captures rather than controlled transformations of identical source images, relied on one reference capture per identity, and evaluated only one pretrained architecture. Statistical independence between identities could not be independently established from the available dataset and acquisition records. Furthermore, the study did not evaluate identification or verification accuracy, demographic fairness, operational false-alarm behaviour, or real-time drift detection.

Future research should investigate controlled illumination transformations, multiple reference captures, additional face-recognition architectures, and direct relationships between embedding displacement and recognition outcomes. Separate temporal monitoring experiments would also be necessary to establish whether these measurements can support reliable drift detection under operational conditions.

Overall, Experiment 20 provides a focused empirical characterisation of reference-relative face embedding displacement across illumination geometry conditions within the evaluated dataset and model. Its findings establish a descriptive basis for further investigation, while the stated methodological limitations define the boundaries of the conclusions that can currently be drawn.

# References
Beveridge, J. R., Draper, B. A., Chang, J.-M., Kirby, M., Kley, H., & Peterson, C. (2009). Principal angles separate subject illumination spaces in YDB and CMU-PIE. *IEEE Transactions on Pattern Analysis and Machine Intelligence, 31*(2), 351–363. https://doi.org/10.1109/TPAMI.2008.200

Han, H., Shan, S., Chen, X., & Gao, W. (2013). A comparative study on illumination preprocessing in face recognition. *Pattern Recognition, 46*(6), 1691–1699. https://doi.org/10.1016/j.patcog.2012.11.022

Moses, Y., Adini, Y., & Ullman, S. (1994). Face recognition: The problem of compensating for changes in illumination direction. In J.-O. Eklundh (Ed.), *Computer vision—ECCV 1994* (Vol. 800, pp. 286–296). Springer. https://doi.org/10.1007/3-540-57956-7_33

Ochoa-Villegas, M. A., Nolazco-Flores, J. A., Barron-Cano, O., & Kakadiaris, I. A. (2015). Addressing the illumination challenge in two-dimensional face recognition: A survey. *IET Computer Vision, 9*(6), 978–992. https://doi.org/10.1049/iet-cvi.2014.0086