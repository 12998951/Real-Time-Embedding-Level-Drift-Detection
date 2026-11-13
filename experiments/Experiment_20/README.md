# Experiment 20

## Title
Paired Identity-Level Evaluation of Within-Identity Embedding Drift Detection Under Multiple Controlled Biometric Acquisition Changes.

## Overview
This experiment investigates reference-relative face embedding displacement across illumination geometry conditions using the Cropped Extended Yale B dataset.

## Research Artifacts

| Component | Location |
|---|---|
| Experimental notebook | [Notebook](notebook/Experiment_20.ipynb) |
| Final research manuscript | [Phase 4.7 Manuscript](manuscript/experiment20_integrated_manuscript_v2_phase47_final.md) |
| Research figures | [Figures](figures/) |
| Supporting audits and closure records | [Supporting Records](supporting_records/) |

## Dataset
The frozen dataset package contains the source archive, embeddings, configuration, audits, and checksum manifest.

The dataset ZIP is maintained separately because of its size. It is not included in this repository.

## Key Findings
- 38 identities and 2,414 images.
- 2,128 non-reference observations across 56 illumination conditions.
- 512-dimensional FaceNet embeddings.
- Friedman chi-square: 76.0.
- p-value: 3.139e-17.
- Kendall's W: 1.0.
- Holm-adjusted pairwise comparisons remained significant.

## Research Limitations
- Acquisition captures are not controlled transformations.
- The dataset archive has not been authenticated against the official UCSD checksum.
- Exact model-weight checksum preservation is incomplete.
- One reference capture per identity was used.
- Only one embedding architecture was evaluated.
- The study does not establish recognition accuracy, operational drift detection, or deployment readiness.

## Audit Status
The Phase 4.7 automated manuscript audit reported one structural check requiring review. The conclusion insertion was verified, but the overall automated audit should not be described as fully passed.

The Phase 5.5 record documents administrative project closure. It does not establish independent scientific validation.

## Repository Structure

- `notebook/` — Experimental notebook
- `manuscript/` — Final integrated manuscript
- `figures/` — Four frozen research figures
- `supporting_records/` — Audit reports, closure records, and artifact index
