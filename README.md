# Real-Time Embedding-Level Drift Detection for Trustworthy Biometric Systems

This repository provides the official implementation of a **real-time, label-free, embedding-level drift detection framework** for biometric systems. The framework enables continuous monitoring of distributional drift directly in latent embedding space, without relying on recognition accuracy or labeled data.

## Motivation
Biometric systems deployed in real-world environments are subject to non-stationary conditions such as illumination changes, ageing, sensor variation, and acoustic noise. These factors introduce representation drift that can silently degrade system trustworthiness. This work addresses the need for **early, interpretable, and modality-agnostic drift detection**.

## Key Contributions
- Real-time embedding-level drift detection without labels  
- Unified drift detector for face and voice biometrics  
- Lightweight statistical drift signal with streaming operation  
- Interpretable drift dynamics (onset, transition, stabilization)  
- Reproducible experimental pipeline  

## Repository Structure
```text
drift_detector/        # core drift detection logic
embeddings/
  ├── face/            # face encoder + illumination drift experiments
  └── voice/           # voice encoder + noise drift experiments
utils/
  └── visualization/  # drift curve plotting
assets/               # drift curve figures
