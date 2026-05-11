# MIMII Anomaly Detection

Industrial audio anomaly detection on the MIMII dataset using classical and deep learning approaches.

## Project Overview

This project explores anomaly detection in industrial machine sounds for predictive maintenance applications.  
The work focuses on comparing:

- A classical baseline model:
  - MFCC + One-Class SVM
- A deep learning approach:
  - Transformer-based Autoencoder

The experiments are conducted on selected subsets of the MIMII dataset.

## Dataset

Dataset: MIMII Dataset  
Machine types used in this project:

- Fan
- Pump
- Valve

The dataset contains both normal and abnormal machine operating sounds.

## Project Structure

```text
mimii-anomaly-detection/
│
├── data/
├── notebooks/
├── src/
├── results/
├── README.md
└── requirements.txt
```

## Technologies

- Python
- librosa
- scikit-learn
- PyTorch
- NumPy
- Matplotlib

## Planned Pipeline

1. Audio preprocessing
2. Feature extraction (MFCC / Mel-Spectrogram)
3. Baseline anomaly detection with One-Class SVM
4. Transformer Autoencoder training
5. Evaluation and comparison

## Goals

- Understand industrial audio anomaly detection workflows
- Compare classical and deep learning methods
- Analyze the strengths and limitations of each approach

## References

- Purohit et al., “MIMII Dataset: Sound Dataset for Malfunctioning Industrial Machine Investigation and Inspection”
- DCASE Challenge – Acoustic Anomaly Detection
