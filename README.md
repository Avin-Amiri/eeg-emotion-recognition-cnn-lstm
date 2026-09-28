````markdown
# A CNN-LSTM Framework for EEG-Based Emotion Recognition

[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00.svg)](https://tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-D00000.svg)](https://keras.io/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of the paper:  
**"A CNN-LSTM Framework for EEG-Based Emotion Recognition"**  
*Zahra Amiri, Azadeh Mansouri*  
*Department of Electrical and Computer Engineering, Kharazmi University, Tehran, Iran*  
*Presented at the 15th International Conference on Computer and Knowledge Engineering (ICCKE 2025)*

---

## 📌 Overview

Accurate emotion recognition from electroencephalogram (EEG) signals requires capturing localized spectral-spatial patterns across critical brain regions as well as sequential dependencies across time segments.

This repository provides an end-to-end deep learning framework that:
1. **Extracts Band Power Spectral Density (PSD):** Decomposes raw EEG signals from 14 salient cortical channels across 5 canonical frequency bands ($\theta$, slow $\alpha$, $\alpha$, $\beta$, $\gamma$) using a sliding window of 4 seconds ($512$ points) with step size of $0.125$ seconds ($16$ points), generating a 70-dimensional spatial-spectral feature vector.
2. **Deep Hierarchical Feature Extraction (3-Stage 1D-CNN):** Employs multi-scale 1D Convolutional blocks with Batch Normalization, Max Pooling, and Dropout to capture hierarchical spatial representations.
3. **Sequence & Dynamics Modeling (Regularized LSTM):** Utilizes an $L_2$-regularized Long Short-Term Memory layer to model temporal contextual transitions across feature representations.
4. **Dense Classification:** Multi-layer perceptron (MLP) with Batch Normalization and Dropout for robust emotion state classification (Arousal / Valence).

Evaluated on the benchmark **DEAP (Database for Emotion Analysis using Physiological Signals)** dataset using stratified splits.

---

## 🏛 Framework Architecture

```
Raw EEG Data (32 Subjects × 40 Trials × 14 Channels × 8064 Samples)
                              │
  [Sliding Window: Size=512 (4s), Step=16 (0.125s), Fs=128 Hz]
                              │
  [Band-Power Feature Extraction (pe.bin_power): θ, slow-α, α, β, γ]
                              │
          Shape: (Batch, 70, 1) — 14 Channels × 5 Spectral Bands
                              │
┌─────────────────────────────┴─────────────────────────────┐
│ Stage 1: Multi-Scale 1D-CNN Spatial-Spectral Extraction   │
│   • Conv1D (256 filters, k=7) + BatchNorm + MaxPool1D(2) │
│   • Conv1D (128 filters, k=5) + BatchNorm + MaxPool1D(2) │
│   • Conv1D (64 filters,  k=3) + BatchNorm + MaxPool1D(2) │
│   • Gaussian Noise Injection (σ = 0.02)                   │
└─────────────────────────────┬─────────────────────────────┘
                              │
┌─────────────────────────────┴─────────────────────────────┐
│ Stage 2: Temporal Sequence Encoding (LSTM)                │
│   • LSTM (128 units, tanh) + L2 Regularization            │
│   • Batch Normalization + Dropout (0.20)                  │
└─────────────────────────────┬─────────────────────────────┘
                              │
┌─────────────────────────────┴─────────────────────────────┐
│ Stage 3: Fully Connected Classifier & Inference           │
│   • Dense (384) + BatchNorm + Dropout (0.20)              │
│   • Dense (64)  + BatchNorm + Dropout (0.20)              │
│   • Dense (32)  + Dropout (0.15)                          │
│   • Output Dense (Softmax) ──► Emotion Classes Logits     │
└───────────────────────────────────────────────────────────┘
```

---

## 📊 Channel & Frequency Configurations

### 1. Selected 14 EEG Channels (10–20 System)
Frontal, Temporal, Parietal, and Occipital channels capturing emotional and cognitive responses:
- `Fp1`, `AF3`, `F3`, `F7`, `FC5`, `T7`, `P7`, `O1`, `Oz`, `Pz`, `Fp2`, `AF4`, `Fz`, `F4`

### 2. Frequency Bands
| Frequency Band | Range (Hz) | Neural Relevance |
|---|---|---|
| **Theta ($\theta$)** | 4 – 8 Hz | Drowsiness, deep emotional states, meditation |
| **Slow Alpha (slow-$\alpha$)** | 8 – 10 Hz | Calmness, resting state |
| **Alpha ($\alpha$)** | 12 – 16 Hz | Relaxed alertness, internal focus |
| **Beta ($\beta$)** | 16 – 25 Hz | Active thinking, focus, emotional arousal |
| **Gamma ($\gamma$)** | 25 – 45 Hz | High-level cognitive processing, multi-modal integration |

---

## 📁 Repository Structure

```text
├── data/
│   └── raw_deap/          # Place DEAP dataset files: s01.dat to s32.dat
├── CNN_LSTM.py            # Complete end-to-end preprocessing, training & evaluation
├── requirements.txt       # Dependencies
├── .gitignore             # Git ignore rules for checkpoints and large arrays
└── README.md
```

---

## ⚙️ Installation & Environment Setup

### 1. Clone & Virtual Environment

```bash
git clone https://github.com/Avin-Amiri/eeg-emotion-recognition-cnn-lstm.git
cd eeg-emotion-recognition-cnn-lstm

python -m venv .venv
source .venv/bin/activate    # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Execution & Usage

### Step 1: Data Preparation
Download the preprocessed Python version (`data_preprocessed_python.zip`) of the **DEAP dataset** and place the participant files inside `data/raw_deap/`:
```text
data/raw_deap/s01.dat
data/raw_deap/s02.dat
...
data/raw_deap/s32.dat
```

### Step 2: Run Training Pipeline
Run the script to extract spectral power features, train the CNN-LSTM network, and evaluate performance:

```bash
python CNN_LSTM.py
```

### Step 3: Outputs & Evaluation
- Trained model checkpoints are saved automatically to `checkpoint.keras`.
- Generates detailed confusion matrix and classification reports (Accuracy, Precision, Recall, F1-Score).

---

## 🔬 Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| Input Dimension | $(N, 70, 1)$ | 14 channels $\times$ 5 frequency bands |
| Window Size | 512 samples (4.0 s) | Sliding time window |
| Step Size | 16 samples (0.125 s) | Sliding window overlap |
| Sampling Rate ($F_s$) | 128 Hz | Downsampled DEAP rate |
| Conv1D Stages | 3 layers | Filter sizes: 256 ($k=7$), 128 ($k=5$), 64 ($k=3$) |
| LSTM Units | 128 | $\text{tanh}$ activation, $L_2$ kernel/recurrent reg ($1\times 10^{-4}$) |
| Optimizer | Adam | Learning Rate: $4 \times 10^{-4}$ |
| Loss Function | Categorical Crossentropy | Multi-class / Binary classification |
| Batch Size | 1024 | Mini-batch sample size |
| Max Epochs | 300 | Guarded by Early Stopping |
| Early Stopping | Patience = 12 | Monitored on `val_loss` ($\text{min\_delta} = 1\times 10^{-4}$) |
| LR Scheduler | ReduceLROnPlateau | Factor = 0.1, Patience = 3, Min LR = $1\times 10^{-5}$ |
| Train / Test Split | 80% / 20% | Stratified shuffle split |

---

## 📖 Citation

If you find this codebase or research useful, please cite our paper:

```bibtex
@inproceedings{amiri2025cnn,
  title={A CNN LSTM Framework for EEG-Based Emotion Recognition},
  author={Amiri, Zahra and Mansouri, Azadeh},
  booktitle={Proceedings of the 15th International Conference on Computer and Knowledge Engineering (ICCKE 2025)},
  year={2025},
  organization={Faculty of Engineering, Kharazmi University}
}
```

---

## 📬 Contact & Inquiries

For technical questions or prospective research discussions, please contact:
- **Avin Amiri** — [zahraamiri@khu.ac.ir](mailto:zahraamiri@khu.ac.ir)
````
