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

The advancement of deep learning architectures, contributes to the effective design of methods for EEG based emotion recognition. In this code, a CNN-LSTM model is presented, where 1D-CNN reduces dimensionality and extracts meaningful features before passing to LSTM, which improves computational efficiency. This architecture provides better detection results compared to using either CNN or LSTM by combining these two model in an efficient way and change the window size in feature extracting to extract better features. The experimental results on the DEAP dataset demonstrate its superior performance.

## 🏛 Framework Architecture

```text
Raw EEG Data (32 Subjects × 40 Trials × 14 Channels × 8064 Samples)
                              │
  [Sliding Window: Size=512 (4s), Step=16 (0.125s), Fs=128 Hz]
                              │
    [Band-Power Feature Extraction (band-power): 4-8, 8-12, 12-16, 16-25, 25-45 Hz]
                              │
          Shape: (Batch, 70, 1) — 14 Channels × 5 Spectral Bands
                              │
┌─────────────────────────────┴─────────────────────────────┐
│ Stage 1: Multi-Scale 1D-CNN Spatial-Spectral Extraction   │
│   • Conv1D (256 filters, k=7) + BN + MaxPool(2) + Drop(0.2) │
│   • Conv1D (128 filters, k=5) + BN + MaxPool(2) + Drop(0.2) │
│   • Conv1D (64 filters,  k=3) + BN + MaxPool(2) + Drop(0.2) │
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
│   • Dense (384, L2=1e-4) + BN + Dropout (0.20)            │
│   • Dense (64,  L2=1e-4) + BN + Dropout (0.20)            │
│   • Dense (32,  L2=1e-4) + Dropout (0.15)                 │
│   • Output Dense (2 units, Softmax) ──► Class Probabilities │
└───────────────────────────────────────────────────────────┘
```

---

## 📊 Channel & Frequency Configurations

### Selected 14 Channels (DEAP indices)
Channels selected by array indices from the DEAP preprocessed dataset:
`channel = [1, 2, 3, 4, 6, 11, 13, 17, 19, 20, 21, 25, 29, 31]`

### Frequency Bands
Spectral power features are extracted across 5 standard EEG bands (`[4, 8, 12, 16, 25, 45]` Hz):

- **Theta ($\theta$):** 4 – 8 Hz
- **Alpha ($\alpha$):** 8 – 12 Hz
- **Low Beta ($\beta_1$):** 12 – 16 Hz
- **High Beta ($\beta_2$):** 16 – 25 Hz
- **Gamma ($\gamma$):** 25 – 45 Hz

---

## 📁 Repository Structure

```text
├── data/
│   └── raw_deap/          # Place DEAP dataset files: s01.dat to s32.dat
├── results/               # Generated artifacts (models, plots, metrics)
├── CNN_LSTM.py            # Complete end-to-end preprocessing, training & evaluation
├── requirements.txt        # Dependencies
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

*Note: The target label is set to Arousal by default (`label_idx = 0`). Modify `label_idx` in `CNN_LSTM.py` for other dimensions (e.g., `1` for Valence).*

### Step 3: Outputs & Evaluation
- Trained model weights: `results/best_model.keras`
- Training curves: `results/accuracy.png` & `results/loss.png`
- Evaluation metrics & confusion matrix: `results/confusion_matrix.png` & `results/results.pkl`

---

## 🔬 Hyperparameters

| Hyperparameter | Value | Description |
|---|---|---|
| Input Dimension | $(N, 70, 1)$ | 14 channels $\times$ 5 frequency bands |
| Window Size | 512 samples (4.0 s) | Sliding time window |
| Step Size | 16 samples (0.125 s) | Sliding window overlap |
| Sampling Rate ($F_s$) | 128 Hz | Downsampled DEAP rate |
| Conv1D Stages | 3 layers | Filter sizes: 256 ($k=7$), 128 ($k=5$), 64 ($k=3$), Dropout: 0.2 |
| LSTM Units | 128 | $\text{tanh}$ activation, $L_2$ kernel/recurrent reg ($1\times 10^{-4}$), bias reg ($1\times 10^{-5}$) |
| Optimizer | Adam | Learning Rate: $4 \times 10^{-4}$ |
| Loss Function | Categorical Crossentropy | Binary classification (threshold: 5.0) |
| Batch Size | 1024 | Mini-batch sample size |
| Max Epochs | 300 | Guarded by Early Stopping |
| Early Stopping | Patience = 12 | Monitored on `val_loss` (`min_delta` = $1\times 10^{-4}$) |
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
```
