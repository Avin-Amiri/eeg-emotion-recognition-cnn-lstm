````markdown
## A CNN LSTM Framework for EEG-Based Emotion Recognition


[![Python 3.9+](https://img.shields.io/badge/Python-3.9%2B-blue.svg)](https://www.python.org/)
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.x-EE4C2C.svg)](https://pytorch.org/)
[![PyG](https://img.shields.io/badge/PyG-PyTorch--Geometric-3C2179.svg)](https://pyg.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



This repository contains the official implementation of the hybrid deep learning model proposed for EEG-based emotion recognition using the **DEAP** dataset. The pipeline extracts power spectral density (PSD) features across standard EEG frequency bands and employs a cascading 1D-CNN and LSTM network to capture spatial-frequency and temporal dynamics for **Valence**, **Arousal**, **dominance** and **liking** classification.

---

## 📌 Architecture Overview

1. **Feature Extraction:**
   - 14 targeted EEG channels (frontal, temporal, and parietal lobes).
   - Frequency band decomposition using PyEEG: Theta (4–8 Hz), Slow Alpha (8–10 Hz), Alpha (8–12 Hz), Beta (12–30 Hz), and Gamma (30–45 Hz).
   - Spectral power features flattened per time window.

2. **Classification Network:**
   - **Spatial/Local Feature Extraction:** 1D Convolutional layers (`Conv1D`) with `BatchNormalization`, `GaussianNoise`, and `Dropout`.
   - **Temporal Modeling:** Long Short-Term Memory (`LSTM`) units to model sequential transitions across EEG segments.
   - **Dense Classifier:** Fully connected layers with Softmax output for 2-class / multi-class classification.

---

## 📂 Project Structure

```text
├── data/
│   └── raw_deap/          # Place s01.dat to s32.dat here (excluded via .gitignore)
├── CNN_LSTM.py            # Main preprocessing, training, and evaluation script
├── requirements.txt       # Python dependencies
├── .gitignore
└── README.md
```

---

## ⚙️ Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Avin-Amiri/eeg-emotion-recognition-cnn-lstm.git
   cd eeg-emotion-recognition-cnn-lstm
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate       # On Linux/macOS
   # or: venv\Scripts\activate    # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 📊 Dataset Preparation

1. Download the preprocessed Python version (`data_preprocessed_python.zip`) of the **DEAP dataset** from the [official DEAP portal](https://www.eecs.qmul.ac.uk/mmv/datasets/deap/).
2. Extract the files (`s01.dat` through `s32.dat`) into the directory:
   ```text
   data/raw_deap/
   ```

---

## 🚀 Usage

Run the complete pipeline (data loading, band-power feature extraction, training, and evaluation):

```bash
python CNN_LSTM.py
```

The script outputs:
- Real-time training metrics (Loss, Accuracy per epoch).
- Confusion Matrix and classification metrics (Precision, Recall, F1-Score).
- Serialized model checkpoints (`.keras`) in the designated artifacts directory.

---

## 📑 Citation

If you find this work or code useful in your research, please cite:

```bibtex
@inproceedings{amiri2025eeg,
  title={A CNN LSTM Framework for EEG-Based Emotion Recognition},
  author={Amiri, Zahra and Mansouri, Azadeh},
  booktitle={Proceedings of the International Conference on Computer and Knowledge Engineering (ICCKE)},
  year={2025}
}
```

---


Distributed under the MIT License. See `LICENSE` for more information.
````
