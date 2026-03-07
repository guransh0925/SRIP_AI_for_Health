# SRIP_AI_for_Health
Sleep apnea detection pipeline using physiological signals (EEG, SpO2) and 1D CNN - SRIP 2026 submission

## Overview
This project builds a pipeline to detect breathing irregularities during sleep using overnight 
physiological signals from 5 participants. The pipeline covers data visualization, signal 
preprocessing, dataset creation, and classification using a 1D Convolutional Neural Network.

## Project Structure
- `Data/` — Raw participant data (AP01-AP05)
- `scripts/` — Processing and training scripts
- `models/` — Model architecture
- `Dataset/` — Processed windowed dataset
- `Visualizations/` — Per-participant PDF reports
- `results/` — Confusion matrix and evaluation outputs

## Setup
Install dependencies:
```
py -m pip install -r requirements.txt
```

## Usage

**Step 1: Generate visualizations**
```
py scripts/vis.py -name "Data/AP01"
```

**Step 2: Create dataset**
```
py scripts/create_dataset.py -in_dir "Data" -out_dir "Dataset"
```

**Step 3: Train and evaluate model**
```
py scripts/train_model.py
```

## Data
Each participant folder contains:
- **Nasal Airflow** (32 Hz) — measures airflow through the nose
- **Thoracic Movement** (32 Hz) — measures chest wall movement during breathing
- **SpO2** (4 Hz) — measures blood oxygen saturation
- **Flow Events** — annotated breathing irregularities (Hypopnea, Obstructive Apnea, Mixed Apnea)
- **Sleep Profile** — sleep stage annotations every 30 seconds

## Methodology

### Q1: Visualization (vis.py)
- Loaded all three physiological signals using Pandas, aligning them by timestamp
- Generated 5-minute window plots across the full 8-hour recording per participant
- Overlaid annotated breathing events (Hypopnea in red, Obstructive Apnea in yellow, 
  Mixed Apnea in purple) on the Nasal Airflow plot
- Saved all pages as a multi-page PDF per participant in the Visualizations/ directory

### Q2: Signal Preprocessing and Dataset Creation (create_dataset.py)
- Applied a 4th-order Butterworth bandpass filter (0.17–0.4 Hz) to all signals to retain 
  only the relevant breathing frequency range and remove noise
- Split signals into 30-second windows with 50% overlap (step size = 15 seconds)
  - Flow and Thorac: 960 samples per window (32 Hz × 30s)
  - SpO2: 120 samples per window (4 Hz × 30s)
- Labelled each window using the flow events file:
  - If a window overlaps by more than 50% with an annotated event, it receives that event label
  - Otherwise it is labelled as Normal
- Final dataset: 8800 windows across 5 participants

| Label | Count |
|-------|-------|
| Normal | 8047 |
| Hypopnea | 590 |
| Obstructive Apnea | 161 |
| Mixed Apnea | 2 |

### Q3: Model Training and Evaluation (train_model.py)
- Trained a 1D CNN with 3 convolutional layers on the windowed signals
- Input: 3-channel signal (Flow, Thorac, SpO2 resampled to 960 samples)
- Architecture: Conv1d(3→16) → Conv1d(16→32) → Conv1d(32→64) → Linear(64→4)
- Used weighted cross-entropy loss to address class imbalance
- Evaluated using Leave-One-Participant-Out Cross-Validation (5 folds)

## Results

| Metric | Value |
|--------|-------|
| Accuracy | 0.4350 |
| Precision (macro) | 0.2408 |
| Recall (macro) | 0.2464 |

### Confusion Matrix Interpretation
The confusion matrix reveals that the model struggles to distinguish between classes, 
largely due to severe class imbalance — 91% of windows are labelled Normal. While class 
weighting improved minority class detection compared to a baseline that predicted everything 
as Normal, performance remains limited. The model successfully detects some Hypopnea and 
Obstructive Apnea events but has a high false positive rate for Normal windows being 
misclassified as events.

## Limitations
- **Class imbalance** — Normal windows vastly outnumber event windows, making it hard 
  for the model to learn minority classes
- **Small dataset** — only 5 participants limits generalization
- **Simple architecture** — a more complex model (e.g. LSTM, Transformer) with more 
  epochs and data augmentation would likely improve performance
- **Mixed Apnea** — only 2 windows across all participants, making it impossible to 
  learn reliably

## Tools Used
- Python, Pandas, NumPy, SciPy, Matplotlib, PyTorch, Scikit-learn
- This project was completed with guidance from Claude (Anthropic). All code was 
  understood and verified by the student.
