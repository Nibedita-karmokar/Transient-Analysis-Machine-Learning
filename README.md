# Transient Thermal Analysis Using Machine Learning

## Overview

This repository contains a machine-learning framework for fast transient thermal analysis of integrated circuits.

The framework uses a ConvLSTM-based neural network to learn the temporal and spatial relationship between power dissipation and temperature evolution. Once trained, the model can predict transient temperature profiles significantly faster than traditional physics-based thermal simulations.

The workflow consists of:

1. Generation of transient thermal simulation data
2. Training a ConvLSTM model using power and temperature maps
3. Predicting temperature evolution from new power traces

---

## Repository Structure

```text
Transient-Analysis-Machine-Learning/

├── Conv_LSTM_Training.py      # Model training
├── ConvLSTM_Pred.py           # Inference/prediction
│
├── Training_data/
│   └── excel_files_new_*/
│       └── tile_data_*.xlsx
│
├── Test_data/
│   └── tile_data_*.xlsx
│
└── README.md
```

---

## Requirements

```bash
pip install tensorflow numpy pandas matplotlib openpyxl
```

---

## Dataset Format

The model operates on transient power and temperature maps stored in Excel files.

Each file contains:

| Column | Description   |
| ------ | ------------- |
| x      | X-coordinate  |
| y      | Y-coordinate  |
| P      | Power density |
| T      | Temperature   |

Example:

```text
x   y   P          T
0   0   1.2e-6     300.1
1   0   8.0e-7     299.8
...
```

---

## Input

### Training Data

Training samples are organized into folders containing time-series data:

```text
Training_data/
└── excel_files_new_16_46.0/
    ├── tile_data_1.xlsx
    ├── tile_data_2.xlsx
    ├── ...
    └── tile_data_42.xlsx
```

Each folder represents a transient thermal simulation sequence.

### Power Maps

The network input consists of:

* 128 × 128 spatial power maps
* 42 temporal frames

Input tensor shape:

```python
(time_steps, height, width)
=
(42, 128, 128)
```

---

## Model Architecture

The framework uses a ConvLSTM-based neural network that captures:

* Spatial thermal coupling
* Temporal heat propagation
* Dynamic thermal behavior

The model learns a mapping:

```text
Power Maps (t)
      ↓
ConvLSTM Network
      ↓
Temperature Maps (t)
```

Training is implemented in:

```text
Conv_LSTM_Training.py
```

---

## Training

Run:

```bash
python Conv_LSTM_Training.py
```

The script:

1. Loads training datasets
2. Normalizes power and temperature maps
3. Builds the ConvLSTM network
4. Trains the model
5. Saves the trained model

Generated model:

```text
U_NET_transient_New_data_prev_1.h5
```

---

## Prediction

Run:

```bash
python ConvLSTM_Pred.py
```

The script:

1. Loads the trained model
2. Loads unseen power traces
3. Predicts transient temperature maps
4. Generates comparison plots

---

## Output

The framework generates:

### Predicted Temperature Maps

* Temperature distribution at each time step
* Spatial hotspot locations

### Thermal Metrics

* Peak temperature
* Average temperature
* Temperature evolution over time

### Visualization

* Ground-truth temperature maps
* Predicted temperature maps
* Prediction error plots

---

## Main Source Files

### Conv_LSTM_Training.py

Responsible for:

* Dataset loading
* Data preprocessing
* Normalization
* ConvLSTM model creation
* Training and model export

### ConvLSTM_Pred.py

Responsible for:

* Model loading
* Inference
* Temperature prediction
* Visualization

---

## Citation

If you use this code in academic work, please cite the associated publication describing the machine-learning-based transient thermal analysis methodology.

```text
Nibedita Karmokar, Sai-Wang Tam, Thanh Viet Dinh, Vidya A. Chhabria, Ramesh Harjani, and Sachin S. Sapatnekar,
“Analyzing the Impact of FinFET Self-Heating on the Performance of RF Power Amplifiers”, in Proc. International
Conference on Computer-Aided Design, IEEE, 2024
```
