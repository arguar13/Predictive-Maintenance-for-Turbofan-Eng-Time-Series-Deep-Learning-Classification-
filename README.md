*Read this in other languages: [Español](README_es.md)*
# Deep Learning Project — Predictive Maintenance for Turbofan Engines (NASA C-MAPSS)

## Overview

This project develops an advanced **Deep Learning benchmark for Predictive Maintenance** using the **NASA C-MAPSS (Commercial Modular Aero-Propulsion System Simulation)** dataset.

Instead of predicting the exact **Remaining Useful Life (RUL)** through regression, the problem is reformulated as a **multiclass multivariate time series classification task**, where engine health is categorized into operational risk levels.

The solution evaluates and compares **five State-of-the-Art Deep Learning architectures** for multivariate time series classification, implemented in **PyTorch** following an end-to-end industrial machine learning workflow.

---

#  About the Dataset

**Dataset:** Turbofan Engine Degradation Simulation Dataset (C-MAPSS)

**Source:** NASA Prognostics Data Repository

---

# General Description

The C-MAPSS dataset simulates the degradation process of turbofan aircraft engines operating under different environmental conditions and fault modes.

Each engine is monitored through multiple sensor channels that capture its thermodynamic behavior over time until failure.

The dataset contains:

* Operational settings
* Sensor measurements
* Engine identifiers
* Complete degradation trajectories


---

# Problem Reformulation

Instead of predicting the exact Remaining Useful Life (RUL), the problem is transformed into a multiclass classification task.

Engine health states are defined as follows:

| Class    | Remaining Useful Life      |
| -------- | -------------------------- |
| Healthy  | RUL > 60 cycles            |
| Alert    | 30 < RUL ≤ 60 cycles       |
| Critical | RUL ≤ 30 cycles            |

This formulation provides more actionable outputs for maintenance planning and industrial decision-making.

---

#  Feature Engineering

The project includes extensive feature engineering techniques designed to enhance degradation pattern detection.

* **Dynamic features:** causal rolling mean and rolling standard deviation (window = 5 cycles) of every sensor, computed per engine.
* **Static features:** engine baseline signature (mean of each sensor over the first 5 cycles) and an embedding of the sub-dataset (FD001–FD004, i.e. operating conditions / fault modes).
* **Data preparation:** Z-score scaling fitted on the training split only and sliding windows of 30 cycles (`[Windows, 30, 66]` tensors), labelled with the class of the last cycle of each window.

---

#  Data Leakage Prevention

A critical aspect of predictive maintenance projects is preventing information leakage.

To ensure realistic evaluation:

* Entire engine trajectories remain in a single split.
* Group-based Train/Validation/Test partitioning (70% / 15% / 15% of the engines, `GroupShuffleSplit`).
* Scaler and invariant-sensor detection are fitted on the training split only.
* No future information is used during feature generation.
* RUL values are removed after target construction.

---

#  Deep Learning Architectures Evaluated

Five advanced architectures were implemented and benchmarked.

## 1. FCN (Fully Convolutional Network)

Traditional strong baseline for time series classification.

---

## 2. InceptionTime

State-of-the-Art convolutional architecture for time series classification.

---

## 3. Time Series Transformer

Transformer-based architecture using self-attention.

---

## 4. PatchTST

Recent State-of-the-Art Transformer architecture.

---

## 5. ConvTransformer

Hybrid CNN-Transformer architecture.

---

#  Training Strategy

The training pipeline includes:

* PyTorch Implementation
* GPU Acceleration
* Early Stopping
* Gradient Clipping
* Class Weight Balancing
* Adam Optimizer
* CrossEntropy Loss
* Fixed random seeds for reproducibility

**Model selection:** the best architecture is chosen by **validation Macro F1**; the test split is only used to report final metrics.

**Evaluation data:** all splits are built from the run-to-failure `train_FD00x` trajectories, because the class labels require the full RUL of every cycle. The official `test_FD00x` / `RUL_FD00x` files are included in `data/` for reference.

---

#  Generated Outputs

The project automatically produces:

*  Benchmark Comparison Table
*  Accuracy Comparison
*  Macro F1 Comparison
*  Training Time Comparison
*  Confusion Matrix
*  Classification Report
*  ROC Curves
*  Engine Degradation Visualizations
*  Best Model Identification

---

#  Results

Test-set metrics (107 unseen engines, 20,641 windows). Training on an NVIDIA GeForce GTX 1650.

| Model                 | Val Macro F1 | Test Macro F1 | Test Accuracy | Training Time (s) |
| --------------------- | :----------: | :-----------: | :-----------: | :---------------: |
| **PatchTST**          | **0.779**    | 0.792         | **0.841**     | 504               |
| TimeSeriesTransformer | 0.774        | **0.794**     | 0.834         | 221               |
| ConvTransformer       | 0.750        | 0.738         | 0.792         | 213               |
| InceptionTime         | 0.723        | 0.753         | 0.803         | 104               |
| FCN (Baseline)        | 0.683        | 0.681         | 0.750         | 134               |

**Selected model: PatchTST** (best validation Macro F1). On the test set it reaches ROC AUC of 0.988 (Critical), 0.967 (Healthy) and 0.895 (Alert). The *Alert* class is the hardest one (F1 = 0.61) because it is the transition zone between healthy operation and imminent failure, while *Critical* engines are detected with a recall of 0.83 and are almost never confused with *Healthy* (5 of 3,317 windows).

Both Transformer-based models clearly outperform the convolutional baselines; TimeSeriesTransformer offers a similar Macro F1 at less than half of PatchTST's training time.

---

#  Project Structure

```
├── data/
│   ├── train_FD001.txt ... train_FD004.txt   # run-to-failure trajectories (used)
│   ├── test_FD001.txt  ... test_FD004.txt    # official NASA test set (reference)
│   ├── RUL_FD001.txt   ... RUL_FD004.txt     # official test RUL (reference)
│   ├── readme.txt                            # dataset description (NASA)
│   └── Damage Propagation Modeling.pdf       # reference paper (Saxena et al., 2008)
├── notebooks/
│   └── Predictive Maintenance.ipynb          # full pipeline: EDA → FE → 5 models → benchmark
├── requirements.txt
├── README.md
└── README_es.md
```

---

#  How to Run

```bash
git clone https://github.com/arguar13/Predictive-Maintenance-for-Turbofan-Eng-Time-Series-Deep-Learning-Classification-.git
cd Predictive-Maintenance-for-Turbofan-Eng-Time-Series-Deep-Learning-Classification-
pip install -r requirements.txt
jupyter notebook "notebooks/Predictive Maintenance.ipynb"
```

The notebook reads the data from the relative `data/` folder and uses the GPU automatically when available (CPU also works).

---

#  Business Impact

This solution can support:

* Predictive Maintenance Scheduling
* Aircraft Fleet Management
* Spare Parts Planning
* Downtime Reduction
* Failure Prevention
* Asset Health Monitoring

The multiclass approach provides interpretable maintenance alerts directly usable by operational teams.

---

#  License

This project is intended for educational, research, and portfolio purposes.

Dataset provided by NASA Ames Research Center.

---

## Author

**Armando Guarnera**
Data Scientist
Argentina