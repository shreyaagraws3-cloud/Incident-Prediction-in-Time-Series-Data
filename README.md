# Incident Prediction in Time-Series ECG Data

## Overview

This project implements a model that predicts whether an **abnormal heartbeat (incident)** will occur within a future time horizon using recent ECG signal history.

The problem is formulated as a **sliding-window time-series classification task**. Each window of past ECG samples is used to predict whether an abnormal beat will occur in the **next H time steps**.

The dataset used is **record 100 from the MIT-BIH Arrhythmia Database**.

---

# Problem Formulation

Given a time series signal:

* **Input window:** previous **W = 360 samples**
* **Prediction horizon:** next **H = 180 samples**

For window (i):

[
X_i = signal[i : i + W]
]

[
y_i = \max(incident[i + W : i + W + H])
]

A window is labeled **1** if any abnormal beat occurs within the prediction horizon.

Abnormal beat symbols used as incidents:

```
V, A, E, F, L, R
```

---

# Pipeline

### 1. Data Loading

ECG signals and annotations are loaded using the `wfdb` library.

### 2. Incident Label Generation

Beat annotations are converted into a **binary incident signal** aligned with ECG samples.

### 3. Sliding Window Dataset

A dataset of windows is created:

* Window length: **360 samples**
* Horizon: **180 samples**

### 4. Feature Engineering

Each window includes:

* **360 raw ECG values**
* Statistical features:

  * mean
  * standard deviation
  * maximum
  * minimum

Total features per window: **364**

### 5. Train/Test Split

The dataset is split **80/20** using a **time-ordered split (no shuffling)** to preserve temporal structure.

### 6. Feature Scaling

Features are standardized using **StandardScaler**.

### 7. Model

A **Balanced Random Forest Classifier** (`BalancedRandomForestClassifier`) is used because the dataset contains **strong class imbalance** between normal and abnormal windows.

---

# Evaluation

Model performance is evaluated using:

* **ROC-AUC**
* **PR-AUC**

PR-AUC is particularly useful because abnormal events are rare.

The project also visualizes:

* predicted probabilities vs true incidents
* ECG signals with abnormal beats
* feature importance across ECG samples

---

## Results

The trained model successfully learns patterns preceding abnormal beats and produces probability scores that increase near incident regions.


