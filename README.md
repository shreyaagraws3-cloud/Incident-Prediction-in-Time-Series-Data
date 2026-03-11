## Incident Prediction in Time Series Data
This project implements a **machine‑learning pipeline** to predict whether an abnormal heartbeat (incident) will occur in the near future based on ECG signals.
The task is formulated as a sliding‑window time‑series classification problem.
The dataset is derived from the **MIT‑BIH Arrhythmia Database (records 100, 101, 102)**.

### Problem Statement
Given a continuous ECG time series:
- **Input window (W)**: 360 samples (~1 second)
- **Prediction horizon (H)**: 180 samples (~0.5 second)
- A window is labeled 1 if any abnormal beat occurs within the next H samples.
Abnormal beat types considered as incidents:
**V, A, E, F, L, R**

**Goal**: Predict the likelihood of an abnormal beat in the next H steps, given the last W steps.

### Pipeline
#### 1. Data Loading
- ECG signals and annotations loaded using wfdb
- Records 100, 101, 102 combined (**~1.95M samples**)
#### 2. Incident Labeling
- Beat annotations converted into a binary incident signal
- Abnormal beats marked as 1
#### 3. Sliding Windows
Each training sample consists of:
- **360 ECG samples**
- **4 statistical features**: mean, std, max, min
- Label: 1 if any incident occurs in the next 180 samples
#### 4. Train / Validation / Test Split
- 80% train / 20% test, preserving temporal order
- 20% of training set used as validation
- Stratified split ensures rare abnormal windows appear in validation
#### 5. Feature Scaling
- Standardized using StandardScaler
#### 6. Model
XGBoost (XGBClassifier) with parameters:
- n_estimators = 300
- max_depth = 5
- learning_rate = 0.05
- subsample = 0.8
- colsample_bytree = 0.8
- scale_pos_weight = num_normal / num_abnormal
- tree_method = "hist"

#### 7. Evaluation
Metrics
- **Precision / Recall / F1‑score** (across multiple thresholds)
- **ROC‑AUC**
- **PR‑AUC**
Visualizations
- Predicted probability vs. true incidents
- ECG signal with abnormal beats highlighted
- Threshold sweep curves
- Feature importance across the input window

##### Results
ROC-AUC : 0.633
PR-AUC: 0.0013


#### 8. Observations
- Predicted probabilities remain extremely low even near incidents
- Threshold tuning (0.01–0.05) does not produce true positives
- Severe class imbalance (41 incidents in 1.95M samples) is the main challenge
- Basic ECG window features are not discriminative enough for rare events

#### Limitations
- Model fails to generalize to unseen test data (no true positives)
- Sliding‑window + simple features insufficient for rare abnormal beats
- Extreme class imbalance remains a bottleneck
- No oversampling or advanced ECG‑specific feature extraction applied

##### Future Improvements
Data Augmentation / Oversampling
Advanced Models - 1D CNN


### Additional Notes / Baseline

An earlier version of the pipeline was run only on record 100 as a baseline / proof-of-concept.

Results and visualizations for record 100 are available in the main branch of this repository.

The combined multi-record version (100, 101, 102) is in the current branch and is used for final analysis.

