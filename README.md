## Time-Series Incident Prediction in ECG Signals
This project implements a machine learning model to predict whether an abnormal heartbeat (incident) will occur within a future horizon, based on recent ECG signal history.
The task uses a sliding-window time-series classification formulation, where each window of past ECG samples is used to predict if an incident occurs in the next H time steps.
The ECG dataset comes from the **MIT-BIH Arrhythmia Database (records 100, 101, 102)**, accessed via PhysioNet.

For testing Two datasets were evaluated:
1. Single‑Record Dataset (Record 100)
- More homogeneous ECG morphology
- Model achieves higher confidence
- Useful for baseline performance
2. Multi‑Record Dataset (Records 100, 101, 102)
- More diverse ECG patterns
- Stronger distribution shift
- Model struggles to generalize
- Highlights the challenge of cross‑patient prediction
  
### Problem Formulation
Input window (W): 360 samples (~1 second)
Prediction horizon (H): 180 samples (~0.5 seconds)
A window is labeled 1 if any abnormal beat occurs in the next H samples.
Abnormal beats considered:
**V, A, E, F, L, R**

### 1. Data Loading
ECG signals and annotations loaded with wfdb
Signals from multiple records are combined for analysis

### 2. Incident Label Generation
Beat annotations converted into binary incident arrays
Abnormal beats marked as 1, normal beats as 0

### 3. Sliding-Window Dataset
For each window of 360 samples, check the next 180 samples
Label as 1 if any incident occurs in the horizon, else 0

###  4. Feature Engineering
Each window includes:
360 raw ECG samples

### 4 statistical features:
mean
standard deviation
maximum
minimum
Total features per window: 364

### 5. Train/Test/Validation Split
First split: 80% of the data is used for training + validation, 20% for testing
Second split: From the 80% training set, 20% is held out as validation

### 6. Feature Scaling
Standardized using StandardScaler to normalize values

### 7. Model
XGBoost (XGBClassifier)
Gradient-boosted tree model suitable for tabular sliding-window features
Class weighting or oversampling not applied in this version

### Evaluation
Metrics used:
ROC-AUC
PR-AUC (more meaningful for rare events)
Precision, Recall, F1 at multiple thresholds

### Visualizations include:
ECG signal with abnormal beats highlighted
Predicted probability vs. true incidents
Threshold sweep curves
Feature importance across ECG window

##### Results Summary
- Model probability rises near true incidents, indicating some pattern learning
- However, maximum predicted probability on the test set is extremely low (~0.06)

##### Limitations
Severe class imbalance 
Raw ECG windows may lack discriminative features
Distribution differences between training and test segments

##### Future Improvements
Add class weights (scale_pos_weight) in XGBoost or Balanced Random Forest
Use oversampling techniques like SMOTE for rare events





