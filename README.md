# Incident-Prediction-in-Time-Series-Data

## Project Overview
Predict whether an **incident** (abnormal heartbeat) will occur within the next **H** time steps using the previous **W** ECG samples. This notebook implements a sliding-window formulation on MIT‑BIH record `100`, builds window-level features, trains classifiers, and evaluates alerting performance with per-alert metrics and lead time.

### Quick configuration used in the notebook
- **Window size**: **W = 360** samples  
- **Prediction horizon**: **H = 180** samples  
- **Labeling rule**: \(y_i = \max(\text{incident}_{i+W : i+W+H-1})\) — a window is positive if any annotated abnormal beat occurs in the next \(H\) samples.  
- **Incident symbols**: **V, A, E, F, L, R** (from MIT‑BIH annotations)  
- **Models demonstrated**: Logistic Regression baseline (suggested) and **BalancedRandomForestClassifier** (used in the notebook)

### What the notebook contains
- **Data loading**: loads MIT‑BIH record `100` via `wfdb` and optionally saves a CSV.  
- **Annotation mapping**: converts beat annotation symbols into a binary `incident` array aligned with ECG samples.  
- **Sliding-window pipeline**: `build_windows(signal, labels, W, H)` that returns \(X\in\mathbb{R}^{N\times W}\) and \(y\in\{0,1\}^N\).  
- **Feature engineering**: raw window samples plus appended per-window statistics (**mean**, **std**, **max**, **min**).  
- **Train/validation/test split**: time-aware split (no shuffling) implemented in the notebook.  
- **Model training**: scaling, class-imbalance handling via `BalancedRandomForestClassifier`, model fitting, and saving predictions/probabilities.  
- **Evaluation**: ROC-AUC, PR-AUC, probability vs true-incident plots, feature importances.  
- **Alert simulation**: streaming-style simulator that triggers alerts when score > threshold, applies debounce of length \(H\), computes per-alert precision, false alarms per time unit, and lead-time distribution.  
- **Threshold sweep**: evaluates precision/recall/F1/avg lead time/false alarms across thresholds and selects a validation-optimal threshold.

### How labels and windows are computed (precise)
- Input windows: \(X_i = \text{signal}[i : i+W-1]\)  
- Label: \(y_i = \max(\text{incident}[i+W : i+W+H-1])\)  
- Implementation detail: windows are generated with stride = 1 by default; stride is configurable for faster experiments.

### Evaluation metrics reported
- **Per-timestep**: AUC-ROC, AUC-PR, Precision, Recall, F1.  
- **Per-alert**: Per-alert precision (TP / (TP + FP)), false alarms per hour/day, **average lead time** for true positives, lead-time histogram.  
- **Calibration**: optional reliability diagram and Platt scaling/isotonic calibration on validation set (not required but included as a notebook cell template).

### Limitations and notes
- Notebook demonstrates a single-record experiment (MIT‑BIH record `100`) — results do not generalize across patients without multi-record training and patient-level splits.  
- Incidents are sparse (beat-level), so class imbalance is strong; the notebook uses a balanced forest but also includes notes on class weights and resampling for other models.  
- Sliding stride = 1 is computationally heavy for long records; use larger stride for faster prototyping but keep stride = 1 for final evaluation/alert simulation.  
- Debounce logic is applied in the alert simulator to avoid duplicate alerts for the same incident; tune debounce length to match operational needs.

### Reproducibility and running the notebook
- **Requirements**: `wfdb`, `numpy`, `pandas`, `scikit-learn`, `imbalanced-learn`, `matplotlib`, `joblib` (install via `pip`).  
- **Seed**: set `np.random.seed(42)` and `random.seed(42)` in the notebook for reproducibility.  
- **Run order**: run cells sequentially: data load → annotation mapping → build windows → train/test split → scaling → model training → evaluation → threshold sweep → alert simulation.  
- The notebook saves key artifacts (predictions, threshold sweep table, lead-time histogram) to local files for inspection.

---

**If you want, I can now produce a compact single-file README.md text ready to paste into your repo (no extra structure).**
