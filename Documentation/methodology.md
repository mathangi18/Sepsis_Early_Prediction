# Methodology

---

## Stage 1 — Exploratory Data Analysis

### Dataset Structure

The dataset is distributed across four independently collected partitions (A–D), each corresponding to a distinct patient cohort. All partitions were loaded and merged into a single longitudinal time-series panel.

| Property | Value |
|---|---|
| Total rows | 602,568 |
| Total patients | 1,275 |
| Partition overlap | None (verified — no shared patient IDs across A–D) |

### Datatypes

After merge, the unified dataset contains:
- **45 float64 columns** — clinical biomarkers and physiological measurements
- **2 int64 columns** — patient identifiers and time-step indices

### Data Quality

| Check | Result |
|---|---|
| Missing values | None detected |
| Zero-variance columns | None detected |

### Distribution Analysis

Several biomarker features exhibit heavy-tailed distributions indicative of acute physiological events. Skew was documented for all 45 float features; the most extreme were flagged for reference (see `Results/figures/top_skewed_histograms.png` and `Results/tables/feature_skewness.csv`). No features were removed at this stage.

### Correlation Structure

High pairwise Pearson correlations were identified among several clinically co-dependent biomarkers:

| Feature Pair | Correlation |
|---|---|
| bicarbonate ↔ base_excess | 0.98 |
| diastolic_bp ↔ mean_bp | 0.89 |
| mean_bp ↔ systolic_bp | 0.86 |
| sodium ↔ chloride | 0.83 |

These correlations reflect known physiological relationships. Features were retained at this stage; redundancy handling is deferred to feature importance analysis.

### Patient-Level Class Imbalance

| Class | Count |
|---|---|
| Non-septic | 979 |
| Septic | 296 |

The dataset is imbalanced at a ~3.3:1 ratio. This is noted for downstream evaluation strategy planning (e.g., stratified splits, class-weighted loss functions, calibration-aware metrics).

---

*Stage 1 established structural validity and data integrity required for onset definition and modeling.*

This project integrates core machine learning concepts from the course
into a structured time-series prediction pipeline.

Supervised Learning:
The task is formulated as a binary classification problem,
where the model predicts whether sepsis onset will occur within
2h, 4h, or 6h.

Linear / Logistic Regression:
Used as baseline models to benchmark neural network performance.

Bias–Variance Tradeoff:
Guides architecture complexity, regularization, and early stopping.

Neurons / MLPs:
Form the foundation of the custom neural network model.

Backpropagation:
Used to update model weights during training.

Activation Functions:
Applied in hidden layers to introduce non-linearity and
in output layer for probability estimation.

Gradient Descent Variants:
Modern optimizers (e.g., AdamW) will be used to stabilize training.

Train/Test Split:
Time-aware splitting at the patient level prevents data leakage.

Evaluation Metrics:
Accuracy, recall, precision, AUROC, and calibration will be used
to evaluate performance under class imbalance.

Interpretability:
Feature importance analysis will link predictions to physiological markers.

Ethics & Bias Awareness:
Error analysis will consider the clinical implications of false negatives
and false positives in early sepsis detection.

## Stage 3 — Baseline Modeling

### Objective
Establish a performance benchmark using a linear model (Logistic Regression) to justify the complexity of subsequent neural network architectures.

### Model Architecture
- **Algorithm**: Logistic Regression with L2 regularization.
- **Optimization**: L-BFGS (via PyTorch implementation for GPU acceleration).
- **Input**: Standard-scaled features (45 physiological markers + time encoding).
- **Target**: Binary classification of sepsis onset within 2h, 4h, and 6h horizons.

### Implementation Details
Due to the dataset size (602k rows), CPU training was inefficient. We implemented a custom PyTorchLogisticRegression estimator that:
1.  Moves data to GPU (CUDA).
2.  Optimizes using 	orch.optim.LBFGS.
3.  Exposes a scikit-learn compatible API (it, predict_proba).

### Evaluation Strategy
- **Metric**: Area Under the Receiver Operating Characteristic (AUROC).
- **Validation**: Patient-level stratified split (80/20) to prevent data leakage.
- **Result**: The baseline model achieved **AUROC ~0.76** across all horizons, setting a strong linear baseline.
