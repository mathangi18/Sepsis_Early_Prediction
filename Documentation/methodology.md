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
