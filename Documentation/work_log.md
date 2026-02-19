# Work Log

---

## Stage 1 — EDA & Structural Validation

| Step | Activity | Notes |
|---|---|---|
| 1 | Partition validation | Loaded partitions A–D; verified schema consistency across all files |
| 2 | ID overlap verification | Confirmed zero overlap of patient IDs across all four partitions |
| 3 | Merge operation | Concatenated all partitions into unified panel: 602,568 rows × 1,275 patients |
| 4 | Datatype & missingness check | 45 float64, 2 int64; no missing values; no zero-variance columns detected |
| 5 | Distribution & skew analysis | Computed skewness for all 45 float features; heavy-tailed biomarkers flagged |
| 6 | Correlation mapping | Computed pairwise Pearson matrix; top correlated pairs extracted and recorded |
| 7 | Patient-level class imbalance analysis | Aggregated sepsis labels per patient: 979 non-septic, 296 septic |
| 8 | Exported figures and tables | All outputs written to `Results/` |

### Exported Artifacts

**Figures:**
- `Results/figures/correlation_matrix.png`
- `Results/figures/top_skewed_histograms.png`
- `Results/figures/patient_level_class_distribution.png`

**Tables:**
- `Results/tables/feature_skewness.csv`
- `Results/tables/patient_level_class_distribution.csv`

**Interim Data:**
- `Data/interim/sepsis_timeseries_full.pkl`
