# Strategic Decisions

---

## Stage 1 Decisions

- **Merge all partitions.** Partitions A–D were verified as independent cohorts (no overlapping patient IDs). Full merge was performed to maximise sample size for EDA.
- **Use pickle for interim storage.** Parquet serialisation caused version conflicts in the local environment. Pickle (`.pkl`) was used for the merged interim dataset to ensure reliable round-trip fidelity.
- **Do not perform additional normalisation.** Initial inspection suggests the dataset may already be pre-standardised. No further normalisation was applied at this stage to avoid double-scaling.
- **Do not remove skewed features at EDA stage.** Heavy-tailed distributions are clinically expected for acute biomarkers (e.g., lactate, troponin). Removal decisions are deferred to feature selection.
- **Retain correlated features.** Highly correlated pairs (e.g., bicarbonate ↔ base_excess at 0.98) reflect known clinical co-dependencies. Redundancy will be handled during feature importance analysis in later stages.
- **Use patient-level imbalance for evaluation planning.** The ~3.3:1 non-septic to septic ratio informs downstream decisions around stratified splitting, class-weighted training, and calibration-aware evaluation metrics.

## Stage 3 Decisions

- **Switch to GPU-Accelerated Logistic Regression.** Scikit-learn's CPU implementation was prohibitively slow for the fully expanded dataset. We implemented a custom PyTorch module (PyTorchLogisticRegression) to enable rapid iteration.
- **Enforce Verification of Data Scaling.** Convergence warnings in the baseline model highlighted the need for scaling. We mandated StandardScaler in the pipeline to ensure reliable gradient descent.
- **Automated Visualization of Metrics.** To prevent manual errors in reading results, we implemented a robust visualization cell that automatically identifies and plots the correct performance metrics (ROC-AUC) from the results dataframe.
- **Maintain Scikit-Learn API Compatibility.** The custom GPU model was designed to inherit from BaseEstimator and ClassifierMixin, ensuring it works seamlessly with existing scikit-learn pipelines and evaluation tools.
