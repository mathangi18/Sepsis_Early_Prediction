# Experimental Design

## Research Question
Can we predict the onset of sepsis at fixed horizons (2h, 4h, 6h) using physiological time-series data from ICU patients with high precision and clinical lead time?

## Study Population
- **Dataset:** Integrated ICU cohort from four partitions.
- **Sample size:** 1,275 patients (602,568 hourly observations).
- **Target labels:** Binary sepsis indicator (1: Sepsis onset within horizon, 0: Otherwise).

## Feature Engineering
- **Physiological Signals:** 40 laboratory markers and 5 vital signs.
- **Preprocessing:** 
    - Panel concatenation and ID verification.
    - Standard scaling for gradient-based convergence.
    - Temporal indexing for sequence-aware modeling.

## Modeling Strategy
- **Baseline:** GPU-accelerated Logistic Regression to establish linear benchmarks.
- **Main Model:** Long Short-Term Memory (LSTM) network to capture temporal dynamics.
- **Validation:** 80/20 patient-stratified split to ensure no data leakage across time-steps of the same patient.

## Evaluation Metrics
- **Primary:** Area Under the Receiver Operating Characteristic (AUROC) curve.
- **Secondary:** Precision-Recall curves to assess performance under class imbalance (~3.3:1).
- **Interpretability:** Feature importance analysis via coefficient analysis and gradient attribution.
