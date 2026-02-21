# 🧠 Early Sepsis Prediction Using Time-Series LSTM

## Overview

Can we predict sepsis hours before clinical recognition — and understand *why* a model makes that prediction?

This project implements a custom LSTM-based neural network to predict sepsis onset **2h, 4h, and 6h in advance** using longitudinal laboratory measurements.

Beyond model performance, this work emphasizes **interpretability**, using permutation importance to analyze how feature relevance shifts across prediction horizons.

---

## Dataset

This project uses the **SepsisExp dataset**.

Dataset credit:  
https://www.cl.uni-heidelberg.de/statnlpgroup/sepsisexp/#data

The dataset contains:
- Time-series laboratory measurements per patient  
- Structured temporal intervals  
- Sepsis onset annotations  

All preprocessing and modeling were conducted locally. No patient-identifiable data is included in this repository.

---

## Model Architecture

Custom LSTM implemented from scratch in PyTorch:

- Input: 12 timesteps × 44 laboratory features  
- LSTM (hidden size = 64)  
- Dropout (0.3)  
- Fully connected layer (64 → 32 → 1)  
- ReLU activation  
- BCEWithLogitsLoss (with class imbalance handling)

Weight initialization:
- Xavier uniform (input weights)
- Orthogonal (recurrent weights)

---

## Training Strategy

- Patient-level train/test split (80/20)
- Adam optimizer (lr = 1e-3)
- Batch size = 256
- Early stopping (patience = 5)
- Class imbalance handled via `pos_weight`
- Evaluation on held-out test set only

Primary metric: **ROC-AUC**

---

## Performance

| Horizon | ROC-AUC |
|----------|----------|
| 2h | ~0.75 |
| 4h | ~0.73 |
| 6h | ~0.73 |

Discrimination improves closer to sepsis onset.

---

## Feature Importance (Permutation-Based)

To avoid black-box opacity, permutation importance was used to measure performance drop when individual feature trajectories were shuffled.

### Key Findings

- Feature importance shifts across prediction horizons.
- 6h emphasizes metabolic and oxygenation markers (e.g., lactate, sodium).
- 4h highlights respiratory and inflammatory features.
- 2h strengthens acute hypoperfusion indicators.
- No small subset of 3–5 laboratory features could approximate full model performance.

This suggests early sepsis detection reflects **distributed multi-system instability**, not a single dominant biomarker.

---

## Minimal Feature Subset Experiments

At 4h:

- Top 3 features → ROC-AUC ≈ 0.63  
- Top 5 features → ROC-AUC ≈ 0.61  

Performance dropped substantially compared to the full model (~0.73), indicating that compression to a narrow lab panel reduces predictive power.

---

## Repository Structure

01_EDA_Sepsis_TimeSeries.ipynb
02_Label_Construction.ipynb
03_Baseline_Models.ipynb
04_Custom_NN_Model.ipynb
05_Final_Model_Evaluation.ipynb
06_Feature_Importance_and_Minimal_Feature_Analysis.ipynb


---

## Key Takeaways

- Early sepsis prediction is a time-evolving problem.
- Interpretability is critical in clinical ML.
- Distributed feature interaction matters more than isolated biomarkers.
- Temporal feature importance analysis reveals physiologically coherent progression patterns.

---

## Future Work

- Remove structural time-index features and re-evaluate performance.
- Explore SHAP or temporal attribution methods.
- Evaluate calibration and decision-curve analysis.
- Compare with transformer-based sequence models.

---

## Research Interests

This project reflects an interest in:

- Time-series modeling  
- Clinical machine learning  
- Model interpretability  
- Early-event detection systems  

Feedback and discussion are welcome.
