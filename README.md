
Sepsis Early Prediction Submission
This project implements a neural network-based solution for the early prediction of sepsis onset using time-series laboratory data. The models are designed to predict sepsis 2 hours, 4 hours, and 6 hours in advance to enable proactive clinical intervention.

Repository Structure
Notebooks/: Contains the end-to-end development pipeline.
01_EDA_Sepsis_TimeSeries.ipynb: Exploratory analysis of physiological markers.
02_Label_Construction.ipynb: Definition of sepsis onset and temporal labeling.
03_Baseline_Models.ipynb: Logistic Regression benchmarks (GPU-accelerated).
04_Custom_NN_Model.ipynb: Experimental design of the LSTM architecture.
05_Final_Model_Evaluation.ipynb: Deterministic performance analysis and ROC/PR evaluation.
06_Feature_Importance_and_Minimal_Feature_Analysis.ipynb: Permutation-based interpretability.
Data/:
raw/: Placeholder. Please place the SepsisExp dataset partitions (A-D) here.
processed/: Intermediate processed tensors (X_train, y_train, etc.) are expected here after running notebooks.
Results/:
model_weights/: Saved PyTorch model states (.pt).
figures/: Visualizations of data distributions and model performance.
tables/: CSV reports of features and results.
requirements.txt: Environment dependencies.
submission_q_and_a.txt: Detailed answers and justifications for the assignment tasks.
Environment Requirements
Python: 3.11+
Key Libraries: PyTorch, Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn.
Hardware: CUDA-enabled GPU is highly recommended for training/inference speed (though the notebooks implement fallbacks).
To set up the environment:

pip install -r requirements.txt
How to Run
Data Placement: Download the SepsisExp dataset and ensure the parquet partitions are in Data/raw/.
Preprocessing: Run 02_Label_Construction.ipynb to generate the labeled dataset for all horizons.
Inference/Evaluation: If you wish to evaluate existing models, open 05_Final_Model_Evaluation.ipynb. It will load the weights from Results/model_weights/ and perform a full evaluation on the test set.
Permutation Importance: Run 06_Feature_Importance_and_Minimal_Feature_Analysis.ipynb to see which features contribute most to the predictions.
Tech Stack
Framework: PyTorch (NN Implementation)
Architecture: LSTM (Long Short-Term Memory) to capture temporal dependencies in patient history.
Acceleration: Custom L-BFGS for Logistic Regression and Adam for LSTM, both GPU-optimized.
For further reference : https://github.com/mathangi18/Sepsis_Early_Prediction.git
