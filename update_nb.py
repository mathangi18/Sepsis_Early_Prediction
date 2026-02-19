import json
import os

path = r'c:\Users\Hope\OneDrive - KTH\StudyPeriod 3\AI & ML\Assignment2\Sepsis_Early_Prediction\Notebooks\03_Baseline_Models.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Cell 10 is index 9
cell = nb['cells'][9]

# New source code for Cell 10
new_source = [
    "import time\n",
    "\n",
    "print(f\"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features...\")\n",
    "start_time = time.time()\n",
    "\n",
    "model = LogisticRegression(  \n",
    "    max_iter=1000,\n",
    "    class_weight='balanced', \n",
    "    solver='saga',  # SAGA is faster for large datasets\n",
    "    n_jobs=-1,      # Use all available CPU cores\n",
    "    random_state=42,\n",
    "    verbose=1\n",
    ")\n",
    "\n",
    "model.fit(X_train, y_train)  \n",
    "\n",
    "duration = time.time() - start_time\n",
    "print(f\"Model trained successfully in {duration:.2f} seconds.\")"
]

cell['source'] = new_source

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print("Notebook updated.")
