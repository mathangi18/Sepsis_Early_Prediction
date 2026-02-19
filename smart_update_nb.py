import json
import os

path = r'c:\Users\Hope\OneDrive - KTH\StudyPeriod 3\AI & ML\Assignment2\Sepsis_Early_Prediction\Notebooks\03_Baseline_Models.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Content signatures
def is_import_cell(source):
    return "import pandas" in source and "sklearn.linear_model" in source and "def " not in source

def is_single_model_cell(source):
    return "model = LogisticRegression" in source and "fit(" in source and "def " not in source

def is_run_baseline_cell(source):
    return "def run_baseline" in source

# New sources
new_imports = [
    "import pandas as pd\n",
    "import numpy as np\n",
    "from pathlib import Path\n",
    "import torch\n",
    "import time\n",
    "import sys\n",
    "\n",
    "# Add parent directory to path to import Scripts\n",
    "sys.path.append('..')\n",
    "from Scripts.gpu_models import PyTorchLogisticRegression\n",
    "\n",
    "from sklearn.model_selection import train_test_split\n",
    "from sklearn.preprocessing import StandardScaler\n",
    "from sklearn.pipeline import Pipeline\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.metrics import (\n",
    "    accuracy_score,\n",
    "    precision_score,\n",
    "    recall_score,\n",
    "    roc_auc_score,\n",
    "    confusion_matrix\n",
    ")\n",
    "\n",
    "print(f\"PyTorch Version: {torch.__version__}\")\n",
    "print(f\"CUDA Available: {torch.cuda.is_available()}\")\n",
    "if torch.cuda.is_available():\n",
    "    print(f\"Device: {torch.cuda.get_device_name(0)}\")\n"
]

new_single_model = [
    "import time\n",
    "\n",
    "print(f\"Training on {X_train.shape[0]} samples with {X_train.shape[1]} features...\")\n",
    "start_time = time.time()\n",
    "\n",
    "# Use PyTorch Logistic Regression on GPU\n",
    "pipeline = Pipeline([\n",
    "    ('scaler', StandardScaler()),\n",
    "    ('clf', PyTorchLogisticRegression(\n",
    "        max_iter=1000,\n",
    "        verbose=True,\n",
    "        device='cuda'\n",
    "    ))\n",
    "])\n",
    "\n",
    "pipeline.fit(X_train, y_train)\n",
    "model = pipeline\n",
    "\n",
    "duration = time.time() - start_time\n",
    "print(f\"Model trained successfully in {duration:.2f} seconds.\")"
]

new_run_baseline = [
    "def run_baseline(df, label_col):\n",
    "    import time\n",
    "    \n",
    "    # Patient-level split\n",
    "    patient_ids = df['id'].unique()\n",
    "    stratify_labels = df.groupby('id')[label_col].max()\n",
    "    \n",
    "    train_ids, test_ids = train_test_split(\n",
    "        patient_ids,\n",
    "        test_size=0.2,\n",
    "        random_state=42,\n",
    "        stratify=stratify_labels\n",
    "    )\n",
    "    \n",
    "    train_df = df[df['id'].isin(train_ids)]\n",
    "    test_df = df[df['id'].isin(test_ids)]\n",
    "    \n",
    "    # Features\n",
    "    feature_cols = df.columns.drop(['id', 'sepsis', label_col])\n",
    "    \n",
    "    X_train = train_df[feature_cols]\n",
    "    y_train = train_df[label_col]\n",
    "    X_test = test_df[feature_cols]\n",
    "    y_test = test_df[label_col]\n",
    "    \n",
    "    # GPU Logistic Regression with Scaling\n",
    "    print(f\"Training for {label_col} on GPU...\")\n",
    "    start = time.time()\n",
    "    \n",
    "    pipeline = Pipeline([\n",
    "        ('scaler', StandardScaler()),\n",
    "        ('clf', PyTorchLogisticRegression(\n",
    "            max_iter=1000,\n",
    "            device='cuda'\n",
    "        ))\n",
    "    ])\n",
    "    \n",
    "    pipeline.fit(X_train, y_train)\n",
    "    \n",
    "    print(f\"Training took {time.time() - start:.2f} seconds\")\n",
    "    \n",
    "    y_pred = pipeline.predict(X_test)\n",
    "    y_prob = pipeline.predict_proba(X_test)[:, 1]\n",
    "    \n",
    "    return {\n",
    "        \"accuracy\": accuracy_score(y_test, y_pred),\n",
    "        \"precision\": precision_score(y_test, y_pred, zero_division=0),\n",
    "        \"recall\": recall_score(y_test, y_pred),\n",
    "        \"roc_auc\": roc_auc_score(y_test, y_prob)\n",
    "    }\n"
]

updated_counts = {"import": 0, "single": 0, "func": 0}

for cell in nb['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        
        if is_import_cell(source):
            cell['source'] = new_imports
            updated_counts["import"] += 1
            print("Updated imports cell.")
            
        elif is_single_model_cell(source):
            cell['source'] = new_single_model
            updated_counts["single"] += 1
            print("Updated single model cell.")
            
        elif is_run_baseline_cell(source):
            cell['source'] = new_run_baseline
            updated_counts["func"] += 1
            print("Updated run_baseline start cell.")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Update complete. Stats: {updated_counts}")
