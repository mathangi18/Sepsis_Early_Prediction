import json
import os
from pathlib import Path

notebook_path = Path(r"c:/Users/Hope/OneDrive - KTH/StudyPeriod 3/AI & ML/Assignment2/Sepsis_Early_Prediction/Notebooks/04_Custom_NN_Model.ipynb")

# Read the notebook
with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Define the header markdown cell
markdown_cell = {
    "cell_type": "markdown",
    "id": "export_header",
    "metadata": {},
    "source": [
        "# Stage 5 — Export Processed Tensors & Scalers\\n",
        "\\n",
        "We export the final processed tensors and scalers to the Results directory \\n",
        "for use in downstream analysis, model serving, or cross-horizon evaluation.\\n",
        "\\n",
        "Directories are created automatically if they do not exist."
    ]
}

# Define the export code cell
code_cell = {
    "cell_type": "code",
    "execution_count": None,
    "id": "export_cell",
    "metadata": {},
    "outputs": [],
    "source": [
        "import torch\\n",
        "import pickle\\n",
        "import os\\n",
        "from pathlib import Path\\n",
        "\\n",
        "# 1. Create Export Directory\\n",
        "EXPORT_DIR = Path(\"../Results/processed_tensors/\")\\n",
        "EXPORT_DIR.mkdir(parents=True, exist_ok=True)\\n",
        "\\n",
        "print(f\"Export Directory: {EXPORT_DIR.absolute()}\")\\n",
        "\\n",
        "# 2. Define Export Mapping\\n",
        "# Mapping internal variables to (filename, type)\\n",
        "export_plan = [\\n",
        "    # 2h Horizon\\n",
        "    ('X_train_tensor', \\\"X_train_2h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('y_train_tensor', \\\"y_train_2h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('X_test_tensor', \\\"X_test_2h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('y_test_tensor', \\\"y_test_2h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('scaler', \\\"scaler_2h.pkl\\\", \\\"scaler\\\"),\\n",
        "    \\n",
        "    # 4h Horizon\\n",
        "    ('X_train_tensor_4h', \\\"X_train_4h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('y_train_tensor_4h', \\\"y_train_4h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('X_test_tensor_4h', \\\"X_test_4h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('y_test_tensor_4h', \\\"y_test_4h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('scaler_4h', \\\"scaler_4h.pkl\\\", \\\"scaler\\\"),\\n",
        "    \\n",
        "    # 6h Horizon\\n",
        "    ('X_train_tensor_6h', \\\"X_train_6h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('y_train_tensor_6h', \\\"y_train_6h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('X_test_tensor_6h', \\\"X_test_6h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('y_test_tensor_6h', \\\"y_test_6h.pt\\\", \\\"tensor\\\"),\\n",
        "    ('scaler_6h', \\\"scaler_6h.pkl\\\", \\\"scaler\\\")\\n",
        "]\\n",
        "\\n",
        "print(\\\"\\\\nStarting Export...\\\")\\n",
        "\\n",
        "# 3. Execute Export\\n",
        "for var_name, filename, obj_type in export_plan:\\n",
        "    save_path = EXPORT_DIR / filename\\n",
        "    \\n",
        "    # Check if variable exists in globals\\n",
        "    if var_name not in globals():\\n",
        "        print(f\\\"[WARNING] Variable {var_name} not found in memory. Skipping.\\\")\\n",
        "        continue\\n",
        "        \\n",
        "    obj = globals()[var_name]\\n",
        "    \\n",
        "    if obj_type == \\\"tensor\\\":\\n",
        "        # Ensure float32 and move to CPU before saving\\n",
        "        tensor_copy = obj.detach().cpu().to(torch.float32)\\n",
        "        torch.save(tensor_copy, save_path)\\n",
        "    else:\\n",
        "        # Save scalers using pickle\\n",
        "        with open(save_path, 'wb') as f:\\n",
        "            pickle.dump(obj, f)\\n",
        "            \\n",
        "    # Print confirmation with file size\\n",
        "    file_size_mb = os.path.getsize(save_path) / (1024 * 1024)\\n",
        "    print(f\\\"- Saved: {filename:<20} | Size: {file_size_mb:>8.2f} MB\\\")\\n",
        "\\n",
        "print(\\\"\\\\nSuccess: All available tensors and scalers exported successfully.\\\")"
    ]
}

# Append the cells
nb['cells'].append(markdown_cell)
nb['cells'].append(code_cell)

# Write the notebook back
with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)

print(f"Successfully added export cell to {notebook_path.name}")
