import json
import os

path = r'c:\Users\Hope\OneDrive - KTH\StudyPeriod 3\AI & ML\Assignment2\Sepsis_Early_Prediction\Notebooks\03_Baseline_Models.ipynb'

with open(path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Validated Visualization Code
viz_code = [
    "import matplotlib.pyplot as plt\n",
    "import seaborn as sns\n",
    "import pandas as pd\n",
    "\n",
    "# Check if results_df exists\n",
    "if 'results_df' not in locals():\n",
    "    print(\"Error: results_df is not defined. Please run the training cell above.\")\n",
    "else:\n",
    "    print(\"Columns found in results_df:\", results_df.columns.tolist())\n",
    "    \n",
    "    # Identify the correct column name for AUC\n",
    "    auc_col = None\n",
    "    possible_names = ['roc_auc', 'AUC', 'auc', 'val_auc', 'roc']\n",
    "    for name in possible_names:\n",
    "        if name in results_df.columns:\n",
    "            auc_col = name\n",
    "            break\n",
    "            \n",
    "    if auc_col:\n",
    "        # Plot\n",
    "        plt.figure(figsize=(10, 6))\n",
    "        sns.set_theme(style=\"whitegrid\")\n",
    "        ax = sns.barplot(x=results_df.index, y=results_df[auc_col], palette='viridis')\n",
    "        \n",
    "        plt.title(f'Baseline Model Performance ({auc_col}) by Horizon', fontsize=16)\n",
    "        plt.xlabel('Prediction Horizon', fontsize=12)\n",
    "        plt.ylabel('Score', fontsize=12)\n",
    "        plt.ylim(0.5, 1.0)\n",
    "        \n",
    "        # Add values on bars\n",
    "        for i, v in enumerate(results_df[auc_col]):\n",
    "            ax.text(i, v + 0.01, f\"{v:.4f}\", ha='center', fontsize=11, fontweight='bold')\n",
    "            \n",
    "        plt.show()\n",
    "    else:\n",
    "        print(f\"Could not find AUC column. Available: {results_df.columns}\")\n",
    "        # Fallback: Plot first column\n",
    "        first_col = results_df.columns[0]\n",
    "        print(f\"Plotting available metric: {first_col}\")\n",
    "        \n",
    "        plt.figure(figsize=(10, 6))\n",
    "        sns.barplot(x=results_df.index, y=results_df[first_col], palette='magma')\n",
    "        plt.title(f'Baseline Model Performance ({first_col})', fontsize=16)\n",
    "        plt.show()\n",
    "\n",
    "    display(results_df) # Show table too\n"
]

# Find and replace the cell
found = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        src = "".join(cell['source'])
        # Identify the visualization cell by imports or previous content
        if "import matplotlib.pyplot" in src and ("sns.barplot" in src or "results_df" in src):
            nb['cells'][i]['source'] = viz_code
            # clear outputs
            nb['cells'][i]['outputs'] = []
            nb['cells'][i]['execution_count'] = None
            found = True
            print(f"Force updated visualization at cell index {i}")
            break

if not found:
    # Append new cell
    new_cell = {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": viz_code
    }
    nb['cells'].append(new_cell)
    print("Appended new visualization cell.")

with open(path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
