import json

path = r'c:\Users\Hope\OneDrive - KTH\StudyPeriod 3\AI & ML\Assignment2\Sepsis_Early_Prediction\Notebooks\03_Baseline_Models.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

found_scaler = False
for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code':
        source = ''.join(cell['source'])
        if 'Scaler' in source or 'scale' in source:
            print(f"Found scaling in Cell {i+1}:")
            print(source)
            found_scaler = True

if not found_scaler:
    print("WARNING: No explicit scaling (StandardScaler/MinMaxScaler) found in code cells.")
