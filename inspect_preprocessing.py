import json

path = r'c:\Users\Hope\OneDrive - KTH\StudyPeriod 3\AI & ML\Assignment2\Sepsis_Early_Prediction\Notebooks\03_Baseline_Models.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
# Print cells 1-9 to see data loading and preprocessing
for i in range(0, 9):
    c = cells[i]
    src = ''.join(c['source'])
    print(f'\n{"="*60}')
    print(f'Cell {i+1} ({c["cell_type"]}):')
    print(src)
