import json

path = r'c:\Users\Hope\OneDrive - KTH\StudyPeriod 3\AI & ML\Assignment2\Sepsis_Early_Prediction\Notebooks\03_Baseline_Models.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
# Print ONLY code cells from 0-9
for i in range(0, 9):
    c = cells[i]
    if c['cell_type'] == 'code':
        src = ''.join(c['source'])
        print(f'\n{"="*60}')
        print(f'Cell {i+1} (code):')
        print(src)
