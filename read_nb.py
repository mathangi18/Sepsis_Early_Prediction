import json

path = r'c:\Users\Hope\OneDrive - KTH\StudyPeriod 3\AI & ML\Assignment2\Sepsis_Early_Prediction\Notebooks\03_Baseline_Models.ipynb'
with open(path, encoding='utf-8') as f:
    nb = json.load(f)

cells = nb['cells']
print(f'Total cells: {len(cells)}')
# Print cells 8-12 (context around cell 10)
for i in range(7, min(12, len(cells))):
    c = cells[i]
    src = ''.join(c['source'])
    print(f'\n{"="*60}')
    print(f'Cell {i+1} ({c["cell_type"]}):')
    print(src)
