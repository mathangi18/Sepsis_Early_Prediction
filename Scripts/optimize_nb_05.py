import json

def optimize_nb_05(nb_path):
    print(f"Reading {nb_path}...")
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    for cell in nb['cells']:
        src_text = ''.join(cell['source'])
        
        # 1. Update Imports to include gc
        if 'import torch' in src_text and 'torch.nn' in src_text:
            if 'import gc' not in src_text:
                cell['source'].insert(0, 'import gc\n')
        
        # 2. Add memory management to train_from_frozen
        if 'def train_from_frozen(horizon):' in src_text:
            lines = cell['source']
            new_lines = []
            for line in lines:
                # Add cleanup inside the validation loop
                if 'preds.extend(probs.cpu().numpy())' in line:
                    new_lines.append(line)
                    new_lines.append('                del xb, outputs, probs\n')
                    continue
                
                # Add cleanup at the end of the function
                if 'return best_auc' in line:
                    new_lines.append('\n    # Final cleanup to prevent kernel crash\n')
                    new_lines.append('    del X_train, y_train, X_test, y_test, train_loader, test_loader, model, optimizer, criterion\n')
                    new_lines.append('    gc.collect()\n')
                    new_lines.append('    torch.cuda.empty_cache()\n')
                    new_lines.append(line)
                    continue
                
                new_lines.append(line)
            cell['source'] = new_lines

    print(f"Writing optimized notebook to {nb_path}...")
    with open(nb_path, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)
    print("Optimization complete.")

if __name__ == "__main__":
    path = r"c:\Users\Hope\OneDrive - KTH\StudyPeriod 3\AI & ML\Assignment2\Sepsis_Early_Prediction\Notebooks\05_Final_Model_Evaluation.ipynb"
    optimize_nb_05(path)
