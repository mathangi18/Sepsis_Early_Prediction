import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from sklearn.metrics import roc_curve, roc_auc_score, precision_recall_curve, auc

class SepsisLSTM(torch.nn.Module):
    def __init__(self, input_size=44, hidden_size=64, dropout=0.3):
        super().__init__()
        self.lstm = torch.nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout = torch.nn.Dropout(dropout)
        self.fc1 = torch.nn.Linear(hidden_size, 32)
        self.relu = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(32, 1)
        
    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        out = self.dropout(out)
        out = self.fc1(out)
        out = self.relu(out)
        out = self.fc2(out)
        return out

def export_plots():
    base = Path('../Results/processed_tensors')
    mpath = Path('../Results/model_weights')
    fpath = Path('../Results/figures')
    fpath.mkdir(parents=True, exist_ok=True)
    
    plt.figure(figsize=(12, 6))
    
    # ROC Curves
    plt.subplot(1, 2, 1)
    for h in ['2h', '4h', '6h']:
        X = torch.load(base/f'X_test_{h}.pt', map_location='cpu')
        y = torch.load(base/f'y_test_{h}.pt', map_location='cpu')
        
        m = SepsisLSTM()
        m.load_state_dict(torch.load(mpath/f'lstm_{h}.pt', map_location='cpu'))
        m.eval()
        
        with torch.no_grad():
            p = torch.sigmoid(m(X)).numpy()
            
        fpr, tpr, _ = roc_curve(y, p)
        plt.plot(fpr, tpr, label=f'{h} (AUC={roc_auc_score(y, p):.4f})')
        
    plt.plot([0, 1], [0, 1], 'k--')
    plt.legend()
    plt.title('ROC Curves Across Horizons')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    
    # PR Curves
    plt.subplot(1, 2, 2)
    for h in ['2h', '4h', '6h']:
        X = torch.load(base/f'X_test_{h}.pt', map_location='cpu')
        y = torch.load(base/f'y_test_{h}.pt', map_location='cpu')
        
        m = SepsisLSTM()
        m.load_state_dict(torch.load(mpath/f'lstm_{h}.pt', map_location='cpu'))
        m.eval()
        
        with torch.no_grad():
            p = torch.sigmoid(m(X)).numpy()
            
        pr, re, _ = precision_recall_curve(y, p)
        plt.plot(re, pr, label=f'{h} (AUC={auc(re, pr):.4f})')
        
    plt.legend()
    plt.title('PR Curves Across Horizons')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    plt.tight_layout()
    plt.savefig(fpath/'combined_evaluation_curves.png')
    print('Saved Results/figures/combined_evaluation_curves.png')

if __name__ == '__main__':
    export_plots()
