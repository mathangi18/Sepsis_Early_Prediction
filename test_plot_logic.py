import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Mock data mimicking expected results
data = {
    '2h': {'accuracy': 0.99, 'roc_auc': 0.76},
    '4h': {'accuracy': 0.99, 'roc_auc': 0.75},
    '6h': {'accuracy': 0.99, 'roc_auc': 0.75}
}
results_df = pd.DataFrame(data).T
print("Mock DataFrame created:")
print(results_df)

# --- Visualization Logic to Test ---
print("\n--- Testing Visualization Logic ---")

# Check if results_df exists (in notebook this checks locals())
if 'results_df' not in locals():
    print("Error: results_df is not defined.")
else:
    print("Columns found in results_df:", results_df.columns.tolist())
    
    # Identify the correct column name for AUC
    auc_col = None
    possible_names = ['roc_auc', 'AUC', 'auc', 'val_auc', 'roc']
    for name in possible_names:
        if name in results_df.columns:
            auc_col = name
            break
            
    if auc_col:
        print(f"Found AUC column: {auc_col}")
        # Plot
        try:
            plt.figure(figsize=(10, 6))
            sns.set_theme(style="whitegrid")
            ax = sns.barplot(x=results_df.index, y=results_df[auc_col], palette='viridis')
            
            plt.title(f'Baseline Model Performance ({auc_col}) by Horizon', fontsize=16)
            plt.xlabel('Prediction Horizon', fontsize=12)
            plt.ylabel('Score', fontsize=12)
            plt.ylim(0.5, 1.0)
            
            # Add values on bars
            for i, v in enumerate(results_df[auc_col]):
                ax.text(i, v + 0.01, f"{v:.4f}", ha='center', fontsize=11, fontweight='bold')
                
            print("Plot command executed successfully (window might not show in headless).")
            # plt.show() # Commented out for headless execution
        except Exception as e:
            print(f"FAILED to plot: {e}")
            raise e
    else:
        print(f"Could not find AUC column. Available: {results_df.columns}")
