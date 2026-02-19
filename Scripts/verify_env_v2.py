
import sys
import torch
import pandas as pd
import sklearn
import matplotlib

print(f"Python Executable: {sys.executable}")
print(f"Python Version: {sys.version}")
print("-" * 30)
print(f"Pandas Version: {pd.__version__}")
print(f"Scikit-learn Version: {sklearn.__version__}")
print(f"Matplotlib Version: {matplotlib.__version__}")
print("-" * 30)
print(f"PyTorch Version: {torch.__version__}")
print(f"CUDA Available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA Device Count: {torch.cuda.device_count()}")
    print(f"Current Device: {torch.cuda.current_device()}")
    print(f"Device Name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA NOT AVAILABLE")
