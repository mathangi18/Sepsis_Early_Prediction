
import torch
import sys

def check_cuda():
    print("Python version:", sys.version)
    print("PyTorch version:", torch.__version__)
    
    if torch.cuda.is_available():
        print(f"CUDA is available! Found {torch.cuda.device_count()} device(s).")
        for i in range(torch.cuda.device_count()):
            print(f"Device {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute Capability: {props.major}.{props.minor}")
    else:
        print("CUDA is NOT available. Running on CPU.")

if __name__ == "__main__":
    check_cuda()
