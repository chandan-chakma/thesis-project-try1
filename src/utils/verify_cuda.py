import torch

def verify_cuda_setup():
    print("\n=== CUDA Setup Verification ===")
    print(f"PyTorch Version: {torch.__version__}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA Version: {torch.version.cuda}")
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")
        
        # Test CUDA memory allocation
        try:
            x = torch.rand(1000, 1000).cuda()
            print("Successfully allocated CUDA tensor")
            del x
            torch.cuda.empty_cache()
        except RuntimeError as e:
            print(f"Error allocating CUDA tensor: {e}")
            
        # Print GPU memory info
        print(f"Total GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        print(f"Current Memory Usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        print(f"Max Memory Usage: {torch.cuda.max_memory_allocated() / 1e9:.2f} GB")
    else:
        print("CUDA is not available. Please check your PyTorch installation.")

if __name__ == "__main__":
    verify_cuda_setup() 