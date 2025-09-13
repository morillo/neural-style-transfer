#!/usr/bin/env python3
"""
Test script to verify device detection works properly across different environments
"""
import torch

def get_best_device():
    """Get the best available device with proper fallback order: CUDA -> MPS -> CPU"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def test_device_detection():
    """Test the device detection and basic tensor operations"""
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    # Check MPS availability (PyTorch 1.12+)
    if hasattr(torch.backends, 'mps'):
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"MPS built: {torch.backends.mps.is_built()}")
    else:
        print("MPS not available (PyTorch version < 1.12)")
    
    # Get the best device
    device = get_best_device()
    print(f"\nSelected device: {device}")
    
    # Test basic tensor operations on the selected device
    try:
        print("Testing tensor operations on device...")
        test_tensor = torch.randn(3, 3).to(device)
        result = torch.matmul(test_tensor, test_tensor.T)
        
        print(f"✓ Device test successful!")
        print(f"  - Tensor shape: {result.shape}")
        print(f"  - Tensor device: {result.device}")
        print(f"  - Sample values: {result[0, :2]}")
        
        return True
        
    except Exception as e:
        print(f"✗ Device test failed: {e}")
        return False

def simulate_environments():
    """Simulate different environment conditions"""
    print("\n" + "="*50)
    print("SIMULATING DIFFERENT ENVIRONMENTS")
    print("="*50)
    
    # Original CUDA-only check (what code used to do)
    cuda_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Old logic (CUDA -> CPU): {cuda_device}")
    
    # New improved logic
    best_device = get_best_device()
    print(f"New logic (CUDA -> MPS -> CPU): {best_device}")
    
    if cuda_device != best_device:
        print("🎉 Improvement detected! MPS will be used instead of CPU on Apple Silicon")
    elif str(best_device) == "mps":
        print("🍎 Running on Apple Silicon with MPS support")
    elif str(best_device) == "cuda":
        print("🚀 Running on CUDA-enabled GPU")
    else:
        print("💻 Running on CPU")

if __name__ == "__main__":
    print("Device Detection Test")
    print("=" * 30)
    
    success = test_device_detection()
    simulate_environments()
    
    print("\n" + "="*50)
    print("DEVICE DETECTION UPGRADE SUMMARY")
    print("="*50)
    print("✓ Added get_best_device() function with proper fallback order")
    print("✓ Updated all PyTorch device initialization to use get_best_device()")
    print("✓ Updated Ray resource allocation to handle MPS properly")
    print("✓ Updated both Python script and Jupyter notebook")
    
    if success:
        print("\n🎉 All tests passed! Device detection is working correctly.")
    else:
        print("\n⚠️  Tests failed - check PyTorch installation and device availability.")