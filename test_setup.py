#!/usr/bin/env python3
"""
Quick test script to verify the installation and basic functionality.
"""

import sys
import importlib

def test_imports():
    """Test all required imports."""
    packages = [
        'mlx.core',
        'numpy', 
        'cv2',
        'matplotlib.pyplot',
        'sklearn',
        'pandas',
        'tqdm',
        'PIL'
    ]
    
    failed_imports = []
    
    for package in packages:
        try:
            importlib.import_module(package)
            print(f"✓ {package}")
        except ImportError as e:
            print(f"✗ {package}: {e}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\nFailed imports: {failed_imports}")
        return False
    else:
        print("\nAll packages imported successfully!")
        return True

def test_mlx_functionality():
    """Test basic MLX functionality."""
    try:
        import mlx.core as mx
        
        # Test basic operations
        a = mx.array([1, 2, 3, 4, 5])
        b = mx.array([2, 3, 4, 5, 6])
        c = a + b
        
        print(f"MLX array addition: {a} + {b} = {c}")
        
        # Test random generation
        random_array = mx.random.uniform(0, 1, (3, 3))
        print(f"MLX random array:\n{random_array}")
        
        return True
        
    except Exception as e:
        print(f"MLX functionality test failed: {e}")
        return False

def test_basic_cv_operations():
    """Test basic computer vision operations."""
    try:
        import numpy as np
        import cv2
        
        # Create a test image
        img = np.zeros((100, 100), dtype=np.uint8)
        cv2.rectangle(img, (20, 20), (80, 80), 255, 2)
        
        # Test Gaussian blur
        blurred = cv2.GaussianBlur(img, (5, 5), 1.0)
        
        print("✓ Basic CV operations working")
        return True
        
    except Exception as e:
        print(f"CV operations test failed: {e}")
        return False

def main():
    """Run all tests."""
    print("Testing Computer Vision Environment Setup")
    print("=" * 50)
    
    # Test imports
    print("Testing imports...")
    imports_ok = test_imports()
    
    if not imports_ok:
        print("Some imports failed. Please install missing packages.")
        sys.exit(1)
    
    print("\nTesting MLX functionality...")
    mlx_ok = test_mlx_functionality()
    
    print("\nTesting CV operations...")
    cv_ok = test_basic_cv_operations()
    
    if imports_ok and mlx_ok and cv_ok:
        print("\n" + "=" * 50)
        print("✓ All tests passed! Environment is ready.")
        print("Run 'python main.py' to start the full pipeline.")
        print("Run 'python demo.py' for demonstrations.")
    else:
        print("\n" + "=" * 50)
        print("✗ Some tests failed. Please check your installation.")

if __name__ == "__main__":
    main()
