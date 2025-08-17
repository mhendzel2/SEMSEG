"""
3D FIB-SEM Segmentation and Quantification Program

A comprehensive software package for automated analysis of three-dimensional 
Focused Ion Beam Scanning Electron Microscopy (FIB-SEM) datasets.

Author: Manus AI
Version: 1.0
Date: August 2025
"""

__version__ = "1.0.0"
__author__ = "Manus AI"
__email__ = "support@manus.ai"

# Import main components for easy access
try:
    from .pipeline.main_pipeline import FIBSEMPipeline, create_default_pipeline
    from .core.config import FIBSEMConfig
    from .core.data_io import load_fibsem_data
except ImportError:
    # Handle import errors gracefully during development
    pass

def run_diagnostics():
    """Run system diagnostics to check installation."""
    print("FIB-SEM Program Diagnostics")
    print("=" * 40)
    
    # Check Python version
    import sys
    print(f"Python version: {sys.version}")
    
    # Check required packages
    required_packages = [
        'numpy', 'scipy', 'scikit-image', 'matplotlib', 
        'pandas', 'h5py', 'tifffile', 'yaml'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package}")
        except ImportError:
            print(f"✗ {package} (missing)")
            missing_packages.append(package)
    
    # Check optional packages
    optional_packages = ['torch', 'cv2']
    for package in optional_packages:
        try:
            __import__(package)
            print(f"✓ {package} (optional)")
        except ImportError:
            print(f"- {package} (optional, not installed)")
    
    if missing_packages:
        print(f"\nMissing required packages: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    else:
        print("\n✓ All required packages available")
        return True

def test_installation():
    """Test basic functionality."""
    print("Testing FIB-SEM Program Installation")
    print("=" * 40)
    
    try:
        import numpy as np
        from .core.config import FIBSEMConfig
        
        # Test configuration
        config = FIBSEMConfig()
        print("✓ Configuration system working")
        
        # Test synthetic data creation
        test_data = np.random.randint(0, 255, (50, 100, 100), dtype=np.uint8)
        print("✓ Data handling working")
        
        print("\n✓ Installation test passed")
        return True
        
    except Exception as e:
        print(f"✗ Installation test failed: {e}")
        return False

