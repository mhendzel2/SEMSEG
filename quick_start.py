#!/usr/bin/env python
"""
Quick Start Script for SEMSEG
Run this to see a simple example of the segmentation workflow
"""

import sys
import os
import numpy as np

# Add SEMSEG to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def quick_start_demo():
    """Run a quick demonstration of SEMSEG capabilities"""
    
    print("="*70)
    print(" SEMSEG - Quick Start Demo")
    print("="*70)
    
    # Import core modules
    print("\n1. Importing SEMSEG modules...")
    try:
        from core import segmentation, preprocessing, config
        from core.config import FIBSEMConfig
        print("   ✓ Modules imported successfully")
    except ImportError as e:
        print(f"   ✗ Import error: {e}")
        print("\n   Make sure you're in the SEMSEG directory!")
        return False
    
    # Create sample data
    print("\n2. Creating sample 3D data (50x256x256)...")
    np.random.seed(42)
    # Create synthetic 3D data with structures
    data = np.zeros((50, 256, 256), dtype=np.float32)
    
    # Add some circular structures
    for i in range(5):
        z = np.random.randint(10, 40)
        y = np.random.randint(50, 206)
        x = np.random.randint(50, 206)
        radius = np.random.randint(15, 30)
        
        for dz in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                for dx in range(-radius, radius+1):
                    if dz**2 + dy**2 + dx**2 <= radius**2:
                        if (0 <= z+dz < 50 and 0 <= y+dy < 256 and 0 <= x+dx < 256):
                            data[z+dz, y+dy, x+dx] = 0.8
    
    # Add noise
    data += np.random.normal(0, 0.1, data.shape)
    data = np.clip(data, 0, 1)
    
    print(f"   ✓ Created synthetic data: shape={data.shape}, dtype={data.dtype}")
    
    # Preprocessing
    print("\n3. Preprocessing data...")
    try:
        preprocessed = preprocessing.remove_noise(data, method='gaussian', sigma=1.0)
        print("   ✓ Applied Gaussian noise reduction")
        
        preprocessed = preprocessing.enhance_contrast(preprocessed, method='clahe')
        print("   ✓ Applied CLAHE contrast enhancement")
    except Exception as e:
        print(f"   ⚠ Preprocessing note: {e}")
        preprocessed = data
    
    # Traditional segmentation
    print("\n4. Testing traditional segmentation methods...")
    
    methods_to_test = [
        ('watershed', {'min_distance': 20, 'threshold_rel': 0.6}),
        ('region_growing', {'seed_threshold': 0.5, 'growth_threshold': 0.1}),
        ('slic', {'n_segments': 100, 'compactness': 10.0}),
    ]
    
    results = {}
    for method, params in methods_to_test:
        try:
            print(f"   - {method}...", end=" ")
            labels = segmentation.segment_traditional(preprocessed, method, params)
            num_labels = len(np.unique(labels)) - 1  # Subtract background
            results[method] = num_labels
            print(f"✓ Found {num_labels} regions")
        except Exception as e:
            print(f"⚠ {e}")
            results[method] = None
    
    # Configuration
    print("\n5. Testing configuration system...")
    try:
        config = FIBSEMConfig()
        watershed_params = config.get_segmentation_params('watershed', 'traditional')
        print(f"   ✓ Config loaded. Watershed params: {watershed_params}")
    except Exception as e:
        print(f"   ⚠ Config note: {e}")
    
    # Summary
    print("\n" + "="*70)
    print(" Demo Complete!")
    print("="*70)
    print("\nSegmentation Results:")
    for method, count in results.items():
        if count is not None:
            print(f"  • {method:20s}: {count:4d} regions")
        else:
            print(f"  • {method:20s}: (not available)")
    
    print("\n" + "="*70)
    print("What's Next?")
    print("="*70)
    print("\n1. Try with your own data:")
    print("   from core.data_io import load_fibsem_data")
    print("   data = load_fibsem_data('your_data.tif')")
    
    print("\n2. Launch the GUI:")
    print("   python launch_gui.py")
    
    print("\n3. Read the guides:")
    print("   - HOW_TO_RUN.md - All methods to run SEMSEG")
    print("   - SEGMENTATION_GUIDE.md - Detailed method descriptions")
    print("   - INSTALLATION.md - Usage examples")
    
    print("\n4. Explore all segmentation methods:")
    traditional_methods = [
        'watershed', 'region_growing', 'graph_cuts', 'active_contours',
        'slic', 'felzenszwalb', 'random_walker', 'thresholding', 'morphology'
    ]
    dl_methods = ['unet_2d', 'unet_3d', 'vnet', 'attention_unet', 'nnunet']
    
    print(f"\n   Traditional: {', '.join(traditional_methods)}")
    print(f"   Deep Learning: {', '.join(dl_methods)}")
    
    print("\n" + "="*70)
    
    return True

def show_quick_help():
    """Show quick usage examples"""
    print("\n" + "="*70)
    print(" Quick Usage Examples")
    print("="*70)
    
    print("\n## Launch GUI:")
    print("   python launch_gui.py")
    
    print("\n## Run from command line:")
    print("   python -m __main__ --run data.tif --method region_growing")
    
    print("\n## Use in Python script:")
    print("   from core.segmentation import segment_traditional")
    print("   labels = segment_traditional(data, 'watershed', params)")
    
    print("\n## Get help:")
    print("   python -m __main__ --help")
    print("   python verify_installation.py")
    
    print("\n" + "="*70 + "\n")

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="SEMSEG Quick Start")
    parser.add_argument("--demo", action="store_true", help="Run full demo with synthetic data")
    parser.add_argument("--help-quick", action="store_true", help="Show quick usage examples")
    
    args = parser.parse_args()
    
    if args.help_quick:
        show_quick_help()
    elif args.demo:
        quick_start_demo()
    else:
        # Default: run demo
        print("\nRunning quick start demo...")
        print("(Use --help-quick for usage examples only)\n")
        quick_start_demo()
