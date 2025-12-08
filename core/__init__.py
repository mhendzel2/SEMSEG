"""
Core functionality for FIB-SEM data processing.

This module contains the fundamental components for data I/O, preprocessing,
and configuration management.
"""

from .data_io import load_fibsem_data, save_fibsem_data, FIBSEMData
from .config import FIBSEMConfig
from .preprocessing import (
    remove_noise, enhance_contrast, remove_artifacts,
    correct_drift, normalize_intensity
)
from .segmentation import (
    segment_traditional, segment_deep_learning, 
    check_dependencies, SegmentationResult
)

__all__ = [
    'load_fibsem_data', 'save_fibsem_data', 'FIBSEMData',
    'FIBSEMConfig',
    'remove_noise', 'enhance_contrast', 'remove_artifacts',
    'correct_drift', 'normalize_intensity',
    'segment_traditional', 'segment_deep_learning',
    'check_dependencies', 'SegmentationResult'
]


def check_all_dependencies() -> dict:
    """
    Check all optional dependencies at startup.
    
    Returns:
        Dictionary with dependency status grouped by category:
        - 'required': Core required packages
        - 'deep_learning': TensorFlow, PyTorch, etc.
        - 'optional': Graph-cuts, SAM3, etc.
    """
    from .segmentation import check_dependencies
    
    required_deps = ['numpy', 'scipy', 'skimage']
    deep_learning_deps = ['tensorflow', 'torch']
    optional_deps = ['maxflow', 'sam3', 'nibabel', 'h5py', 'zarr']
    
    return {
        'required': check_dependencies(required_deps),
        'deep_learning': check_dependencies(deep_learning_deps),
        'optional': check_dependencies(optional_deps)
    }


def print_dependency_status():
    """Print a formatted summary of dependency availability."""
    status = check_all_dependencies()
    
    print("\n" + "=" * 50)
    print("SEMSEG Dependency Status")
    print("=" * 50)
    
    for category, deps in status.items():
        print(f"\n{category.upper()}:")
        for dep, available in deps.items():
            symbol = "✓" if available else "✗"
            print(f"  {symbol} {dep}")
    
    print("\n" + "=" * 50 + "\n")

